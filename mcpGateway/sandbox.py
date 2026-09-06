# -*- coding: utf-8 -*-
"""M4（D8 沙箱）进程级资源限制——跨平台双轨实现。

设计依据：`设计/06_横切能力设计.md` §2.5（M4 文档同步项）/
`设计/02_现状盘点与缺陷分析.md` D8 / `设计/11_开工顺序清单.md` M4。

背景（为什么是双轨，而不是一套代码）：
- M3 常驻化后，3 个 MCP server（sql_bill / llm_base / llm_finance）是**独立常驻子进程**，
  工具在 server 进程内执行（`startup/bootstrap.py:start_mcp_servers` Popen 拉起）。
  常驻 = 失去"短连接一次一进程"的天然隔离（`03` §9.8.2 ④ 风险1），一个失控工具
  （爆内存 / 死循环）会拖垮整个 server → 所有 Agent 一起挂。
- 进程级资源限制必须**在子进程创建时或创建后立即**施加：
  - **Windows**：进程无法自我限制内存（无 POSIX `resource` 模块）→ 由父进程用
    **Job Object** 给子进程挂硬上限（`JOB_OBJECT_LIMIT_PROCESS_MEMORY`），超限即被 OS 终止。
    Job 由 bootstrap 在 `subprocess.Popen` 后立刻挂载。
  - **POSIX**：无法事后约束已启动的子进程（只能 preexec_fn 或子进程自设）→ server 进程
    启动时 **self-prison**（`resource.setrlimit`），3 个 server 的 `__main__` 各调一次。

★ 沙箱定位：**加固而非故障点**。所有施加动作失败只告警、不阻断启动——
可用性优先，沙箱绝不成为"给现有功能引入的新单点"。

★ **已知覆盖边界**：stdio 回滚模式（`BILLAGENT_MCP_STDIO_FALLBACK=1`）下 server 由
`client.py` 的 `stdio_client` 按需 Popen 拉起，**不经 bootstrap** → 不挂 Job Object。
该模式仅用于联调排障（`03` §9.8.2 ④），生产主路径为常驻 HTTP（沙箱生效）。

★ CPU 时间为何不做进程级限制：常驻 server 的 CPU 是正常工作属性（uvicorn 并发服务），
设 `RLIMIT_CPU` 会把正常长跑的 server 误杀。CPU 失控由**引擎层工具级超时**
（`ToolSpec.timeout`，async 可取消，`08` §1.7）承担；`cpu_seconds` 参数仅面向
未来一次性执行场景（D8 "代码执行用 subprocess + 受限工作目录"），常驻路径不传。
"""
import ctypes
import os

from loguru import logger

# 默认内存上限（MB）：未显式传参时回退 config.SANDBOX_MEMORY_MB（环境变量可配）
_DEFAULT_MEMORY_MB = 2048

# ── Windows Job Object 常量 ──
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100   # 进程内存在限制（ProcessMemoryLimit 生效）
_JOB_OBJECT_EXTENDED_LIMIT_INFO = 9             # JobObjectExtendedLimitInformation
_PROCESS_SET_QUOTA = 0x0100                     # AssignProcessToJobObject 所需权限
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_INFORMATION = 0x0400

# 已挂载 Job 的句柄表（pid → job handle）。★不能 CloseHandle：Job 句柄关闭即失效，
# 限制随之解除；句柄随 bootstrap 进程生命周期自然回收即可。
_ACTIVE_JOBS: dict[int, int] = {}


# ═══════════ Job Object 结构体（Windows x64，ctypes 按自然对齐）═══════════
# 参考 winnt.h：BasicLimitInformation 64B + IO_COUNTERS 48B + 4 个 SIZE_T = 144B。
class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _memory_mb_from_config() -> int:
    """沙箱内存上限的单一来源：config.SANDBOX_MEMORY_MB（环境变量可配）。"""
    try:
        from config.config import SANDBOX_MEMORY_MB  # noqa: PLC0415
        return SANDBOX_MEMORY_MB
    except Exception:  # pragma: no cover —— config 缺失时兜底，不阻断沙箱逻辑
        return _DEFAULT_MEMORY_MB


def _resolve_memory_mb(memory_mb: int | None) -> int:
    """显式传参优先；None → config.SANDBOX_MEMORY_MB；config 缺失 → 模块默认。"""
    if memory_mb is not None:
        return memory_mb
    return _memory_mb_from_config()


def apply_process_limits(memory_mb: int | None = None,
                         cpu_seconds: int | None = None) -> bool:
    """当前进程自我限制（self-prison）——由 3 个 MCP server 的 `__main__` 启动时调用。

    - POSIX：`RLIMIT_AS`（虚拟地址空间硬上限，防爆内存）+ `RLIMIT_CORE=0`（防 core dump 写盘）；
      显式传 `cpu_seconds` 时叠加 `RLIMIT_CPU`（仅面向一次性执行场景，常驻 server 不传）。
    - Windows：无 `resource` 模块 → no-op 告警，限制由父进程 Job Object 承担（`apply_job_object_limits`）。

    :return: True=限制已生效；False=平台不支持 / 被显式关闭 / 施加失败（只告警不抛出）。
    """
    mb = _resolve_memory_mb(memory_mb)
    if mb <= 0:
        logger.warning("[sandbox] 进程内存限制已关闭（memory_mb<=0），跳过 self-prison")
        return False
    if os.name != "posix":
        logger.debug("[sandbox] 当前平台无 POSIX setrlimit，内存限制由父进程 Job Object 承担（Windows）")
        return False
    import resource  # noqa: PLC0415 —— POSIX 专用模块，Windows 下 import 即崩，故函数内导入

    limit_bytes = mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError) as e:  # pragma: no cover —— 平台差异兜底
        logger.warning(f"[sandbox] RLIMIT 施加失败（沙箱降级，不阻断启动）: {e}")
        return False
    logger.info(f"[sandbox] self-prison 生效：RLIMIT_AS={mb}MB / RLIMIT_CORE=0"
                + (f" / RLIMIT_CPU={cpu_seconds}s" if cpu_seconds else ""))
    return True


def apply_job_object_limits(proc, memory_mb: int | None = None) -> bool:
    """给已拉起的子进程挂 Job Object 内存硬上限（Windows；由 bootstrap 调用）。

    - Windows：`JOB_OBJECT_LIMIT_PROCESS_MEMORY`——进程 committed memory 超过上限时
      **新分配失败**（进程内表现为 MemoryError / 崩溃），由 OS 保证，无法绕过。
    - POSIX：no-op（无法事后约束子进程，限制由 server 自身 self-prison 承担）。

    :param proc: `subprocess.Popen` 句柄（须仍在运行）。
    :return: True=限制已生效；False=平台不支持 / 关闭 / 挂载失败（只告警不抛出）。
    """
    mb = _resolve_memory_mb(memory_mb)
    if mb <= 0:
        logger.warning(f"[sandbox] pid={getattr(proc, 'pid', '?')} 跳过 Job 挂载（memory_mb<=0）")
        return False
    if os.name != "nt":
        logger.debug("[sandbox] POSIX 平台由 server self-prison 承担限制，跳过 Job Object")
        return False
    if proc.poll() is not None:
        logger.warning(f"[sandbox] 子进程 pid={proc.pid} 已退出，跳过 Job 挂载")
        return False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ★restype 必须显式 HANDLE：默认 c_int 会把 64 位句柄截断成 32 位 → 后续调用全部失败
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.OpenProcess.restype = ctypes.c_void_p

    try:
        job = kernel32.CreateJobObjectW(None, f"mcp_sandbox_{proc.pid}")
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = mb * 1024 * 1024
        if not kernel32.SetInformationJobObject(
                job, _JOB_OBJECT_EXTENDED_LIMIT_INFO,
                ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())

        h_process = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_QUERY_INFORMATION,
            False, proc.pid)
        if not h_process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel32.AssignProcessToJobObject(job, h_process):
                # 常见失败：父进程本身已在某 Job（IDE/shell 环境），子进程自动继承，
                # Windows 8+ 嵌套 Job 下仍可能 assign 失败 → 降级告警，不阻断启动。
                err = ctypes.WinError(ctypes.get_last_error())
                logger.warning(f"[sandbox] AssignProcessToJobObject 失败（沙箱降级，不阻断启动）: {err}")
                return False
        finally:
            kernel32.CloseHandle(h_process)
    except Exception as e:  # noqa: BLE001 —— 沙箱失败只告警
        logger.warning(f"[sandbox] Job Object 挂载失败（沙箱降级，不阻断启动）: {e}")
        return False

    _ACTIVE_JOBS[proc.pid] = job
    logger.info(f"[sandbox] Job Object 生效：pid={proc.pid} 进程内存硬上限={mb}MB（超限即被 OS 终止）")
    return True


__all__ = ["apply_process_limits", "apply_job_object_limits"]
