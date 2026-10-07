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
# ★M23（D26）L3 内核层：Job 句柄**全部关闭**时终止 Job 内所有进程。
#   语义链：父进程死亡（**任何方式**，含 `taskkill /F` / 崩溃）→ OS 回收 `_ACTIVE_JOBS`
#          持有的句柄 → Job 关闭 → **内核**终止 Job 内全部 MCP 子进程 → 不留孤儿。
#   ★为何常驻期不会误杀：`_ACTIVE_JOBS` 在父进程存活期**始终持有句柄**，触发条件不可达。
#   ★这是唯一能覆盖"父进程被剥夺执行权"的机制（信号 handler 对此无能为力）。
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFO = 9             # JobObjectExtendedLimitInformation
_PROCESS_SET_QUOTA = 0x0100                     # AssignProcessToJobObject 所需权限
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_INFORMATION = 0x0400

# ── POSIX 父死信号常量（M23 / D26 L3 内核层）──
_PR_SET_PDEATHSIG = 1   # Linux `prctl` 选项号：父进程死亡时向本进程投递指定信号

# 已挂载 Job 的句柄表（pid → job handle）。★不能 CloseHandle：Job 句柄关闭即失效，
# 限制随之解除；句柄随 bootstrap 进程生命周期自然回收即可。
_ACTIVE_JOBS: dict[int, int] = {}


# ═══════════ Job Object 结构体（Windows x64，ctypes 按自然对齐）═══════════
# 参考 winnt.h：BasicLimitInformation 64B + IO_COUNTERS 48B + 4 个 SIZE_T = 144B。
class _IO_COUNTERS(ctypes.Structure):
    """
    函数/类功能与逻辑描述：
        映射 Windows `IO_COUNTERS` 结构体，用于填充 Job Object 的扩展限制信息。
        本模块只用到 BasicLimitInformation 与 ProcessMemoryLimit，IO 计数仅为满足
        结构体布局（字段偏移必须与 winnt.h 完全一致，否则 SetInformationJobObject 会失败），
        因此这里不做任何业务读写。
    构造入参说明：
        无（由 ctypes 按 _fields_ 声明构造，各计数默认 0）。
    返回值说明：
        无（纯数据结构；不提供业务方法）。
    """
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    """
    函数/类功能与逻辑描述：
        映射 Windows `JOBOBJECT_BASIC_LIMIT_INFORMATION`，承载 Job 的基础限制标志与工作集限制。
        本模块实际只使用 LimitFlags（置 `_JOB_OBJECT_LIMIT_PROCESS_MEMORY`），
        其余字段（时间限制、工作集、优先级、亲和性）必须按 winnt.h 布局保留以满足对齐要求。
    构造入参说明：
        无（由 ctypes 按 _fields_ 声明构造，各字段默认 0）。
    返回值说明：
        无（纯数据结构；不提供业务方法）。
    """
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
    """
    函数/类功能与逻辑描述：
        映射 Windows `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`，是传给
        `SetInformationJobObject(..., JobObjectExtendedLimitInformation, ...)` 的实参类型。
        真正生效的两个字段是 `BasicLimitInformation.LimitFlags`（开启进程内存限制）
        与 `ProcessMemoryLimit`（上限字节数）。
    构造入参说明：
        无（由 ctypes 按 _fields_ 声明构造；业务字段由 apply_job_object_limits 填充）。
    返回值说明：
        无（纯数据结构；不提供业务方法）。
    """
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _memory_mb_from_config() -> int:
    """
    函数功能与逻辑描述：
        沙箱内存上限的单一来源读取：从 `config.config` 取 `SANDBOX_MEMORY_MB`（该常量支持环境变量配置）。
        刻意在函数内做延迟导入（noqa: PLC0415），避免模块级导入 config 造成环状依赖与
        导入期副作用。config 缺失或导入异常时**不抛出**，回退模块常量 `_DEFAULT_MEMORY_MB`
        （2048），以贯彻「沙箱失败不阻断启动」的定位。
    入参说明：
        无。
    返回值说明：
        int：内存上限（单位 MB）。正常路径为 config.SANDBOX_MEMORY_MB；
            config 不可用时为 `_DEFAULT_MEMORY_MB`（2048）。
    """
    try:
        from config.config import SANDBOX_MEMORY_MB  # noqa: PLC0415
        return SANDBOX_MEMORY_MB
    except Exception:  # pragma: no cover —— config 缺失时兜底，不阻断沙箱逻辑
        return _DEFAULT_MEMORY_MB


def _resolve_memory_mb(memory_mb: int | None) -> int:
    """
    函数功能与逻辑描述：
        内存上限参数的三级优先级解析：显式传参 > config.SANDBOX_MEMORY_MB > 模块默认值。
        调用方传 0 或负数会被原样返回，由调用方据此判定「限制已关闭」
        （各公开函数均在入口判 mb <= 0 后跳过施加动作并告警）。
    入参说明：
        memory_mb (int | None)：显式指定的内存上限（MB）；传 None 表示回退到 config / 默认值。
            注意 0 不等于 None：0 表示显式关闭限制。
    返回值说明：
        int：最终生效的内存上限（MB），可能是 0 或负数（表示关闭）。
    """
    if memory_mb is not None:
        return memory_mb
    return _memory_mb_from_config()


def apply_parent_death_signal(sig: int | None = None) -> bool:
    """
    函数功能与逻辑描述：
        M23（D26）L3 内核层（POSIX 分支）：给**当前进程**挂「父进程死亡信号」——
        调 `prctl(PR_SET_PDEATHSIG, sig)`，父进程一旦死亡，内核即向本进程投递 `sig`。
        定位：这是**唯一能覆盖"父进程被强杀 / 崩溃"的机制** —— 由内核在核心态保证，
        不依赖父进程执行任何清理代码（对比 `shutdown_system` 要求父进程有机会执行）。

        为什么必须由子进程自己调：POSIX 下父进程**无法事后**给已启动的子进程施加该属性
        （只能靠 `preexec_fn` 或子进程自设）。故落点选在 `apply_process_limits` 内部 ——
        那是 3 个 MCP server `__main__` 的既有自调用钩子，**调用方零改动**。

        ★竞态处理（关键）：若**设置时父进程已死**，`prctl` 会设置成功但 PDEATHSIG
        **永不触发**（"父死亡"这个事件已经过去了）→ 本进程会静默变成孤儿。
        因此设置后**必须立即检查 `os.getppid()`**：若为 1（已被 init 收养），
        说明父进程已死，本进程**主动退出**。

        目标信号默认 `SIGTERM`（而非 `SIGKILL`）：给 server 侧 uvicorn 留自行优雅收尾的机会。
        非 POSIX 平台直接返回 False —— 该侧由 Job Object 的
        `_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 承担（见 `apply_job_object_limits`）。
        沙箱定位使然：任何失败只告警、不抛出，绝不阻断 server 启动。
    入参说明：
        sig (int | None)：父进程死亡时向本进程投递的信号；None（默认）表示 `signal.SIGTERM`。
    返回值说明：
        bool：True 表示已成功挂载；False 表示平台非 POSIX、libc 不可用或施加失败
            （均已记录告警日志）。**注意**：检测到"父已死"时本函数**不返回**
            （主动 `os._exit(0)`）。
    """
    if os.name != "posix":
        logger.debug("[sandbox] 当前平台无 prctl，父死信号由 Windows Job Object 承担")
        return False
    import signal as _signal  # noqa: PLC0415 —— 延迟导入，与 resource 同风格
    target = _signal.SIGTERM if sig is None else sig
    try:
        # CDLL(None)：取主程序全局符号表（含 libc），比写死 "libc.so.6" 更通用 ——
        #   musl 系统（Alpine）不存在 libc.so.6，写死会让本函数在那些平台静默失效。
        libc = ctypes.CDLL(None, use_errno=True)
        ret = libc.prctl(_PR_SET_PDEATHSIG, target, 0, 0, 0)
        if ret != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) 返回非 0")
        # ★竞态检查：设置时父已死 → PDEATHSIG 永不触发，此处主动了断（否则"设了保护仍是孤儿"）
        if os.getppid() == 1:
            logger.warning("[sandbox] 检测到父进程已死亡（已被 init 收养），主动退出以避免孤儿残留")
            os._exit(0)
    except Exception as e:  # noqa: BLE001 —— 沙箱失败只告警
        logger.warning(f"[sandbox] 父死信号挂载失败（沙箱降级，不阻断启动）: {e}")
        return False
    logger.info(f"[sandbox] 父死信号已挂载：父进程退出即向本进程投递信号 {target}")
    return True


def apply_process_limits(memory_mb: int | None = None,
                         cpu_seconds: int | None = None) -> bool:
    """
    函数功能与逻辑描述：
        当前进程自我限制（self-prison），由 3 个 MCP server 的 `__main__` 在启动时各调用一次。
        平台分轨：POSIX 下施加 `RLIMIT_AS`（虚拟地址空间硬上限，防爆内存）与
        `RLIMIT_CORE=0`（禁写 core dump，避免磁占用）；仅当显式传入 cpu_seconds 时
        才叠加 `RLIMIT_CPU`（常驻 server 不传，理由见模块头「CPU 时间为何不做进程级限制」）。
        Windows 下无 `resource` 模块，本函数退化为 no-op 并返回 False，
        内存限制改由父进程经 apply_job_object_limits 挂 Job Object 承担。
        `resource` 为 POSIX 专用模块，Windows 下 import 即崩，故在函数内延迟导入。
        沙箱施加失败只告警不抛出（可用性优先）。
    入参说明：
        memory_mb (int | None)：内存上限（MB），默认 None 表示走
            _resolve_memory_mb 的三级回退；传 <=0 表示显式关闭限制。
        cpu_seconds (int | None)：CPU 时间上限（秒），默认 None 表示不限制；
            仅面向未来一次性执行场景，常驻 MCP server 不应传。
    返回值说明：
        bool：True 表示限制已生效；False 表示平台不支持（Windows）、
            被显式关闭（memory_mb<=0）或施加失败（均已记录告警日志）。
    """
    mb = _resolve_memory_mb(memory_mb)
    if mb <= 0:
        logger.warning("[sandbox] 进程内存限制已关闭（memory_mb<=0），跳过 self-prison")
        return False
    if os.name != "posix":
        logger.debug("[sandbox] 当前平台无 POSIX setrlimit，内存限制由父进程 Job Object 承担（Windows）")
        return False
    import resource  # noqa: PLC0415 —— POSIX 专用模块，Windows 下 import 即崩，故函数内导入

    # ★M23（D26）L3：父死信号与资源限制同属"self-prison"，一并施加。
    #   放在 rlimit 之前 —— 生命周期保护优先于资源保护（孤儿残留比内存超限更危险）。
    apply_parent_death_signal()

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
    """
    函数功能与逻辑描述：
        给已拉起的子进程挂 Windows Job Object 内存硬上限（由 bootstrap 在 Popen 之后调用）。
        原理：Job 上设 `JOB_OBJECT_LIMIT_PROCESS_MEMORY` 后，进程 committed memory 超过
        ProcessMemoryLimit 时**新内存分配直接失败**（进程内表现为 MemoryError / 崩溃），
        由 OS 在核心态保证，进程自身无法绕过。POSIX 下无法事后约束子进程，
        本函数直接 no-op（限制由 server 自身 self-prison 承担）。
        关键实现细节：`CreateJobObjectW` / `OpenProcess` 的 restype 必须显式声明为
        c_void_p，否则 ctypes 默认 c_int 会把 64 位句柄截断成 32 位，导致后续调用全部失败。
        `AssignProcessToJobObject` 常见失败原因是父进程本身已在某个 Job 中（IDE / shell 环境），
        子进程被自动继承；此时降级为告警而不阻断启动。
        成功挂载的 Job 句柄记入模块级 `_ACTIVE_JOBS` **且不 CloseHandle**——
        句柄一旦关闭 Job 即失效、限制随之解除，句柄随 bootstrap 进程生命周期自然回收。
    入参说明：
        proc：`subprocess.Popen` 句柄，须仍在运行；函数会读取其 pid 并调用 poll() 判活。
        memory_mb (int | None)：内存上限（MB），默认 None 表示走 _resolve_memory_mb 的三级回退；
            传 <=0 表示显式关闭限制。
    返回值说明：
        bool：True 表示 Job 已挂载生效；False 表示平台非 Windows、限制被显式关闭、
            子进程已退出、AssignProcessToJobObject 失败或发生异常（均已记录告警日志）。
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
        # ★M23（D26）：在既有内存标志基础上**按位或**叠加 KILL_ON_JOB_CLOSE ——
        #   父进程死亡即由内核回收 Job 内进程。**必须"或"而非替换**，否则丢失 M4 的内存硬上限。
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
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


__all__ = ["apply_process_limits", "apply_job_object_limits", "apply_parent_death_signal"]
