# -*- coding: utf-8 -*-
"""M23（D26）三层进程退出防御单元测试。

覆盖（对齐 `设计/11` §3 M23 验收表与 `设计/21` §七）：

| 用例 | 验证点 | 对应层 |
|---|---|---|
| `test_job_limit_flags_contains_kill_on_job_close` | Job 标志位**按位或**含 `0x2000`，且**不丢** `0x100`（M4 内存限制） | L3-a |
| `test_job_object_limits_skip_on_non_windows` | 非 Windows 返回 False（由服务端 self-prison 承担） | L3-a |
| `test_parent_death_signal_non_posix_returns_false` | 非 POSIX 返回 False 且**不抛** | L3-b |
| `test_parent_death_signal_posix_calls_prctl_with_sigterm` | POSIX 下 `prctl` 选项号与目标信号均正确 | L3-b |
| `test_parent_death_signal_prctl_failure_degrades` | `prctl` 返回非 0 → 告警降级、返回 False、**不抛** | L3-b |
| `test_parent_death_signal_libc_unavailable_degrades` | `CDLL` 抛错 → 降级不抛（沙箱是加固非故障点） | L3-b |
| `test_parent_death_signal_orphan_self_exits` | 设置时父已死（`getppid()==1`）→ **主动退出**，不留孤儿 | L3-b 竞态 |
| `test_process_limits_skips_death_signal_when_disabled` | `memory_mb<=0`（沙箱总开关关闭）→ PDEATHSIG **一并跳过** | L3 开关 |
| `test_webui_installs_sigterm_handler` | `webUI/app.py` 注册 `SIGTERM` 且 `main()` 内调用安装函数 | L2 |
| `test_chat_loop_has_finally_shutdown` | CLI 循环有 `finally: await shutdown_system()` 收尾 | L2 |

★设计说明（为什么 L2 用"源码级静态断言"而非行为断言）：
    `webUI/app.py` 的导入链会拉起 gradio（实测 **8s**）并触发日志轮转副作用，
    对单测而言代价过高；而"是否注册了某信号"本身是**声明式**的，静态断言已足够锁定。
    这与本仓库既有惯例一致（见 `test_webui_bill_panel.py` 的
    `test_webui_source_contains_no_bill_write_calls`）。
★不拉起真实 MCP 服务：L3 全部以假 os / 假 ctypes 验证"施加契约"，不 spawn 子进程。
"""
import ctypes
from pathlib import Path

import pytest

from mcpGateway import sandbox

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ===================== ① L3-a：Windows Job Object =====================
def _fake_kernel32(captured: dict):
    """构造一个假的 kernel32：记录下发给 OS 的 LimitFlags 与内存上限。

    ★CreateJobObjectW / OpenProcess 刻意用「函数对象作为实例属性」而非绑定方法 ——
      sandbox 会对它们赋 `restype`（64 位句柄截断防护），而**绑定方法不支持属性赋值**。
    """

    class _FakeKernel32:
        def __init__(self):
            self.CreateJobObjectW = lambda *a, **k: 1    # 非 0 即视为成功（句柄用 1 代替）
            self.OpenProcess = lambda *a, **k: 2

        def SetInformationJobObject(self, job, info_class, info_ptr, size):
            # 从指针还原结构体，捕获**真正会下发到内核**的标志位
            info = ctypes.cast(
                info_ptr,
                ctypes.POINTER(sandbox._JOBOBJECT_EXTENDED_LIMIT_INFORMATION),
            ).contents
            captured["flags"] = info.BasicLimitInformation.LimitFlags
            captured["mem_limit"] = info.ProcessMemoryLimit
            return 1

        def AssignProcessToJobObject(self, *a, **k):
            return 1

        def CloseHandle(self, *a, **k):
            return None

    return _FakeKernel32()


class _FakeProc:
    """最小 Popen 替身：仅需 pid 与 poll()。"""
    pid = 4242

    def poll(self):
        return None


class _FakeOs:
    """假 os 模块：避免 monkeypatch 全局 os.name 影响同期其它测试。"""
    name = "nt"


def test_job_limit_flags_contains_kill_on_job_close(monkeypatch):
    """★核心：Job 标志位必须是「内存限制 或 父死回收」，缺任一都算回归。"""
    captured: dict = {}
    monkeypatch.setattr(sandbox, "os", _FakeOs)
    monkeypatch.setattr(sandbox.ctypes, "WinDLL", lambda *a, **k: _fake_kernel32(captured))

    assert sandbox.apply_job_object_limits(_FakeProc(), memory_mb=512) is True

    flags = captured["flags"]
    assert flags & sandbox._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, \
        "M23：缺 KILL_ON_JOB_CLOSE → 强杀场景仍会留孤儿"
    assert flags & sandbox._JOB_OBJECT_LIMIT_PROCESS_MEMORY, \
        "M23 红线：不得替换 LimitFlags，否则丢失 M4 的内存硬上限"
    assert captured["mem_limit"] == 512 * 1024 * 1024


def test_job_object_limits_skip_on_non_windows(monkeypatch):
    """POSIX 下不挂 Job（限制由服务端 self-prison 承担），返回 False 且不抛。"""

    class _PosixOs:
        name = "posix"

    monkeypatch.setattr(sandbox, "os", _PosixOs)
    assert sandbox.apply_job_object_limits(_FakeProc(), memory_mb=512) is False


# ===================== ② L3-b：POSIX 父死信号 =====================
def _fake_libc(calls: dict, ret: int = 0):
    """构造假 libc：记录 prctl 的实参，并按 ret 模拟成功 / 失败。"""

    class _FakeLibc:
        def prctl(self, *args):
            calls["args"] = args
            return ret

    return _FakeLibc()


class _PosixOsWithPpid:
    """假 os：可控 getppid / _exit，用于验证竞态分支。"""
    name = "posix"

    def __init__(self, ppid: int, exits: dict):
        self._ppid = ppid
        self._exits = exits

    def getppid(self):
        return self._ppid

    def _exit(self, code):
        self._exits["code"] = code
        raise SystemExit(code)          # 阻断后续语句，模拟真实 os._exit 不返回


def test_parent_death_signal_non_posix_returns_false(monkeypatch):
    """非 POSIX 直接返回 False（该侧由 Job Object 承担），**不抛**。"""
    monkeypatch.setattr(sandbox, "os", _FakeOs)          # name == "nt"
    assert sandbox.apply_parent_death_signal() is False


def test_parent_death_signal_posix_calls_prctl_with_sigterm(monkeypatch):
    """POSIX：prctl 选项号为 PR_SET_PDEATHSIG、目标信号默认为 SIGTERM。"""
    import signal as _signal

    calls: dict = {}
    exits: dict = {}
    monkeypatch.setattr(sandbox, "os", _PosixOsWithPpid(ppid=1234, exits=exits))
    monkeypatch.setattr(sandbox.ctypes, "CDLL", lambda *a, **k: _fake_libc(calls))

    assert sandbox.apply_parent_death_signal() is True

    assert calls["args"][0] == sandbox._PR_SET_PDEATHSIG
    assert calls["args"][1] == _signal.SIGTERM, "红线：目标信号须为 SIGTERM 而非 SIGKILL"
    assert "code" not in exits, "父进程存活时不得主动退出"


def test_parent_death_signal_prctl_failure_degrades(monkeypatch):
    """prctl 返回非 0 → 告警降级、返回 False、**不抛**（沙箱是加固非故障点）。"""
    calls: dict = {}
    exits: dict = {}
    monkeypatch.setattr(sandbox, "os", _PosixOsWithPpid(ppid=1234, exits=exits))
    monkeypatch.setattr(sandbox.ctypes, "CDLL", lambda *a, **k: _fake_libc(calls, ret=-1))

    assert sandbox.apply_parent_death_signal() is False


def test_parent_death_signal_libc_unavailable_degrades(monkeypatch):
    """libc 不可用（如极端裁剪环境）→ 降级不抛。"""
    exits: dict = {}
    monkeypatch.setattr(sandbox, "os", _PosixOsWithPpid(ppid=1234, exits=exits))

    def _boom(*a, **k):
        raise OSError("libc 不可用")

    monkeypatch.setattr(sandbox.ctypes, "CDLL", _boom)
    assert sandbox.apply_parent_death_signal() is False


def test_parent_death_signal_orphan_self_exits(monkeypatch):
    """★竞态红线：设置时父已死（已被 init 收养）→ 必须主动退出，否则仍是孤儿。"""
    calls: dict = {}
    exits: dict = {}
    monkeypatch.setattr(sandbox, "os", _PosixOsWithPpid(ppid=1, exits=exits))
    monkeypatch.setattr(sandbox.ctypes, "CDLL", lambda *a, **k: _fake_libc(calls))

    with pytest.raises(SystemExit):
        sandbox.apply_parent_death_signal()

    assert exits.get("code") == 0


def test_process_limits_skips_death_signal_when_disabled(monkeypatch):
    """沙箱总开关关闭（memory_mb<=0）时，PDEATHSIG 也必须一并跳过（零新增配置项）。"""
    called: list = []
    monkeypatch.setattr(sandbox, "apply_parent_death_signal",
                        lambda *a, **k: called.append(1))

    assert sandbox.apply_process_limits(memory_mb=0) is False
    assert called == [], "沙箱关闭时不得仍挂父死信号"


# ===================== ③ L2：信号与 CLI 收尾（源码级） =====================
def test_webui_installs_sigterm_handler():
    """`webUI/app.py` 必须注册 SIGTERM，且 `main()` 内调用安装函数（M23/L2）。"""
    text = (_REPO_ROOT / "webUI" / "app.py").read_text(encoding="utf-8")

    assert "import signal" in text, "缺少 signal 导入"
    assert "signal.SIGTERM" in text, "未注册 SIGTERM → kill -15 / docker stop 下 finally 不执行"
    assert "signal.signal(" in text, "未见实际注册调用"
    assert "KeyboardInterrupt" in text, "未复用既有 KeyboardInterrupt 冒泡路径"
    assert "_install_signal_handlers()" in text, "main() 未调用安装函数"


def test_chat_loop_has_finally_shutdown():
    """CLI 入口必须用 finally 收尾 —— 否则输入 exit 后 MCP 子进程必然泄漏（D26 ③）。"""
    text = (_REPO_ROOT / "startup" / "chat_loop.py").read_text(encoding="utf-8")

    assert "shutdown_system" in text, "未引入 shutdown_system"
    assert "finally:" in text, "循环未用 try/finally 包裹"
    assert "await shutdown_system()" in text, "finally 内未执行实际收敛"


def test_compose_comment_no_longer_claims_tini_fixes_shutdown():
    """`docker_compose.yml` 注释不得再宣称「tini 使 shutdown_system 生效」（D26 ④ 已修正）。"""
    text = (_REPO_ROOT / "utils" / "docker_compose.yml").read_text(encoding="utf-8")

    assert "M23" in text, "注释未标注修正"
    assert "tini **只解决" in text or "只解决「信号送达」" in text, \
        "未写清 tini 的能力边界（送达 / 僵尸回收 ≠ 执行清理）"
