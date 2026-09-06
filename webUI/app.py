# -*- coding: utf-8 -*-
"""消费管家智能体 - Gradio WebUI 入口

复用 CLI 对话循环（startup/chat_loop.py）的调用链：
bootstrap_all() -> AGENT_GRAPH["orchestrator_agent"].ainvoke(init_state) -> final_reply
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser

# Windows 下输出被重定向到文件时，Python 会退回系统 locale(GBK)，导致中文日志乱码。
# 显式把标准输出/错误流改为 UTF-8，保证任何启动方式下的日志编码一致。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

# 修复 gradio_client 1.3.0 无法解析布尔 JSON-Schema 的缺陷（背景见该模块 docstring）
from utils.gradio_compat import apply_gradio_patch  # noqa: E402

apply_gradio_patch()

from startup.bootstrap import bootstrap_all, shutdown_system, AGENT_GRAPH
from memory.short_memory import get_session_memory
from utils.common import db

# 进程级固定会话ID：多轮对话复用同一 session，保证短时记忆与草稿缓存生效
SESSION_ID = str(uuid.uuid4())

# 常驻事件循环：bootstrap_all 会 spawn 后台 worker（消费 a2a 总线任务），
# 必须保持事件循环存活，不能用 asyncio.run()（run 结束会关闭 loop 导致 worker 被取消）
_loop: asyncio.AbstractEventLoop | None = None


def init_system() -> None:
    """一次性初始化底层资源（数据库 / RAG / 后台 Worker）"""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(bootstrap_all())


def shutdown() -> None:
    """优雅关闭后台 Worker"""
    if _loop is not None:
        try:
            _loop.run_until_complete(shutdown_system())
        except Exception:
            pass


def respond(user_input: str, history: list) -> tuple:
    """处理一轮对话：落库用户提问 -> 执行编排 Agent -> 返回回复"""
    history = history or []
    if not user_input or not user_input.strip():
        return history, ""

    # 与 chat_loop.py 一致：先落库用户提问，与 collect_node 写入的回复成对
    get_session_memory(SESSION_ID).add_msg("user", user_input)

    init_state = {
        "user_input": user_input,
        "session_id": SESSION_ID,
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent",
    }

    try:
        orchestrator_graph = AGENT_GRAPH["orchestrator_agent"]

        # M7（D3 接入点①）：与 chat_loop 同构——每轮建 run_id + root span，
        #   run_id 由 dispatch_node 注入 task_data 下发 worker（见 `11` §3 M7）
        from utils.tracer import run_scope, span

        async def _trace_run():
            with run_scope(session_id=SESSION_ID):
                with span("orchestrator.run", kind="orchestrator",
                          agent_id="orchestrator_agent"):
                    return await orchestrator_graph.ainvoke(init_state)

        result_state = _loop.run_until_complete(_trace_run())
        reply = result_state.get("final_reply") or "无返回结果"
    except Exception as e:  # noqa: BLE001
        reply = f"【系统异常】执行失败：{str(e)}"

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def new_session() -> list:
    """一键清理会话：重置会话 ID + 清空聊天记录"""
    global SESSION_ID
    SESSION_ID = str(uuid.uuid4())
    return []


# ===================== 预算与偏好设置（WebUI 表单直写 user_config） =====================
# 背景：user_config 读取侧早已工具化（orchestrator/finance 走 config.get_latest），但写入侧
# 此前没有受控入口（Agent 侧零写权限），表现为"预算不可配置"。这里用 WebUI 表单直写，
# 不经过 LLM/Agent，值 100% 确定性落库；保存后新一轮对话的 finance/预警即读最新一行生效。

# 五分类与账单表 bill 的 CHECK 约束完全一致（见 database/init_db.py）
_CATEGORIES = ("餐饮", "交通", "住宿", "购物", "娱乐")

# quota_mode 存储值 -> 下拉展示文案（DB CHECK 枚举见 init_db.py，值必须取枚举原值）
_QUOTA_LABELS = {
    "auto": "自动（优先按我的设定，其次习惯、再城市估算）",
    "user": "只按我设置的分品类预算",
    "habit": "只按历史消费习惯估算",
    "city": "只按城市消费水平估算",
}
# 展示名 -> 枚举值（防御：极端情况下 Gradio 交互可能回传 label 而非 value）
_QUOTA_BY_LABEL = {label: val for val, label in _QUOTA_LABELS.items()}


def _normalize_quota(val) -> str:
    """将下拉值归一化为 DB 枚举（auto/user/habit/city），非法值兜底 auto。"""
    if isinstance(val, str) and val in _QUOTA_LABELS:
        return val
    if isinstance(val, str) and val in _QUOTA_BY_LABEL:
        return _QUOTA_BY_LABEL[val]
    return "auto"


def _city_choices() -> list:
    """城市下拉候选 = city_price 表既有城市（价格对标数据最全）。"""
    rows = db.query_sql("SELECT DISTINCT city FROM city_price ORDER BY city")
    return [r["city"] for r in rows] if rows else []


def _parse_latest_cfg() -> dict:
    """读 user_config 最新一行（与 config.get_latest 同构）；无数据返回空 dict。"""
    rows = db.get_latest_user_config()
    return dict(rows[0]) if rows else {}


def _cfg_summary_markdown() -> str:
    """当前配置摘要卡（纯确定性文本，供页面顶部与保存后刷新展示）。"""
    cfg = _parse_latest_cfg()
    if not cfg:
        return (
            "**当前预算配置：尚未设置**\n\n"
            "系统目前没有任何预算基准：理财分析会提示你先设置月度预算，超支/单品类额度预警不触发。"
            "在下方面板填写并保存即可启用。"
        )
    city = (cfg.get("city") or "").strip()
    budget = cfg.get("month_budget")
    mode = cfg.get("consume_mode") or "正常"
    quota = (cfg.get("quota_mode") or "auto") if cfg.get("quota_mode") in _QUOTA_LABELS else "auto"
    try:
        raw_cb = cfg.get("category_budget") or ""
        cb = {k: v for k, v in json.loads(raw_cb).items() if isinstance(v, (int, float)) and v > 0} if raw_cb else {}
    except (TypeError, ValueError):
        cb = {}
    remind = (cfg.get("remind_pref") or "").strip()

    lines = [
        "**当前预算配置**（user_config 最新行）",
        f"- 所在城市：{city or '（未填写；对话未提及时按默认兜底）'}",
        f"- 月度预算：{f'{budget:g} 元' if budget else '未设置（不触发超支预警）'}",
        f"- 消费模式：{mode}",
        f"- 分品类预算：{'、'.join(f'{k} {v:g} 元' for k, v in cb.items()) if cb else '未单独设置'}",
        f"- 品类额度评估：{_QUOTA_LABELS[quota]}",
        f"- 提醒偏好：{remind or '未设置'}",
    ]
    return "\n".join(lines)


def _parse_category_budget(raw: str) -> dict:
    """category_budget 列 JSON 解析为 dict（容错损坏值）。"""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def load_config_panel():
    """页面加载时回填表单：读最新配置填入各控件，并渲染当前配置摘要卡。"""
    cfg = _parse_latest_cfg()
    choices = _city_choices()
    city_val = (cfg.get("city") or "").strip()
    if city_val and city_val not in choices:
        # 现库城市不在 city_price（可自定义城市）时，前置注入当前值避免 Gradio 告警
        choices = [city_val] + choices
    cb = _parse_category_budget(cfg.get("category_budget"))
    quota_val = cfg.get("quota_mode") if cfg.get("quota_mode") in _QUOTA_LABELS else "auto"
    return (
        gr.update(choices=choices, value=city_val or None),
        gr.update(value=cfg.get("month_budget")),
        gr.update(value=cfg.get("consume_mode") if cfg.get("consume_mode") in ("节俭", "正常", "宽松") else "正常"),
        gr.update(value=_QUOTA_LABELS[quota_val]),
        gr.update(value=cb.get("餐饮")),
        gr.update(value=cb.get("交通")),
        gr.update(value=cb.get("住宿")),
        gr.update(value=cb.get("购物")),
        gr.update(value=cb.get("娱乐")),
        gr.update(value=(cfg.get("remind_pref") or "").strip()),
        gr.update(value=_cfg_summary_markdown()),
    )


def save_config_panel(city, month_budget, consume_mode, quota_mode,
                      cb_food, cb_trans, cb_lodge, cb_shop, cb_fun, remind_pref):
    """保存表单 -> 直写 user_config（单行覆盖语义）。返回 (保存结果卡, 当前配置摘要卡)。"""
    def _fail(msg: str):
        return (f"**保存失败：{msg}**", gr.update())

    city = (city or "").strip() if isinstance(city, str) else ""
    if not city:
        return _fail("所在城市不能为空")

    if month_budget is not None:
        try:
            month_budget = float(month_budget)
        except (TypeError, ValueError):
            return _fail("月度预算必须是数字")
        if month_budget < 0:
            return _fail("月度预算不能为负数")
        if month_budget == 0:
            month_budget = None  # 0 无预算意义，按"未设置"处理

    if consume_mode not in ("节俭", "正常", "宽松"):
        consume_mode = "正常"
    quota_mode = _normalize_quota(quota_mode)

    category_budget = {}
    for cat_name, val in zip(_CATEGORIES, (cb_food, cb_trans, cb_lodge, cb_shop, cb_fun)):
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            return _fail(f"「{cat_name}」预算必须是数字")
        if val < 0:
            return _fail(f"「{cat_name}」预算不能为负数")
        if val > 0:
            category_budget[cat_name] = round(val, 2)
    cb_json = json.dumps(category_budget, ensure_ascii=False) if category_budget else None

    remind_pref = (remind_pref or "").strip() if isinstance(remind_pref, str) else ""
    ok = db.upsert_user_config(
        city=city,
        month_budget=month_budget,
        consume_mode=consume_mode,
        category_budget=cb_json,
        quota_mode=quota_mode,
        remind_pref=remind_pref or None,
    )
    if not ok:
        return _fail("写入数据库失败，请重试")
    return (
        "**已保存并生效** —— 新一轮记账/统计/理财/超支预警将按最新配置计算。",
        gr.update(value=_cfg_summary_markdown()),
    )


# 显眼的对话输入区样式：大字号 + 聚焦高亮光圈 + 加大操作按钮
_CUSTOM_CSS = """
#chat-input textarea {
    font-size: 18px !important;
    line-height: 1.7 !important;
    padding: 14px 18px !important;
}
#chat-input textarea::placeholder {
    color: #aab2bf !important;
    font-size: 15px !important;
    opacity: 1 !important;
}
#chat-input {
    border: 2px solid var(--border-color-primary, #c9d2e0) !important;
    border-radius: 14px !important;
}
#chat-input:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb, 0 6px 18px rgba(37, 99, 235, 0.20) !important;
}
#input-row button {
    min-width: 92px !important;
    min-height: 52px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}
#header-card {
    text-align: center;
    padding: 4px 0 2px;
}
#header-card h2 {
    font-size: 26px !important;
    margin-bottom: 6px !important;
}
"""


# 顶栏初始文案 / 退出后结束卡文案（后端直接替换 Markdown 内容，不依赖浏览器脚本）
_HEADER_TEXT = (
    "## 消费管家智能体\n"
    "支持记账、开销统计、同类消费对比、理财分析、超支预警。\n"
    "示例：`记午饭56元` · `这个月花了多少` · `打车贵不贵` · `给我点省钱建议`"
)
_EXIT_TEXT = (
    "## 🟢 系统已安全退出\n\n"
    "**所有后台服务与数据通道均已关闭**，本窗口可安全关闭。"
)


def _do_exit() -> None:
    """真正执行退出：优雅关闭 Worker 与 MCP 常驻服务后结束进程。"""
    print("收到安全退出确认，正在优雅关闭后台服务...", flush=True)
    shutdown()
    print("后台 Worker 与 MCP 常驻服务均已关闭，进程退出。", flush=True)
    os._exit(0)


def _request_exit(armed: bool):
    """安全退出（页面按钮）：首次点击进入确认态，再次点击才真正退出。

    第二次点击的"页面收尾"由后端直接完成（必然生效，不依赖前端脚本）：
    - 顶栏立即替换为"系统已安全退出"结束卡，输入区/按钮全部隐藏；
    - 同时延迟 1.5 秒执行退出链：优雅关闭 Worker 与 MCP 常驻服务后 os._exit，
      双击 bat 的 cmd 黑窗随之自动关闭；
    - 页面侧 JS 仅作附加尝试：独立应用窗口（--app，见 open_app_window）允许
      window.close()，若打开方式为普通标签页则被浏览器安全策略拒绝——此时
      页面停留在结束卡，手动关闭即可。
    """
    if not armed:
        # 第一次点击：仅按钮变色进入"待确认"，不执行任何关闭、不关页面
        return (
            True,
            gr.update(value="再点一次，确认退出", variant="primary"),
            gr.update(), gr.update(), gr.update(), gr.update(),
        )
    threading.Timer(1.5, _do_exit).start()
    return (
        False,
        gr.update(visible=False),            # 退出按钮隐藏
        gr.update(value=_EXIT_TEXT),         # 顶栏替换为结束卡
        gr.update(visible=False),            # 输入框隐藏
        gr.update(visible=False),            # 发送按钮隐藏
        gr.update(visible=False),            # 新会话按钮隐藏
    )


# 页面侧附加脚本：仅在第二次点击（armed=True）时尝试关闭自身窗口。
# 对 --app 独立窗口有效；普通标签页会被浏览器拒绝（静默），不影响上述后端结束卡。
# 延迟 300ms 是为先让 gradio 退出请求发出，避免脚本同步动作打断请求。
_EXIT_CLOSE_JS = r"""
(_armed) => {
  if (!_armed) { return; }
  setTimeout(() => {
    try { window.close(); } catch (e) {}
  }, 300);
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="消费管家智能体",
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    ) as demo:
        # 顶栏同时充当"退出结束卡"（退出确认后被 _request_exit 替换为 _EXIT_TEXT）
        header = gr.Markdown(_HEADER_TEXT, elem_id="header-card")

        # ---- 预算与偏好设置：WebUI 表单直写 user_config（不经过 LLM，值确定性落库） ----
        with gr.Accordion("预算与偏好设置", open=False):
            cur_cfg_md = gr.Markdown()
            gr.Markdown(
                "填写并保存后即刻生效：新一轮记账/统计/理财/超支预警会按最新配置计算。"
                "城市支持手动输入其它城市（下拉为物价对标库已有城市）。"
            )
            with gr.Row():
                city_dd = gr.Dropdown(
                    label="所在城市",
                    allow_custom_value=True,
                    scale=2,
                    info="候选为物价对标库城市，可手动输入其它城市",
                )
                budget_num = gr.Number(
                    label="月度总预算（元）",
                    minimum=0,
                    precision=2,
                    scale=1,
                    info="留空或填 0 = 不设预算上限",
                )
                mode_dd = gr.Dropdown(
                    label="消费模式",
                    choices=["节俭", "正常", "宽松"],
                    value="正常",
                    info="影响单笔消费\"贵不贵\"的判定阈值",
                )
                quota_dd = gr.Dropdown(
                    label="品类额度评估方式",
                    choices=list(_QUOTA_LABELS.items()),
                    value=_QUOTA_LABELS["auto"],  # (label, value) 选项的初始选中须传显示 label
                    info="单品类消费合理性基准的来源",
                )
            gr.Markdown("**分品类预算（可选，单位：元）**：给某一类设上限后，该类消费超合理额度会预警。留空 = 不单独设限。")
            with gr.Row():
                cb_food = gr.Number(label="餐饮", minimum=0, precision=2)
                cb_trans = gr.Number(label="交通", minimum=0, precision=2)
                cb_lodge = gr.Number(label="住宿", minimum=0, precision=2)
                cb_shop = gr.Number(label="购物", minimum=0, precision=2)
                cb_fun = gr.Number(label="娱乐", minimum=0, precision=2)
            remind_tb = gr.Textbox(
                label="提醒偏好（可选）",
                placeholder="如：语气简洁；记账成功时顺带提示本月剩余预算",
                max_lines=2,
            )
            save_cfg_btn = gr.Button("保存配置", variant="primary")
            save_msg_md = gr.Markdown()

        chatbot = gr.Chatbot(type="messages", height=480, label="对话")
        with gr.Row(elem_id="input-row"):
            msg = gr.Textbox(
                elem_id="chat-input",
                placeholder="想记一笔？直接输入，如：记午饭56元（Enter 发送）",
                scale=8,
                container=False,
            )
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("新会话")
            exit_state = gr.State(False)
            exit_btn = gr.Button("退出系统", variant="stop")
        send_btn.click(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(new_session, inputs=None, outputs=chatbot)
        # 预算配置表单：保存 -> 单行覆盖写 user_config；页面加载 -> 回填当前配置
        save_cfg_btn.click(
            save_config_panel,
            inputs=[city_dd, budget_num, mode_dd, quota_dd,
                    cb_food, cb_trans, cb_lodge, cb_shop, cb_fun, remind_tb],
            outputs=[save_msg_md, cur_cfg_md],
        )
        demo.load(
            load_config_panel,
            inputs=None,
            outputs=[city_dd, budget_num, mode_dd, quota_dd,
                     cb_food, cb_trans, cb_lodge, cb_shop, cb_fun, remind_tb, cur_cfg_md],
        )
        exit_btn.click(
            _request_exit,
            inputs=[exit_state],
            outputs=[exit_state, exit_btn, header, msg, send_btn, clear_btn],
            # 第二次点击时 JS 尽力关闭窗口（对 --app 窗口有效），页面收尾由后端保证
            js=_EXIT_CLOSE_JS,
        )
    return demo


def _cleanup_stale_mcp_ports() -> None:
    """释放上次运行残留的 MCP 常驻服务端口（8001/8002/8003）。

    一键启动健壮性：bootstrap_all 通过 Popen 拉起 sql_bill / llm_base / llm_finance
    常驻进程，父进程退出后它们不会自动消失；若端口仍被占用，下次启动会直接
    raise。此处仅结束命令行含本项目路径（billAgent）的监听进程，避免误杀其它程序。
    """
    if os.name != "nt":
        return
    for port in (8001, 8002, 8003):
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
            continue  # 端口空闲，无需清理
        except OSError:
            pass
        finally:
            probe.close()
        try:
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    f"Get-NetTCPConnection -LocalPort {port} -State Listen "
                    "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess",
                ],
                capture_output=True, text=True, timeout=10,
            ).stdout.split()
        except Exception:  # noqa: BLE001
            continue
        for pid in out:
            try:
                cmdline = subprocess.run(
                    [
                        "powershell", "-NoProfile", "-Command",
                        f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine",
                    ],
                    capture_output=True, text=True, timeout=10,
                ).stdout or ""
            except Exception:  # noqa: BLE001
                cmdline = ""
            if cmdline.strip() and "billAgent" in cmdline:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


def _port_in_use(port: int) -> bool:
    """7860 是否已有服务在监听（用于判断"系统是否已在运行"）。"""
    s = socket.socket()
    try:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


# Edge / Chrome 常见安装路径（探测存在后以其 --app 模式打开独立应用窗口）
_BROWSER_APP_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def open_app_window() -> bool:
    """以浏览器 --app 独立窗口打开 WebUI。

    --app 窗口无地址栏、更像桌面应用，且允许页面脚本 window.close() 关闭自身，
    因此点"退出系统"可让窗口连同黑窗一起消失；找不到 Edge/Chrome 时退回默认
    浏览器（普通标签页无法被网页脚本自关，由页面结束卡兜底）。
    """
    url = "http://127.0.0.1:7860"
    for exe in _BROWSER_APP_CANDIDATES:
        if os.path.exists(exe):
            try:
                subprocess.Popen(
                    [exe, f"--app={url}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except OSError:
                continue
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    return False


def _open_app_window_delayed() -> None:
    """launch 就绪后延迟打开 --app 窗口（避免与端口绑定产生竞态）。"""
    threading.Timer(1.2, open_app_window).start()


def main() -> None:
    # ★单实例保护：7860 已有服务 = 系统正在运行（上一个 cmd 窗口未关 / 重复双击 bat）。
    #   此时直接打开既有页面并退出本进程——**绝不再执行端口清理**，
    #   否则会误杀在跑系统自己的 MCP 服务（8001-8003）。
    if _port_in_use(7860):
        print("消费管家智能体 已在运行：http://127.0.0.1:7860")
        print("如需重启，请先关闭原启动窗口（或在其上按 Ctrl+C），再双击 start_webui.bat")
        open_app_window()
        return

    # 一键启动健壮性：先释放残留 MCP 端口，再让 gradio 的 localhost 探测绕过系统代理
    _cleanup_stale_mcp_ports()
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    try:
        init_system()
        print("===== 消费管家智能体启动完成，正在打开应用窗口 =====")
    except Exception as e:  # noqa: BLE001
        print(f"【初始化失败】{e}")
        sys.exit(1)

    demo = build_demo()
    try:
        # inbrowser=False：改由 _open_app_window_delayed 以 --app 独立窗口打开。
        # 普通标签页受浏览器策略限制无法被脚本关闭，独立窗口可以被 window.close() 关闭。
        _open_app_window_delayed()
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
