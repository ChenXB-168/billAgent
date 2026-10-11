# -*- coding: utf-8 -*-
"""消费管家智能体 - Gradio WebUI 入口

复用 CLI 对话循环（startup/chat_loop.py）的调用链：
bootstrap_all() -> AGENT_GRAPH["orchestrator_agent"].ainvoke(init_state) -> final_reply
"""
import asyncio
import json
import os
import signal
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
# M15（P4）：账单面板与「定位辅助」——★只读展示 + 只把目标写进输入框，绝不直连写库
#（红线与理由见 `webUI/bill_panel.py` 模块头；独立成模块便于单测与整体下线回退）
from webUI.bill_panel import (
    bill_choices, on_delete_selected, on_edit_selected, page_label, page_view,
    panel_updates, render_bill_panel,
)

# 进程级固定会话ID：多轮对话复用同一 session，保证短时记忆与草稿缓存生效
SESSION_ID = str(uuid.uuid4())

# 常驻事件循环：bootstrap_all 会 spawn 后台 worker（消费 a2a 总线任务），
# 必须保持事件循环存活，不能用 asyncio.run()（run 结束会关闭 loop 导致 worker 被取消）
_loop: asyncio.AbstractEventLoop | None = None


def init_system() -> None:
    """
    函数功能与逻辑描述：
        WebUI 启动期的一键初始化：新建进程级事件循环并注册为当前 loop，再在其上同步驱动
        bootstrap_all()。初始化动作与 CLI 入口（startup/chat_loop.py）完全同源：数据库连通校验、
        用户文档库索引加载（M8 起由 userDocs 接管，旧 ragKnowledge 已退役）、拉起 3 个 MCP
        常驻服务（sql_bill/llm_base/llm_finance）并健康检查、A2A 总线清理 + Durable 崩溃恢复、
        以 create_task 挂起 4 个子 Agent 常驻 worker。
        关键约束：必须 new_event_loop + run_until_complete，不可用 asyncio.run()——bootstrap_all
        spawn 的后台 worker 需要 loop 持续存活，asyncio.run 返回即关闭 loop 会把 worker 取消。
    入参说明：
        无。
    返回值说明：
        无（副作用为模块级变量 _loop 被赋值为一个未关闭的事件循环，供 respond 复用）。
    """
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(bootstrap_all())


def shutdown() -> None:
    """
    函数功能与逻辑描述：
        优雅关闭后台服务：在常驻事件循环上驱动 shutdown_system()——置全局 SHUTDOWN_FLAG 让各
        worker 消费循环自然退出，cancel 并 gather 收敛全部 WORKER_TASKS，最后关闭 3 个 MCP
        常驻服务进程，避免孤儿进程。整个关闭过程吞异常（except 后 pass），保证退出链不因单个
        子服务异常而中断。
    入参说明：
        无。
    返回值说明：
        无（副作用为停止全部后台 Worker 与 MCP 常驻服务；_loop 为 None 时直接跳过，不做任何事）。
    """
    if _loop is not None:
        try:
            _loop.run_until_complete(shutdown_system())
        except Exception:
            pass


def _install_signal_handlers() -> None:
    """
    函数功能与逻辑描述：
        M23（D26）L2 信号层：把 `SIGTERM` / `SIGINT` 统一转成 `KeyboardInterrupt`，
        从而**复用既有的异常冒泡路径** —— `demo.launch()` 收到该异常后退出阻塞，
        `main()` 的 `finally: shutdown()` 随即执行（优雅关闭 Worker 与 MCP 常驻服务）。
        为什么必须注册：Python 对 `SIGINT`（Ctrl+C）**默认**就转 `KeyboardInterrupt`，
        故 Ctrl+C 一直可用；但 `SIGTERM`（`kill -15` / `docker stop` / 系统关机）
        的默认行为是**直接终止进程** —— 不抛异常、**不执行 `finally`**、不跑 `atexit`，
        导致 `shutdown_system()` 根本不执行、3 个 MCP 子进程**变孤儿**。
        ★本函数刻意"抛异常"而非"直接调 shutdown()"：后者需在信号上下文里处理事件循环
        跨线程投递（`call_soon_threadsafe`）等复杂度，而浮出到 `finally` 是**已验证**的路径。
        边界：`signal.signal` **只能在主线程**注册（否则抛 `ValueError`），`main()` 在主线程
        满足该约束；仍以 try/except 包裹 —— 注册失败只打印告警、不阻断启动
        （此时退化为"仅 Ctrl+C 可优雅退出"，属已知边界；强杀场景由 L3 内核层兜底）。
    入参说明：
        无。
    返回值说明：
        无（副作用为注册信号处理器；失败只打印告警，不抛异常）。
    """

    def _raise_keyboard_interrupt(signum, frame):  # noqa: ARG001 —— 形参由 signal 规范固定
        raise KeyboardInterrupt(f"收到信号 {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _raise_keyboard_interrupt)
        except (ValueError, OSError) as e:  # 非主线程 / 平台不支持
            print(f"[M23] 信号 {sig} 注册失败（优雅退出降级）: {e}")


def respond(user_input: str, history: list) -> tuple:
    """
    函数功能与逻辑描述：
        单轮对话回调（Gradio 事件处理）：先把用户提问写入短期记忆（与 collect_node 写入的
        Agent 回复成对，构成完整会话上下文）→ 组装编排器 init_state → 在 M7 埋点范围内驱动
        编排图并取 final_reply。
        边界与异常：空输入（None 或纯空白）短路返回，不落库、不调用 Agent；Agent 执行异常被
        except 捕获后转为「【系统异常】执行失败：…」文案回复，不向上抛出，保证页面不崩。
    入参说明：
        user_input (str)：输入框文本；None/空白视为空输入，直接短路。
        history (list)：Gradio Chatbot(type="messages") 的消息列表，元素形如
            {"role": "user"/"assistant", "content": str}；None 按空列表处理。
    返回值说明：
        tuple：4 元组，元素顺序与 outputs=[chatbot, msg, bill_panel, bill_pick] 一一对应
            - 第 1 项 (list)：追加本轮 user/assistant 消息后的完整历史，回填 chatbot。
            - 第 2 项 (str)：固定空串，用于清空输入框 msg。
            - 第 3 项 (str)：账单面板 HTML（M15 P4：每轮刷新，让刚记的那笔立刻可见）。
            - 第 4 项 (dict)：目标选择器的 gr.update（与面板原子刷新，见 bill_panel.panel_updates）。
            空输入短路时不刷新面板（返回空 gr.update），避免无谓的库查询。
    """
    history = history or []
    if not user_input or not user_input.strip():
        return history, "", gr.update(), gr.update()

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
        "cancelled": False,      # M16：本轮是否已被用户取消（停止按钮可置为 True）
        "abort_reason": None,    # M19：终止原因（None/user_cancel/task_timeout/task_error）
    }

    try:
        orchestrator_graph = AGENT_GRAPH["orchestrator_agent"]

        # M7（D3 接入点①）：与 chat_loop 同构——每轮建 run_id + root span，
        #   run_id 由 dispatch_node 注入 task_data 下发 worker（见 `11` §3 M7）
        from utils.tracer import run_scope, span

        async def _trace_run():
            """
            函数功能与逻辑描述：
                respond 的 M7 埋点内层包装：在 run_scope（绑定本进程固定 SESSION_ID）内开启
                orchestrator.run 根 span，再 await 编排图 ainvoke。作为协程交给 _loop
                run_until_complete 驱动，保证埋点上下文与图执行同处一个事件循环。
            入参说明：
                无（闭包捕获外层 respond 的 init_state、orchestrator_graph 与 SESSION_ID）。
            返回值说明：
                dict：编排图输出的完整 state（与 init_state 同构并含 final_reply 等
                    被各节点写回的字段）；异常不在此捕获，上抛给 respond 统一转文案。
            """
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
    # M15（P4）：每轮对话后刷新账单面板与目标选择器（同一实现取数，与 Agent 认知同源）
    return history, "", *panel_updates()


def new_session() -> list:
    """
    函数功能与逻辑描述：
        「新会话」按钮回调：重新生成进程级 SESSION_ID，从而切断上一会话的短期记忆——
        对话历史、三类账单半成品草稿（bill_add/bill_edit/bill_delete）与追问轮次计数
        在 memory/short_memory.py 中均按 session_id 分区存放，换号即等价于全新会话。
        仅重置会话号，不关闭事件循环、不重启任何后台 Worker，MCP 常驻服务保持存活。
    入参说明：
        无（由 clear_btn.click 触发，inputs=None，不接收界面控件值）。
    返回值说明：
        list：空列表 []，作为 Chatbot 的新值以清空页面聊天记录
            （绑定关系见 build_demo 中 clear_btn.click(..., outputs=chatbot)）。
    """
    global SESSION_ID
    SESSION_ID = str(uuid.uuid4())
    return []


def on_stop(history: list) -> list:
    """
    函数功能与逻辑描述：
        「停止本轮」按钮回调（M16 用户取消）：把当前会话全部在途任务（submitted / running）
        标记为 `cancelled` 并中断正在执行的子任务 runner。★本回调**只发令、不等编排图返回**——
        正在跑的 respond 回调占用着事件循环，本回调必须立即返回、用户才能看到反馈。
        编排器侧的处理：worker 的 `track()` 捕获取消后回传取消结果 → `wait_result_node`
        识别到即终止后续波次 → `collect_node` 产出**取消回执**而不是业务回复
        （用户已要求停止，再回一条"已为您记账…"等于无视指令）。
        取消是**终态**：重启后不会被重投、也不会再次自动回复（区别于 `interrupted`）。
        ★线程安全：`asyncio.Task.cancel()` 非线程安全，故用 `run_coroutine_threadsafe`
        把取消动作投递回 `_loop` 所在线程执行，而非在此线程直接调用。
    入参说明：
        history (list)：Gradio Chatbot(type="messages") 的当前消息列表；None 按空处理。
    返回值说明：
        list：追加一条停止回执后的完整历史，回填 chatbot（不触发面板刷新，避免无谓查询）。
    """
    history = list(history or [])
    try:
        from startup.task_manager import get_task_manager

        async def _do_cancel() -> dict:
            """在 _loop 线程内执行取消（Task.cancel 必须与目标任务同线程）。"""
            return get_task_manager().cancel_session(SESSION_ID)

        future = asyncio.run_coroutine_threadsafe(_do_cancel(), _loop)
        res = future.result(timeout=5)
        total = len(res.get("cancelled") or [])
        hit = len(res.get("interrupted") or [])
        if total == 0:
            text = "当前没有正在执行的任务。"
        else:
            text = f"已停止本轮请求（共 {total} 个任务，其中 {hit} 个正在执行已中断）。"
    except Exception as e:  # noqa: BLE001 —— 停止失败只提示，不影响页面
        text = f"停止失败：{e}"
    history.append({"role": "assistant", "content": text})
    return history


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
    """
    函数功能与逻辑描述：
        把「品类额度评估方式」下拉的回传值归一化为 user_config.quota_mode 的 DB 枚举
        （auto/user/habit/city，见 database/init_db.py 的 CHECK 约束）。兼容两种回传形态：
        先按枚举值（value）命中，再按展示 label 反查（_QUOTA_BY_LABEL 兜底，防 Gradio
        极端情况下回传 label）；两者都不命中则兜底 "auto"，保证落库值永不越界。
    入参说明：
        val：下拉回传值，正常为枚举字符串 auto/user/habit/city，
            也可能为 _QUOTA_LABELS 中的展示文案，或 None / 非字符串等异常值。
    返回值说明：
        str：合法 DB 枚举值，取值 auto/user/habit/city；无法识别时返回 "auto"。
    """
    if isinstance(val, str) and val in _QUOTA_LABELS:
        return val
    if isinstance(val, str) and val in _QUOTA_BY_LABEL:
        return _QUOTA_BY_LABEL[val]
    return "auto"


def _city_choices() -> list:
    """
    函数功能与逻辑描述：
        生成「所在城市」下拉的候选列表，取自 city_price（物价对标）表中已入库的城市去重
        结果——该表的城市覆盖最全，且后续溢价评估正依赖它，选它可减少用户填错城市名。
        仅做查询，不写库、不改动任何表结构与数据。
    入参说明：
        无。
    返回值说明：
        list：城市名字符串列表，按 city 升序去重；表为空或无数据时返回空列表 []。
    """
    rows = db.query_sql("SELECT DISTINCT city FROM city_price ORDER BY city")
    return [r["city"] for r in rows] if rows else []


def _parse_latest_cfg() -> dict:
    """
    函数功能与逻辑描述：
        读取 user_config 表按 create_time 倒序的第一行作为「当前生效配置」。取数口径与
        Agent 侧事实工具 config.get_latest（绑定 mcpGateway/fact_tools.get_latest_user_config）
        完全一致，均为 db.get_latest_user_config()，因此 WebUI 表单展示值 = Agent 实际计算依据。
        只读，不写库。
    入参说明：
        无。
    返回值说明：
        dict：该行的列名->列值映射（city / month_budget / consume_mode / category_budget /
            quota_mode / remind_pref / create_time 等）；表为空时返回空 dict {}。
    """
    rows = db.get_latest_user_config()
    return dict(rows[0]) if rows else {}


def _cfg_summary_markdown() -> str:
    """
    函数功能与逻辑描述：
        渲染「当前预算配置」摘要卡 Markdown（纯确定性文本，不调用 LLM）：
        无配置时给出"尚未设置"提示并说明后果（不触发超支/单品类额度预警）；
        有配置时逐项列出城市、月预算、消费模式、分品类预算、品类额度评估、提醒偏好。
        其中 category_budget 为 JSON 字符串，解析容错（仅保留正数项，损坏值按无处理），
        quota_mode 越界值按 "auto" 展示，避免 KeyError 导致页面渲染失败。
    入参说明：
        无。
    返回值说明：
        str：多行 Markdown 文本；调用方以 gr.update(value=...) 写回
            「预算与偏好设置」面板内的摘要 Markdown 组件（cur_cfg_md），
            由 load_config_panel（页面加载）与 save_config_panel（保存成功后）两处刷新。
    """
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
    """
    函数功能与逻辑描述：
        把 user_config.category_budget 列的 JSON 字符串解析为 dict，供表单回填各分品类输入框。
        纯解析、无副作用；防御性容错：空值/非 JSON 文本/解析结果为非 dict（如数组、数字）时
        一律按空 dict 处理，保证页面回填不因历史脏数据抛异常。
    入参说明：
        raw (str)：category_budget 列的原始 JSON 文本；None 或空串表示未设置。
    返回值说明：
        dict：品类名->预算金额映射（如 {"餐饮": 800.0}）；无法解析或非 dict 时返回空 dict {}。
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def load_config_panel():
    """
    函数功能与逻辑描述：
        页面加载回调（demo.load）：读 user_config 最新行回填预算表单各控件，并渲染配置摘要卡。
        关键逻辑：① 城市候选取自 city_price，若库中当前城市不在候选内（用户曾手填自定义城市），
        则把该值前置注入候选，避免 Gradio 因"值不在选项内"告警；② consume_mode 越界兜底"正常"；
        ③ category_budget 经 JSON 解析容错后按五分类取值；④ quota_mode 越界兜底"auto"并
        映射为下拉展示 label（(label, value) 选项初始选中须传 label）。
        只读，不写库。
    入参说明：
        无（demo.load 绑定 inputs=None）。
    返回值说明：
        tuple：11 元组，元素顺序与 outputs 列表逐一位置对应，不可调换：
            1 (gr.update) 城市下拉 city_dd：choices=城市候选，value=当前城市或 None；
            2 (gr.update) 月度总预算 budget_num；
            3 (gr.update) 消费模式 mode_dd；
            4 (gr.update) 品类额度评估方式 quota_dd（传展示 label）；
            5-9 (gr.update) 分品类预算 cb_food/cb_trans/cb_lodge/cb_shop/cb_fun；
            10 (gr.update) 提醒偏好 remind_tb；
            11 (gr.update) 配置摘要 Markdown cur_cfg_md。
    """
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
    """
    函数功能与逻辑描述：
        「保存配置」按钮回调：把表单值直接写入 user_config（单行覆盖语义，复用
        db.upsert_user_config；不经过 LLM/Agent，值 100% 确定性落库）。校验链按序短路：
        城市非空 -> 月预算可转 float 且 >=0（0 视为"未设置"存 NULL）-> 消费模式越界兜底"正常"
        -> quota_mode 经 _normalize_quota 归一化 -> 五个分品类预算可转 float 且 >=0（None 跳过、
        0 视为未设置不写入 JSON）。全部通过后落库，写失败返回失败卡。
        保存成功后下一轮记账/统计/理财/超支预警即读最新一行生效。
    入参说明：
        city：所在城市文本，空白视为非法（必填）。
        month_budget：月度总预算，可 None/空；单位元。
        consume_mode：消费模式，取值 节俭/正常/宽松，越界按"正常"。
        quota_mode：品类额度评估方式，取值 auto/user/habit/city 或其展示 label。
        cb_food / cb_trans / cb_lodge / cb_shop / cb_fun：五分类预算（餐饮/交通/住宿/购物/娱乐），
            单位元，可 None（不设该品类额度）。
        remind_pref：提醒偏好文本，可 None/空（落库为 NULL）。
    返回值说明：
        tuple：2 元组，元素顺序与 outputs=[save_msg_md, cur_cfg_md] 位置对应：
            - 第 1 项 (str)：保存结果 Markdown（成功提示或"**保存失败：<原因>**"）；
            - 第 2 项 (gr.update | dict)：成功时以 gr.update(value=最新摘要卡) 刷新配置摘要，
              失败时返回 gr.update()（无参 = 不改动任何组件）。
    """
    def _fail(msg: str):
        """
        函数功能与逻辑描述：
            保存失败统一收口：组装"保存失败"结果卡，并保持配置摘要卡不变（空 gr.update，
            不带任何参数即不更新组件）。
        入参说明：
            msg (str)：失败原因文案，拼接进 Markdown。
        返回值说明：
            tuple：2 元组，与 save_config_panel 的正常返回同构
                （第 1 项为失败卡文本，第 2 项为无操作的 gr.update()）。
        """
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
    """
    函数功能与逻辑描述：
        真正执行退出的收尾动作，由 _request_exit 用 threading.Timer(1.5) 延迟调用
        （延迟是留时间给 Gradio 把第二次点击的响应/结束卡送达前端）。先打印确认日志，
        同步调用 shutdown() 优雅关闭 Worker 与 MCP 常驻服务，最后 os._exit(0) 强退进程——
        用 os._exit 而非 sys.exit 是为了跳过解释器清理，确保双击 bat 的 cmd 黑窗随之关闭。
    入参说明：
        无。
    返回值说明：
        无（副作用为关闭后台服务并终止进程；os._exit 之后函数不会返回）。
    """
    print("收到安全退出确认，正在优雅关闭后台服务...", flush=True)
    shutdown()
    print("后台 Worker 与 MCP 常驻服务均已关闭，进程退出。", flush=True)
    os._exit(0)


def _request_exit(armed: bool):
    """
    函数功能与逻辑描述：
        「退出系统」按钮回调（exit_btn.click，双击确认式安全退出）：
        以 exit_state（gr.State）作为确认态标记，首次点击只把标记置 True 并让按钮变为"待确认"，
        不执行任何关闭动作（防误触）；再次点击（armed=True）才真正退出。
        第二次点击的"页面收尾"全部由后端直接完成，必然生效、不依赖前端脚本：
        - 顶栏立即替换为"系统已安全退出"结束卡（_EXIT_TEXT），输入框/发送/新会话/退出按钮全部隐藏；
        - 同时启动 threading.Timer(1.5, _do_exit)：优雅关闭 Worker 与 MCP 常驻服务后 os._exit，
          双击 bat 的 cmd 黑窗随之自动关闭；
        - 页面侧 JS（_EXIT_CLOSE_JS）仅作附加尝试：--app 独立窗口（见 open_app_window）允许
          window.close()，普通标签页会被浏览器安全策略拒绝——此时页面停留在结束卡，手动关闭即可。
    入参说明：
        armed (bool)：确认态标记，来自 exit_state 组件；False=首次点击（进入待确认），
            True=已处于待确认，执行真正退出。
    返回值说明：
        tuple：6 元组，元素顺序与 outputs=[exit_state, exit_btn, header, msg, send_btn, clear_btn]
            逐一位置对应（顺序错位会关错组件）：
            首次点击 -> (True, 按钮改为"再点一次，确认退出"并转 primary, 其余 4 项空 gr.update())；
            再次点击 -> (False, 退出按钮隐藏, 顶栏替换为 _EXIT_TEXT, 输入框隐藏, 发送按钮隐藏,
            新会话按钮隐藏)。
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
    """
    函数功能与逻辑描述：
        构建并返回 Gradio 界面（Blocks）及全部事件绑定，是 WebUI 的布局唯一入口。
        布局自上而下：顶栏 header（同时充当退出结束卡）→「预算与偏好设置」折叠面板
        （摘要卡 + 城市/月预算/消费模式/额度评估 + 五分品类预算 + 提醒偏好 + 保存按钮）
        → 聊天区 chatbot → 输入行（输入框 msg / 发送 / 新会话 / 退出系统 + exit_state）。
        事件绑定：send_btn.click 与 msg.submit 同绑 respond（inputs=[msg, chatbot]，
        outputs=[chatbot, msg]）；clear_btn.click 绑 new_session（outputs=chatbot）；
        save_cfg_btn.click 绑 save_config_panel（10 个输入控件，outputs=[save_msg_md, cur_cfg_md]）；
        demo.load 绑 load_config_panel（回填 11 个组件）；exit_btn.click 绑 _request_exit
        （inputs=[exit_state]，outputs 6 项，并附加 _EXIT_CLOSE_JS）。
        本函数只做声明式组装，不启动服务、不访问数据库。
    入参说明：
        无。
    返回值说明：
        gr.Blocks：已装配全部组件与事件回调的界面对象（尚未 launch），供 main() 调用 launch。
    """
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

        # ---- M15（P4）：账单面板 + 目标选择器的「定位辅助」 ----
        # ★这两个控件**只读不写**：面板展示最近账单（含 id），选择器 + 两个按钮只把目标翻译成
        #   一句话写进输入框，真正的改/删仍由对话链路执行（删除还会先二次确认）。
        #   红线与理由见 `webUI/bill_panel.py` 模块头。
        # ★默认收起：面板展开时是 20 行长列表，会让首屏显得冗长、盖住对话区；
        #   用户需要指认某条账时再展开，展开态由 Gradio 自行记住（会话内不重置）。
        with gr.Accordion("最近账单 · 快速定位", open=False):
            bill_panel = gr.HTML(render_bill_panel())
            bill_pick = gr.Radio(
                label="目标选择器（只列当前页账单；选中后点下方按钮，会把它写进输入框）",
                choices=bill_choices(),
                value=None,
            )
            # 按天翻页：每页 7 天；页码在 [0, 总页数-1] 内收敛，点到底/到顶只是停住（不报错）
            with gr.Row():
                page_prev_btn = gr.Button("← 更早 7 天")
                page_next_btn = gr.Button("更新 7 天 →")
            page_info = gr.Markdown(page_label(0))
            bill_page = gr.State(0)
            with gr.Row():
                pick_edit_btn = gr.Button("修改选中")
                pick_del_btn = gr.Button("删除选中")
            gr.Markdown(
                "说明：此处只帮你**把目标写进输入框**（例如「把 id=515 那笔改成 」），"
                "补充好新值后发送即可；删除会先向你二次确认，不会直接执行。"
            )

        chatbot = gr.Chatbot(type="messages", height=480, label="对话")
        with gr.Row(elem_id="input-row"):
            msg = gr.Textbox(
                elem_id="chat-input",
                placeholder="想记一笔？直接输入，如：记午饭56元（Enter 发送）",
                scale=8,
                container=False,
            )
            send_btn = gr.Button("发送", variant="primary")
            stop_btn = gr.Button("停止本轮")
            clear_btn = gr.Button("新会话")
            exit_state = gr.State(False)
            exit_btn = gr.Button("退出系统", variant="stop")
        # 定位辅助：★outputs 只有 msg（输入框）——不触发任何工具调用、不写库
        pick_edit_btn.click(on_edit_selected, inputs=[bill_pick], outputs=msg)
        pick_del_btn.click(on_delete_selected, inputs=[bill_pick], outputs=msg)
        # 按天翻页：+1 = 更早 7 天（页码 +1）/ -1 = 更新 7 天（页码 -1）。
        # outputs 四项与 page_view 的返回顺序**逐一对应**：面板 HTML / 选择器 / 页码文案 / 页码 state。
        page_prev_btn.click(lambda p: page_view(1, p), inputs=[bill_page],
                            outputs=[bill_panel, bill_pick, page_info, bill_page])
        page_next_btn.click(lambda p: page_view(-1, p), inputs=[bill_page],
                            outputs=[bill_panel, bill_pick, page_info, bill_page])
        # 对话：outputs 追加面板与选择器，实现「每轮刷新」（刚记的那笔立刻可见）
        send_btn.click(respond, inputs=[msg, chatbot],
                       outputs=[chatbot, msg, bill_panel, bill_pick])
        msg.submit(respond, inputs=[msg, chatbot],
                   outputs=[chatbot, msg, bill_panel, bill_pick])
        # M16 用户取消：只发令（标记 cancelled + 中断子任务），不等编排图返回；
        #   正在跑的 respond 会在收口后给出取消回执而非业务回复。
        stop_btn.click(on_stop, inputs=[chatbot], outputs=[chatbot])
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
    """
    函数功能与逻辑描述：
        一键启动健壮性防护：bootstrap_all 通过 Popen 拉起 sql_bill / llm_base / llm_finance
        三个 MCP 常驻进程（端口 8001/8002/8003，见 config.MCP_SERVER_PORTS），父进程退出后
        它们不会自动消失；端口若仍被占用，下次 start_mcp_servers 会直接 raise 导致启动失败。
        做法：逐个端口先 bind 探测是否空闲（空闲则跳过），忙则用 PowerShell
        Get-NetTCPConnection 取监听进程 PID，再查其 CommandLine，仅当命令行含本项目
        路径关键字 "billAgent" 时才 taskkill，避免误杀其它程序占用的同号端口。
        非 Windows（os.name != "nt"）直接返回，不做任何处理；单个端口探测/查询异常均静默跳过。
    入参说明：
        无（端口 8001/8002/8003 为硬编码，与 config.MCP_SERVER_PORTS 对齐）。
    返回值说明：
        无（副作用为可能强杀残留的 MCP 监听进程）。
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
    """
    函数功能与逻辑描述：
        通用端口占用探测：对 127.0.0.1 的目标端口发起一次 TCP 连接，能连上即视为已有
        服务在监听。当前唯二调用点是 main() 的 7860 单实例保护（判断"系统是否已在运行"，
        是则只打开页面不再重复启动）；本函数本身与端口号无关，可复用于任意端口。
        探测用短超时 0.5 秒，socket 在 finally 中必然关闭，不泄漏句柄。
    入参说明：
        port (int)：待探测的本地 TCP 端口号，调用方传 7860（Gradio 服务端口）。
    返回值说明：
        bool：True=端口已被监听（connect_ex 返回 0）；False=未监听或连接失败。
    """
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
    """
    函数功能与逻辑描述：
        以浏览器 --app 独立窗口方式打开 WebUI（固定地址 http://127.0.0.1:7860）。
        --app 窗口无地址栏、更像桌面应用，且允许页面脚本 window.close() 关闭自身，
        因此点"退出系统"可让窗口连同 cmd 黑窗一起消失；按 _BROWSER_APP_CANDIDATES 顺序
        探测本机 Edge/Chrome 可执行文件，命中即 Popen 拉起并返回 True（stdout/stderr 丢弃，
        不阻塞主进程）；单个候选启动抛 OSError 时继续试下一个。
        所有候选都不存在时退回 webbrowser.open（默认浏览器、普通标签页，无法被网页脚本自关，
        由页面结束卡兜底），该分支无论成败都返回 False。
    入参说明：
        无。
    返回值说明：
        bool：True=已用 Edge/Chrome 的 --app 模式打开；False=退回默认浏览器或打开异常
            （仅表示未启用 --app 模式，不代表启动失败）。
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
    """
    函数功能与逻辑描述：
        延迟 1.2 秒后用 threading.Timer 在后台线程调用 open_app_window：demo.launch 绑定
        7860 端口是阻塞调用，必须先把"打开窗口"排到定时器上再进入 launch，否则会因端口
        尚未就绪而打开空白页/连接失败。定时器不 join，主线程照常进入 launch。
    入参说明：
        无。
    返回值说明：
        无（副作用为启动一个 1.2 秒后触发的后台定时器线程）。
    """
    threading.Timer(1.2, open_app_window).start()


def main() -> None:
    """
    函数功能与逻辑描述：
        WebUI 进程主入口，按序完成：① 单实例保护——_port_in_use(7860) 为真说明系统已在运行
        （上一个 cmd 窗口未关 / 重复双击 start_webui.bat），此时仅打开既有页面并直接 return，
        绝不执行端口清理，否则会误杀在跑系统自己的 MCP 服务（8001-8003）；② 释放残留 MCP
        端口并设置 NO_PROXY/no_proxy 让 Gradio 的 localhost 探测绕过系统代理；
        ③ init_system() 初始化（失败打印【初始化失败】并 sys.exit(1)）；④ build_demo()
        构建界面，先排定延迟打开 --app 窗口，再 demo.launch(server_name/port, inbrowser=False)
        阻塞提供 7860 服务；⑤ finally 中 shutdown() 兜底优雅关闭后台服务。
    入参说明：
        无。
    返回值说明：
        无（副作用为启动/复用 WebUI 服务进程；初始化失败时进程以退出码 1 结束）。
    """
    # ★M23（D26）L2：先挂信号处理器（必须在主线程，且应尽早）——
    #   否则 `kill -15` / `docker stop` 下 finally 不执行，MCP 常驻子进程会变孤儿。
    _install_signal_handlers()

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
