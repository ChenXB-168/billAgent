# -*- coding: utf-8 -*-
"""
需求5 意图短板修复 端到端测试：真实 LLM 全链路验证三个意图场景。

M4 修复内容（agents/orchestrator/nodes.py + prompts 模板）：
1. 多意图参数抽取：'花了300买耳机，值吗' → bill+finance，且 finance 任务必须携带金额/品类参数
   （启发式 has_finance_review 把"记账上下文合理性疑问"映射为 finance 意图；金额上下文补全扩展到 finance）
2. 防误记账：'别记这笔' 等拒绝记账表述 → 即使模型规划 bill 也拦截，bill 表零新增
3. 单意图不脑补：'奶茶15块' → 仅 bill，不附带 stat/finance/price 多余任务

数据隔离（每个用例前后自动恢复，测试后数据零残留）：
- bill 表：MAX(id) 快照法，运行后删除 id > 快照 的新增记录
- user_config 表：全列备份恢复（场景1 finance 需预算）
- monthly_habit / pkl：测试前清空，避免历史习惯干扰 finance 推算
"""
import json
import uuid
from datetime import date

import pytest

from agents.orchestrator.graph import orchestrator_graph
from memory.long_memory import LONG_MEM_PATH, _load_mem, clear_all_long_mem
from utils.common import db

TODAY = date.today()

# 场景输入（带 [意图e2e] 标记便于排查，不影响语义）
MULTI_INTENT_INPUT = "花了300买耳机，值吗 [意图e2e]"
REFUSE_BILL_INPUT = "别记这笔，看看我最近吃火锅多不多 [意图e2e]"
SINGLE_BILL_INPUT = "奶茶15块 [意图e2e]"


async def _run_graph(user_input: str):
    """
    函数功能与逻辑描述：
        辅助协程：以随机 session_id（前缀 intent_）构造 orchestrator 初始 state 并 await 执行
        orchestrator_graph.ainvoke，返回终态 state 供用例断言 task_plan 的任务集合与执行状态。
        运行时依赖真实外部 LLM（意图规划与各 Agent 解析）、本地账单库与常驻子 Agent worker。
    入参说明：
        user_input (str)：用户本轮自然语言输入，作为 state["user_input"]。
    返回值说明：
        dict：orchestrator 执行完成后的 final state（含 task_plan / final_reply / error_msg）。
    """
    init_state = {
        "user_input": user_input,
        "session_id": f"intent_{uuid.uuid4().hex[:8]}",
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent",
    }
    return await orchestrator_graph.ainvoke(init_state)


def _max_id() -> int:
    """
    函数功能与逻辑描述：
        取 bill 表当前最大 id 作为快照基线，供用例/fixture 判定"运行后是否新增账单"，
        是"防误记账"与"单笔记账"两类断言的数据基准。
    入参说明：
        无。
    返回值说明：
        int：bill 表当前 MAX(id)；表为空或查询返回空列表时返回 0。
    """
    rows = db.query_sql("SELECT MAX(id) AS m FROM bill")
    return int(rows[0]["m"] or 0) if rows else 0


def _new_bills(after_id: int) -> list:
    """
    函数功能与逻辑描述：
        查询 id 大于快照基线的账单记录（按 id 升序），仅取 id / amount / category / remark
        四列，用于断言新增笔数与金额（如场景1 应恰好入库 1 笔 300 元）。
    入参说明：
        after_id (int)：快照基线，即 _max_id() 在用例开始前取得的值。
    返回值说明：
        list[dict]：新增账单行（列名 → 值）；无新增时返回空列表。
    """
    return db.query_sql("SELECT id, amount, category, remark FROM bill WHERE id > ? ORDER BY id", (after_id,))


def _backup_user_config() -> list:
    """
    函数功能与逻辑描述：
        备份 user_config 表全部行，供 fixture 测试后恢复，避免测试预算配置残留
        （场景1 的 finance 分析需要预算存在）。
    入参说明：
        无。
    返回值说明：
        list[dict]：user_config 全表行（列名 → 值）；表为空时返回空列表。
    """
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    """
    函数功能与逻辑描述：
        恢复 user_config：先全表 DELETE，再按备份行的全部列动态拼 INSERT 重插，
        以兼容并保留 remind_pref / quota_mode / category_budget 等新增列。
    入参说明：
        backup (list)：_backup_user_config() 备份的行列表；为空列表时仅执行清空。
    返回值说明：
        无（副作用：覆盖 user_config 表内容）。
    """
    db.execute_sql("DELETE FROM user_config")
    for r in backup:
        cols = list(r.keys())
        placeholders = ",".join(["?"] * len(cols))
        db.execute_sql(
            f"INSERT INTO user_config ({','.join(cols)}) VALUES ({placeholders})",
            [r[c] for c in cols],
        )


def _set_budget(amount: float):
    """
    函数功能与逻辑描述：
        清空 user_config 后写入唯一一条测试预算（广州 / 消费模式"正常" / budget_type "month"），
        保证 get_latest_user_config 取到的就是本用例设定的预算（finance 无预算会走追问）。
    入参说明：
        amount (float)：月度总预算，单位元。
    返回值说明：
        无（副作用：改写 user_config 表内容）。
    """
    db.execute_sql("DELETE FROM user_config")
    db.add_user_config("广州", amount, "正常", "month")


def _backup_pkl() -> bytes:
    """
    函数功能与逻辑描述：
        以二进制方式备份长期记忆持久化文件 long_consume_mem.pkl（字节级快照），
        供 fixture 测试后原样写回；os 模块在本函数内按需导入，未在模块顶层 import os。
    入参说明：
        无。
    返回值说明：
        bytes | None：文件字节内容；文件不存在时返回 None（无可恢复内容）。
    """
    if __import__("os").path.exists(LONG_MEM_PATH):
        with open(LONG_MEM_PATH, "rb") as f:
            return f.read()
    return None


def _restore_pkl(blob: bytes):
    """
    函数功能与逻辑描述：
        恢复长期记忆：先 clear_all_long_mem() 清空内存文本/FAISS 索引并删除 pkl 文件，
        再把备份字节原样写回并调用 _load_mem() 重载内存与重建索引；blob 为 None 时只清空。
    入参说明：
        blob (bytes | None)：_backup_pkl() 取得的字节快照；None 表示测试前无该文件。
    返回值说明：
        无（副作用：改写 LONG_MEM_PATH 文件与 long_memory 模块级全局状态）。
    """
    clear_all_long_mem()
    if blob is not None:
        with open(LONG_MEM_PATH, "wb") as f:
            f.write(blob)
        _load_mem()


def _collect_diagnostics(result: dict) -> str:
    """
    函数功能与逻辑描述：
        断言失败时的诊断文本拼装：输出 final_reply，并遍历 task_plan 逐任务打印
        raw_segments / status / result（JSON 截断 300 字符），便于定位意图规划与任务裁剪问题。
    入参说明：
        result (dict)：orchestrator 执行后的 final state。
    返回值说明：
        str：多行诊断文本（1 行 final_reply + 每个任务 1 行）。
    """
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(
            f"[{tname}] segs={tinfo.get('raw_segments')} status={tinfo.get('status')} "
            f"result={json.dumps(res, ensure_ascii=False, default=str)[:300]}"
        )
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _data_isolation():
    """
    函数功能与逻辑描述：
        pytest fixture（function 作用域，autouse=True），为本文件每个用例提供数据隔离，
        不向用例注入任何数据。setup：记录 bill 的 MAX(id) 快照，备份 user_config 全表与
        long_consume_mem.pkl 字节快照，再清空 monthly_habit 与长期记忆，避免历史习惯干扰
        finance 推算与账单预警；teardown：删除新增账单、恢复配置与 pkl 快照并重载内存，
        保证测试后数据零残留。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无显式返回值（yield 仅用于分离 setup/teardown，不向用例提供数据）。
    """
    bill_before = _max_id()
    cfg_backup = _backup_user_config()
    pkl_backup = _backup_pkl()
    # 测试前清空历史习惯/长期记忆，避免干扰 finance 推算与账单预警
    db.execute_sql("DELETE FROM monthly_habit")
    clear_all_long_mem()
    yield
    db.execute_sql("DELETE FROM bill WHERE id > ?", (bill_before,))
    _restore_user_config(cfg_backup)
    _restore_pkl(pkl_backup)


@pytest.mark.asyncio
async def test_e2e_multi_intent_params():
    """
    函数功能与逻辑描述：
        场景「多意图参数抽取」（M4）：先备份基线并把预算设为 2000 元，输入
        “花了300买耳机，值吗 [意图e2e]”跑完整 orchestrator 全链路，断言：
        ① task_plan 同时含 bill 与 finance（合理性疑问由启发式 has_finance_review 兜底补齐）；
        ② finance 任务的 raw_segments 必须携带金额"300"与品类"耳机"（金额上下文补全扩展到 finance）；
        ③ bill / finance 的 status 均为 done，bill 表恰好新增 1 笔且金额为 300.0，final_reply 非空。
        运行时前置依赖：真实外部 LLM（规划与解析）、本地 bill / user_config 库、常驻子 Agent worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    _set_budget(2000.0)
    bill_before = _max_id()

    result = await _run_graph(MULTI_INTENT_INPUT)
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① ★2026-09-17 期望更新（用户确认）：本输入**无记账指令**（"花了300买耳机，值吗"）→
    #   **不生成 bill**（`system.md` 2026-09-05 规则：只有"记/记账/记录/记一下"等动作词才算
    #   记账指令，"花了/买了"只是开销陈述）；合理性疑问映射 finance。
    #   原期望含 bill，基于 M4 期"金额+消费实体即记账"的旧规则，已被取代。
    assert "bill" not in tp, f"无记账指令不得生成 bill: {tp.keys()}\n{diag}"
    assert "finance" in tp, f"未触发 finance 分析任务（合理性疑问应映射 finance）: {tp.keys()}\n{diag}"

    # ② finance 参数完整性：raw_segments 必须携带金额与消费品类（金额上下文补全扩展到 finance）
    fin_segs = " ".join(tp["finance"].get("raw_segments") or [])
    assert "300" in fin_segs, f"finance 参数缺失金额: {fin_segs}\n{diag}"
    assert "耳机" in fin_segs, f"finance 参数缺失品类: {fin_segs}\n{diag}"

    # ③ finance 执行成功 + **零记账**（与 ① 同口径：本输入本就不该记账，故断言表内零新增）
    assert tp["finance"].get("status") == "done", diag
    new_bills = _new_bills(bill_before)
    assert new_bills == [], f"无记账指令不得入库: {new_bills}\n{diag}"
    assert result.get("final_reply"), "final_reply 为空"


@pytest.mark.asyncio
async def test_e2e_refuse_bill_no_record():
    """
    函数功能与逻辑描述：
        场景「防误记账」（M4）：输入含拒绝记账表述的“别记这笔，看看我最近吃火锅多不多 [意图e2e]”，
        断言：① task_plan 中不含 bill（启发式 has_refuse_bill 在计划裁剪阶段拦截，
        即使模型规划了 bill 也会被剔除）；② bill 表零新增（任务层拦截 + 表级断言双保险）；
        ③ final_reply 非空且不含"记账成功"（拒绝记账场景不得声称已记账）。
        运行时前置依赖：真实外部 LLM（规划与解析）、本地 bill 库、常驻子 Agent worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    bill_before = _max_id()

    result = await _run_graph(REFUSE_BILL_INPUT)
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① 绝不派发 bill 任务（启发式 has_refuse_bill 拦截，即使模型规划了 bill）
    assert "bill" not in tp, f"拒绝记账场景仍派发 bill: {tp.keys()}\n{diag}"

    # ② bill 表零新增（双保险：任务层拦截 + 表级断言）
    new_bills = _new_bills(bill_before)
    assert new_bills == [], f"拒绝记账场景 bill 表不应新增: {new_bills}\n{diag}"

    # ③ 用户仍得到响应（统计/提示或未识别提示，均不得是记账）
    assert result.get("final_reply"), "final_reply 为空"
    assert "记账成功" not in (result.get("final_reply") or ""), "拒绝记账场景回复不应声称记账成功"


@pytest.mark.asyncio
async def test_e2e_stat_multi_window_compare():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（方案 `设计/13` 第 2 步验收）：**多时间窗统计**端到端验收。
        输入"这个月跟上个月比，花得多了还是少了"，断言 stat 任务成功，且结果的 data
        **同时包含两个时间窗**（`groups` 两项、区间互不相同）—— 验证"多组查询"表达力已打通。
        改造前该能力**不存在**：IR 只有一个 `filter`，两个时间窗无处安放（模型只能二选一，
        或写成一个跨两个月的大区间），"对比"必然失效。
        运行时前置依赖：真实外部 LLM（规划 + IR 解析）、本地 bill 库、常驻子 Agent worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    result = await _run_graph("这个月跟上个月比，花得多了还是少了 [多窗口e2e]")
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    assert "stat" in tp, f"未触发统计任务: {tp.keys()}\n{diag}"
    assert tp["stat"].get("status") == "done", diag

    data = ((tp["stat"].get("result") or {}).get("data")) or {}
    groups = data.get("groups")
    assert isinstance(groups, list) and len(groups) == 2, f"应有恰好两个时间窗: {data}\n{diag}"
    ranges = [g.get("time_range") for g in groups]
    assert ranges[0] != ranges[1], f"两个窗口不得相同（否则'对比'无意义）: {ranges}\n{diag}"
    assert result.get("final_reply"), "final_reply 为空"


@pytest.mark.asyncio
async def test_e2e_multi_bill_one_utterance():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（方案 `设计/13` 第 1 步验收）：**一句多笔账**端到端验收。
        输入"记午饭30，记打车20 [多笔e2e]"，断言：
        ① 命中 bill 任务且 status 为 done；② bill 表**恰好新增 2 笔**；
        ③ 两笔的金额/类目各自正确（30→餐饮、20→交通），**不得串账**；
        ④ final_reply 非空且点名两笔的金额。
        这是改造前**必然失败**的场景：主 Agent 只派一条 bill，而子层是单笔硬编码
        （只读单个 amount/category、只 INSERT 一次），且 `json_parser` 会把多个同构 JSON
        浅合并成一笔 → 两笔只落一笔（静默丢账）。
        运行时前置依赖：真实外部 LLM（规划 + 抽取）、本地 bill 库、常驻子 Agent worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    bill_before = _max_id()

    result = await _run_graph("记午饭30，记打车20 [多笔e2e]")
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① 触发记账任务并成功
    assert "bill" in tp, f"未触发记账任务: {tp.keys()}\n{diag}"
    assert tp["bill"].get("status") == "done", diag

    # ② 恰好两笔入库
    new_bills = _new_bills(bill_before)
    assert len(new_bills) == 2, f"一句两笔应恰好入库 2 笔: {new_bills}\n{diag}"

    # ③ 金额与类目各自正确（防串账）
    by_amount = {float(b["amount"]): b for b in new_bills}
    assert 30.0 in by_amount, f"缺少 30 元那笔: {new_bills}\n{diag}"
    assert 20.0 in by_amount, f"缺少 20 元那笔: {new_bills}\n{diag}"
    assert by_amount[30.0]["category"] == "餐饮", f"30 元应为餐饮: {by_amount[30.0]}\n{diag}"
    assert by_amount[20.0]["category"] == "交通", f"20 元应为交通: {by_amount[20.0]}\n{diag}"

    # ④ 回复点名两笔
    reply = result.get("final_reply") or ""
    assert reply, "final_reply 为空"
    assert "30" in reply and "20" in reply, f"回复应同时点名两笔: {reply}\n{diag}"


@pytest.mark.asyncio
async def test_e2e_single_bill_no_extra():
    """
    函数功能与逻辑描述：
        场景「单意图不脑补」（M4）：输入“奶茶15块 [意图e2e]”，断言：
        ① task_plan 的键集合恰为 ["bill"]（金额 + 消费实体词场景由 _is_force_bill 兜底补齐 bill，
        非记账意图的 stat / finance / price 任务被裁剪）；② bill 任务 status 为 done，
        bill 表恰好新增 1 笔，final_reply 非空。
        运行时前置依赖：真实外部 LLM（规划与解析）、本地 bill 库、常驻子 Agent worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    bill_before = _max_id()

    result = await _run_graph(SINGLE_BILL_INPUT)
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① ★2026-09-17 期望更新（用户确认）：本输入**无明确记账指令**（"奶茶15块"不含"记/帮我记"
    #   等动作词，"花了/买了"只是开销陈述）→ **不生成 bill**，按"比价咨询"处理 →
    #   任务集合**恰为 ["price"]**。原期望 ["bill"] 基于 M4 期"金额+消费实体即记账"的旧规则，
    #   已被 `system.md`（2026-09-05）"必须显式记账指令才记账"取代。
    assert list(tp.keys()) == ["price"], f"无记账指令应只比价、不得脑补记账: {tp.keys()}\n{diag}"

    # ② 零记账（数据层双保险：无论任务层如何裁剪，bill 表都不得新增）
    new_bills = _new_bills(bill_before)
    assert new_bills == [], f"无记账指令不得入库: {new_bills}\n{diag}"
    assert result.get("final_reply"), "final_reply 为空"
