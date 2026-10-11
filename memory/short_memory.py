# ==============================================
# 短期会话记忆 - 纯内存的对话历史 + 账单草稿 + 追问轮次计数
# 说明：三个全局字典按 session_id 分桶，进程内共享；离线项目不依赖 Redis
# ==============================================
from typing import Dict, List, Optional, Any

# 全局会话内存存储，key = session_id，离线项目无需Redis
SESSION_MEM: Dict[str, List[dict]] = {}
# 新增：独立草稿缓存池，和对话历史完全隔离，互不干扰
SESSION_DRAFT_CACHE: Dict[str, Dict[str, Optional[Dict]]] = {}
# ★M5（D5 T2）：追问轮次计数（会话级，`session_id -> 追问key -> 轮次`）。
#   独立字典而非塞进 SESSION_DRAFT_CACHE：`export_all()` 会把草稿池整体下发给子 Agent，
#   计数混进去会污染 draft_store 契约；且 T3 放弃清草稿时不应连带清掉计数语义。
SESSION_ASK_ROUND: Dict[str, Dict[str, int]] = {}


class ShortSessionMemory:
    """
    函数/类功能与逻辑描述：
        短期会话记忆门面：以 session_id 为键读写三个模块级全局字典——SESSION_MEM（user/agent
        对话历史）、SESSION_DRAFT_CACHE（三类账单半成品草稿）、SESSION_ASK_ROUND（M5 追问轮次计数）；
        本类不持有这些状态，所有实例共享同一份全局存储，故同一 sid 复用即天然单例语义。
        纯内存、无持久化、无锁（进程内单实例调用），不依赖 Redis。
    构造入参说明：
        session_id (str)：会话唯一标识，作为三个全局字典的键。
    返回值说明：
        构造返回 ShortSessionMemory 实例，仅完成该会话在全局存储中的注册（幂等，已存在则不动）。
    """

    def __init__(self, session_id: str):
        """
        函数功能与逻辑描述：
            注册会话：确保 SESSION_MEM[self.sid]（对话历史，初始 []）与 SESSION_DRAFT_CACHE[self.sid]
            （三类草稿 bill_add / bill_edit / bill_delete，初始全 None）存在；已存在的会话保持原状不重置（幂等）。
            不在此初始化 SESSION_ASK_ROUND——追问计数由 incr_ask_round 首次调用时惰性建键。
        入参说明：
            session_id (str)：会话唯一标识，直接存入实例属性 self.sid。
        返回值说明：
            无（仅初始化实例属性与全局存储，无其它副作用）。
        """
        self.sid = session_id
        # 对话历史初始化（幂等：已存在则保留）
        if self.sid not in SESSION_MEM:
            SESSION_MEM[self.sid] = []
        # 草稿空间初始化（三类账单半成品，幂等：已存在则保留）
        if self.sid not in SESSION_DRAFT_CACHE:
            SESSION_DRAFT_CACHE[self.sid] = {
                "bill_add": None,
                "bill_edit": None,
                "bill_delete": None
            }

    def add_msg(self, role: str, content: str):
        """
        函数功能与逻辑描述：
            向本会话对话历史尾部追加一条消息；不区分角色、不去重、不设条数上限，
            条数裁剪统一由读取侧的 get_history 滑动窗口负责。
        入参说明：
            role (str)：消息角色，取值 "user" / "agent"。
            content (str)：消息正文。
        返回值说明：
            无（副作用：向全局 SESSION_MEM[self.sid] 追加 {"role","content"} 一条）。
        """
        SESSION_MEM[self.sid].append({"role": role, "content": content})

    def get_history(self, limit: int = 3) -> str:
        """
        函数功能与逻辑描述：
            读取本会话最近 limit 轮对话，拼接为纯文本（每行 "role: content"）供注入子 Agent 提示词。
            裁剪策略：记 msg_max = limit * 2（1 轮 = user + agent 两条消息），仅保留消息列表**末尾**
            最多 msg_max 条，更早的历史直接丢弃且不可恢复（纯内存滑窗，不落盘）。
            边界：未校验 limit 合法性——limit=0 时切片 [-0:] 等价于取全部消息，负数会从头部截断；
            生产调用方应传正整数（默认 3，对应最多 6 条消息）。
        入参说明：
            limit (int)：保留的对话轮数，默认 3。
        返回值说明：
            str：拼接后的历史文本，每条一行并以换行结尾；历史不足 msg_max 条时返回全部已有消息；
                空会话返回 ""。
        """
        msg_max = limit * 2  # 一轮对话2条消息，换算最大消息条数
        all_msg = SESSION_MEM[self.sid]
        # 取末尾最多 msg_max 条消息
        hist = all_msg[-msg_max:] if len(all_msg) > msg_max else all_msg

        text = ""
        for item in hist:
            text += f"{item['role']}: {item['content']}\n"
        return text

    def clear(self):
        """
        函数功能与逻辑描述：
            会话结束清理：把本会话对话历史 SESSION_MEM[self.sid] 重置为 []，并把
            SESSION_DRAFT_CACHE[self.sid] 的三类草稿一律重置为 None。
            仅改写已存在的键，不新增；**不清理** SESSION_ASK_ROUND（追问轮次计数）——
            该计数由 reset_ask_round 按 key 单独清零，清空会话不改变"已追问轮次"语义。
        入参说明：
            无。
        返回值说明：
            无（副作用：改写全局 SESSION_MEM / SESSION_DRAFT_CACHE 中本会话条目）。
        """
        if self.sid in SESSION_MEM:
            SESSION_MEM[self.sid] = []
        # 同步清空草稿缓存
        if self.sid in SESSION_DRAFT_CACHE:
            SESSION_DRAFT_CACHE[self.sid] = {
                "bill_add": None,
                "bill_edit": None,
                "bill_delete": None
            }

    # ===================== 新增草稿操作接口（完全新增，不改动原有任何方法） =====================
    def set_draft(self, draft_type: str, data: Optional[Dict[str, Any]]):
        """
        函数功能与逻辑描述：
            保存账单半成品草稿：直接覆盖写入 SESSION_DRAFT_CACHE[self.sid][draft_type]。
            仅在会话已注册（草稿池 key 已存在）时写入，否则直接返回，不隐式创建会话。
        入参说明：
            draft_type (str)：草稿类别，取值 "bill_add" / "bill_edit" / "bill_delete"。
            data (Optional[Dict[str, Any]])：半成品账单字典；无信息时传 None（表示该类草稿为空）。
        返回值说明：
            无（副作用：改写全局 SESSION_DRAFT_CACHE 中本会话的对应草稿槽）。
        """
        if self.sid not in SESSION_DRAFT_CACHE:
            return
        SESSION_DRAFT_CACHE[self.sid][draft_type] = data

    def get_draft(self, draft_type: str) -> Optional[Dict[str, Any]]:
        """
        函数功能与逻辑描述：
            读取指定类别的草稿，供多轮追问拼接（上一轮半成品 + 本轮新增信息）使用；只读不改写。
        入参说明：
            draft_type (str)：草稿类别，取值 "bill_add" / "bill_edit" / "bill_delete"。
        返回值说明：
            Optional[Dict[str, Any]]：对应草稿字典；会话未注册或该槽为 None 时返回 None。
        """
        if self.sid not in SESSION_DRAFT_CACHE:
            return None
        return SESSION_DRAFT_CACHE[self.sid].get(draft_type)

    def clear_draft(self, draft_type: str):
        """
        函数功能与逻辑描述：
            单独清空某一类草稿（入库成功 / 追问超限放弃时调用）。
            ★M5：只清理**已存在**的 key——否则会给草稿池引入非草稿 key（如 agent_id），
            污染 `export_all()` 下发给子 Agent 的 `draft_store` 契约。
        入参说明：
            draft_type (str)：草稿类别，取值 "bill_add" / "bill_edit" / "bill_delete"。
        返回值说明：
            无（副作用：把对应槽置为 None；会话或该 key 不存在时不动作）。
        """
        store = SESSION_DRAFT_CACHE.get(self.sid)
        if store is not None and draft_type in store:
            store[draft_type] = None

    # ===================== M5（D5）：追问轮次计数（会话级）=====================
    def incr_ask_round(self, ask_key: str) -> int:
        """
        函数功能与逻辑描述：
            追问轮次 +1 并返回当前轮次，用于拦截「无限追问」缺陷
            （`02` D5 / `01` §3.4）。首次调用时惰性创建该会话的计数桶。
            ★为什么必须**会话级**（`03` §8.2 T9 定稿 B 方案）：graph state 每次 `ainvoke`
            都会重建，无法承载跨轮次计数；而 `dispatch_round` 是**调度轮次**（防任务 DAG
            死循环、上限 20、每次新用户输入即重置为 0），**不是**追问轮次——
            这正是"无限追问"缺陷的根因。计数只增不减，清零需显式调用 reset_ask_round。
        入参说明：
            ask_key (str)：追问场景标识（通常为 agent_id），同一会话下按 key 分别计数。
        返回值说明：
            int：该 ask_key 自上次清零以来累计的追问轮次（从 1 开始）。
        """
        rounds = SESSION_ASK_ROUND.setdefault(self.sid, {})
        rounds[ask_key] = rounds.get(ask_key, 0) + 1
        return rounds[ask_key]

    def get_ask_round(self, ask_key: str) -> int:
        """
        函数功能与逻辑描述：
            读取当前追问轮次，只读不修改；用于在追问前判断是否已达上限。
            会话或 key 不存在时统一返回 0，不做惰性建键。
        入参说明：
            ask_key (str)：追问场景标识（通常为 agent_id）。
        返回值说明：
            int：当前累计追问轮次；未追问过（会话或 key 不存在）时返回 0。
        """
        return SESSION_ASK_ROUND.get(self.sid, {}).get(ask_key, 0)

    def reset_ask_round(self, ask_key: str) -> None:
        """
        函数功能与逻辑描述：
            清零该 key 的追问计数（任务成功 / 业务性拒绝 / 放弃后调用）——
            避免历史追问累计，导致下一次记账一开始就被误判"已超限"。
            用 pop(key, None) 移除而非置 0，使 get_ask_round 仍返回 0，语义等价且不留空桶；
            会话本身不存在时为空操作。
        入参说明：
            ask_key (str)：追问场景标识（通常为 agent_id）。
        返回值说明：
            无（副作用：从 SESSION_ASK_ROUND 中移除本会话该 key 的计数）。
        """
        if self.sid in SESSION_ASK_ROUND:
            SESSION_ASK_ROUND[self.sid].pop(ask_key, None)

    def export_all(self) -> Dict[str, Any]:
        """
        函数功能与逻辑描述：
            导出完整会话数据，用于经 A2A 下发给业务 Agent。固定导出两项：
            history_text（经 get_history 按默认 3 轮裁剪的历史文本）与
            draft_store（三类草稿的**浅拷贝**，防止子 Agent 侧改动污染全局草稿池）。
            ★刻意不导出 SESSION_ASK_ROUND：追问计数属编排侧控制态，
            混入会造成 draft_store 契约污染（见模块头 M5 注释）。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Dict[str, Any]：{"history_text": str, "draft_store": Dict}，
                其中 draft_store 为 {"bill_add","bill_edit","bill_delete"} 三个槽的副本。
        """
        return {
            "history_text": self.get_history(),
            "draft_store": SESSION_DRAFT_CACHE[self.sid].copy()
        }


# 全局获取工具（函数名、入参完全不变，上层调用代码无需修改）
def get_session_memory(session_id: str) -> ShortSessionMemory:
    """
    函数功能与逻辑描述：
        获取指定会话的短期记忆门面实例。因 ShortSessionMemory 本身不持有会话状态
        （全部状态在模块级全局字典中），这里每次调用都新建轻量对象，语义上等价于单例，
        无需缓存或去重；构造过程幂等，已注册会话不会被重置。
    入参说明：
        session_id (str)：会话唯一标识。
    返回值说明：
        ShortSessionMemory：绑定该 session_id 的记忆门面实例。
    """
    return ShortSessionMemory(session_id)
