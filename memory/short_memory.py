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
    def __init__(self, session_id: str):
        self.sid = session_id
        # 原有对话历史初始化，完全保留你的逻辑
        if self.sid not in SESSION_MEM:
            SESSION_MEM[self.sid] = []
        # 新增草稿空间初始化
        if self.sid not in SESSION_DRAFT_CACHE:
            SESSION_DRAFT_CACHE[self.sid] = {
                "bill_add": None,
                "bill_edit": None,
                "bill_delete": None
            }

    def add_msg(self, role: str, content: str):
        """role: user / agent"""
        SESSION_MEM[self.sid].append({"role": role, "content": content})

    def get_history(self, limit: int = 3) -> str:
        """
        读取最近N轮上下文，拼接成文本注入提示词
        :param limit: 保留对话轮数，1轮=user+agent两条消息
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
        """会话结束清空记忆"""
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
        保存账单半成品草稿
        draft_type: bill_add / bill_edit / bill_delete
        data: 半成品账单字典，无信息传None
        """
        if self.sid not in SESSION_DRAFT_CACHE:
            return
        SESSION_DRAFT_CACHE[self.sid][draft_type] = data

    def get_draft(self, draft_type: str) -> Optional[Dict[str, Any]]:
        """读取对应类型草稿，多轮拼接使用"""
        if self.sid not in SESSION_DRAFT_CACHE:
            return None
        return SESSION_DRAFT_CACHE[self.sid].get(draft_type)

    def clear_draft(self, draft_type: str):
        """单独清空某一类草稿（入库成功 / 追问超限时调用）

        ★M5：只清理**已存在**的 key——否则会给草稿池引入非草稿 key（如 agent_id），
        污染 `export_all()` 下发给子 Agent 的 `draft_store` 契约。
        """
        store = SESSION_DRAFT_CACHE.get(self.sid)
        if store is not None and draft_type in store:
            store[draft_type] = None

    # ===================== M5（D5）：追问轮次计数（会话级）=====================
    def incr_ask_round(self, ask_key: str) -> int:
        """追问轮次 +1，返回当前轮次。

        ★为什么必须**会话级**（`03` §8.2 T9 定稿 B 方案）：graph state 每次 `ainvoke`
        都会重建，无法承载跨轮次计数；而 `dispatch_round` 是**调度轮次**（防任务 DAG
        死循环、上限 20、每次新用户输入即重置为 0），**不是**追问轮次——
        这正是"无限追问"缺陷的根因（`02` D5 / `01` §3.4）。
        """
        rounds = SESSION_ASK_ROUND.setdefault(self.sid, {})
        rounds[ask_key] = rounds.get(ask_key, 0) + 1
        return rounds[ask_key]

    def get_ask_round(self, ask_key: str) -> int:
        """读取当前追问轮次（未追问过返回 0）"""
        return SESSION_ASK_ROUND.get(self.sid, {}).get(ask_key, 0)

    def reset_ask_round(self, ask_key: str) -> None:
        """清零该 key 的追问计数（任务成功 / 业务性拒绝 / 放弃后调用）——
        避免历史追问累计，导致下一次记账一开始就被误判"已超限"。"""
        if self.sid in SESSION_ASK_ROUND:
            SESSION_ASK_ROUND[self.sid].pop(ask_key, None)

    def export_all(self) -> Dict[str, Any]:
        """导出完整会话数据，下发A2A给业务Agent"""
        return {
            "history_text": self.get_history(),
            "draft_store": SESSION_DRAFT_CACHE[self.sid].copy()
        }

# 全局获取工具（函数名、入参完全不变，上层调用代码无需修改）
def get_session_memory(session_id: str) -> ShortSessionMemory:
    return ShortSessionMemory(session_id)