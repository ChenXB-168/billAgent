"""M12（D6）ContextManager —— 分层上下文预算 + 溢出截断。

设计出处：`08` §5.2 M12（接口雏形）+ `02` D6（缺陷证据/方案）。本文件为**执行时行为权威**，
接入后回写 `05`（D6 执行时行为）与 `10`（ContextManager 契约，v2.11）。

分层预算（08 表，8k 窗口示例比例，静态分层——某层不参与时份额空置为宽松余量，不赠与他人，
避免"全部让给一层"击穿总窗口）：

| 层 | 预算比例（扣除 system 预留后） | overflow | 红线 |
|---|---|---|---|
| system | 固定预留 min(400, 输入*15%) | 不可截 | 超限仅 WARN，保留全文（防幻觉角色规则） |
| long_memory | 20% | reduce_top_k（保留头部） | 检索结果靠前段最相关 |
| user_docs | 25% | reduce_top_k（保留头部） | **最低保留 1 段**（`05` §6.1.1 宁可不给不可拖慢） |
| task_results | 30% | 保序逐块放入，当前块内部截尾 | 受保护块（确定性事实底稿）**不截**（R9） |
| history | 25% | 滑窗 keep_tail（保留最近） | — |

关键属性：
1. **预算内逐字节零改动**（所有层估算 ≤ 预算即原样返回）→ 常规输入回归零风险。
2. token 估算为**纯字符近似**（`estimate_tokens`），不引第三方 tokenizer（踩坑库 #9/#10）；
   中文字符权重偏高 = 估高 → 更早触发截断（保守方向，宁可少给不可溢出）。
3. ⚠️ **顶层只 import 标准库**，严禁 import 任何项目内模块（踩坑库 #12：`utils.common`
   import 即连库等模块级副作用）。组件无状态、纯函数，可跨 async 安全调用。
4. 默认窗口换算：`context_window - output_reserve`（本地 ollama `num_ctx=4096` -
   输出预留 1024，即 `ContextManager.output_reserve` 默认值）；M14 的
   `Capabilities.context_window` 落地后由调用方传实际窗口接管。
"""
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------- token 近似估算（零第三方依赖） ----------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
_ASCII_RE = re.compile(r"[A-Za-z0-9]")


def estimate_tokens(text: str) -> int:
    """
    函数功能与逻辑描述：
        字符级近似 token 估算，纯字符近似、不引第三方 tokenizer（踩坑库 #9/#10）；
        按正则分类统计各类字符数后加权求和并向上取整。中文字符权重偏高 = 估高 →
        更早触发截断（保守方向，宁可少给不可溢出）。经验权重（保守偏高，只用于预算
        判断，不追求与真实 tokenizer 一致）：CJK 字符/全角标点 1.5 token/字、
        ASCII 字母数字 0.3 token/字符、其余（空格/换行/半角符号等）0.6 token/字符。
        边界处理：空文本短路返回 0。
    入参说明：
        text (str)：待估算的原始文本，可为空串。
    返回值说明：
        int：估算 token 数，为非负整数；空文本返回 0。
    """
    if not text:
        return 0
    n_cjk = len(_CJK_RE.findall(text))
    n_ascii = len(_ASCII_RE.findall(text))
    n_other = len(text) - n_cjk - n_ascii
    return math.ceil(n_cjk * 1.5 + n_ascii * 0.3 + n_other * 0.6)


# ---------------- 分层预算常量 ----------------
# system 固定预留（08 表 ~400）：min(SYSTEM_RESERVE_RATIO*输入, SYSTEM_RESERVE_MAX)
SYSTEM_RESERVE_RATIO: float = 0.15
SYSTEM_RESERVE_MAX: int = 400
# 其余四层比例（相对 system 预留后的可分配池）
LAYER_RATIOS: Dict[str, float] = {
    "long_memory": 0.20,
    "user_docs": 0.25,
    "task_results": 0.30,
    "history": 0.25,
}

# ---------------- 溢出策略 ----------------
OVERFLOW_FIXED = "fixed"          # 不可截
OVERFLOW_REDUCE = "reduce_top_k"  # 段落化后保留头部（最低 1 段）
OVERFLOW_TRUNCATE = "truncate"    # 保序逐块放入、当前块内部截尾（task_results）
OVERFLOW_KEEP_TAIL = "keep_tail"  # 滑窗：保留最近（history）


@dataclass
class Budget:
    """
    函数/类功能与逻辑描述：
        单层预算卡（接口雏形见 `08` §5.2 M12），承载某一层（long_memory / user_docs /
        task_results / history）的最大 token 预算与溢出策略；为纯数据容器（dataclass），
        不持有连接、缓存等运行时资源。溢出策略取值见模块常量 OVERFLOW_FIXED /
        OVERFLOW_REDUCE / OVERFLOW_TRUNCATE / OVERFLOW_KEEP_TAIL。
    构造入参说明：
        layer (str)：层名标识，与 self._budgets 的键一致。
        max_tokens (int)：该层最大 token 预算，>0 时该层视为启用。
        overflow (str)：溢出策略，默认 OVERFLOW_FIXED（不可截）。
    返回值说明：
        构造返回 Budget 实例；启用判定见 `enabled` 属性。
    """
    layer: str
    max_tokens: int
    overflow: str = OVERFLOW_FIXED

    @property
    def enabled(self) -> bool:
        """
        函数功能与逻辑描述：
            判断本层预算是否生效（max_tokens > 0），供调用方跳过未启用层。
        入参说明：
            无（仅读 self.max_tokens）。
        返回值说明：
            bool：预算 >0 返回 True，否则 False。
        """
        return self.max_tokens > 0


@dataclass
class ContextResult:
    """
    函数/类功能与逻辑描述：
        分层处理结果载体（dataclass），记录各层处理后的文本、近似 token 数与是否发生
        截断；layers 键仅含调用方传入的非空层，未传入或处理后为空的层不出现在键中。
        纯数据容器，无副作用。
    构造入参说明：
        layers (Dict[str, str])：层名 → 处理后文本，默认空字典。
        per_layer_tokens (Dict[str, int])：层名 → 处理后的近似 token 数，默认空字典。
        truncated (bool)：本次装配是否发生任一层截断，默认 False。
    返回值说明：
        构造返回 ContextResult 实例；按层取文本见 `layer` 方法。
    """
    layers: Dict[str, str] = field(default_factory=dict)
    per_layer_tokens: Dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def layer(self, name: str) -> str:
        """
        函数功能与逻辑描述：
            按层名取回处理后的文本，对缺失层做空串兜底，避免调用方 KeyError。
        入参说明：
            name (str)：层名，如 long_memory / user_docs / task_results / history。
        返回值说明：
            str：该层文本；层不存在时返回 ""。
        """
        return self.layers.get(name, "")


def _fit_prefix(text: str, max_tokens: int) -> str:
    """
    函数功能与逻辑描述：
        二分求满足预算的最大前缀（截尾保留头部，safe 方向）；预算充足时原样返回，
        否则二分搜索最大可放入前缀并 rstrip 去尾部空白。
        边界处理：max_tokens <= 0 直接返回空串；文本为空时估算为 0、原样返回空串。
    入参说明：
        text (str)：待截断文本。
        max_tokens (int)：可用 token 预算；<=0 视为无可分配预算。
    返回值说明：
        str：保留头部的截断结果；预算<=0 或文本为空时为 ""。
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip()


def _fit_suffix(text: str, max_tokens: int) -> str:
    """
    函数功能与逻辑描述：
        二分求满足预算的最大后缀（滑窗保留最近）；预算充足时原样返回，否则二分搜索
        最大可放入后缀并 lstrip 去首部空白。对应 history 层 OVERFLOW_KEEP_TAIL 策略。
        边界处理：max_tokens <= 0 直接返回空串；文本为空时估算为 0、原样返回空串。
    入参说明：
        text (str)：待截断文本。
        max_tokens (int)：可用 token 预算；<=0 视为无可分配预算。
    返回值说明：
        str：保留尾部的截断结果；预算<=0 或文本为空时为 ""。
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[len(text) - mid:]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[len(text) - lo:].lstrip()


def _split_paras(text: str) -> List[str]:
    """
    函数功能与逻辑描述：
        按连续换行切分段落，过滤空段并去除各段首尾空白，供 reduce_top_k 段落化
        保留头部使用。
    入参说明：
        text (str)：待切分的原始文本。
    返回值说明：
        List[str]：非空段落列表（已 strip）；无有效内容时返回空列表。
    """
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def _reduce_keep_head(text: str, max_tokens: int, min_kept: int = 1) -> str:
    """
    函数功能与逻辑描述：
        reduce_top_k 溢出策略实现（对应 long_memory / user_docs 的 OVERFLOW_REDUCE）：
        段落化后从头部累加保留，放不下时对当前段内部截尾并停止；段落不足 min_kept 时
        尽力保留第 1 段头部（宁少勿空）。
        边界处理：无有效段落时返回空串；首段即超预算时保留其头部而非丢弃。
    入参说明：
        text (str)：待压缩文本。
        max_tokens (int)：可用 token 预算。
        min_kept (int)：最低保留段数，默认 1；当前实现用于兜底保证至少保留第 1 段头部。
    返回值说明：
        str：以 "\n" 拼接的保留段落；无有效段落时为 ""。
    """
    paras = _split_paras(text)
    if not paras:
        return ""
    kept: List[str] = []
    used = 0
    for p in paras:
        pt = estimate_tokens(p)
        if used + pt <= max_tokens:
            kept.append(p)
            used += pt
            continue
        if not kept:  # 首段即超预算 → 内部截头保留
            kept.append(_fit_prefix(p, max_tokens))
            used = estimate_tokens(kept[-1])
        break
    if not kept and paras:
        kept.append(_fit_prefix(paras[0], max_tokens))
    return "\n".join(kept)


def _budget_blocks(blocks: List[Tuple[str, bool]], max_tokens: int) -> Tuple[List[str], bool]:
    """
    函数功能与逻辑描述：
        task_results 层块级保序治理：按顺序逐块放入，放不下的当前块内部截尾（保留头部）
        并停止后续块；受保护块（protected=True，确定性底稿）**不截**——宁超预算也不丢
        确定性事实（R9），超预算时记 WARN 并将剩余预算置 0。
    入参说明：
        blocks (List[Tuple[str, bool]])：块列表 `[(text, protected)]`；text 应为独立完整
            文本（如单个 agent 的结果文本），protected=True 表示确定性底稿不可截。
        max_tokens (int)：本层可用 token 预算。
    返回值说明：
        Tuple[List[str], bool]：第 1 项为保序保留的块文本列表（可能为空）；
            第 2 项为是否发生截断或受保护块超预算。
    """
    out: List[str] = []
    remain = max_tokens
    truncated = False
    for text, protected in blocks:
        t = estimate_tokens(text)
        if t <= remain:
            out.append(text)
            remain -= t
            continue
        if protected:
            out.append(text)
            truncated = True
            logger.warning(
                "[ContextManager] task_results 受保护块超预算仍完整保留 "
                "(R9 确定性事实不截) text_tokens=%d remain=%d", t, remain)
            remain = 0
            continue
        truncated = True
        if remain > 0:
            head = _fit_prefix(text, remain)
            if head:
                out.append(head)
        break
    return out, truncated


class ContextManager:
    """
    函数/类功能与逻辑描述：
        分层上下文预算管理器（纯函数，无状态，可复用/跨 async 安全）；按 system /
        long_memory / user_docs / task_results / history 五层装配预算，各层溢出时按
        对应策略截断，静态分层——某层不参与时份额空置为宽松余量、不赠与他人，避免
        "全部让给一层"击穿总窗口。组件不感知模板，拼接交由调用方。
    构造入参说明：
        context_window (int)：所选后端 context 窗口（默认 4096，对齐本地 ollama
            `num_ctx=4096`；M14 `Capabilities.context_window` 落地后由调用方传实际值）。
        output_reserve (int)：输出预留 token（对齐 num_predict / max_tokens，默认 1024）。
    返回值说明：
        构造返回 ContextManager 实例（无状态纯组件）；分层装配结果见 `build_context`。
    """

    def __init__(self, context_window: int = 4096, output_reserve: int = 1024):
        """
        函数功能与逻辑描述：
            依据输入窗口与输出预留推导各层预算：输入预算 = max(128, context_window -
            output_reserve)；system 固定预留 = min(SYSTEM_RESERVE_MAX,
            输入预算 * SYSTEM_RESERVE_RATIO)；其余四层按 LAYER_RATIOS 比例切分
            system 预留后的可分配池，并绑定各自溢出策略（long_memory/user_docs 用
            OVERFLOW_REDUCE、task_results 用 OVERFLOW_TRUNCATE、history 用
            OVERFLOW_KEEP_TAIL）。仅初始化实例字段，无外部副作用。
        入参说明：
            context_window (int)：所选后端 context 窗口，默认 4096。
            output_reserve (int)：输出预留 token，默认 1024；输入预算最低保底 128。
        返回值说明：
            无（仅设置 self._input_budget / self._system_budget / self._layer_pool /
            self._budgets 等实例属性）。
        """
        self._input_budget = max(128, context_window - output_reserve)
        self._system_budget = min(SYSTEM_RESERVE_MAX,
                                  round(self._input_budget * SYSTEM_RESERVE_RATIO))
        self._layer_pool = self._input_budget - self._system_budget
        self._budgets: Dict[str, Budget] = {
            "long_memory": Budget("long_memory",
                                  round(self._layer_pool * LAYER_RATIOS["long_memory"]),
                                  OVERFLOW_REDUCE),
            "user_docs": Budget("user_docs",
                                round(self._layer_pool * LAYER_RATIOS["user_docs"]),
                                OVERFLOW_REDUCE),
            "task_results": Budget("task_results",
                                   round(self._layer_pool * LAYER_RATIOS["task_results"]),
                                   OVERFLOW_TRUNCATE),
            "history": Budget("history",
                              round(self._layer_pool * LAYER_RATIOS["history"]),
                              OVERFLOW_KEEP_TAIL),
        }

    @property
    def input_budget(self) -> int:
        """
        函数功能与逻辑描述：
            暴露扣除输出预留后的输入预算（只读），供调用方查询总可用输入 token 额度。
        入参说明：
            无（仅读 self._input_budget）。
        返回值说明：
            int：输入预算 token 数，保底 >=128。
        """
        return self._input_budget

    @property
    def system_budget(self) -> int:
        """
        函数功能与逻辑描述：
            暴露 system 层固定预留预算（只读），供调用方预估 system 层占用上限。
        入参说明：
            无（仅读 self._system_budget）。
        返回值说明：
            int：system 固定预留 token 数，上限为 SYSTEM_RESERVE_MAX(400)。
        """
        return self._system_budget

    # ------------------------------------------------------------------
    def budget_for(self, layer: str) -> int:
        """
        函数功能与逻辑描述：
            查询某层预算 token（collect 等按块治理的调用方用）；直接透传对应 Budget 的
            max_tokens，不触发截断计算。
        入参说明：
            layer (str)：层名，须为已注册层（long_memory / user_docs / task_results /
                history）；传入未知层名会抛 KeyError。
        返回值说明：
            int：该层预算 token 数。
        """
        return self._budgets[layer].max_tokens

    def budget_blocks(self, blocks: List[Tuple[str, bool]]) -> Tuple[List[str], bool]:
        """
        函数功能与逻辑描述：
            task_results 层按块治理（collect 用，需保留块列表时）：以本层预算调用模块级
            `_budget_blocks` 保序放入，放不下当前块内部截尾并停止；受保护块不截（R9）。
        入参说明：
            blocks (List[Tuple[str, bool]])：块列表 `[(text, protected)]`，protected=True
                表示确定性底稿不可截。
        返回值说明：
            Tuple[List[str], bool]：保留块列表与是否发生截断（语义同 `_budget_blocks`）。
        """
        return _budget_blocks(blocks, self._budgets["task_results"].max_tokens)

    def truncate_text(self, text: str, max_tokens: int, mode: str = "tail") -> str:
        """
        函数功能与逻辑描述：
            对单段文本直接按预算截断，供不经分层装配的调用方按需使用。
        入参说明：
            text (str)：待截断文本，空串短路返回 ""。
            max_tokens (int)：可用 token 预算。
            mode (str)：截断方向，默认 "tail"（保留头部，走 `_fit_prefix`）；
                传 "head" 时保留尾部（滑窗，走 `_fit_suffix`）。
        返回值说明：
            str：截断后的文本；空输入或预算<=0 时为 ""。
        """
        if not text:
            return ""
        if mode == "head":
            return _fit_suffix(text, max_tokens)
        return _fit_prefix(text, max_tokens)

    # ------------------------------------------------------------------
    def build_context(
        self,
        *,
        system: str = "",
        history: str = "",
        long_memory: str = "",
        user_docs: str = "",
        task_results: Optional[List[Tuple[str, bool]]] = None,
    ) -> ContextResult:
        """
        函数功能与逻辑描述：
            分层预算装配主入口：system 层不可截（红线），超限仅 WARN 并保留全文；
            long_memory / user_docs 走 OVERFLOW_REDUCE（段落化保留头部，最低保留 1 段）、
            history 走 OVERFLOW_KEEP_TAIL（滑窗保留最近）、task_results 走块级保序治理。
            预算内逐字节零改动；仅处理调用方传入的非空层。组件不感知模板，避免与各 agent
            的 jinja 渲染耦合。
        入参说明：
            system (str)：system 角色/规则文本，不可截，默认 ""。
            history (str)：历史对话文本，滑窗保留最近，默认 ""。
            long_memory (str)：长期记忆检索文本，段落化保留头部，默认 ""。
            user_docs (str)：用户文档文本，段落化保留头部且最低保留 1 段，默认 ""。
            task_results (Optional[List[Tuple[str, bool]]])：块列表 `[(text, protected)]`；
                protected=True 的块（确定性底稿）不截（R9）；块内容应是一个独立完整文本
                （如单个 agent 的结果文本），默认 None（该层不参与）。
        返回值说明：
            ContextResult：layers 仅含调用方传入的非空层（处理结果），per_layer_tokens
                记录各层近似 token 数，任一层发生截断时 truncated 为 True。
        """
        if not system:
            sys_tokens = 0
        else:
            sys_tokens = estimate_tokens(system)
            if sys_tokens > self._system_budget:
                # system 不可截（红线）：超限仅 WARN 并保留全文——宁可可观测地超，不丢角色规则
                logger.warning(
                    "[ContextManager] system 层超固定预算不可截: %d > %d (保留全文)",
                    sys_tokens, self._system_budget)

        result = ContextResult()
        # 层文本 → (预算, 处理函数)，只处理调用方传入的非空层
        handlers: Dict[str, Tuple[Budget, object]] = {
            "history": (self._budgets["history"], _fit_suffix),
            "long_memory": (self._budgets["long_memory"], _reduce_keep_head),
            "user_docs": (self._budgets["user_docs"], _reduce_keep_head),
        }
        for name, (budget, fn) in handlers.items():
            text = {"history": history, "long_memory": long_memory,
                    "user_docs": user_docs}[name]
            if not text:
                continue
            tokens = estimate_tokens(text)
            if tokens <= budget.max_tokens:
                result.layers[name] = text
                result.per_layer_tokens[name] = tokens
                continue
            if budget.overflow == OVERFLOW_REDUCE:
                processed = fn(text, budget.max_tokens, 1)
            else:
                processed = fn(text, budget.max_tokens)
            result.layers[name] = processed
            result.per_layer_tokens[name] = estimate_tokens(processed)
            result.truncated = True
            logger.warning(
                "[ContextManager] 层 '%s' 超预算截断: %d → %d tokens (预算 %d)",
                name, tokens, result.per_layer_tokens[name], budget.max_tokens)

        # task_results：块级保序处理（列表在 layers 中还原拼接，供按块的调用方使用）
        if task_results:
            total = sum(estimate_tokens(t) for t, _ in task_results)
            if total <= self._budgets["task_results"].max_tokens:
                result.layers["task_results"] = "\n".join(
                    t for t, _ in task_results)
                result.per_layer_tokens["task_results"] = total
            else:
                blocks, truncated = self.budget_blocks(task_results)
                result.layers["task_results"] = "\n".join(blocks)
                result.per_layer_tokens["task_results"] = sum(
                    estimate_tokens(b) for b in blocks)
                result.truncated = result.truncated or truncated
                if truncated:
                    logger.warning(
                        "[ContextManager] task_results 超预算截断: %d → %d tokens",
                        total, result.per_layer_tokens["task_results"])

        return result


# 模块级共享实例（无状态纯组件，可安全跨 async 复用；调用方可按后端传窗口自建）
context_manager = ContextManager()
