# ==============================================
# JSON 输出解析器 - 带自校验与错误反馈重试
# 面向 1.8B 小模型设计：模型常输出 markdown 代码块、键名带空格、多段并列 JSON，
# 本模块统一做清洗 + 递归键名校验 + 缺字段重试，是 5 个现役 agent 与 orchestrator 的共用出口。
# ==============================================
import json
import re
from typing import Awaitable, Callable, Optional
from utils.retry_utils import async_retry

# 业务校验回调签名：入参为已解析出的结果字典，返回错误文案表示不通过，返回 None 表示通过
ValidationFunc = Callable[[dict], Optional[str]]


def clean_markdown_code_block(text: str) -> str:
    """
    函数功能与逻辑描述：
        剥离 LLM 输出外层包裹的 markdown 代码围栏（```json ... ``` 或 ``` ... ```），
        只返回围栏内部的正文。小模型习惯把 JSON 包在代码块里，不剥离会导致后续解析失败。
        仅匹配第一段围栏；未匹配到围栏时原样返回入参，不做任何裁剪或抛错。
    入参说明：
        text (str)：模型原始输出文本。
    返回值说明：
        str：围栏内的正文；无围栏时返回原文本本身（可能含首尾空白，不做 strip）。
    """
    pattern = r"```(?:json)?\n?([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text


def normalize_keys(data):
    """
    函数功能与逻辑描述：
        递归清洗字典的所有 key，移除其中全部空格。1.8B 小模型经常在 JSON 键名中插入空格
        （如 "consume_ date"、"category_ in"），导致标准键名校验失败，
        因此必须在 schema 校验前调用。对 list 递归下探，对其它类型原样返回；
        不修改入参对象，始终返回新构造的结构（非 dict/list 的叶子节点为同一引用）。
    入参说明：
        data：待清洗的对象，可为 dict / list / 任意标量；非容器类型直接透传。
    返回值说明：
        dict | list | any：清洗后的同构结构（dict 的键已去空格）；
            入参为标量时返回其本身。
    """
    if isinstance(data, dict):
        new_dict = {}
        for raw_key, value in data.items():
            clean_key = raw_key.replace(" ", "")
            new_dict[clean_key] = normalize_keys(value)
        return new_dict
    if isinstance(data, list):
        return [normalize_keys(item) for item in data]
    return data


def _try_loads_merged(raw: str):
    """
    函数功能与逻辑描述：
        兼容 LLM 输出多个并列顶层 JSON 对象的情形。策略为先整体 json.loads（对象/数组均原样返回）；
        失败时按 "}...{" 边界拆分顶层对象、逐个尝试解析，再**按各段 key 集合是否一致**分流：
        ① **各段 key 集合完全一致** → 判为「同构并列记录」（如一次多笔记账的
           `{"amount":30,...}, {"amount":20,...}`）→ **返回列表，绝不合并**；
           若走浅合并，同名键会互相覆盖 → **静默丢数据**（两笔账只落一笔）。
        ② **各段 key 集合不同** → 判为「同一逻辑对象被拆成多段」（如 stat 的
           `{filter...}, {agg_ops...}`）→ 保持**浅合并为单字典**（不可删除，否则 stat IR 解析会挂）。
        拆分后连一段都解析不出则返回失败。
    入参说明：
        raw (str)：候选 JSON 文本（通常已经过 clean_markdown_code_block 与正则截取）。
    返回值说明：
        tuple：二元组 (data, err_msg)
            - 成功：(dict | list, None)——整体解析结果，或按上述分流得到的 dict / list。
            - 失败：(None, 错误摘要)，如 "JSON语法错误: 非标准JSON"（拆不出多段）
              或 "JSON语法错误: 无法拆分解析"（拆出多段但无一可解析）。
    """
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    # 按 "}...{" 拆分顶层对象
    parts = re.split(r"\}\s*,?\s*\{", raw)
    if len(parts) < 2:
        return None, "JSON语法错误: 非标准JSON"
    objs = []
    for i, part in enumerate(parts):
        # 按段位置补齐花括号 —— 首段已自带前导 "{"、末段已自带结尾 "}"，中间段两者都缺。
        #   若无条件 `"{" + part + "}"`，会把首段拼成 "{{…"、末段拼成 "…}}" →
        #   **首末段永远解析失败**，只剩中间段能解析（"单笔也能成功"由此掩盖了缺陷）。
        frag = part
        if i > 0:
            frag = "{" + frag
        if i < len(parts) - 1:
            frag = frag + "}"
        try:
            obj = json.loads(frag)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    if not objs:
        return None, "JSON语法错误: 无法拆分解析"
    if len(objs) == 1:
        return objs[0], None
    # ★分流：各段 key 集合完全一致 → 同构并列记录（多笔）→ 返回列表，不做浅合并（防同名覆盖丢数据）
    keysets = [frozenset(o.keys()) for o in objs]
    if all(ks == keysets[0] for ks in keysets):
        return objs, None
    # 各段 key 集合不同 → 同一对象的拆分片段（stat 的 filter/agg_ops 场景）→ 沿用浅合并
    merged = {}
    for obj in objs:
        merged.update(obj)
    return merged, None


@async_retry(max_times=0)  # 关闭装饰器重试，由内部业务循环接管
async def parse_json_output(
    llm_call_func,
    sys_prompt: str,
    user_prompt: str,
    schema_example: str,
    validate: Optional[ValidationFunc] = None,
    max_self_retry: int = 2,
    agent_tag: str = "unknown",
    ignore_missing_keys: Optional[list] = None
):
    """
    函数功能与逻辑描述：
        带自校验 + 错误反馈重试的 JSON 解析器，是全项目 LLM 结构化输出的统一出口。
        单轮流程：先调用 llm_call_func 取原始输出，经 try_extract 清洗并解析，再依次做
        ① JSON 解析失败重试 ② 业务规则校验（validate 回调）重试 ③ 基于 schema_example
        提取字段集合做「必需字段」校验重试。每次失败都会把具体错误拼进下一轮的追加提示
        （【修正提示】/【重要修正提示】），让模型自我修正，最多 max_self_retry 轮。
        嵌套函数 try_extract 负责单轮提纯：剥离 markdown 围栏 → 拦截 FINANCE_ERROR 快速失败
        → 正则截取首个 JSON 对象 → 调用 _try_loads_merged；解析成功后统一做键名去空格
        （normalize_keys）。嵌套函数 _has_key 用于递归判断某个 schema 字段是否出现在结果树的
        任意层级，避免嵌套字段（stat 的 filter/agg_ops、orchestrator 的 tasks）被误判为缺失。
        特例规则：当结果中 need_more_info 为 True 时，只要求输出 prompt 追问话术，
        不再做常规必需字段检查。
    入参说明：
        llm_call_func：异步 LLM 调用函数，签名为
            await llm_call_func(sys_prompt, user_prompt, agent_tag=...)，返回原始文本。
        sys_prompt (str)：系统提示词。
        user_prompt (str)：用户提示词；重试时以它为基准追加错误提示，不会累积污染原文。
        schema_example (str)：JSON 示例文本，既用于提取必需字段集合，也用于回填修正提示。
        validate (Optional[ValidationFunc])：业务校验回调，返回字符串表示错误信息、返回 None 表示通过；默认 None（跳过该校验）。
        max_self_retry (int)：最大自我修正重试次数，默认 2（即最多 3 轮调用）。
        agent_tag (str)：当前调用 Agent 标识，透传给 LLM 调用函数用于差异化推理参数，默认 "unknown"。
        ignore_missing_keys (Optional[list])：必需字段校验豁免列表。小模型常漏输出某些字段，
            若调用方有业务兜底（默认值 / 追问），可在列表中声明这些字段，缺失时不再反复重试；默认 None。
    返回值说明：
        dict | list：解析并通过全部校验的结构化结果（键名已去空格）。
            通常为 dict；当模型输出**多个同构并列对象**（如一次多笔记账）时，`_try_loads_merged`
            会返回 **list**（避免浅合并导致同名键互相覆盖、静默丢数据），此处原样透传。
    异常说明：
        RuntimeError：当模型输出以 FINANCE_ERROR 开头时抛出，提示「LLM调用失败（非模型输出）」
            并附带前 200 字符原文（避免把 API 错误响应的 JSON 片段当模型输出误判重试）。
        ValueError：重试次数耗尽仍失败时抛出，文案区分三类：
            "多次尝试仍然解析失败，最后错误：..."、"多次修正仍然不满足业务规则：..."、
            "多次修正仍然缺少必需字段：..."。
    """
    current_user_prompt = user_prompt
    attempt = 0

    async def try_extract(content: str):
        """
        函数功能与逻辑描述：
            单轮输出的提纯函数：先剥离 markdown 代码围栏，再拦截 FINANCE_ERROR 前缀做快速失败，
            然后用正则截取第一个 {...} 片段并交由 _try_loads_merged 解析（兼容多段并列对象）。
            FINANCE_ERROR 快速失败：LLM 调用失败时返回的错误文本里常内嵌 API 错误响应的 JSON
            片段（如 402 配额耗尽的 {"error":{...}}），若不拦截会被当作模型输出解析成功，
            进而以「缺少必需字段」这类误导信息重试整轮——既浪费调用，又掩盖真实故障。
        入参说明：
            content (str)：本轮 LLM 的原始输出文本。
        返回值说明：
            tuple：二元组 (data, err_msg)。成功为 (dict, None)；
                找不到 JSON 对象时为 (None, "无法找到JSON对象")；
                找到但解析失败时透传 _try_loads_merged 的错误文案。
        异常说明：
            RuntimeError：输出以 FINANCE_ERROR 开头时抛出（非模型输出，立即失败）。
        """
        content = clean_markdown_code_block(content)
        # 失败即快速失败：错误文本内嵌的 JSON 片段若被当作模型输出，会以"缺少必需字段"
        #   这类误导信息重试整轮，既浪费调用又掩盖真实故障（配额耗尽 / 网络异常）。
        #   理由详见本函数 docstring。
        if content.lstrip().startswith("FINANCE_ERROR"):
            raise RuntimeError(f"LLM调用失败（非模型输出）：{content[:200]}")
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return None, "无法找到JSON对象"
        raw_json_str = json_match.group()
        return _try_loads_merged(raw_json_str)

    while attempt <= max_self_retry:
        raw_output = await llm_call_func(sys_prompt, current_user_prompt, agent_tag=agent_tag)
        result, err_msg = await try_extract(raw_output)
        # 解析成功后先清洗键名（去空格），再做 schema 校验，避免小模型键名空格导致误判缺失
        if result is not None:
            result = normalize_keys(result)

        # 1. JSON解析失败
        if err_msg is not None:
            attempt += 1
            if attempt > max_self_retry:
                raise ValueError(f"多次尝试仍然解析失败，最后错误：{err_msg}")
            # 追加错误提示，继续重试
            current_user_prompt = (
                user_prompt
                + f"\n\n【修正提示】上一轮输出错误：{err_msg}\n严格按照示例输出纯JSON：{schema_example}"
            )
            continue

        # 2. JSON解析成功，执行业务规则校验（日期/字段规则）
        if validate is not None:
            validate_err = validate(result)
            if validate_err is not None:
                attempt += 1
                if attempt > max_self_retry:
                    raise ValueError(f"多次修正仍然不满足业务规则：{validate_err}")
                current_user_prompt = (
                    user_prompt
                    + f"\n\n【重要修正提示】你的输出不符合业务约束：{validate_err}\n请严格修正，只输出合法JSON。示例：{schema_example}"
                )
                continue

        # 2.5 通用必需字段校验（基于 schema_example 提取字段集合，支持嵌套结构）
        # 防御 LLM 漏字段/拼错字段名（如 consume_ date），缺失即重试
        # 注意：schema 示例可能含嵌套结构（stat 的 filter/agg_ops、orchestrator 的 tasks），
        # 必须递归检查键是否存在于结果树任意层级，否则嵌套字段会被误判为缺失
        schema_keys = re.findall(r'"(\w+)"\s*:', schema_example)
        if schema_keys:

            def _has_key(node, key):
                """
                函数功能与逻辑描述：
                    递归判断指定键名是否出现在结果树的任意层级，用于必需字段校验时兼容嵌套结构
                    （如 stat 的 filter/agg_ops、orchestrator 的 tasks）。dict 命中即真，
                    未命中则继续下探其 values；list 逐个下探元素；标量返回假。
                入参说明：
                    node：当前遍历到的节点，可为 dict / list / 标量。
                    key (str)：待查找的键名。
                返回值说明：
                    bool：出现在任意层级则 True，否则 False。
                """
                if isinstance(node, dict):
                    if key in node:
                        return True
                    return any(_has_key(v, key) for v in node.values())
                if isinstance(node, list):
                    return any(_has_key(item, key) for item in node)
                return False

            # `result` 可能是 **list**（多个同构并列对象，见 `_try_loads_merged` 的分流）。
            #   list 形态没有 need_more_info 语义，直接按"每个元素都需含必需字段"校验即可；
            #   若无条件调 `result.get(...)`，多笔场景会抛 AttributeError。
            if isinstance(result, dict) and result.get("need_more_info") is True:
                schema_err = (
                    "need_more_info=true 时必须输出 prompt 追问话术"
                    if "prompt" in schema_keys and not result.get("prompt")
                    else None
                )
            else:
                missing = [k for k in schema_keys if not _has_key(result, k)]
                # 调用方声明豁免的字段缺失时不判错，交给业务侧兜底（默认值/追问）
                if ignore_missing_keys:
                    missing = [k for k in missing if k not in ignore_missing_keys]
                schema_err = f"缺少必需字段: {missing}" if missing else None
            if schema_err is not None:
                attempt += 1
                if attempt > max_self_retry:
                    raise ValueError(f"多次修正仍然缺少必需字段：{schema_err}")
                current_user_prompt = (
                    user_prompt
                    + f"\n\n【重要修正提示】你的输出缺少必需字段：{schema_err}\n请严格补全所有字段，只输出纯JSON。示例：{schema_example}"
                )
                continue

        # 全部校验通过，返回结果
        return result

    raise RuntimeError("超出最大自修正重试次数")
