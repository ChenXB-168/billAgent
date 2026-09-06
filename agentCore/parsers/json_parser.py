import json
import re
from typing import Awaitable, Callable, Optional
from utils.retry_utils import async_retry

ValidationFunc = Callable[[dict], Optional[str]]

def clean_markdown_code_block(text: str) -> str:
    pattern = r"```(?:json)?\n?([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text


def normalize_keys(data):
    """
    递归清洗字典所有key，移除全部空格。
    1.8B小模型经常在JSON键名中插入空格（如 "consume_ date"、"category_ in"），
    导致标准键名校验失败。必须在 schema 校验前调用。
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
    兼容 LLM 输出多个并列顶层 JSON 对象（如 stat 场景 {filter...}, {agg_ops...}）。
    整体解析失败时，按顶层对象拆分后逐个解析并浅合并。
    :return: (data, err_msg)
    """
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    # 按 "}...{" 拆分顶层对象
    parts = re.split(r"\}\s*,?\s*\{", raw)
    if len(parts) < 2:
        return None, "JSON语法错误: 非标准JSON"
    merged = {}
    for part in parts:
        frag = "{" + part + "}"
        try:
            obj = json.loads(frag)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            merged.update(obj)
    if merged:
        return merged, None
    return None, "JSON语法错误: 无法拆分解析"

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
    带自校验+错误反馈重试的JSON解析器
    :param validate: 业务校验回调，返回字符串=错误信息，返回None=校验通过
    :param max_self_retry: 最大自我修正重试次数
    :param agent_tag: 当前调用Agent标识，透传给LLM调用函数用于差异化推理参数
    :param ignore_missing_keys: 通用字段校验豁免列表。小模型经常漏输出某些字段，
        若调用方有业务兜底（默认值/追问），可在列表中声明这些字段，缺失时不再反复重试
    """
    current_user_prompt = user_prompt
    attempt = 0

    async def try_extract(content: str):
        content = clean_markdown_code_block(content)
        # ★2026-08-31 修复（M1 实测）：LLM 调用失败时返回的是 FINANCE_ERROR 文本，
        #   其中常内嵌 API 错误响应的 JSON 片段（如 402 配额耗尽的 {"error":{...}}）。
        #   若不拦截，该片段会被下面当作模型输出解析成功，进而以"缺少必需字段：
        #   ['tasks','pre_check_pass']"这类误导信息重试整轮——既浪费调用，又掩盖
        #   真实故障（配额耗尽/网络异常）。失败即快速失败，交由上层直接暴露原因。
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
                if isinstance(node, dict):
                    if key in node:
                        return True
                    return any(_has_key(v, key) for v in node.values())
                if isinstance(node, list):
                    return any(_has_key(item, key) for item in node)
                return False

            if result.get("need_more_info") is True:
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