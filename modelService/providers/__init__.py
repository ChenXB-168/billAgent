# ==============================================
# 模型后端 Provider 集合 - 统一导出本地/外部两个后端实现
# 边界：Provider 只管「组装请求 → 发请求 → 归一响应」，路由/重试/fallback 全在 ModelRouter
# ==============================================
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider

__all__ = ["OllamaProvider", "OpenAICompatProvider"]
