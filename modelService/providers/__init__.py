"""M14（D17）模型后端 Provider 集合。

OllamaProvider（本地）/ OpenAICompatProvider（外部）——各自只做：
把**已归一参数**组装成本后端请求体 → 发请求 → 把本后端响应归一成 ChatResult / ChatError。
❌ 不决定走哪个后端（路由在 ModelRouter）；❌ 不做跨端参数转译；❌ 不处理 fallback（在 Router）。
"""
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider

__all__ = ["OllamaProvider", "OpenAICompatProvider"]
