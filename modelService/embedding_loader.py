import sys
from pathlib import Path
from typing import List

from config.config import EMBEDDING_ENABLED

# ===================== 全局路径 & 设备配置 =====================
# 项目根目录（自动适配路径，不用手动修改）
BASE_DIR = Path(__file__).parent.parent
# 向量模型本地路径
EMB_MODEL_PATH = BASE_DIR / "modelService" / "embedding" / "bge-small-zh-v1.5"
# 强制使用CPU（适配你的离线CPU部署）
DEVICE = "cpu"


class EmbeddingService:
    """
    向量化服务：文本转向量数组，供 RAG / 记忆语义检索使用。
    设计：单例加载，模型只加载一次，常驻内存，反复调用。

    2026-09-05 懒加载重构（本地 embedding 默认启用，不拖慢启动）：
      - **构造不再加载模型**，也不再于 import 期拉起 torch / sentence_transformers
        （二者延迟到首次真正编码时才 import）——因此 `import modelService.embedding_loader`
        （orchestrator / bootstrap 启动链必经）不再造成 ~10s 冷启动延迟；
      - 首次调用 `get_embedding / get_batch_embedding` 时经 `_ensure_loaded()` 加载一次并常驻；
        加载失败（模型缺失 / import 错误）→ 本进程内自动降级返回空向量，
        下游 user_docs（BM25）/ long_memory（jieba 关键词）有守卫，不会崩；
      - `EMBEDDING_ENABLED=False`（BILLAGENT_EMBEDDING=0）时彻底跳过加载，
        仅用于特殊"纯外部 API 极速演示"部署。
    """

    def __init__(self):
        self.model = None
        self._enabled = EMBEDDING_ENABLED
        self._load_attempted = False  # 尝试过一次（无论成败）即不再重复 import，避免每调用都白耗
        if not self._enabled:
            print("Embedding 按配置禁用（BILLAGENT_EMBEDDING=0）→ 检索自动降级 BM25/jieba",
                  file=sys.stderr)

    def _ensure_loaded(self) -> None:
        """懒加载入口：首次真正编码时才加载模型（一次性 ~10s，进程内常驻）。"""
        if not self._enabled or self.model is not None or self._load_attempted:
            return
        self._load_attempted = True
        try:
            # 延迟导入：仅在真正要加载模型时才拉起 torch / sentence_transformers
            from sentence_transformers import SentenceTransformer
            # 加载本地模型，指定运行设备
            self.model = SentenceTransformer(str(EMB_MODEL_PATH), device=DEVICE)
            print("Embedding 向量模型加载成功（首次使用懒加载，本进程已就绪）", file=sys.stderr)
        except Exception as e:
            print(f"向量模型加载失败（后续检索自动降级 BM25/jieba，重启进程可重试）：{str(e)}",
                  file=sys.stderr)

    def get_embedding(self, text: str) -> List[float]:
        """
        对外核心接口：单条文本转向量
        :param text: 输入文本（账单、问题、知识库内容）
        :return: 浮点型向量列表；模型不可用/失败时返回 []（下游据此降级）
        """
        self._ensure_loaded()
        # 模型未加载（禁用/失败）直接返回空
        if not self.model:
            return []
        try:
            # 文本编码为向量，不返回张量，转为普通列表
            vec_result = self.model.encode(text, convert_to_tensor=False)
            return vec_result.tolist()
        except Exception as e:
            print(f"单文本向量化失败：{str(e)}", file=sys.stderr)
            return []

    def get_batch_embedding(self, text_list: List[str]) -> List[List[float]]:
        """
        对外接口：批量文本转向量（用于RAG知识库建库）
        :param text_list: 文本列表
        :return: 二维向量数组；模型不可用/失败时返回 []
        """
        self._ensure_loaded()
        if not self.model or not text_list:
            return []
        try:
            batch_result = self.model.encode(text_list, convert_to_tensor=False)
            return batch_result.tolist()
        except Exception as e:
            print(f"批量向量化失败：{str(e)}", file=sys.stderr)
            return []


# 全局单例：整个项目只实例化一次。注意：懒加载——import 本模块只创建空壳，
# 模型在首次 get_embedding / get_batch_embedding 时才加载。
embedding_client = EmbeddingService()
