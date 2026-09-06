import torch, sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from pathlib import Path

# ===================== 全局路径配置 =====================
BASE_DIR = Path(__file__).parent.parent
RERANK_MODEL_PATH = BASE_DIR / "modelService" / "embedding" / "bge-reranker-base"
DEVICE = "cpu"

class RerankService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载本地重排模型+分词器"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(RERANK_MODEL_PATH))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(RERANK_MODEL_PATH)
            ).to(DEVICE)
            self.model.eval()  # 推理模式，关闭训练梯度，提速减内存
            print("Rerank 重排模型加载完成，常驻内存", file=sys.stderr)
        except Exception as e:
            print(f"重排模型加载失败：{str(e)}", file=sys.stderr)

    def rerank(self, query: str, doc_list: List[str], top_k=None):
        """
        重排核心接口
        :param query: 用户提问
        :param doc_list: 初步召回的文档列表
        :return: 排序后的列表 (文档文本, 相似度分数)，分数越高越相关
        """
        if not self.model or not self.tokenizer or not doc_list:
            return []

        pairs = [[query, doc] for doc in doc_list]
        results = []

        with torch.no_grad():  # 关闭梯度，大幅降低CPU占用
            for pair in pairs:
                inputs = self.tokenizer(
                    pair[0], pair[1],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(DEVICE)
                score = self.model(**inputs).logits.view(-1).float().item()
                results.append((pair[1], score))

        # 按分数从高到低排序
        results.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            results = results[:top_k]
        return results

# 全局单例
rerank_client = RerankService()

# # ========== 自测代码 ==========
# if __name__ == "__main__":
#     # 模拟RAG场景：用户提问 + 初步召回的多条文档
#     user_query = "我这个月餐饮花了多少钱？"
#     recall_docs = [
#         "本月餐饮总支出300元",
#         "本月交通支出120元",
#         "上个月娱乐消费200元",
#         "本月餐饮包含晚餐、奶茶共300元"
#     ]
#     # 重排打分
#     sorted_docs = rerank_client.rerank(user_query, recall_docs)
#     print("重排结果（分数越高越相关）：")
#     for doc, score in sorted_docs:
#         print(f"分数：{score:.4f} | 内容：{doc}")