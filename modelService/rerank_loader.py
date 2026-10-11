import torch, sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from pathlib import Path

# ===================== 全局路径配置 =====================
BASE_DIR = Path(__file__).parent.parent
RERANK_MODEL_PATH = BASE_DIR / "modelService" / "embedding" / "bge-reranker-base"
DEVICE = "cpu"

class RerankService:
    """
    函数/类功能与逻辑描述：
        检索重排服务：对向量 / BM25 初筛召回的多条文档按「与 query 的相关性」二次打分并降序排序，
        供 RAG 精排使用。构造即加载本地 bge-reranker-base（CPU）；加载失败仅降级（self.model 为空），
        由 rerank 侧守卫返回空列表，不抛异常。
    构造入参说明：
        无。
    返回值说明：
        构造返回 RerankService 实例；构造副作用为立即执行 _load_model()，加载模型到 self.model / self.tokenizer。
    """

    def __init__(self):
        """
        函数功能与逻辑描述：
            构造即初始化模型与分词器占位，并立即调用 _load_model() 完成本地模型加载
            （与 EmbeddingService 的懒加载不同，本服务构造即加载）。
        入参说明：
            无。
        返回值说明：
            无（副作用：写入 self.model / self.tokenizer 并触发 _load_model()）。
        """
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """
        函数功能与逻辑描述：
            加载本地重排模型 + 分词器：从 RERANK_MODEL_PATH 读取 bge-reranker-base，模型
            .to(DEVICE)（CPU）并置 eval() 关闭梯度提速减内存；加载异常仅打印 stderr 提示、
            self.model 保持为 None，由 rerank 侧守卫降级。
        入参说明：
            无。
        返回值说明：
            无（副作用：写 self.tokenizer / self.model；成功或失败均打印一条 stderr 提示）。
        """
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
        函数功能与逻辑描述：
            重排核心接口：将 query 与每条文档组成 pair，经 tokenizer（truncation、max_length=512）
            送入模型取 logits 分数，按分数从高到低排序；top_k 为真值时截断取前 top_k 条。
            模型 / 分词器未就绪或 doc_list 为空时直接返回空列表；推理在 torch.no_grad() 下进行。
        入参说明：
            query (str)：用户提问（重排基准）。
            doc_list (List[str])：初步召回的文档列表。
            top_k (int)：可选，默认 None；为真值时只保留前 top_k 条，否则返回全部。
        返回值说明：
            list：排序后的 [(文档文本, 相似度分数), ...]，分数越高越相关；未就绪 / 入参为空时返回 []。
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