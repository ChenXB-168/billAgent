from modelscope import snapshot_download
import os, sys

# 模型保存目录
SAVE_DIR = "./modelService/embedding"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 下载 Embedding 向量模型 bge-small-zh-v1.5
emb_model_dir = snapshot_download(
    model_id="BAAI/bge-small-zh-v1.5",
    local_dir=os.path.join(SAVE_DIR, "bge-small-zh-v1.5")
)
print(f"Embedding 模型下载完成：{emb_model_dir}", file=sys.stderr)

# 2. 下载 Rerank 重排模型 bge-reranker-base
rerank_model_dir = snapshot_download(
    model_id="BAAI/bge-reranker-base",
    local_dir=os.path.join(SAVE_DIR, "bge-reranker-base")
)
print(f"Rerank 模型下载完成：{rerank_model_dir}", file=sys.stderr)