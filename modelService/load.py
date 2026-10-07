# ==============================================
# 向量与重排模型下载脚本（一次性初始化用，非运行期依赖）
# 用途：从 ModelScope 拉取 bge-small-zh-v1.5（Embedding）与 bge-reranker-base（Rerank）
#       到 modelService/embedding/ 下，之后由 embedding_loader / rerank_loader 本地加载
# 注意：本脚本为顶层直执行脚本，无函数；重复执行会走 ModelScope 缓存，不会重复下载全量
# ==============================================
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
