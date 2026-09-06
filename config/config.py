# ==============================================
# 全局配置文件 - billAgent 项目专用
# 规范：全部业务常量大写，分层隔离，推理参数与业务参数分离
# ==============================================
from pathlib import Path
import os
from typing import Dict, Final

# --------------------------
# 基础路径常量
# --------------------------
BASE_DIR: Final[Path] = Path(__file__).parent.parent

# --------------------------
# 加载项目根 .env（密钥不入库）
# --------------------------
# load_dotenv 默认 override=False —— **不覆盖**已存在的环境变量，
# 因此 README 的 `$env:BILLAGENT_*` 会话级设置优先级更高，两种方式可共存。
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # python-dotenv 缺失时静默降级为"纯环境变量"模式
    pass

# 数据库
DB_PATH: Final[Path] = BASE_DIR / "database" / "bill.db"
# M5（D2）：任务状态**独立库**——Durable 每次状态流转一次写，与业务库同库会争抢 SQLite
# 写锁；独立库也便于清理与备份隔离（`04` §2.4 / `08` §6 Q4 推荐）
TASK_STATE_DB: Final[Path] = BASE_DIR / "database" / "task_state.db"
# 启动重投窗口（秒）：status='submitted' 超过该时长的任务视为**过期遗留**（测试/演示残留，
# 崩溃前排队任务只会在队列里停留毫秒级），不再自动重投、直接标 failed，避免每次演示启动
# 都自动执行一批历史脏任务（会真的调 LLM / 写账）。
REQUEUE_STALE_AFTER_SEC: Final[float] = float(os.getenv("BILLAGENT_REQUEUE_STALE_AFTER", "1800"))
# 日志
LOG_PATH: Final[Path] = BASE_DIR / "logs" / "app.log"
AUDIT_LOG_PATH: Final[Path] = BASE_DIR / "logs" / "audit.log"
# RAG & Embedding & LORA 模型路径
RAG_PATH: Final[Path] = BASE_DIR / "ragKnowledge"
# 用户文档库（用户笔记 / 「用户资料」；01 §3.5 新增需求，04 §3.5 设计，M8 交付 memory/user_docs.py）
USER_DOCS_PATH: Final[Path] = BASE_DIR / "userDocs"
# 用户文档库检索默认返回条数（`memory/user_docs.py` search_user_docs top_k）
USER_DOCS_TOP_K: Final[int] = int(os.getenv("BILLAGENT_USER_DOCS_TOP_K", "3"))
# 用户文档库文档数上限（`10` §5.5 红线：超出明确报错）
USER_DOCS_MAX_DOCS: Final[int] = int(os.getenv("BILLAGENT_USER_DOCS_MAX_DOCS", "1000"))
# 用户文档超长行硬切长度（分块，复用 split_text 语义）
USER_DOCS_CHUNK_SIZE: Final[int] = int(os.getenv("BILLAGENT_USER_DOCS_CHUNK_SIZE", "128"))
# 用户文档操作超时阈值（秒，预留扩展：当前实现为同步构建，由调用方 to_thread 控制）
USER_DOCS_TIMEOUT: Final[float] = float(os.getenv("BILLAGENT_USER_DOCS_TIMEOUT", "60"))
MODEL_PATH: Final[Path] = BASE_DIR / "modelService"
LORA_DATASET: Final[Path] = MODEL_PATH / "loraDataset"
LORA_WEIGHT: Final[Path] = MODEL_PATH / "loraWeight"
EMB_MODEL_PATH: Final[Path] = MODEL_PATH / "embedding"

# 本地向量模型开关（用户文档 M8 语义检索 / 长期记忆向量路由用，见 modelService/embedding_loader.py）：
#   **默认启用（=1）**——本地 bge embedding 是系统既有向量能力，与"本地 LLM"开关无关。
#   embedding_loader 为**懒加载**：进程启动不加载模型、不拉起 torch/sentence-transformers；
#   首次真正向量编码（用户文档索引/检索、长期记忆索引）时才加载一次并常驻，启动不再变慢。
#   BILLAGENT_EMBEDDING=0 仅用于特殊"纯外部 API 极速演示"场景：跳过加载，检索自动降级
#   BM25 / jieba 关键词（`memory/user_docs.py` / `memory/long_memory.py` 均有降级守卫，不崩）。
EMBEDDING_ENABLED: Final[bool] = os.getenv("BILLAGENT_EMBEDDING", "1") == "1"

# 自动初始化目录，首次运行不会报目录不存在
def init_project_dirs():
    dir_list = [
        os.path.dirname(DB_PATH),
        os.path.dirname(LOG_PATH),
        os.path.dirname(AUDIT_LOG_PATH),
        RAG_PATH,
        USER_DOCS_PATH,
        LORA_DATASET,
        LORA_WEIGHT,
        EMB_MODEL_PATH
    ]
    for d in dir_list:
        os.makedirs(d, exist_ok=True)

# 程序启动自动创建目录
init_project_dirs()

# --------------------------
# Ollama 模型通信基础配置
# --------------------------
OLLAMA_API_URL: Final[str] = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL_NAME: Final[str] = "qwen-bill-1.5-1.8b-q4km"
FINANCE_LORA_MODEL: Final[str] = "qwen-bill-1.5-1.8b-q4km"
MODEL_TIMEOUT: Final[int] = 240  # CPU长文本超时，单位秒
# M14（D17）：OllamaProvider 上下文窗口（对齐 num_ctx=4096，`Capabilities.context_window` 单一来源）
OLLAMA_CONTEXT_WINDOW: Final[int] = int(os.getenv("BILLAGENT_OLLAMA_CONTEXT_WINDOW", "4096"))

# --------------------------
# 外部API强模型配置（OpenAI兼容协议 /v1/chat/completions）
# 主Agent(orchestrator_agent) 与 理财Agent(finance_agent) 优先走外部强模型
# 未配置 api_key 时自动回退本地 Ollama 小模型，保证链路不中断
# 可通过环境变量覆盖，避免密钥硬编码进仓库
# --------------------------
EXTERNAL_LLM_BASE_URL: Final[str] = os.getenv("BILLAGENT_EXT_LLM_BASE_URL", "").rstrip("/")
EXTERNAL_LLM_API_KEY: Final[str] = os.getenv("BILLAGENT_EXT_LLM_API_KEY", "")
# 主Agent强模型名，默认兼容 DeepSeek 风格（可按需改 qwen-plus / gpt-4o 等）
EXTERNAL_ORCH_MODEL: Final[str] = os.getenv("BILLAGENT_EXT_ORCH_MODEL", "deepseek-chat")
# 理财Agent强模型名
EXTERNAL_FINANCE_MODEL: Final[str] = os.getenv("BILLAGENT_EXT_FINANCE_MODEL", "deepseek-chat")
# 外部API超时（秒），外网链路更慢，放宽到5分钟
EXTERNAL_LLM_TIMEOUT: Final[int] = int(os.getenv("BILLAGENT_EXT_TIMEOUT", "300"))
# M14（D17）：OpenAICompatProvider 上下文窗口（保守默认 8k，`Capabilities.context_window` 单一来源；
#   可按实际所选模型窗口覆盖，如 glm-4-flash 128k / deepseek 32k 设为更大值以放宽容错）
EXTERNAL_CONTEXT_WINDOW: Final[int] = int(os.getenv("BILLAGENT_EXT_CONTEXT_WINDOW", "8192"))
# 是否启用外部API（base_url + api_key 齐备才启用）
EXTERNAL_LLM_ENABLED: Final[bool] = bool(EXTERNAL_LLM_BASE_URL and EXTERNAL_LLM_API_KEY)

# --------------------------
# 外部模型路由：哪些 Agent 走外部强模型
# --------------------------
# 默认仅主调度走外部 —— 本地 1.8B 兜底，保证「不配任何 API 也能跑通全流程」(`01` §7 本地可用性)
EXTERNAL_LLM_AGENTS: Final[frozenset] = frozenset({"orchestrator_agent"})
# 测试/联调开关（BILLAGENT_FORCE_EXTERNAL=1）：**所有** Agent 一律走外部强模型。
# 用途：验证 harness 可用性与正确性时，绕开本地 1.8B 的 CPU 推理——
#   实测本地单次调用可达 189s，e2e 4 条用例耗时 36 分钟，且小模型 JSON 输出不稳定。
#   与 `05` §9.8.7 P2「子 Agent 全走外部 API」方向一致。
# 未配 API 时本开关自动失效（仍要求 EXTERNAL_LLM_ENABLED=True）。
FORCE_EXTERNAL_LLM: Final[bool] = os.getenv("BILLAGENT_FORCE_EXTERNAL", "0") == "1"
# 硬开关（BILLAGENT_DISABLE_LOCAL_LLM=1）：**封死**本地 Ollama 通道。
# 任何走到 `ollama_base_call` 的调用**立即抛错**，而不是回退或静默失败。
# 与 FORCE_EXTERNAL_LLM 的区别：后者是"优先走外部，失败仍可回退本地"；
# 前者是"直接禁用本地，走到即判定为路由缺陷并报错"——排查期用来杜绝意外走本地。
DISABLE_LOCAL_LLM: Final[bool] = os.getenv("BILLAGENT_DISABLE_LOCAL_LLM", "0") == "1"

# --------------------------
# MCP 常驻服务（M3，`03_逻辑架构.md` §9.8.2 / `07_部署架构.md` §4 决策1）
# --------------------------
# 绑定 127.0.0.1 不对外暴露（§9.8.2 ④ 风险7）；端口基址 8000 起逐 +1，
# 环境变量可覆盖：BILLAGENT_MCP_PORT（基址）/ BILLAGENT_MCP_STDIO（回滚开关）。
MCP_HOST: Final[str] = "127.0.0.1"
MCP_PORT_BASE: Final[int] = int(os.getenv("BILLAGENT_MCP_PORT", "8000"))
# 回滚开关（M3 回滚要求）：BILLAGENT_MCP_STDIO=1 切回 stdio 短连接
MCP_STDIO_FALLBACK: Final[bool] = os.getenv("BILLAGENT_MCP_STDIO", "0") == "1"
# server_key → 端口（`07` §4 决策1：bill=8001 / llm_base=8002 / llm_finance=8003）
MCP_SERVER_PORTS: Final[Dict[str, int]] = {
    "sql_bill": MCP_PORT_BASE + 1,
    "llm_base": MCP_PORT_BASE + 2,
    "llm_finance": MCP_PORT_BASE + 3,
}
# MCP 挂载路径（FastMCP streamable-http 默认 /mcp，显式声明避免魔法值）
MCP_HTTP_PATH: Final[str] = "/mcp"

# --------------------------
# M4 沙箱（D8）：进程级资源限制 + 工具级超时（`06` §2.5 / `11` M4）
#   内存硬上限：Windows 由 bootstrap 挂 Job Object；POSIX 由 3 个 server self-prison
#   （`mcpGateway/sandbox.py`）。单 server 常驻峰值约 200~400MB，2048MB 留足余量。
SANDBOX_MEMORY_MB: Final[int] = int(os.getenv("BILLAGENT_SANDBOX_MEMORY_MB", "2048"))
# 工具级超时（引擎层 `asyncio.wait_for(ToolSpec.timeout)`）：
#   SQL=30s：SQLite 本地库 + busy_timeout 5s，正常调用毫秒级（M3 实测 135ms），30s 极宽裕；
#   LLM=300s：M3 常驻化消除 26~29s 冷启动后，本地 1.8B 单次实测 189s、外部 API 更快，
#   300s 是防挂死兜底（原 900s 是 M1 stdio 冷启动期的"绝不误杀"值，见 registry._build_platform）。
#   ≤0 = 关闭超时（M4 回滚要求"超时阈值可配置关闭"）。
TOOL_TIMEOUT_SQL: Final[float] = float(os.getenv("BILLAGENT_TOOL_TIMEOUT_SQL", "30"))
TOOL_TIMEOUT_LLM: Final[float] = float(os.getenv("BILLAGENT_TOOL_TIMEOUT_LLM", "300"))

# --------------------------
# M5（D5）：追问治理（`01` §3.4 / §8 Q5 决策 / 技术待办 T2 / T3）
#   规则：信息不全最多追问 3 轮，第 3 轮仍不满足 → **直接放弃并报错**（不再追问、也不静默跳过）
ASK_ROUND_LIMIT: Final[int] = int(os.getenv("BILLAGENT_ASK_ROUND_LIMIT", "3"))
# 放弃提示语：`01` §3.4 明文规定，改动即属需求变更
ASK_GIVEUP_TIP: Final[str] = "信息不全，执行失败，请重新来"

# --------------------------
# M6（P0-2 依赖动态化，`03` §9.8.5 / D-b）：`final_deps = 静态表 ∪ LLM 输出的 deps`
#   ★安全不变式：LLM **只能增加**依赖、不能删除静态表依赖 ⇒ 动态性只往"更保守（更串行）"
#     方向走，最坏退化为改造前的串行，**绝不比现状更差**。
#   0 = 关闭动态依赖，纯静态表（M6 回滚开关）
DYNAMIC_DEPS_ENABLED: Final[bool] = os.getenv("BILLAGENT_DYNAMIC_DEPS", "1") != "0"
# 调度轮次上限：M6 起按**波次**递增（单次请求最多 2~3 波），由 20 下调至 8（`03` §9.8.4 ②）
MAX_DISPATCH_ROUND: Final[int] = int(os.getenv("BILLAGENT_MAX_DISPATCH_ROUND", "8"))
# M6（P1 批量派发，`03` §9.8.4）：0 = **退回单派发**（`ready[0]`）——M6 回滚开关，
#   也是同环境 A/B 对比（并行 vs 串行）的测量手段
BATCH_DISPATCH_ENABLED: Final[bool] = os.getenv("BILLAGENT_BATCH_DISPATCH", "1") != "0"

# --------------------------
# 可观测开关（D3 Tracer，见 `03_逻辑架构.md` §9.3）
# --------------------------
# BILLAGENT_TRACER_ENABLED=0 可完全关闭埋点（默认开启）。
# ★设计铁律（`03` §9.3.1「支撑层」）：可观测属横切能力，**失败不得阻断主链路**——
#   Tracer 内部所有异常自行吞掉，最坏情况只是统计缺失，绝不影响业务返回。
TRACER_ENABLED: Final[bool] = os.getenv("BILLAGENT_TRACER_ENABLED", "1") == "1"

# LLM通用停止符，全局统一
LLM_STOP_TOKENS: Final[list] = ["###", "-----"]

# --------------------------
# LLM 推理基础超参（所有Agent继承基础值）
# --------------------------
BASE_LLM_OPTIONS: Final[Dict] = {
    "temperature": 0.3,
    "num_predict": 1000,
    "num_ctx": 4096,
    "top_p": 0.8,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "stop": LLM_STOP_TOKENS
}

# 各Agent独立差异化推理参数，key统一 agent_xxx 格式
AGENT_LLM_CONFIG: Final[Dict[str, Dict]] = {
    # 理财Agent：极低随机、四段式固定输出
    # ★2026-08-31 num_predict 上调至 2048：外部 hy3 为推理模型，800 的预算
    #   可能被 reasoning 占满导致 content 为空（同 bill_agent 缺陷）。
    "finance_agent": {
        "temperature": 0.1,
        "num_predict": 2048,
        "stop": ["五、", "更多省钱方案"]
    },
    # 物价对比Agent：temperature=0 禁止编造金额、城市
    "price_agent": {
        "temperature": 0.0,
        "num_predict": 2048,
        "top_p": 0.6
    },
    # 记账账单Agent：简短分类总结
    # ★2026-08-31 num_predict 400→2048（M1 实测发现的缺陷）：
    #   该参数会透传为外部 API 的 max_tokens；外部 hy3 为推理模型，400 的预算
    #   会被较长 reasoning 占满，导致 message.content 为空、下游解析失败。
    "bill_agent": {
        "temperature": 0.2,
        "num_predict": 2048
    },
    # 统计报表Agent：低随机，百分比表格固定格式
    "stat_agent": {
        "temperature": 0.1,
        "num_predict": 2048
    },
    # 调度分发Agent（orchestrator）：无差异化参数，复用BASE_LLM_OPTIONS
    "orchestrator_agent": {}
}

# --------------------------
# 外部 API（OpenAI 兼容）推理参数层 —— 与本地 Ollama 参数严格分离
# --------------------------
# ★语义隔离（T4，见 `08_架构演进方案与决策记录.md` 第 7 章 7.1）：
#   num_predict（本地"预测多少 token"的软约束）≠ max_tokens（外部"输出硬上限"，
#   且**含 reasoning 消耗**）。二者语义不同，**禁止互转**——
#   external_llm_call 只认本层白名单，ollama_base_call 只认 BASE_LLM_OPTIONS。
#   历史缺陷：num_predict=400 被隐式透传为 max_tokens，hy3 推理模型的 reasoning
#   吃满预算导致 content 为空、连续重试烧 token（M1 实测）。
EXTERNAL_BASE_LLM_OPTIONS: Final[Dict] = {
    "temperature": 0.3,
    "max_tokens": 4096,   # 推理模型：reasoning + 正文预算（400 会被 reasoning 挤空 content）
}

# 各 Agent 外部差异化参数（仅走外部的 Agent 需要；其余 fallback EXTERNAL_BASE_LLM_OPTIONS）
AGENT_EXTERNAL_LLM_CONFIG: Final[Dict[str, Dict]] = {
    # 调度分发Agent（orchestrator）：复用外部基础层，无额外差异
    "orchestrator_agent": {"max_tokens": 4096},
    # 理财Agent：极低随机，四段式固定输出（外部走强模型时同样保持低随机）
    "finance_agent": {"temperature": 0.1, "max_tokens": 4096},
}

# --------------------------
# 全局业务配置（独立块，与LLM推理参数隔离）
# --------------------------
# 用户默认城市，price_agent无提取城市时兜底读取
DEFAULT_USER_CITY: Final[str] = "深圳"

# 三档消费阈值系数（与 user_config 表 CHECK 约束('节俭','正常','宽松')全链路统一中文）
CONSUMPTION_MODES: Final[Dict[str, float]] = {
    "节俭": 0.1,
    "正常": 0.3,
    "宽松": 0.6
}

# --------------------------
# 数据库权限常量（避免硬编码字符串）
# --------------------------
TABLE_BILL: Final[str] = "bill"
PERM_NONE: Final[str] = "none"
PERM_READ: Final[str] = "read"
PERM_WRITE: Final[str] = "write"

# RBAC 权限映射，统一key格式 agent_xxx
AGENT_ROLE_PERMISSION: Final[Dict[str, Dict[str, str]]] = {
    "agent_dispatcher": {TABLE_BILL: PERM_NONE, "other_tables": PERM_NONE},
    "bill_agent": {TABLE_BILL: PERM_WRITE, "other_tables": PERM_WRITE},
    "stat_agent": {TABLE_BILL: PERM_READ, "other_tables": PERM_READ},
    "price_agent": {TABLE_BILL: PERM_READ, "other_tables": PERM_READ},
    "finance_agent": {TABLE_BILL: PERM_READ, "other_tables": PERM_READ}
}

# --------------------------
# Redis 会话存储配置
# --------------------------
REDIS_HOST: Final[str] = "127.0.0.1"
REDIS_PORT: Final[int] = 6379
REDIS_DB: Final[int] = 0
SESSION_EXPIRE_SEC: Final[int] = 3600  # 会话缓存1小时

# 对外导出可使用的常量
__all__ = [
    "BASE_DIR",
    "DB_PATH", "TASK_STATE_DB", "REQUEUE_STALE_AFTER_SEC", "LOG_PATH", "AUDIT_LOG_PATH",
    "RAG_PATH", "USER_DOCS_PATH", "USER_DOCS_TOP_K", "USER_DOCS_MAX_DOCS",
    "USER_DOCS_CHUNK_SIZE", "USER_DOCS_TIMEOUT",
    "MODEL_PATH", "LORA_DATASET", "LORA_WEIGHT", "EMB_MODEL_PATH", "EMBEDDING_ENABLED",
    "OLLAMA_API_URL", "OLLAMA_MODEL_NAME", "FINANCE_LORA_MODEL", "MODEL_TIMEOUT",
    "OLLAMA_CONTEXT_WINDOW", "EXTERNAL_CONTEXT_WINDOW",
    "EXTERNAL_LLM_BASE_URL", "EXTERNAL_LLM_API_KEY",
    "EXTERNAL_ORCH_MODEL", "EXTERNAL_FINANCE_MODEL",
    "EXTERNAL_LLM_TIMEOUT", "EXTERNAL_LLM_ENABLED",
    "EXTERNAL_LLM_AGENTS", "FORCE_EXTERNAL_LLM", "DISABLE_LOCAL_LLM", "TRACER_ENABLED",
    "MCP_HOST", "MCP_PORT_BASE", "MCP_STDIO_FALLBACK", "MCP_SERVER_PORTS", "MCP_HTTP_PATH",
    "SANDBOX_MEMORY_MB", "TOOL_TIMEOUT_SQL", "TOOL_TIMEOUT_LLM",
    "ASK_ROUND_LIMIT", "ASK_GIVEUP_TIP",
    "DYNAMIC_DEPS_ENABLED", "MAX_DISPATCH_ROUND", "BATCH_DISPATCH_ENABLED",
    "LLM_STOP_TOKENS", "BASE_LLM_OPTIONS", "AGENT_LLM_CONFIG",
    "EXTERNAL_BASE_LLM_OPTIONS", "AGENT_EXTERNAL_LLM_CONFIG",
    "DEFAULT_USER_CITY", "CONSUMPTION_MODES",
    "TABLE_BILL", "PERM_NONE", "PERM_READ", "PERM_WRITE", "AGENT_ROLE_PERMISSION",
    "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "SESSION_EXPIRE_SEC",
    "init_project_dirs"
]