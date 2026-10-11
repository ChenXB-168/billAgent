# ==============================================
# 全局配置文件 - billAgent 项目专用
# 规范：全部业务常量大写，分层隔离，推理参数与业务参数分离
# ==============================================
from pathlib import Path
import os
import json
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

# --------------------------
# 让 localhost 调用绕过系统代理（单一来源）
# --------------------------
# 背景（`07` 踩坑记录）：Windows 上若存在注册表代理（本机实测为 127.0.0.1:7892），
#   httpx 连接 127.0.0.1:8001/8002/8003 的 MCP 常驻服务时会走 `proxy_request`，
#   报 "Connection…/unhandled errors in a TaskGroup（httpcore._connect）" 而全部失败，
#   表象常像"服务启动慢"或"探活超时"，实为请求全被代理吞掉。
# 放在 config 作为单一来源：凡 import config 的进程一律自动生效，各入口无需各自 setdefault。
# 注：放在 load_dotenv 之后，使 .env 中显式配置的 NO_PROXY 优先。
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

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
    """
    函数功能与逻辑描述：
        按常量清单一次性创建运行期依赖目录（库目录、日志目录、RAG / 用户文档、
        LoRA 数据集 / 权重、embedding 模型目录），exist_ok=True 保证重复执行幂等、
        不会清空既有目录内容。本函数在模块导入期被无条件调用一次，只建目录、
        不读数据库、不加载模型。
        边界处理：依赖 os.makedirs 在创建失败时直接抛出异常（启动期"快速失败"）——
        目录缺失会连锁导致日志 / 建库 / 模型加载失败，静默吞错反而更难排查。
    入参说明：
        无。
    返回值说明：
        无（仅副作用：创建 dir_list 中尚不存在的目录）。
    """
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
# M22（弃用本地通道 + 超时预算收紧）：**编排器等待上限**（秒，默认 150，
#   可由 BILLAGENT_MODEL_TIMEOUT 覆盖；原为"未开放环境变量覆盖"）。
#   ★语义变更：本值原为"本地 Ollama 单次请求超时"（240，按 CPU 长文本推理 **189s 实测**设）。
#   2026-09-15（M15 P0-a）起生产全链路走外部强模型，单次调用为**秒级** → 240s 只会让
#   故障时用户**白等 4 分钟**（`设计/20`）。现收紧为 150s 并开放 env 配置；
#   `mcpGateway/a2a_queue.py` 的 wait_result 与 `agents/orchestrator` 的等待均**引用本值**
#   （单一来源，不再各自硬编码 240）。
MODEL_TIMEOUT: Final[int] = int(os.getenv("BILLAGENT_MODEL_TIMEOUT", "150"))
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
# 外部 API 单次调用超时（秒，默认 60）。★M22：原 300s 是按"本地 1.8B 单次 189s"的**旧基线**
#   设的（`设计/20`）；外部强模型（deepseek-chat）单次为**秒级**，60s 已留足余量。
#   本值是超时链的**最内环**：EXTERNAL_LLM_TIMEOUT(60) < TOOL_TIMEOUT_LLM(60) ≈
#   TASK_TIMEOUT(120) < MODEL_TIMEOUT(150)。
EXTERNAL_LLM_TIMEOUT: Final[int] = int(os.getenv("BILLAGENT_EXT_TIMEOUT", "60"))
# M14（D17）：OpenAICompatProvider 上下文窗口（保守默认 8k，`Capabilities.context_window` 单一来源；
#   可按实际所选模型窗口覆盖，如 glm-4-flash 128k / deepseek 32k 设为更大值以放宽容错）
EXTERNAL_CONTEXT_WINDOW: Final[int] = int(os.getenv("BILLAGENT_EXT_CONTEXT_WINDOW", "8192"))
# 是否启用外部API（base_url + api_key 齐备才启用）
EXTERNAL_LLM_ENABLED: Final[bool] = bool(EXTERNAL_LLM_BASE_URL and EXTERNAL_LLM_API_KEY)

# --------------------------
# 外部模型路由：哪些 Agent 走外部强模型
# --------------------------
# ★2026-09-15（M15 P0-a）：默认**全链路走外部强模型**——项目已决定不再使用本地 1.8B。
#   可用 BILLAGENT_EXT_AGENTS（逗号分隔）按需收窄，如 "orchestrator_agent,finance_agent"。
#   未配 API（EXTERNAL_LLM_ENABLED=False）时本名单自动失效、全部回退本地——
#   保证 `01` §6 非功能需求「本地可用性」/ §7.1 约束 R5「必须本地可跑」仍然成立
#   （故**不**默认开启 DISABLE_LOCAL_LLM；需要「绝不回退本地」时由 env 显式打开）。
_ALL_AGENTS: Final[tuple] = (
    "orchestrator_agent", "bill_agent", "stat_agent", "price_agent", "finance_agent",
)
_EXT_AGENTS_ENV: Final[str] = os.getenv("BILLAGENT_EXT_AGENTS", "").strip()
EXTERNAL_LLM_AGENTS: Final[frozenset] = (
    frozenset(a.strip() for a in _EXT_AGENTS_ENV.split(",") if a.strip())
    if _EXT_AGENTS_ENV else frozenset(_ALL_AGENTS)
)
# 测试/联调开关（BILLAGENT_FORCE_EXTERNAL=1）：**所有** Agent 一律走外部强模型。
# 用途：验证 harness 可用性与正确性时，绕开本地 1.8B 的 CPU 推理——
#   实测本地单次调用可达 189s，e2e 4 条用例耗时 36 分钟，且小模型 JSON 输出不稳定。
# ⚠️2026-09-15 语义提醒：全链路走外部后，**生产形态应由 EXTERNAL_LLM_AGENTS 表达**
#   （默认已含 5 个 Agent），本开关仅作兜底（优先级高于名单，见 `09` §6.3 决策表序 3）。
# 未配 API 时本开关自动失效（仍要求 EXTERNAL_LLM_ENABLED=True）。
FORCE_EXTERNAL_LLM: Final[bool] = os.getenv("BILLAGENT_FORCE_EXTERNAL", "0") == "1"
# 硬开关（BILLAGENT_DISABLE_LOCAL_LLM=1）：**封死**本地 Ollama 通道。
# 任何走到 `ollama_base_call` 的调用**立即抛错**，而不是回退或静默失败。
# 与 FORCE_EXTERNAL_LLM 的区别：后者是"优先走外部，失败仍可回退本地"；
# 前者是"直接禁用本地，走到即判定为路由缺陷并报错"——排查期用来杜绝意外走本地。
# ★M22（2026-10-09）：**默认开启**——本地通道已【彻底弃用】（用户决策，见 `设计/20`）。
#   代价：放弃 `01` §7.1 的 **R5「必须本地可跑」** 与 §6 的「本地可用性 / 零成本默认路径」
#   两条非功能需求（已在 `01` 标注变更）。
#   ⚠️ 因此**必须配置外部 API**：未配时 `EXTERNAL_LLM_ENABLED=False` 而本地被封 →
#   `ModelRouter.resolve` **立即抛 RuntimeError** 暴露配置矛盾（**有意的快速失败**，
#   而非静默降级；有专项用例 `test_resolve_disable_local_goes_external` 锁定该行为）。
#   临时恢复本地兜底：设 `BILLAGENT_DISABLE_LOCAL_LLM=0`（代码与 Provider 均保留，属软弃用）。
DISABLE_LOCAL_LLM: Final[bool] = os.getenv("BILLAGENT_DISABLE_LOCAL_LLM", "1") == "1"

# --------------------------
# MCP 常驻服务（M3，`03_逻辑架构.md` §9.8.2 / `07_部署架构.md` §7 待确认决策点#1）
# --------------------------
# 绑定 127.0.0.1 不对外暴露（`03` §9.8.2 ④ 风险7「新增监听端口」的缓解项）；
# 端口基址默认 8000、各 server 逐个 +1，环境变量可覆盖：
# BILLAGENT_MCP_PORT（基址）/ BILLAGENT_MCP_STDIO（回滚开关）。
MCP_HOST: Final[str] = "127.0.0.1"
MCP_PORT_BASE: Final[int] = int(os.getenv("BILLAGENT_MCP_PORT", "8000"))
# 回滚开关（M3 回滚要求）：BILLAGENT_MCP_STDIO=1 切回 stdio 短连接
MCP_STDIO_FALLBACK: Final[bool] = os.getenv("BILLAGENT_MCP_STDIO", "0") == "1"
# server_key → 端口（`07` §7 决策点#1 已落地：基址 8000 起 +1/+2/+3 ⇒
# 默认 sql_bill=8001 / llm_base=8002 / llm_finance=8003；键名以本字典为准，
# 消费方：startup/bootstrap.py、mcpGateway/client.py 及 3 个 server 的 __main__）
MCP_SERVER_PORTS: Final[Dict[str, int]] = {
    "sql_bill": MCP_PORT_BASE + 1,
    "llm_base": MCP_PORT_BASE + 2,
    "llm_finance": MCP_PORT_BASE + 3,
}
# MCP 挂载路径（FastMCP streamable-http 默认 /mcp，显式声明避免魔法值）
MCP_HTTP_PATH: Final[str] = "/mcp"
# M21：MCP 服务**运行期**心跳探测间隔（秒，默认 60，<=0 = 关闭）。
#   为什么需要：启动期健康检查只覆盖"服务能不能起来"；服务在**运行中**挂掉时，主 Agent
#   唯一的感知途径是"工具调用失败 → 编排器盲等 240s 超时"，且**分不清"慢"还是"死"**。
#   本看门狗把"240s 盲等才发现"缩短为"≤1 个探测周期发现"，并给出可告警的明确信号。
#   代价：每次探活 = 建 HTTP 连接 + 完成一次 MCP initialize 握手（有开销，故间隔不宜过短）。
MCP_HEALTH_INTERVAL: Final[float] = float(os.getenv("BILLAGENT_MCP_HEALTH_INTERVAL", "60"))

# --------------------------
# M4 沙箱（D8）：进程级资源限制 + 工具级超时（`06` §2.5 / `11` M4）
#   内存硬上限（单位 MB，默认 2048，可由 BILLAGENT_SANDBOX_MEMORY_MB 覆盖）：
#   Windows 由 bootstrap 挂 Job Object；POSIX 由 3 个 server self-prison
#   （`mcpGateway/sandbox.py`；优先级：显式传参 > 本常量 > 模块默认 2048）。
#   单 server 常驻峰值约 200~400MB，2048MB 留足余量。
SANDBOX_MEMORY_MB: Final[int] = int(os.getenv("BILLAGENT_SANDBOX_MEMORY_MB", "2048"))
# 工具级超时（引擎层 `asyncio.wait_for(ToolSpec.timeout)`）：
#   SQL=30s：SQLite 本地库 + busy_timeout 5s，正常调用毫秒级（M3 实测 135ms），30s 极宽裕；
#   LLM=60s：★M22 收紧（原 300s 按"本地 1.8B 单次实测 189s"的**旧基线**所设，见 `设计/20`）。
#   生产全链路走外部强模型（**秒级**），60s 为防挂死兜底，同时是超时链的最内环之一：
#   **工具(60) < 任务(120) < 编排器等待(150)**。
#   ≤0 = 关闭超时（M4 回滚要求"超时阈值可配置关闭"）。
TOOL_TIMEOUT_SQL: Final[float] = float(os.getenv("BILLAGENT_TOOL_TIMEOUT_SQL", "30"))
TOOL_TIMEOUT_LLM: Final[float] = float(os.getenv("BILLAGENT_TOOL_TIMEOUT_LLM", "60"))

# --------------------------
# M19（D20）：请求级终止语义 + worker 侧任务级超时（`设计/17`）
#   ① ABORT_ON_TASK_FAILURE：任务级失败（超时 / 执行异常）→ **终止本轮**（不再派发后续波次、
#      不产出业务回复），与 M16 用户取消**共用收口路径**、按 `abort_reason` 区分回执文案。
#      =0 退回改造前行为（失败结果溜进汇总素材 → 用户看到"看似正常但无数据"的回复），
#      即 M19 回滚开关。
#   ② TASK_TIMEOUT：worker 侧单任务执行上限，用途是**防"挂死的任务永久占住 worker 消费循环"**
#      （worker 是单协程串行消费，一个卡死的 runner 会让该 Agent 通道持续堆积、彻底停摆）。
#      ★M22 收紧至 120s（原 300）：仍 > 单次工具超时(60)、且 < 编排器等待(150)，形成递增链
#      **工具(60) < 任务(120) < 编排器等待(150)** —— 这解决了 M19 登记的"超时预算倒挂"。
#      （M19 原文按"本地 189s"基线论证；M22 按**外部形态**重排。）
#      ✅ M19 遗留待办 **T-3（超时预算递增链）由 M22 关闭**。
TASK_TIMEOUT: Final[float] = float(os.getenv("BILLAGENT_TASK_TIMEOUT", "120"))
ABORT_ON_TASK_FAILURE: Final[bool] = os.getenv("BILLAGENT_ABORT_ON_FAIL", "1") != "0"

# --------------------------
# M5（D5）：追问治理（`01` §3.4 / §8 Q5 决策 / 技术待办 T2 / T3）
#   规则：信息不全最多追问 3 轮，第 3 轮仍不满足 → **直接放弃并报错**（不再追问、也不静默跳过）
ASK_ROUND_LIMIT: Final[int] = int(os.getenv("BILLAGENT_ASK_ROUND_LIMIT", "3"))
# 放弃提示语：`01` §3.4 明文规定，改动即属需求变更
ASK_GIVEUP_TIP: Final[str] = "信息不全，执行失败，请重新来"

# --------------------------
# M16（D18）：意图层能力补齐（`设计/14` / 契约 `09` §2.14）
#   ① 契约表：四基元的必需参数声明（plan_node 据此产出 missing）
#   ② 阈值：低置信触发澄清
#   ③ 开关：OOD 判定总开关（降级回滚）
# --------------------------
_AGENT_CONTRACTS_PATH: Final[Path] = BASE_DIR / "config" / "agent_contracts.json"


def _load_agent_contracts() -> Dict:
    """读契约表 JSON；失败返回 {}（业务侧随即退化为「只看 LLM 自报」，不阻断主链路）。"""
    try:
        with open(_AGENT_CONTRACTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


AGENT_CONTRACTS: Final[Dict] = _load_agent_contracts()
# 低置信阈值：confidence < 本值 → 走候选澄清（建议 0.6，需实测校准）
INTENT_CONFIDENCE_THRESHOLD: Final[float] = float(
    os.getenv("BILLAGENT_INTENT_CONFIDENCE_THRESHOLD", "0.6"))
# OOD 判定总开关：=0 关闭域外兜底分支（M16 回滚开关）
INTENT_OOD_ENABLED: Final[bool] = os.getenv("BILLAGENT_INTENT_OOD_ENABLED", "1") != "0"
# M17（D19）：保守规划总开关——=0 关闭 inferable 档的保守规划（走回"缺参即追问"旧行为）
INTENT_INFERABLE_ENABLED: Final[bool] = os.getenv("BILLAGENT_INTENT_INFERABLE_ENABLED", "1") != "0"

# --------------------------
# M20（D21）：A2A 队列容量治理（拷打 F3 —— 有界 + 满时拒绝 + 深度可观测）
#   背景：`asyncio.Queue()` 原未传 maxsize（无界）；`send_task` 无满时策略、无深度指标。
#   策略：满时**拒绝**（返回 False + WARNING），**绝不阻塞** —— 阻塞会把派发方
#         （Orchestrator，用户请求持有者）一并挂死（故障扩散）；且 send_task 是
#         **同步方法**（契约 `09` §3.5 红线 2），想阻塞也做不到。
#   单机单用户实际深度仅个位数（单请求最多 4 子任务）；100 是**防御性上限**，
#   覆盖"用户连发 + 子 Agent 消费慢"的最坏情况。
# --------------------------
A2A_QUEUE_MAXSIZE: Final[int] = int(os.getenv("BILLAGENT_A2A_QUEUE_MAXSIZE", "100"))
# 堆积预警阈值：队列深度达到即打 WARNING（不阻断）—— 队列深度是"饱和度"黄金信号
A2A_QUEUE_WARN_DEPTH: Final[int] = int(os.getenv("BILLAGENT_A2A_QUEUE_WARN_DEPTH", "80"))

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
# 可观测总开关（环境变量 BILLAGENT_TRACER_ENABLED，默认 "1" 开启；=0 完全关闭埋点，
# 此时 trace_span / llm_call_stat 两表均零写入）。消费方：utils/tracer.py
# （`_insert_span_row` / `record_llm_call` 入口前置判断）、utils/trace_view.py
# （无记录时提示先确认本开关为 1）。
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

# 各 Agent 独立差异化推理参数；key 统一为 `xxx_agent` 格式，与
# mcpGateway/rbac_config.py 的 AGENT_PERMISSION_MAP 键名保持一致；
# 未登记 / 空 dict（如 orchestrator_agent）表示复用 BASE_LLM_OPTIONS。
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

# 各 Agent 外部差异化参数（M15 P0-a 起 5 个 Agent 均在此显式声明；未声明者 fallback EXTERNAL_BASE_LLM_OPTIONS）
AGENT_EXTERNAL_LLM_CONFIG: Final[Dict[str, Dict]] = {
    # 调度分发Agent（orchestrator）：复用外部基础层，无额外差异
    "orchestrator_agent": {"max_tokens": 4096},
    # 理财Agent：极低随机，四段式固定输出（外部走强模型时同样保持低随机）
    "finance_agent": {"temperature": 0.1, "max_tokens": 4096},
    # ★2026-09-15（M15 P0-a）补全 3 个子 Agent：均为**结构化输出**任务（抽取 / IR 生成），
    #   用低随机保证稳定性；不补则会 fallback 基础层 temperature=0.3。
    "bill_agent": {"temperature": 0.1, "max_tokens": 4096},   # 账单字段抽取
    "stat_agent": {"temperature": 0.1, "max_tokens": 4096},   # 统计 IR 生成
    "price_agent": {"temperature": 0.1, "max_tokens": 4096},  # 物价信息提取
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

# 注：Agent 权限（RBAC）配置的权威来源在 mcpGateway/rbac_config.py 的 AGENT_PERMISSION_MAP。

# --------------------------
# Redis 会话存储配置（**预留**：当前 Python 侧无消费点）
# --------------------------
# 现网短期会话记忆为**进程内内存**实现（memory/short_memory.py 的 SESSION_MEM /
# SESSION_DRAFT_CACHE / SESSION_ASK_ROUND；该模块明确「离线项目不依赖 Redis」）。
# 以下常量仅经 config.__all__ 对外暴露，并由 utils/docker_compose.yml 向容器注入
# 同名环境变量（REDIS_HOST / REDIS_PORT），供将来切换外部 Redis 会话存储时使用。
REDIS_HOST: Final[str] = "127.0.0.1"
REDIS_PORT: Final[int] = 6379
REDIS_DB: Final[int] = 0
SESSION_EXPIRE_SEC: Final[int] = 3600  # 会话缓存 1 小时（3600 秒），预留

# --------------------------
# M15（`设计/12` §4.2.4 / §4.2.5 / §4.9）：账单改删 + 子层自主循环
# --------------------------
# 自主循环总开关（降级回滚用，`设计/12` §4.7 第 2 层）：=0 时 bill_agent 不走 tool loop，
# 退回「确定性目标消解」（施工序 P3a），**功能不丢**——工具层与注册保持不变。
BILL_FC_ENABLED: Final[bool] = os.getenv("BILLAGENT_BILL_FC", "1") != "0"
# 自主循环最大轮次：防模型无限试探（`设计/12` §4.2.3「必须做，不是可选」）。
# 达上限仍无终态 → 转追问（失败矩阵见 `设计/12` §4.6）。
FC_MAX_TURNS: Final[int] = int(os.getenv("BILLAGENT_FC_MAX_TURNS", "4"))
# 可操作窗口下限（`设计/12` §4.2.5）：用户"刚才那笔"只能指代最近 N 笔。
# ★recall 语义：`bill.recent_list(limit)` 的 limit 由**服务端 clamp** 到 [本值, RECENT_BILL_MAX]——
#   P0-b 实测模型会传 limit=1，窗口过小会漏掉真正目标（`设计/12` §5.1 修正 1）。
RECENT_BILL_WINDOW: Final[int] = int(os.getenv("BILLAGENT_RECENT_BILL_WINDOW", "3"))
# `bill.recent_list` 的 limit 上限（防模型一次拉全表，撑爆上下文预算 D6）
RECENT_BILL_MAX: Final[int] = int(os.getenv("BILLAGENT_RECENT_BILL_MAX", "20"))
# UI 账单面板展示条数（`设计/12` §4.9）：与可操作窗口 RECENT_BILL_WINDOW **是两个数**——
#   面板可展示更多（只读），但超窗口行需标注"超出可操作范围"。
UI_BILL_PANEL_LIMIT: Final[int] = int(os.getenv("BILLAGENT_UI_BILL_PANEL_LIMIT", "20"))
# UI 面板的**取数上限**（`设计/12` §4.6："面板展示条数与可操作窗口是两个数"）：
#   面板按天分页浏览，而 20 笔的上限（为模型上下文预算 D6 而设）实测常常只够 3 天，
#   翻页形同虚设。★该值**只作用于 UI 读数**（`bill_tools.recent_bills_for_ui`，
#   不经工具注册、模型不可见），故**不放宽 `RECENT_BILL_MAX`**——模型侧可见面语义原样不动。
UI_BILL_FETCH_LIMIT: Final[int] = int(os.getenv("BILLAGENT_UI_BILL_FETCH_LIMIT", "200"))

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
    "MCP_HEALTH_INTERVAL",
    "SANDBOX_MEMORY_MB", "TOOL_TIMEOUT_SQL", "TOOL_TIMEOUT_LLM",
    "TASK_TIMEOUT", "ABORT_ON_TASK_FAILURE",
    "ASK_ROUND_LIMIT", "ASK_GIVEUP_TIP",
    "AGENT_CONTRACTS", "INTENT_CONFIDENCE_THRESHOLD", "INTENT_OOD_ENABLED",
    "INTENT_INFERABLE_ENABLED",
    "A2A_QUEUE_MAXSIZE", "A2A_QUEUE_WARN_DEPTH",
    "DYNAMIC_DEPS_ENABLED", "MAX_DISPATCH_ROUND", "BATCH_DISPATCH_ENABLED",
    "LLM_STOP_TOKENS", "BASE_LLM_OPTIONS", "AGENT_LLM_CONFIG",
    "EXTERNAL_BASE_LLM_OPTIONS", "AGENT_EXTERNAL_LLM_CONFIG",
    "DEFAULT_USER_CITY", "CONSUMPTION_MODES",
    "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "SESSION_EXPIRE_SEC",
    "BILL_FC_ENABLED", "FC_MAX_TURNS", "RECENT_BILL_WINDOW", "RECENT_BILL_MAX",
    "UI_BILL_PANEL_LIMIT", "UI_BILL_FETCH_LIMIT",
    "init_project_dirs"
]