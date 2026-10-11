"""项目全局统一配置导出入口"""
from .config import (
    BASE_DIR,
    DB_PATH,
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    LOG_PATH,
    RAG_PATH,
    MODEL_PATH,
    LORA_DATASET,
    LORA_WEIGHT,
    CONSUMPTION_MODES
)

__all__ = [
    "BASE_DIR",
    "DB_PATH",
    "OLLAMA_API_URL",
    "OLLAMA_MODEL_NAME",
    "LOG_PATH",
    "RAG_PATH",
    "MODEL_PATH",
    "LORA_DATASET",
    "LORA_WEIGHT",
    "CONSUMPTION_MODES",
]