from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None

    VLLM_HOST: str = "vllm"
    VLLM_PORT: int = 8000
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    VLLM_API_KEY: str = "secret"

    OLLAMA_PORT: int= 11434
    OLLAMA_MODEL: str = "qwen3:30b-a3b"

    EMBEDDING_MODEL_NAME: str = "sergeyzh/BERTA"
    EMBEDDING_DIM: int = 768

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

settings = Settings()

QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
# VLLM_BASE_URL = f"http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1"

print("--- Configuration Loaded ---")
print(f"Qdrant Host: {settings.QDRANT_HOST}")
print(f"Qdrant Port: {settings.QDRANT_PORT}")
# print(f"vLLM Host: {settings.VLLM_HOST}")
# print(f"vLLM Port: {settings.VLLM_PORT}")
# print(f"vLLM Model: {settings.VLLM_MODEL}")
print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
print(f"Embedding Dim: {settings.EMBEDDING_DIM}")
