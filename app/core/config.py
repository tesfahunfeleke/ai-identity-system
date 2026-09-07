"""
Centralized application configuration.
Everything that varies between environments lives here.
"""
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "AI Identity System"
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- Structured memory ---
    database_url: str = Field(default=f"sqlite:///{BASE_DIR / 'data' / 'identity.db'}")

    # --- Semantic memory ---
    chroma_persist_dir: str = Field(default=str(BASE_DIR / "data" / "chroma"))

    # --- LLM Configuration (Phase 3) ---
    LLM_PROVIDER: str = Field(default="openai")
    OPENAI_API_KEY: str = Field(default="")
    ANTHROPIC_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()