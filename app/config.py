from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field


class Settings(BaseSettings):
    database_url: SecretStr
    jwt_secret_key: SecretStr
    access_token_expire_minutes: int = Field(default=30, ge=1)
    base_url: str = Field(default="http://quotes.toscrape.com", description="Base URL for scraping")
    max_pages: int = Field(default=10, ge=1, description="Max pages to scrape")
    topic: str = Field(default="perseverance", description="Default topic for quotes")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()