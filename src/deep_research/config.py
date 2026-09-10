from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DR_",
    )

    # API keys — env vars prefixed with DR_ to avoid conflicts
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""

    # Model tiers — all use sonnet (proxy only supports this model)
    model_haiku: str = "claude-sonnet-4-6"
    model_sonnet: str = "claude-sonnet-4-6"
    model_opus: str = "claude-sonnet-4-6"

    # Token budgets
    max_input_tokens: int = 100_000
    max_output_tokens: int = 8_192
    max_total_tokens: int = 500_000

    # Search config
    search_results_per_query: int = 5
    firecrawl_scrape_max_chars: int = 15_000
    deep_scrape_top_n: int = 3  # scrape full pages for top-N search results

    # Concurrency limits
    llm_concurrency: int = 3
    search_concurrency: int = 2

    # Retry
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    search_max_retries: int = 2
    search_retry_base_delay: float = 0.5

    # Logging
    verbose: bool = False

    # Checkpoint / resume
    checkpoint_dir: str = ".runs"
