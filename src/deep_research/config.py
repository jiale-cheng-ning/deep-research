from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "DR_",
    }
