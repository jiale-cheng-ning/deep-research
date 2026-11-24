from unittest.mock import AsyncMock

import httpx

from deep_research.config import Settings
from deep_research.search import SearchTools


async def test_firecrawl_search_reads_v2_web_results(monkeypatch):
    tools = SearchTools(Settings(firecrawl_api_key="test", search_results_per_query=3))
    result = {"title": "Example", "url": "https://example.org", "markdown": "Page text"}
    response = httpx.Response(200, json={"success": True, "data": {"web": [result]}})
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(tools, "_request_with_retry", request)

    results = await tools.search("topic", source="firecrawl")

    request.assert_awaited_once_with(
        "POST",
        "https://api.firecrawl.dev/v2/search",
        headers={"Authorization": "Bearer test"},
        json={"query": "topic", "limit": 3, "scrapeOptions": {"formats": ["markdown"]}},
        timeout=60,
    )
    assert results == [{
        "title": "Example",
        "url": "https://example.org",
        "content": "Page text",
        "score": 0.5,
        "source": "firecrawl",
    }]


async def test_firecrawl_scrape_reads_v2_markdown(monkeypatch):
    tools = SearchTools(Settings(firecrawl_api_key="test", firecrawl_scrape_max_chars=4))
    response = httpx.Response(200, json={"success": True, "data": {"markdown": "Page text"}})
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(tools, "_request_with_retry", request)

    assert await tools.scrape_url("https://example.org") == "Page"
    request.assert_awaited_once_with(
        "POST",
        "https://api.firecrawl.dev/v2/scrape",
        headers={"Authorization": "Bearer test"},
        json={"url": "https://example.org", "formats": ["markdown"]},
        timeout=60,
    )
