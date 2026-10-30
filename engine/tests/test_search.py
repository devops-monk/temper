import json

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import search, tools
from temper_engine.app import create_app
from temper_engine.tools import Call

AUTH = {"Authorization": "Bearer secret"}

DUCK = """<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=x">Example <b>A</b></a></h2>
<a class="result__snippet" href="#">First   <b>snippet</b></a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">An ad</a></div>
<div class="result"><a class="result__a" href="https://example.org/b">B</a></div>"""


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_duckduckgo_results_are_unwrapped_and_ads_dropped():
    name, results = search.search("q", client=client(lambda r: httpx.Response(200, text=DUCK)))
    assert name == "DuckDuckGo" and results == [{"title": "Example A", "url": "https://example.org/a", "snippet": "First snippet"}, {"title": "B", "url": "https://example.org/b", "snippet": ""}]
    with pytest.raises(ValueError, match="asking for a check"):
        search.search("q", client=client(lambda r: httpx.Response(202, text="<p>anomaly detected</p>")))


def test_providers_with_keys_and_addresses(monkeypatch):
    seen = {}

    def handler(request):
        seen[request.url.host] = request
        if request.url.host == "api.search.brave.com":
            return httpx.Response(200, json={"web": {"results": [{"title": "T", "url": "https://x.org", "description": "d"}]}})
        if request.url.host == "api.tavily.com":
            return httpx.Response(200, json={"results": [{"title": "T", "url": "https://y.org", "content": "c"}]})
        return httpx.Response(200, json={"results": [{"title": "S", "url": "https://z.org", "content": "s"}]})

    search.prefs.update("web_search", {"provider": "brave", "keys": {}})
    with pytest.raises(ValueError, match="needs an API key"):
        search.search("q", client=client(handler))
    search.prefs.update("web_search", {"provider": "brave", "keys": {"brave": "bk", "tavily": "tk"}, "searxng_url": "http://localhost:8888"})
    assert search.search("q", client=client(handler))[1][0]["url"] == "https://x.org" and seen["api.search.brave.com"].headers["x-subscription-token"] == "bk"
    search.prefs.update("web_search", {"provider": "tavily", "keys": {"tavily": "tk"}})
    assert search.search("q", client=client(handler))[1][0]["snippet"] == "c" and seen["api.tavily.com"].headers["authorization"] == "Bearer tk"
    search.prefs.update("web_search", {"provider": "searxng", "searxng_url": "http://localhost:8888"})
    assert search.search("q", client=client(handler))[1][0]["url"] == "https://z.org" and seen["localhost"].url.params["format"] == "json"
    with pytest.raises(ValueError, match="API key was refused"):
        search.prefs.update("web_search", {"provider": "tavily", "keys": {"tavily": "bad"}})
        search.search("q", client=client(lambda r: httpx.Response(401)))


def test_the_tool_numbers_results_and_marks_them_as_the_webs(monkeypatch):
    monkeypatch.setattr(search, "search", lambda q, count=6, client=None: ("DuckDuckGo", [{"title": "T", "url": "https://x.org", "snippet": "s"}]))
    step = tools.run(Call("web_search", {"query": "llamas"}), ["web_search"])
    assert step["result"].startswith("Results from DuckDuckGo for 'llamas'") and "1. T\n   https://x.org\n   s" in step["result"] and "read_web_page" in step["result"]
    assert tools.TOOLS["web_search"].mode == "ask"
    assert "Give what to search for" in tools.run(Call("web_search", {}), ["web_search"])["result"]


def test_settings_keep_keys_masked_and_saved():
    with TestClient(create_app(token="secret")) as c:
        assert c.get("/api/search", headers=AUTH).json()["provider"] == "duckduckgo"
        shown = c.put("/api/search", headers=AUTH, json={"provider": "brave", "key": "bk"}).json()
        assert shown["keys"] == {"brave": search.MASK} and next(p for p in shown["providers"] if p["id"] == "brave")["set"]
        c.put("/api/search", headers=AUTH, json={"provider": "brave", "key": search.MASK})
        assert search.settings()["keys"] == {"brave": "bk"} and "bk" not in json.dumps(c.get("/api/search", headers=AUTH).json())
        assert c.put("/api/search", headers=AUTH, json={"provider": "searxng", "url": "localhost"}).status_code == 422
        assert c.put("/api/search", headers=AUTH, json={"provider": "nope"}).status_code == 422
        assert c.put("/api/search", headers=AUTH, json={"provider": "searxng", "url": "http://localhost:8888/"}).json()["searxng_url"] == "http://localhost:8888"
        assert search.settings()["keys"] == {"brave": "bk"}
