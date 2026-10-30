import json
import queue
import sys
import textwrap

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import mcp, tools
from temper_engine.app import create_app
from temper_engine.tools import Call

AUTH = {"Authorization": "Bearer secret"}

# An MCP server over stdio: two pages of tools, one that fails, and a request back to the client.
SERVER = textwrap.dedent(
    """
    import json, sys
    TOOLS = [
        {"name": "add", "description": "Adds two numbers", "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}},
        {"name": "shout", "description": "Upper-cases text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
        {"name": "broken", "description": "Always fails"},
    ]
    def send(m):
        sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
    asked = {}
    for line in sys.stdin:
        m = json.loads(line)
        if "method" not in m:  # the client's answer to our roots/list
            asked["roots"] = m
            continue
        if "id" not in m:
            continue
        method, p = m["method"], m.get("params") or {}
        if method == "initialize":
            sys.stderr.write("starting up\\n")
            send({"jsonrpc": "2.0", "id": "srv1", "method": "roots/list"})
            result = {"protocolVersion": p["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "demo", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": TOOLS[:2], "nextCursor": "p2"} if not p.get("cursor") else {"tools": TOOLS[2:]}
        elif method == "tools/call":
            a = p["arguments"]
            if p["name"] == "add":
                result = {"content": [{"type": "text", "text": str(a["a"] + a["b"])}]}
            elif p["name"] == "shout":
                result = {"content": [{"type": "text", "text": a["text"].upper()}, {"type": "image", "data": "", "mimeType": "image/png"}]}
            elif p["name"] == "roots":
                result = {"content": [{"type": "text", "text": json.dumps(asked.get("roots"))}]}
            else:
                result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
        else:
            send({"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "no such method"}}); continue
        send({"jsonrpc": "2.0", "id": m["id"], "result": result})
    """
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPER_HOME", str(tmp_path / "home"))
    yield tmp_path
    mcp.manager.close_all()
    mcp.manager.client = None


@pytest.fixture
def stdio_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    mcp.save({"demo": {"command": sys.executable, "args": [str(script)]}})
    return "demo"


def test_a_stdio_server_lists_its_tools_page_by_page_and_runs_them(stdio_server):
    conn = mcp.manager.connection(stdio_server)
    assert [t["name"] for t in conn.tools] == ["add", "shout", "broken"] and conn.info["serverInfo"]["name"] == "demo"
    assert conn.call("add", {"a": 2, "b": 3}) == ("5", False)
    text, failed = conn.call("shout", {"text": "hi"})
    assert text.startswith("HI\n[an image") and not failed
    assert conn.call("broken", {}) == ("it broke", True)
    # The server's own request (roots/list) was declined rather than left hanging.
    assert json.loads(conn.call("roots", {})[0])["error"]["code"] == -32601
    assert mcp.manager.connection(stdio_server) is conn  # kept connected


def test_a_server_that_cant_start_says_why():
    mcp.save({"gone": {"command": "temper-no-such-command-xyz"}})
    with pytest.raises(mcp.McpError, match="Is it installed"):
        mcp.manager.connection("gone")
    assert "Is it installed" in mcp.manager.errors["gone"]


def streamable(seen):
    """A Streamable HTTP server: JSON answers, a session id, and tools/list sent as an event stream."""

    def handler(request):
        if request.method == "DELETE":
            seen.append("closed")
            return httpx.Response(200)
        m = json.loads(request.content)
        seen.append((m.get("method"), request.headers.get("mcp-session-id"), request.headers.get("authorization")))
        if "id" not in m:
            return httpx.Response(202)
        if m["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": mcp.PROTOCOL, "capabilities": {}, "serverInfo": {"name": "web"}}}, headers={"mcp-session-id": "s-1"})
        if m["method"] == "tools/list":
            body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': {'tools': [{'name': 'weather', 'description': 'Gets the weather'}]}})}\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": m["id"], "result": {"content": [{"type": "text", "text": "sunny"}]}})

    return handler


def test_a_streamable_http_server_keeps_its_session_and_headers():
    seen = []
    mcp.manager.client = httpx.Client(transport=httpx.MockTransport(streamable(seen)))
    mcp.save({"web": {"url": "https://mcp.example/mcp", "headers": {"Authorization": "Bearer k"}}})
    conn = mcp.manager.connection("web")
    assert [t["name"] for t in conn.tools] == ["weather"] and conn.call("weather", {}) == ("sunny", False)
    assert seen[0] == ("initialize", None, "Bearer k")
    assert all(s[1] == "s-1" for s in seen[1:])  # every request after initialize carries the session
    mcp.manager.disconnect("web")
    assert seen[-1] == "closed"


def test_an_older_sse_server_takes_requests_where_it_says():
    events: queue.Queue = queue.Queue()

    def stream():
        yield b"event: endpoint\ndata: /messages?session=7\n\n"
        while (item := events.get(timeout=5)) is not None:
            yield f"event: message\ndata: {json.dumps(item)}\n\n".encode()

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, content=stream(), headers={"content-type": "text/event-stream"})
        assert request.url.path == "/messages" and request.url.params["session"] == "7"
        m = json.loads(request.content)
        if "id" in m:
            result = {"protocolVersion": "2024-11-05", "capabilities": {}} if m["method"] == "initialize" else {"tools": [{"name": "lookup"}]} if m["method"] == "tools/list" else {"content": [{"type": "text", "text": "found"}]}
            events.put({"jsonrpc": "2.0", "id": m["id"], "result": result})
        return httpx.Response(202)

    mcp.manager.client = httpx.Client(transport=httpx.MockTransport(handler))
    mcp.save({"old": {"url": "https://old.example/sse", "type": "sse"}})
    conn = mcp.manager.connection("old")
    assert [t["name"] for t in conn.tools] == ["lookup"] and conn.call("lookup", {}) == ("found", False)
    events.put(None)


def test_servers_are_added_pasted_masked_and_set_tool_by_tool(stdio_server):
    mcp.manager.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    with TestClient(create_app(token="secret")) as c:
        r = c.post("/api/mcp/import", headers=AUTH, json={"text": json.dumps({"mcpServers": {"gh": {"url": "https://x.example/mcp", "headers": {"Authorization": "Bearer ghp_secret"}}}})})
        assert r.json()["added"] == ["gh"]
        listed = {s["name"]: s for s in c.get("/api/mcp", headers=AUTH).json()["servers"]}
        assert listed["gh"]["config"]["headers"] == {"Authorization": mcp.MASK} and "ghp_secret" not in json.dumps(listed)
        # Saving the form back with the masked value keeps the real one.
        gh = c.post("/api/mcp", headers=AUTH, json={"name": "gh", "url": "https://x.example/mcp", "headers": {"Authorization": mcp.MASK}}).json()
        assert mcp.load()["gh"]["headers"]["Authorization"] == "Bearer ghp_secret"
        assert gh["state"] == "error" and "401" in gh["error"] and "API key" in gh["error"]
        # A command typed as one line is split.
        c.post("/api/mcp", headers=AUTH, json={"name": "line", "command": "temper-no-such-command-xyz --port 3"})
        assert mcp.load()["line"] == {"command": "temper-no-such-command-xyz", "args": ["--port", "3"]}
        demo = c.post(f"/api/mcp/{stdio_server}/connect", headers=AUTH).json()
        assert demo["state"] == "connected" and {t["name"]: t["mode"] for t in demo["tools"]} == {"add": "ask", "shout": "ask", "broken": "ask"}
        demo = c.put(f"/api/mcp/{stdio_server}", headers=AUTH, json={"tools": {"add": "auto", "broken": "off"}}).json()
        assert {t["name"]: t["mode"] for t in demo["tools"]} == {"add": "auto", "shout": "ask", "broken": "off"}
        assert c.put(f"/api/mcp/{stdio_server}", headers=AUTH, json={"enabled": False}).json()["state"] == "off"
        assert "gh" not in [s["name"] for s in c.delete("/api/mcp/gh", headers=AUTH).json()["servers"]]


def test_a_chats_servers_become_tools_it_can_call(stdio_server):
    mcp.save({**mcp.load(), "demo": {**mcp.load()["demo"], "tools": {"add": "auto", "broken": "off"}}})
    offered = tools.offer(["calculator", "mcp:demo"])
    assert offered.names == ["calculator", "mcp__demo__add", "mcp__demo__shout"] and offered.approve() == ["mcp__demo__shout"]
    assert [s["function"]["name"] for s in offered.specs()] == offered.names
    assert offered.run(Call("mcp__demo__add", '{"a": 1, "b": 41}'))["result"] == "42"
    assert offered.run(Call("calculator", {"expression": "2+2"}))["result"] == "2+2 = 4"
    assert "mcp__demo__shout" in offered.prompt()


def test_a_large_catalog_is_searched_instead_of_sent(stdio_server, monkeypatch):
    monkeypatch.setattr(mcp, "CATALOG_LIMIT", 1)
    offered = tools.offer(["mcp:demo"])
    assert offered.searching and offered.names == [mcp.SEARCH] and [s["function"]["name"] for s in offered.specs()] == [mcp.SEARCH]
    step = offered.run(Call(mcp.SEARCH, {"query": "add numbers"}))
    assert step["result"].startswith("Tools you can call now:\n- mcp__demo__add")
    assert "mcp__demo__add" in offered.names and offered.run(Call("mcp__demo__add", {"a": 1, "b": 1}))["result"] == "2"
    assert offered.run(Call(mcp.SEARCH, {"query": "zebra"}))["error"]


def test_bm25_ranks_the_tool_that_matches_best():
    entries = [{"name": n, "tool": n, "server": "s", "description": d, "parameters": {}} for n, d in [("create_issue", "Opens a GitHub issue"), ("list_repos", "Lists repositories"), ("send_email", "Sends an email message")]]
    assert [e["name"] for e in mcp.search(entries, "open an issue on github")][0] == "create_issue"
    assert [e["name"] for e in mcp.search(entries, "email")] == ["send_email"]
    assert mcp.search(entries, "weather") == []


def test_ready_made_servers_are_added_with_what_they_need(tmp_path, monkeypatch):
    mcp.manager.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    monkeypatch.setattr(mcp.shutil, "which", lambda name: "/usr/bin/npx" if name == "npx" else None)
    with TestClient(create_app(token="secret")) as c:
        assert {p["id"]: p["added"] for p in c.get("/api/mcp/presets", headers=AUTH).json()["presets"]} == {"filesystem": False, "github": False, "blender": False}
        assert c.post("/api/mcp/presets/filesystem", headers=AUTH, json={"value": str(tmp_path / "nope")}).status_code == 422
        c.post("/api/mcp/presets/filesystem", headers=AUTH, json={"value": str(tmp_path)})
        assert mcp.load()["files"]["args"] == ["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path.resolve())]
        assert c.post("/api/mcp/presets/github", headers=AUTH, json={"value": " "}).status_code == 422
        github = c.post("/api/mcp/presets/github", headers=AUTH, json={"value": "ghp_x"}).json()
        assert github["config"]["headers"]["Authorization"] == mcp.MASK and mcp.load()["github"]["headers"]["Authorization"] == "Bearer ghp_x"
        assert c.get("/api/mcp/presets", headers=AUTH).json()["presets"][0]["added"]
        monkeypatch.setattr(mcp.shutil, "which", lambda name: None)
        assert c.post("/api/mcp/presets/filesystem", headers=AUTH, json={"value": str(tmp_path)}).status_code == 409
        assert c.post("/api/mcp/presets/nope", headers=AUTH, json={}).status_code == 404
