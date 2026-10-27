from fastapi.testclient import TestClient

from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def call(method, path, status=200, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == status, r.text
    return r.json()


def test_prompts_are_saved_searched_and_ranked_by_use():
    review = call("POST", "/api/prompts", json={"name": "Code review", "text": "Review this {{language}} code for bugs:\n{{code}}\nFocus on {{ language }} idioms."})
    assert review["variables"] == ["language", "code"]
    summary = call("POST", "/api/prompts", json={"name": "Summarise", "text": "Summarise in three bullet points."})
    call("POST", "/api/prompts", 409, json={"name": "code REVIEW", "text": "again"})
    assert [p["name"] for p in call("GET", "/api/prompts", params={"q": "bullet"})["prompts"]] == ["Summarise"]
    call("POST", f"/api/prompts/{summary['id']}/used")
    assert [p["name"] for p in call("GET", "/api/prompts")["prompts"]] == ["Summarise", "Code review"]
    edited = call("PUT", f"/api/prompts/{review['id']}", json={"name": "Review", "text": "Review it."})
    assert edited["variables"] == [] and edited["name"] == "Review"
    call("DELETE", f"/api/prompts/{review['id']}")
    call("DELETE", f"/api/prompts/{review['id']}", 404)
    assert len(call("GET", "/api/prompts")["prompts"]) == 1
