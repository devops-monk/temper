import httpx
import pytest

from temper_engine import remote


def test_bucket_addresses_become_public_urls_and_file_types_are_read():
    assert remote.url("s3://my-bucket/data/qa.jsonl") == "https://my-bucket.s3.amazonaws.com/data/qa.jsonl"
    assert remote.url("gs://b/x.csv") == "https://storage.googleapis.com/b/x.csv"
    assert remote.is_remote("https://x/y.jsonl") and not remote.is_remote("~/data/y.jsonl") and not remote.is_remote("openai/gsm8k")
    assert remote.suffix("https://x/y.parquet?download=true") == ".parquet"
    with pytest.raises(ValueError):
        remote.suffix("https://x/page.html")


def test_remote_files_are_fetched_once_and_previewed_from_their_first_rows(monkeypatch):
    served = []
    body = "".join(f'{{"text": "row {i}"}}\n' for i in range(1000))

    def handler(request):
        served.append(str(request.url))
        if "private" in str(request.url):
            return httpx.Response(403)
        if request.url.path.endswith(".csv"):
            return httpx.Response(200, text="q,a\n1,2\n3,4\n")
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "stream", lambda method, u, **kw: httpx.Client(transport=transport).stream(method, u, **{k: v for k, v in kw.items() if k != "follow_redirects"}))
    assert remote.head_rows("https://example.com/d.jsonl", 3) == [{"text": "row 0"}, {"text": "row 1"}, {"text": "row 2"}]
    assert remote.head_rows("s3://bucket/t.csv", 5) == [{"q": "1", "a": "2"}, {"q": "3", "a": "4"}]
    path = remote.fetch("https://example.com/d.jsonl")
    assert path.read_text() == body and remote.fetch("https://example.com/d.jsonl") == path
    assert served.count("https://example.com/d.jsonl") == 2  # the preview, then one download
    with pytest.raises(ValueError, match="isn't public"):
        remote.fetch("https://example.com/private.jsonl")
