import httpx
import pytest

from temper_engine import remote


@pytest.fixture(autouse=True)
def no_aws(monkeypatch, tmp_path):
    """No AWS credentials but the ones a test gives, whatever this computer has."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))


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


def test_requests_are_signed_like_awss_own_example():
    from datetime import datetime, timezone

    headers = remote.sign("GET", "https://examplebucket.s3.amazonaws.com/test.txt",
                          {"key": "AKIAIOSFODNN7EXAMPLE", "secret": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "region": "us-east-1"},
                          {"Range": "bytes=0-9", "x-amz-content-sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
                          datetime(2013, 5, 24, tzinfo=timezone.utc))
    assert headers["Authorization"].endswith("Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")


def test_credentials_come_from_the_aws_tools_files(monkeypatch, tmp_path):
    assert remote.s3_credentials() is None
    (tmp_path / "credentials").write_text("[default]\naws_access_key_id = AK1\naws_secret_access_key = S1\n[work]\naws_access_key_id = AK2\naws_secret_access_key = S2\n")
    (tmp_path / "config").write_text("[default]\nregion = eu-west-1\n[profile work]\nregion = ap-south-1\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    assert remote.s3_credentials() == {"key": "AK1", "secret": "S1", "token": None, "region": "eu-west-1"}
    monkeypatch.setenv("AWS_PROFILE", "work")
    assert remote.s3_credentials()["key"] == "AK2" and remote.s3_credentials()["region"] == "ap-south-1"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK3")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "S3")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert remote.s3_credentials() == {"key": "AK3", "secret": "S3", "token": None, "region": "us-west-2"}


def test_a_private_bucket_is_read_signed_in_its_own_region(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET")
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("authorization", "")))
        if not request.headers.get("authorization", "").startswith("AWS4-HMAC-SHA256 Credential=AK/"):
            return httpx.Response(403)
        if request.url.host == "data.s3.us-east-1.amazonaws.com":
            return httpx.Response(301, headers={"x-amz-bucket-region": "eu-central-1"})
        return httpx.Response(200, text='{"text": "secret row"}\n')

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "stream", lambda method, u, **kw: httpx.Client(transport=transport).stream(method, u, **{k: v for k, v in kw.items() if k != "follow_redirects"}))
    assert remote.head_rows("s3://data/train.jsonl", 5) == [{"text": "secret row"}]
    assert seen[-1][0] == "data.s3.eu-central-1.amazonaws.com" and "/eu-central-1/s3/aws4_request" in seen[-1][1]
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://account.r2.cloudflarestorage.com")
    remote.fetch("s3://data/train.jsonl", refresh=True)
    assert seen[-1][0] == "account.r2.cloudflarestorage.com"
