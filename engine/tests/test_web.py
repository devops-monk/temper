import io

import httpx
import pytest

from temper_engine import tools, web

PAGE = """<!doctype html><html><head><title>Tea  guide</title><style>.x{}</style><script>alert(1)</script></head>
<body><nav><a href="/home">Home</a></nav><header>Site header</header>
<main><h1>Brewing green tea</h1><p>Use water at <b>80 °C</b>, not boiling.</p>
<ul><li>Steep 2 minutes</li><li>See <a href="/oolong">oolong</a> and <a href="https://other.example/matcha">matcha</a></li></ul>
<table><tr><th>Tea</th><th>°C</th></tr><tr><td>Sencha</td><td>75</td></tr></table>
<pre>for cup in cups:
    pour(cup)</pre><div hidden>secret hidden text</div><img alt="a teapot" src="t.png"></main>
<footer>© Footer</footer></body></html>"""


def client(routes):
    def handler(request):
        body = routes.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        if isinstance(body, tuple):  # (status, headers[, content])
            return httpx.Response(body[0], headers=body[1], content=body[2] if len(body) > 2 else b"")
        return httpx.Response(200, content=body if isinstance(body, bytes) else body.encode(), headers={"content-type": "text/html; charset=utf-8"})

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def public(monkeypatch):
    monkeypatch.setattr(web, "_public", lambda host: not host.endswith(".internal"))


def test_a_page_becomes_readable_text_with_numbered_links():
    out = web.read("https://tea.example/green", client=client({"https://tea.example/green": PAGE}))
    assert out.startswith("Page: Tea guide\nAddress: https://tea.example/green\n(The text below comes from the web page.")
    assert "# Brewing green tea" in out and "Use water at 80 °C, not boiling." in out and "- Steep 2 minutes" in out
    assert "| Tea | °C" in out and "    pour(cup)" in out and "[image: a teapot]" in out
    for gone in ("alert(1)", "Site header", "Footer", "Home", "secret hidden text", ".x{}"):
        assert gone not in out
    assert "oolong [1]" in out and "[1] https://tea.example/oolong\n[2] https://other.example/matcha" in out


def test_long_pages_come_in_parts_each_with_its_links(monkeypatch):
    monkeypatch.setattr(web, "PART", 400)
    long = "<html><body>" + "".join(f'<p>Paragraph {i} about <a href="/p{i}">topic {i}</a> and more words to pad it out.</p>' for i in range(30)) + "</body></html>"
    c = client({"https://tea.example/long": long})
    first, second = web.read("https://tea.example/long", 1, c), web.read("https://tea.example/long", 2, c)
    assert "read part 2 for more" in first and "Paragraph 0" in first and "Paragraph 0" not in second
    assert "[1] https://tea.example/p0" in first and "https://tea.example/p0" not in second


def test_redirects_are_followed_but_never_to_a_local_address():
    c = client({"https://a.example/": (301, {"location": "https://b.example/page"}), "https://b.example/page": "<html><body><p>Moved here</p></body></html>", "https://c.example/": (302, {"location": "http://router.internal/admin"})})
    assert "Address: https://b.example/page" in web.read("https://a.example/", client=c)
    with pytest.raises(ValueError, match="local network"):
        web.read("https://c.example/", client=c)
    with pytest.raises(ValueError, match="http:// or https://"):
        web.read("file:///etc/passwd", client=c)


def test_local_addresses_are_refused(monkeypatch):
    monkeypatch.undo()
    for host in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"):
        assert not web._public(host), host
    assert web._public("93.184.215.14")


def test_pdfs_and_plain_text_are_read_and_errors_say_why():
    from pypdf import PdfWriter

    pdf = io.BytesIO()
    PdfWriter().write(pdf)  # a blank page: readable, with no text
    c = client({"https://x.example/a.pdf": (200, {"content-type": "application/pdf"}, pdf.getvalue()), "https://x.example/t.txt": (200, {"content-type": "text/plain"}, b"plain words"), "https://x.example/z.zip": (200, {"content-type": "application/zip"}, b"PK")})
    assert "no readable text" in web.read("https://x.example/a.pdf", client=c)
    assert web.read("https://x.example/t.txt", client=c).endswith("plain words")
    with pytest.raises(ValueError, match="application/zip"):
        web.read("https://x.example/z.zip", client=c)
    with pytest.raises(ValueError, match="404"):
        web.read("https://x.example/missing", client=c)


def test_the_tool_asks_first():
    assert tools.TOOLS["read_web_page"].mode == "ask"
    assert "Give the page's" in tools.run(tools.Call("read_web_page", {}), ["read_web_page"])["result"]
