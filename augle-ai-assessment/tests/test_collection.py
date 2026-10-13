"""Collection: parsers, acceptance rules, de-duplication, resumable manifest, and real-browser integration.

The Wikimedia endpoints are replaced by (a) a fake in-memory session and (b) a local mock HTTP server
that mimics the Commons URL scheme. These tests verify OUR logic and the Playwright plumbing; they say
nothing about live Wikimedia availability or markup (the sandbox has no access to it).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pytest

from augle.collection import (
    CommonsAdapter,
    Deduplicator,
    FetchError,
    ImageCollector,
    PlaywrightSession,
    RejectedImageError,
    license_allowed,
    parse_imageinfo,
    prepare_image,
    strip_html,
    titles_from_hrefs,
)
from augle.config import CollectionConfig
from augle.imaging import decode_image, dhash, encode_jpeg, hamming_distance


def texture(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    """Distinct, high-frequency test image (so perceptual hashes of different seeds differ)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg(seed: int, h: int = 240, w: int = 320, q: int = 90) -> bytes:
    return encode_jpeg(texture(seed, h, w), q)


SMALL_CFG = dict(
    min_long_side=300,
    min_short_side=200,
    max_stored_side=400,
    request_delay_s=0.0,
    results_per_page=5,
    max_pages_per_query=2,
    queries=["robot"],
    target_images=50,
    max_retries=2,
)


def meta_json(
    title: str,
    url: str,
    *,
    w=320,
    h=240,
    size=10_000,
    mime="image/jpeg",
    lic="CC BY-SA 4.0",
    artist="<a href='x'>Ann &amp; Bo</a>",
) -> dict:
    return {
        "title": title,
        "pageid": 1,
        "imageinfo": [
            {
                "url": url,
                "descriptionurl": f"https://commons.example/wiki/{title.replace(' ', '_')}",
                "width": w,
                "height": h,
                "size": size,
                "mime": mime,
                "extmetadata": {
                    "LicenseShortName": {"value": lic},
                    "LicenseUrl": {"value": "https://cc.example/by-sa"},
                    "Artist": {"value": artist},
                    "Credit": {"value": "<b>Own work</b>"},
                    "ImageDescription": {"value": "<p>A robot</p>"},
                },
            }
        ],
    }


# ----------------------------------------------------------------- parsers ----
def test_titles_from_hrefs_normalises_dedupes_and_filters():
    hrefs = [
        "/wiki/File:KUKA_robot.jpg",
        "https://commons.wikimedia.org/wiki/File:KUKA_robot.jpg#filehistory",
        "/wiki/File:Caf%C3%A9_arm.png",
        "/wiki/Category:Robots",
        "/w/index.php?title=File:Foo.jpg",
        "/wiki/File:ABB_IRB.jpg",
    ]
    assert titles_from_hrefs(hrefs) == ["File:KUKA robot.jpg", "File:Café arm.png", "File:ABB IRB.jpg"]


def test_strip_html_and_license_rules():
    assert strip_html("<a href='x'>Ann &amp; Bo</a>\n <b>Carl</b>") == "Ann & Bo Carl"
    cfg = CollectionConfig()
    ok = lambda name: license_allowed(name, cfg.allowed_license_prefixes, cfg.denied_license_substrings)  # noqa: E731
    for name in ("CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 4.0", "cc0 1.0", "CC0", "Public domain", "PD-old-70"):
        assert ok(name), name
    for name in (
        "CC BY-NC 4.0",
        "CC BY-NC-SA 2.0",
        "CC BY-ND 2.0",
        "CC BY-NC-ND 4.0",
        "",
        "All rights reserved",
        "GFDL",
    ):
        assert not ok(name), name
    # the deny list wins even if a caller configures a permissive prefix
    assert not license_allowed("CC BY-NC 4.0", ["CC BY"], ["-NC"]) and license_allowed("CC BY-NC 4.0", ["CC BY"])


def test_parse_imageinfo_handles_both_formatversions_and_missing_pages():
    page = meta_json("File:A.jpg", "https://u/a.jpg")
    v2 = {"query": {"pages": [page, {"title": "File:Gone.jpg", "missing": True}]}}
    v1 = {"query": {"pages": {"1": page, "-1": {"title": "File:Gone.jpg", "missing": ""}}}}
    for payload in (v2, v1):
        out = parse_imageinfo(payload)
        assert len(out) == 1
        m = out[0]
        assert (m.title, m.original_url, m.width, m.height, m.mime) == (
            "File:A.jpg",
            "https://u/a.jpg",
            320,
            240,
            "image/jpeg",
        )
        assert (
            m.author == "Ann & Bo"
            and m.credit == "Own work"
            and m.license_short == "CC BY-SA 4.0"
            and m.description == "A robot"
        )
        assert m.page_url.endswith("File:A.jpg")
    assert parse_imageinfo({}) == []


def test_adapter_urls():
    a = CommonsAdapter("https://commons.example/", 50)
    q = parse_qs(urlparse(a.search_url("kuka robot", 100)).query)
    assert q["search"] == ["kuka robot"] and q["ns6"] == ["1"] and q["offset"] == ["100"] and q["limit"] == ["50"]
    u = urlparse(a.imageinfo_url(["File:A b.jpg", "File:C.png"]))
    assert u.path == "/w/api.php" and parse_qs(u.query)["titles"] == ["File:A b.jpg|File:C.png"]


# ------------------------------------------------------- acceptance rules ----
def test_prepare_image_rules():
    cfg = CollectionConfig(**SMALL_CFG)
    ok = prepare_image(jpeg(1, 500, 700), cfg)
    assert max(ok.width, ok.height) == 400 and ok.original_width == 700 and ok.original_height == 500  # down-scaled
    assert decode_image(ok.jpeg_bytes).shape[:2] == (ok.height, ok.width)
    with pytest.raises(RejectedImageError) as e:
        prepare_image(jpeg(1, 100, 150), cfg)
    assert e.value.reason == "too_small"
    with pytest.raises(RejectedImageError) as e:
        prepare_image(b"definitely not an image", cfg)
    assert e.value.reason == "decode_error"
    with pytest.raises(RejectedImageError):
        prepare_image(b"", cfg)


def test_deduplicator_exact_and_near():
    img = texture(3)
    near = cv2.GaussianBlur(cv2.resize(cv2.resize(img, (160, 120)), (320, 240)), (3, 3), 0)
    assert hamming_distance(dhash(img), dhash(near)) <= 6
    dd = Deduplicator(max_hamming=8)
    dd.add_known("sha-a", dhash(img))
    assert dd.duplicate_reason("sha-a", 0xFFFF) == "duplicate_exact"
    assert dd.duplicate_reason("sha-b", dhash(near)) == "duplicate_near"
    assert dd.duplicate_reason("sha-c", dhash(texture(99))) is None


# ---------------------------------------------------- collector (fake session) ----
class FakeSession:
    """TEST DOUBLE for the browser: serves canned search hrefs, API JSON and image bytes."""

    def __init__(self, files: dict[str, dict], blobs: dict[str, bytes], fail_urls: set[str] | None = None) -> None:
        self.files, self.blobs, self.fail_urls = files, blobs, fail_urls or set()
        self.byte_requests: list[str] = []

    def get_attrs(self, url, css, attr, *, scroll_rounds=0):
        if "offset=0" not in url:
            return []
        return ["/wiki/" + t.replace(" ", "_") for t in self.files]

    def get_json(self, url):
        titles = parse_qs(urlparse(url).query)["titles"][0].split("|")
        return {"query": {"pages": [self.files.get(t, {"title": t, "missing": True}) for t in titles]}}

    def get_bytes(self, url):
        self.byte_requests.append(url)
        if url in self.fail_urls:
            raise FetchError(f"GET {url} failed: HTTP 503")
        return self.blobs[url]


def build_fake() -> tuple[FakeSession, dict[str, str]]:
    """One file per acceptance scenario; returns the session and title -> expected outcome."""
    files, blobs, expected = {}, {}, {}

    def add(name, data, outcome, **kw):
        title, url = f"File:{name.replace('_', ' ')}.jpg", f"https://img.example/{name}.jpg"  # API titles use spaces
        files[title] = meta_json(title, url, **kw)
        blobs[url] = data
        expected[title] = outcome

    add("good_one", jpeg(10), "accepted")
    add("good_two", jpeg(11), "accepted")
    add("nc_license", jpeg(12), "license_not_allowed", lic="CC BY-NC 2.0")
    add("no_license", jpeg(13), "license_not_allowed", lic="")
    add("a_png_as_gif", jpeg(14), "mime_not_allowed", mime="image/gif")
    add("huge", jpeg(15), "file_too_large", size=99_000_000)
    add("tiny_meta", jpeg(16), "too_small_metadata", w=100, h=80)
    add("really_tiny", jpeg(17, 100, 120), "too_small", w=320, h=240)  # metadata lies, decode catches it
    add("corrupt", b"\xff\xd8garbage-bytes", "decode_error")
    add("dup_exact", jpeg(10), "duplicate_exact")
    add("dup_near", encode_jpeg(cv2.GaussianBlur(texture(10), (3, 3), 0), 70), "duplicate_near")
    add("net_fail", jpeg(18), "download_error")
    files["File:ghost.jpg"] = {"title": "File:ghost.jpg", "missing": True}
    expected["File:ghost.jpg"] = "metadata_missing"
    session = FakeSession(files, blobs, fail_urls={"https://img.example/net_fail.jpg"})
    return session, expected


def test_collector_applies_every_rule_and_records_provenance(tmp_path):
    cfg = CollectionConfig(**SMALL_CFG)
    session, expected = build_fake()
    collector = ImageCollector(cfg, session, CommonsAdapter("https://commons.example", 50), tmp_path)
    summary = collector.run()
    got = {r["title"]: r["reason"] for r in collector.rejections}
    got.update({r.title: "accepted" for r in collector.records})
    assert got == expected
    assert summary.accepted == 2 and summary.attempted == len(expected)
    assert summary.rejected_by_reason["license_not_allowed"] == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    rec = manifest["accepted"][0]
    assert (
        rec["license_short"] == "CC BY-SA 4.0"
        and rec["author"] == "Ann & Bo"
        and rec["original_url"].startswith("https://img.example/")
    )
    assert (
        rec["page_url"].startswith("https://commons.example/wiki/")
        and rec["query"] == "robot"
        and rec["sha256_original"]
    )
    for r in collector.records:  # stored file exists, decodes, and matches recorded size
        stored = decode_image((tmp_path / r.file_name).read_bytes())
        assert stored.shape[:2] == (r.height, r.width) and max(r.height, r.width) <= 400
    assert manifest["summary"]["accepted"] == 2 and manifest["collection_config"]["min_long_side"] == 300


def test_collector_stops_at_target_and_resumes_from_manifest(tmp_path):
    cfg = CollectionConfig(**{**SMALL_CFG, "target_images": 1})
    session, _ = build_fake()
    first = ImageCollector(cfg, session, CommonsAdapter("https://commons.example", 50), tmp_path)
    assert first.run().accepted == 1
    # titles after the cut-off were never examined, so they must NOT be recorded as seen
    # (the API reported 'ghost' missing up-front, which counts as examined)
    assert first.seen_titles == {"File:good one.jpg", "File:ghost.jpg"}
    # raise the target and resume: known titles are not fetched again, the duplicate guard is restored from disk
    cfg2 = CollectionConfig(**{**SMALL_CFG, "target_images": 10})
    session2, expected = build_fake()
    second = ImageCollector(cfg2, session2, CommonsAdapter("https://commons.example", 50), tmp_path)
    assert len(second.records) == 1
    assert second.run().accepted == 2
    assert "https://img.example/good_one.jpg" not in session2.byte_requests
    reasons = {r["title"]: r["reason"] for r in second.rejections}
    assert reasons["File:dup exact.jpg"] == "duplicate_exact"  # detected against the image stored by the first run
    titles = [r.title for r in second.records]
    assert len(set(titles)) == len(titles) == 2
    # transient failures are retried on resume rather than being remembered as permanent rejections
    third = ImageCollector(cfg2, build_fake()[0], CommonsAdapter("https://commons.example", 50), tmp_path)
    assert all(r["reason"] != "download_error" for r in third.rejections)


def test_collector_survives_search_failure(tmp_path):
    class Broken(FakeSession):
        def get_attrs(self, *a, **k):
            raise FetchError("search page unavailable")

    cfg = CollectionConfig(**SMALL_CFG)
    collector = ImageCollector(cfg, Broken({}, {}), CommonsAdapter("https://commons.example", 50), tmp_path)
    assert collector.run().accepted == 0 and (tmp_path / "manifest.json").is_file()


# ------------------------------------------- real browser vs local mock server ----
class MockCommons(BaseHTTPRequestHandler):
    """Serves a Commons-like search page, imageinfo API and image bytes; records headers and flakiness."""

    files: dict[str, dict] = {}
    blobs: dict[str, bytes] = {}
    seen_user_agents: list[str] = []
    flaky_hits = 0

    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        MockCommons.seen_user_agents.append(self.headers.get("User-Agent", ""))
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/w/index.php":
            offset = int(q.get("offset", ["0"])[0])
            links = (
                "".join(
                    f'<li class="mw-search-result"><a href="/wiki/{t.replace(" ", "_")}">{t}</a></li>'
                    for t in list(self.files)[offset : offset + 5]
                )
                if offset < 100
                else ""
            )
            html = f'<html><body><div id="mw-content-text"><ul class="mw-search-results">{links}</ul></div><a href="/wiki/File:Sidebar_logo.png">logo</a></body></html>'
            self._send(200, html.encode(), "text/html")
        elif url.path == "/w/api.php":
            titles = q["titles"][0].split("|")
            pages = [self.files.get(t, {"title": t, "missing": True}) for t in titles]
            self._send(200, json.dumps({"query": {"pages": pages}}).encode(), "application/json")
        elif url.path == "/img/flaky.jpg":
            MockCommons.flaky_hits += 1
            if MockCommons.flaky_hits == 1:
                self._send(503, b"busy", "text/plain", {"Retry-After": "0"})
            else:
                self._send(200, self.blobs["/img/flaky.jpg"], "image/jpeg")
        elif url.path in self.blobs:
            self._send(200, self.blobs[url.path], "image/jpeg")
        else:
            self._send(404, b"nope", "text/plain")


@pytest.fixture()
def mock_server():
    MockCommons.files, MockCommons.blobs, MockCommons.seen_user_agents, MockCommons.flaky_hits = {}, {}, [], 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockCommons)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base
    server.shutdown()
    thread.join(timeout=5)


@pytest.mark.browser
def test_collector_with_real_playwright_chromium_against_mock_server(mock_server, tmp_path):
    base = mock_server
    for i, (name, lic) in enumerate(
        [("one", "CC BY 4.0"), ("two", "CC0"), ("flaky", "Public domain"), ("nc", "CC BY-NC 4.0")]
    ):
        title = f"File:Robot {name}.jpg"
        MockCommons.files[title] = meta_json(title, f"{base}/img/{name}.jpg", lic=lic)
        MockCommons.blobs[f"/img/{name}.jpg"] = jpeg(100 + i)
    MockCommons.files["File:Robot missing.jpg"] = {"title": "File:Robot missing.jpg", "missing": True}
    cfg = CollectionConfig(**{**SMALL_CFG, "base_url": base})
    ua = "AugleTestBot/0.1 (pytest; contact: test@example.org)"
    try:
        session = PlaywrightSession(
            user_agent=ua, headless=True, executable_path=None, timeout_ms=15_000, delay_s=0.0, max_retries=3
        )
        session.__enter__()
    except Exception as exc:  # no launchable Chromium in this environment
        pytest.skip(f"Playwright Chromium not launchable: {exc}")
    try:
        collector = ImageCollector(cfg, session, CommonsAdapter(base, 50), tmp_path)
        summary = collector.run()
        # direct checks of the session primitives through the real browser
        hrefs = session.get_attrs(
            f"{base}/w/index.php?search=x&offset=0", "#mw-content-text a[href*='/wiki/File:']", "href"
        )
        assert all("Sidebar_logo" not in h for h in hrefs) and len(hrefs) == 5
        assert (
            session.get_json(CommonsAdapter(base).imageinfo_url(["File:Robot one.jpg"]))["query"]["pages"][0]["title"]
            == "File:Robot one.jpg"
        )
        with pytest.raises(FetchError, match="HTTP 404"):
            session.get_bytes(f"{base}/img/does-not-exist.jpg")
    finally:
        session.__exit__(None, None, None)
    assert summary.accepted == 3, summary
    assert summary.rejected_by_reason == {"license_not_allowed": 1, "metadata_missing": 1}
    assert MockCommons.flaky_hits >= 2  # the 503 was retried and then succeeded
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 3
    assert ua in set(MockCommons.seen_user_agents) and all(
        u == ua for u in MockCommons.seen_user_agents
    )  # page loads AND API calls identify us
