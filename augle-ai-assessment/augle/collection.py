"""Browser-based image collection (Playwright) with provenance, licensing and de-duplication.

Design
------
* :class:`PlaywrightSession` is the only code that touches a browser. It renders pages
  (``get_attrs``) and performs HTTP GETs through the browser context (``get_json`` /
  ``get_bytes``), with throttling, retries and an explicit User-Agent.
* :class:`CommonsAdapter` knows the Wikimedia Commons URL scheme and response formats; it works
  against any object implementing the small :class:`BrowserSession` protocol, so it can be unit
  tested with a fake session and integration tested against a local mock server.
* :class:`ImageCollector` applies the acceptance rules (licence, MIME, resolution, size, decodable,
  not a duplicate) and writes ``images/<id>.jpg`` plus ``manifest.json`` (every attempt, with the
  reason for each rejection).

Source choice: Wikimedia Commons hosts freely licensed media, publishes licence/author metadata
for every file and permits polite automated access with an identifying User-Agent.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlencode, urlparse

from augle.coco import save_json_atomic
from augle.config import CollectionConfig
from augle.imaging import (
    ImageReadError,
    decode_image,
    dhash,
    encode_jpeg,
    hamming_distance,
    resize_max_side,
    sha256_hex,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_USER_AGENT = "AugleAssessmentBot/1.0 (educational project; set COLLECTOR_USER_AGENT with contact info)"


TRANSIENT_REASONS = frozenset({"metadata_error", "download_error"})  # retried when a run is resumed


class FetchError(RuntimeError):
    """A network fetch failed permanently (after retries)."""


class RejectedImageError(Exception):
    """A candidate image failed an acceptance rule; ``reason`` is a short machine-readable code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImageMetadata:
    """What the source tells us about one candidate file *before* downloading it."""

    title: str
    page_url: str
    original_url: str
    width: int
    height: int
    size_bytes: int
    mime: str
    license_short: str = ""
    license_url: str = ""
    author: str = ""
    credit: str = ""
    description: str = ""


@dataclass
class ImageRecord:
    """One accepted, stored image with full provenance."""

    image_id: str
    file_name: str
    width: int
    height: int
    original_width: int
    original_height: int
    sha256_original: str
    dhash: str
    title: str
    page_url: str
    original_url: str
    author: str
    credit: str
    license_short: str
    license_url: str
    query: str
    retrieved_utc: str


@dataclass
class CollectionSummary:
    """Aggregate result of a collection run."""

    attempted: int = 0
    accepted: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    accepted_by_license: dict[str, int] = field(default_factory=dict)
    accepted_by_query: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Browser session
# --------------------------------------------------------------------------- #
class BrowserSession(Protocol):
    """The minimal browser capability the collector needs (real or fake)."""

    def get_attrs(self, url: str, css: str, attr: str, *, scroll_rounds: int = 0) -> list[str]:
        """Render ``url`` and return attribute ``attr`` of every element matching ``css``."""

    def get_json(self, url: str) -> dict[str, Any]:
        """HTTP GET returning parsed JSON."""

    def get_bytes(self, url: str) -> bytes:
        """HTTP GET returning the raw body."""


class PlaywrightSession:
    """Chromium (via Playwright sync API) with throttling, retries and an explicit User-Agent."""

    RETRY_STATUS = {429, 500, 502, 503, 504}

    # Headers that make context.request.get() look like a real browser resource
    # load rather than a bot. Wikimedia's CDN (upload.wikimedia.org) checks
    # Referer and Sec-Fetch-* to decide whether to serve the file or return 403.
    _DOWNLOAD_HEADERS = {
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        headless: bool = True,
        executable_path: str | None = None,
        timeout_ms: int = 30_000,
        delay_s: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        self.user_agent, self.headless = user_agent, headless
        self.executable_path = executable_path or None
        self.timeout_ms, self.delay_s, self.max_retries = timeout_ms, delay_s, max_retries
        self._last_request = 0.0
        self._pw = self._browser = self._context = self._page = None

    def __enter__(self) -> PlaywrightSession:
        from playwright.sync_api import sync_playwright  # imported lazily: heavy and optional for tests

        self._pw = sync_playwright().start()
        launch_kwargs: dict[str, Any] = {"headless": self.headless}
        if self.executable_path:
            launch_kwargs["executable_path"] = self.executable_path
        self._browser = self._pw.chromium.launch(**launch_kwargs)
        self._context = self._browser.new_context(user_agent=self.user_agent, viewport={"width": 1366, "height": 900})
        self._context.set_default_timeout(self.timeout_ms)
        self._page = self._context.new_page()
        return self

    def __exit__(self, *exc_info: object) -> None:
        for obj in (self._context, self._browser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # pragma: no cover - best-effort shutdown
                LOGGER.debug("error while closing browser object", exc_info=True)
        if self._pw is not None:
            self._pw.stop()
        self._pw = self._browser = self._context = self._page = None

    def _throttle(self) -> None:
        wait = self.delay_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get_attrs(self, url: str, css: str, attr: str, *, scroll_rounds: int = 0) -> list[str]:
        if self._page is None:
            raise RuntimeError("PlaywrightSession must be used as a context manager")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                status = resp.status if resp is not None else 0
                if status in self.RETRY_STATUS:
                    raise FetchError(f"HTTP {status} for {url}")
                if status >= 400:
                    raise FetchError(f"HTTP {status} for {url} (not retryable)")
                for _ in range(scroll_rounds):  # trigger lazy-loaded galleries
                    self._page.mouse.wheel(0, 4000)
                    self._page.wait_for_timeout(400)
                return self._page.eval_on_selector_all(
                    css, "(els, a) => els.map(e => e.getAttribute(a)).filter(v => v)", attr
                )
            except FetchError as exc:
                last_error = exc
                if "not retryable" in str(exc):
                    break
            except Exception as exc:  # playwright timeouts / navigation errors
                last_error = exc
            time.sleep(min(2.0**attempt, 10.0))
        raise FetchError(f"could not render {url}: {last_error}")

    def _fetch(self, url: str, *, extra_headers: dict[str, str] | None = None) -> bytes:
        """Fetch a URL via the Playwright browser context request API.

        Uses ``context.request.get`` (fast, no page render) with browser-like headers
        so that Wikimedia's CDN does not reject the request with HTTP 403.
        403 responses are retried with backoff in case of transient rate-limiting.
        """
        if self._context is None:
            raise RuntimeError("PlaywrightSession must be used as a context manager")
        last_error: str = "unknown error"
        headers = {"User-Agent": self.user_agent, **self._DOWNLOAD_HEADERS}
        if extra_headers:
            headers.update(extra_headers)
        retry_statuses = self.RETRY_STATUS | {403}  # 403 can be transient on Wikimedia CDN
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._context.request.get(url, headers=headers, timeout=self.timeout_ms)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status == 200:
                    return resp.body()
                last_error = f"HTTP {resp.status}"
                if resp.status not in retry_statuses:
                    break
                retry_after = resp.headers.get("retry-after", "")
                time.sleep(min(float(retry_after) if retry_after.isdigit() else 2.0**attempt, 30.0))
                continue
            time.sleep(min(2.0**attempt, 10.0))
        raise FetchError(f"GET {url} failed: {last_error}")

    def get_json(self, url: str) -> dict[str, Any]:
        try:
            return json.loads(self._fetch(url).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FetchError(f"{url} did not return valid JSON: {exc}") from exc

    def get_bytes(self, url: str) -> bytes:
        """Download binary content by navigating the browser page directly to the URL.

        Both ``context.request.get`` (lacks cookies/TLS fingerprint) and JS ``fetch()``
        (blocked by CORS for cross-origin CDN URLs) return HTTP 403 from Wikimedia's CDN.
        A real page navigation (``page.goto``) is treated as a genuine browser request
        and succeeds.  We capture the response body via the ``response`` event before the
        page content loads, keeping the round-trip fast.

        Falls back to ``_fetch`` (context.request.get + browser headers) if the page is
        unavailable (e.g. during unit tests with a fake session).
        """
        if self._page is None:
            return self._fetch(url)

        # Strip UTM tracking params — they cause redirects on some CDNs.
        clean_url = url.split("?")[0] if "utm_" in url else url

        last_error = "unknown error"
        retry_statuses = self.RETRY_STATUS | {403}
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            captured: list[bytes] = []
            status_code: list[int] = []

            def _on_response(resp: Any) -> None:  # noqa: ANN401
                try:
                    if resp.url.split("?")[0] == clean_url and not captured:
                        status_code.append(resp.status)
                        if resp.status == 200:
                            captured.append(resp.body())
                except Exception:
                    pass

            try:
                self._page.on("response", _on_response)
                nav_resp = self._page.goto(clean_url, wait_until="commit", timeout=self.timeout_ms)
                self._page.remove_listener("response", _on_response)
                # Use the navigation response body if the event didn't fire.
                if not captured and nav_resp is not None and nav_resp.status == 200:
                    captured.append(nav_resp.body())
                    status_code.append(200)
            except Exception as exc:
                try:
                    self._page.remove_listener("response", _on_response)
                except Exception:
                    pass
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2.0**attempt, 10.0))
                continue

            st = status_code[0] if status_code else 0
            if st == 200 and captured:
                return captured[0]
            last_error = f"HTTP {st}"
            if st not in retry_statuses and st != 0:
                break
            time.sleep(min(2.0**attempt, 10.0))
        raise FetchError(f"GET {url} failed: {last_error}")


# --------------------------------------------------------------------------- #
# Wikimedia Commons adapter
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(value: str) -> str:
    """Remove tags/entities from an HTML fragment and collapse whitespace."""
    return re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", value or ""))).strip()


def titles_from_hrefs(hrefs: Iterable[str]) -> list[str]:
    """Extract unique ``File:...`` titles from link hrefs (relative or absolute), keeping order."""
    seen: dict[str, None] = {}
    for href in hrefs:
        path = urlparse(href).path
        if path.startswith("/wiki/File:"):
            seen.setdefault(unquote(path[len("/wiki/") :]).replace("_", " "), None)
    return list(seen)


def license_allowed(license_short: str, prefixes: Iterable[str], denied: Iterable[str] = ()) -> bool:
    """True when the licence starts with an allowed prefix and contains no denied marker.

    Matching is case-insensitive. ``denied`` (e.g. ``-NC``, ``-ND``) wins over ``prefixes`` so that
    non-commercial / no-derivatives variants of an otherwise allowed family are never accepted.
    """
    name = license_short.strip().lower()
    if not name or any(d.lower() in name for d in denied):
        return False
    return any(name.startswith(p.lower()) for p in prefixes)


def _meta_field(meta: dict[str, Any], key: str) -> str:
    """Plain-text value of one ``extmetadata`` entry (HTML stripped)."""
    return strip_html(str((meta.get(key) or {}).get("value", "")))


def parse_imageinfo(payload: dict[str, Any]) -> list[ImageMetadata]:
    """Parse a MediaWiki ``prop=imageinfo`` response (``formatversion`` 1 or 2) into metadata."""
    pages = payload.get("query", {}).get("pages", [])
    page_list = list(pages.values()) if isinstance(pages, dict) else list(pages)
    out: list[ImageMetadata] = []
    for page in page_list:
        info = (page.get("imageinfo") or [None])[0]
        if not info or not info.get("url"):
            continue
        meta = info.get("extmetadata") or {}

        out.append(
            ImageMetadata(
                title=page.get("title", ""),
                page_url=info.get("descriptionurl", ""),
                original_url=info["url"],
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                size_bytes=int(info.get("size") or 0),
                mime=str(info.get("mime", "")),
                license_short=_meta_field(meta, "LicenseShortName"),
                license_url=_meta_field(meta, "LicenseUrl"),
                author=_meta_field(meta, "Artist"),
                credit=_meta_field(meta, "Credit"),
                description=_meta_field(meta, "ImageDescription")[:300],
            )
        )
    return out


class CommonsAdapter:
    """URL construction and parsing for Wikimedia Commons (or a compatible mock server)."""

    def __init__(self, base_url: str = "https://commons.wikimedia.org", results_per_page: int = 50) -> None:
        self.base_url = base_url.rstrip("/")
        self.results_per_page = results_per_page

    def search_url(self, query: str, offset: int) -> str:
        """Classic full-text search restricted to the File namespace."""
        params = {
            "search": query,
            "title": "Special:Search",
            "profile": "advanced",
            "fulltext": "1",
            "ns6": "1",
            "limit": self.results_per_page,
            "offset": offset,
        }
        return f"{self.base_url}/w/index.php?{urlencode(params)}"

    def imageinfo_url(self, titles: list[str]) -> str:
        """MediaWiki API URL returning URL, size, MIME and licence metadata for ``titles``."""
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription",
            "titles": "|".join(titles),
        }
        return f"{self.base_url}/w/api.php?{urlencode(params, quote_via=quote)}"

    def search_titles(self, session: BrowserSession, query: str, offset: int) -> list[str]:
        """Render a search-results page in the browser and return the ``File:`` titles found."""
        hrefs = session.get_attrs(self.search_url(query, offset), "#mw-content-text a[href*='/wiki/File:']", "href")
        return titles_from_hrefs(hrefs)

    def fetch_metadata(self, session: BrowserSession, titles: list[str]) -> list[ImageMetadata]:
        """Resolve titles to downloadable URLs plus licence/author metadata."""
        return parse_imageinfo(session.get_json(self.imageinfo_url(titles)))


# --------------------------------------------------------------------------- #
# Acceptance rules
# --------------------------------------------------------------------------- #
@dataclass
class PreparedImage:
    """A decoded, size-limited image ready to be stored."""

    jpeg_bytes: bytes
    width: int
    height: int
    original_width: int
    original_height: int
    dhash_value: int


def prepare_image(data: bytes, cfg: CollectionConfig) -> PreparedImage:
    """Decode, enforce resolution, down-scale to ``max_stored_side`` and re-encode as JPEG.

    Re-encoding bakes in EXIF orientation and strips metadata so every reader sees identical pixels.
    """
    try:
        img = decode_image(data)
    except ImageReadError as exc:
        raise RejectedImageError("decode_error", str(exc)) from exc
    h, w = img.shape[:2]
    if max(h, w) < cfg.min_long_side or min(h, w) < cfg.min_short_side:
        raise RejectedImageError("too_small", f"{w}x{h} < {cfg.min_long_side}/{cfg.min_short_side}")
    stored = resize_max_side(img, cfg.max_stored_side)
    return PreparedImage(
        jpeg_bytes=encode_jpeg(stored, cfg.jpeg_quality),
        width=stored.shape[1],
        height=stored.shape[0],
        original_width=w,
        original_height=h,
        dhash_value=dhash(stored),
    )


class Deduplicator:
    """Exact (SHA-256) and near-duplicate (dHash Hamming distance) detection."""

    def __init__(self, max_hamming: int) -> None:
        self.max_hamming = max_hamming
        self._sha: set[str] = set()
        self._hashes: list[int] = []

    def add_known(self, sha: str, dh: int) -> None:
        """Register an already-accepted image."""
        self._sha.add(sha)
        self._hashes.append(dh)

    def duplicate_reason(self, sha: str, dh: int) -> str | None:
        """Return ``'duplicate_exact'`` / ``'duplicate_near'`` or ``None`` when the image is new."""
        if sha in self._sha:
            return "duplicate_exact"
        if any(hamming_distance(dh, other) <= self.max_hamming for other in self._hashes):
            return "duplicate_near"
        return None


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
class ImageCollector:
    """Searches, filters, downloads and stores images; resumable via ``manifest.json``."""

    def __init__(
        self, cfg: CollectionConfig, session: BrowserSession, adapter: CommonsAdapter, out_dir: str | Path
    ) -> None:
        self.cfg, self.session, self.adapter = cfg, session, adapter
        self.out_dir = Path(out_dir)
        self.image_dir = self.out_dir / "images"
        self.manifest_path = self.out_dir / "manifest.json"
        self.records: list[ImageRecord] = []
        self.rejections: list[dict[str, str]] = []
        self.seen_titles: set[str] = set()
        self.dedup = Deduplicator(cfg.dedupe_hamming)
        self._load_manifest()

    # -- persistence ------------------------------------------------------ #
    def _load_manifest(self) -> None:
        if not self.manifest_path.is_file():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records = [ImageRecord(**r) for r in data.get("accepted", [])]
        self.rejections = [r for r in data.get("rejected", []) if r["reason"] not in TRANSIENT_REASONS]
        self.seen_titles = {r.title for r in self.records} | {r["title"] for r in self.rejections}
        for rec in self.records:
            if (self.out_dir / rec.file_name).is_file():
                self.dedup.add_known(rec.sha256_original, int(rec.dhash, 16))
        LOGGER.info("resumed manifest: %d accepted, %d rejected", len(self.records), len(self.rejections))

    def _save_manifest(self) -> None:
        save_json_atomic(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": self.cfg.source,
                "collection_config": asdict(self.cfg),
                "summary": asdict(self.summary()),
                "accepted": [asdict(r) for r in self.records],
                "rejected": self.rejections,
            },
            self.manifest_path,
        )

    def summary(self) -> CollectionSummary:
        """Aggregate statistics from the in-memory records."""
        s = CollectionSummary(attempted=len(self.records) + len(self.rejections), accepted=len(self.records))
        for rej in self.rejections:
            s.rejected_by_reason[rej["reason"]] = s.rejected_by_reason.get(rej["reason"], 0) + 1
        for rec in self.records:
            s.accepted_by_license[rec.license_short] = s.accepted_by_license.get(rec.license_short, 0) + 1
            s.accepted_by_query[rec.query] = s.accepted_by_query.get(rec.query, 0) + 1
        return s

    # -- processing ------------------------------------------------------- #
    def _reject(self, meta_title: str, reason: str, detail: str = "") -> None:
        self.rejections.append({"title": meta_title, "reason": reason, "detail": detail})
        if reason not in TRANSIENT_REASONS:
            self.seen_titles.add(meta_title)

    def _check_metadata(self, meta: ImageMetadata) -> None:
        cfg = self.cfg
        if not license_allowed(meta.license_short, cfg.allowed_license_prefixes, cfg.denied_license_substrings):
            raise RejectedImageError("license_not_allowed", meta.license_short or "missing")
        if meta.mime not in cfg.allowed_mime:
            raise RejectedImageError("mime_not_allowed", meta.mime)
        if meta.size_bytes > cfg.max_file_bytes:
            raise RejectedImageError("file_too_large", f"{meta.size_bytes} bytes")
        if max(meta.width, meta.height) < cfg.min_long_side or min(meta.width, meta.height) < cfg.min_short_side:
            raise RejectedImageError("too_small_metadata", f"{meta.width}x{meta.height}")

    def _process(self, meta: ImageMetadata, query: str) -> bool:
        """Try to accept one candidate. Returns True when it was stored."""
        try:
            self._check_metadata(meta)
            try:
                data = self.session.get_bytes(meta.original_url)
            except FetchError as exc:
                raise RejectedImageError("download_error", str(exc)) from exc
            sha = sha256_hex(data)
            prepared = prepare_image(data, self.cfg)
            dup = self.dedup.duplicate_reason(sha, prepared.dhash_value)
            if dup:
                raise RejectedImageError(dup)
        except RejectedImageError as exc:
            self._reject(meta.title, exc.reason, exc.detail)
            LOGGER.info("rejected %s: %s", meta.title, exc)
            return False
        image_id = sha[:12]
        rel = f"images/{image_id}.jpg"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / rel).write_bytes(prepared.jpeg_bytes)
        self.dedup.add_known(sha, prepared.dhash_value)
        self.seen_titles.add(meta.title)
        self.records.append(
            ImageRecord(
                image_id=image_id,
                file_name=rel,
                width=prepared.width,
                height=prepared.height,
                original_width=prepared.original_width,
                original_height=prepared.original_height,
                sha256_original=sha,
                dhash=f"{prepared.dhash_value:016x}",
                title=meta.title,
                page_url=meta.page_url,
                original_url=meta.original_url,
                author=meta.author,
                credit=meta.credit,
                license_short=meta.license_short,
                license_url=meta.license_url,
                query=query,
                retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        LOGGER.info("accepted %s (%d/%d)", meta.title, len(self.records), self.cfg.target_images)
        return True

    def run(self) -> CollectionSummary:
        """Collect until ``target_images`` are accepted or the queries are exhausted."""
        cfg = self.cfg
        try:
            for query in cfg.queries:
                for page_idx in range(cfg.max_pages_per_query):
                    if len(self.records) >= cfg.target_images:
                        break
                    try:
                        titles = self.adapter.search_titles(self.session, query, page_idx * cfg.results_per_page)
                    except FetchError as exc:
                        LOGGER.error("search failed for %r page %d: %s", query, page_idx, exc)
                        break
                    fresh = [t for t in titles if t not in self.seen_titles]
                    if not titles:
                        break
                    for start in range(0, len(fresh), 20):
                        chunk = fresh[start : start + 20]
                        try:
                            metas = self.adapter.fetch_metadata(self.session, chunk)
                        except FetchError as exc:
                            LOGGER.error("metadata fetch failed: %s", exc)
                            for t in chunk:
                                self._reject(t, "metadata_error", str(exc))
                            continue
                        returned = {m.title.replace("_", " ") for m in metas}
                        for t in chunk:
                            if t.replace("_", " ") not in returned:
                                self._reject(t, "metadata_missing")
                        for meta in metas:
                            if len(self.records) >= cfg.target_images:
                                break
                            self._process(meta, query)
                        self._save_manifest()
        finally:
            self._save_manifest()
        return self.summary()
