"""Retrieval layer for the data auditor: chunking, embeddings, vector store, retriever.

* **Chunking** turns the COCO JSON, collection metadata, training log and run configuration into
  small, traceable documents with stable ids (``image:<id>``, ``training:epoch:<n>``, ...).
* **Embedders** are pluggable: a dependency-free signed feature-hashing embedder (offline default),
  any OpenAI-compatible ``/embeddings`` endpoint, or ``sentence-transformers`` if installed.
* **Vector stores**: a persistent exact-cosine flat index (NumPy, default) and an optional
  ChromaDB adapter. Both implement :class:`VectorStore`.
* Everything retrieved is **untrusted data** (image titles/authors come from the web): chunk text is
  sanitised and later presented to the LLM inside delimited evidence blocks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from augle.audit_tools import AuditContext, analyze_bbox_areas, analyze_class_imbalance, get_dataset_statistics
from augle.coco import save_json_atomic
from augle.utils import file_sha256

CHUNKER_VERSION = "4"
MAX_CHUNK_CHARS = 1800


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    """One retrievable chunk with a stable id and provenance metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """A retrieval result."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize(text: Any, max_len: int = 120) -> str:
    """Make untrusted text safe to embed in an evidence block (no tags, control chars, newlines)."""
    out = _CONTROL.sub("", str(text)).replace("<", "(").replace(">", ")")
    out = re.sub(r"\s+", " ", out).strip()
    return out[:max_len]


def _f(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHUNK_CHARS else text[: MAX_CHUNK_CHARS - 3] + "..."


def build_documents(
    ctx: AuditContext,
    metrics: dict[str, Any] | None = None,
    dataset_meta: dict[str, Any] | None = None,
) -> list[Document]:
    """Create traceable documents from the dataset, collection metadata and training run."""
    coco = ctx.coco()
    docs: list[Document] = []
    stats = get_dataset_statistics(ctx, {})
    imbalance = analyze_class_imbalance(ctx, {})
    areas = analyze_bbox_areas(ctx, {})
    names = {c["id"]: c["name"] for c in coco.get("categories", [])}

    docs.append(
        Document(
            "dataset:summary",
            _clip(
                "Topic: dataset overview, number of images, annotations and categories, counts per category, image "
                f"sizes, licences. Dataset summary (COCO file): {stats['num_images']} images, "
                f"{stats['num_annotations']} annotations, "
                f"{stats['num_categories']} categories. Annotations per category: "
                f"{json.dumps(stats['annotations_per_category'])}. "
                f"{stats['images_without_annotations']} images have no annotations. Annotations per image: "
                f"{json.dumps(stats['annotations_per_image'])}. Image size range px: "
                f"{json.dumps(stats['image_size_px'])}. "
                f"{stats['annotations_with_segmentation']} annotations carry segmentation polygons. "
                f"Images per licence: {json.dumps({sanitize(k): v for k, v in stats['images_per_license'].items()})}."
            ),
            {"source": "coco", "kind": "summary"},
        )
    )
    if "error" not in imbalance:
        docs.append(
            Document(
                "dataset:class_balance",
                _clip(
                    "Topic: class imbalance, class distribution, majority and minority class, imbalance ratio. "
                    f"Class balance: counts {json.dumps(imbalance['class_counts'])}, frequencies "
                    f"{json.dumps(imbalance['class_frequencies'])}. "
                    f"Majority class {imbalance['majority_class']}, minority class {imbalance['minority_class']}, "
                    f"imbalance ratio (max/min) {_f(imbalance['imbalance_ratio_max_over_min'])}, normalised entropy "
                    f"{_f(imbalance['normalized_entropy'])}, heuristic label {imbalance['heuristic_label']}. "
                    f"Images containing each class: {json.dumps(imbalance['images_containing_class'])}."
                ),
                {"source": "coco", "kind": "class_balance"},
            )
        )
    th = areas["thresholds_used"]
    docs.append(
        Document(
            "dataset:bbox_areas",
            _clip(
                "Topic: bounding box area statistics, box size distribution, normalised area, invalid geometry, "
                f"small box thresholds. Bounding-box area statistics over {areas['annotations_analyzed']} "
                "annotations. Area in px^2 distribution: "
                f"{json.dumps({k: round(v, 1) for k, v in areas['area_px_distribution'].items()})}. Normalised area "
                "(fraction of image) "
                "distribution: "
                f"{json.dumps({k: round(v, 5) for k, v in areas['normalized_area_distribution'].items()})}. "
                f"{areas['suspiciously_small_count']} suspiciously small boxes at thresholds "
                f"area<{th['abs_area_threshold_px']} px^2, "
                f"normalised area<{th['norm_area_threshold']}, min side<{th['min_side_px']} px. "
                f"Invalid geometry counts: {json.dumps({k: v['count'] for k, v in areas['invalid_geometry'].items()})}."
            ),
            {"source": "coco", "kind": "bbox_areas"},
        )
    )
    small_lines = [
        f"ann {e['annotation_id']} image {e['image_id']} ({sanitize(e['file_name'], 60)}) {e['category']} "
        f"bbox {e['bbox_xywh']} area {e['area_px']} px^2 normalised {e['normalized_area']}"
        for e in areas["smallest_examples"][:12]
    ]
    docs.append(
        Document(
            "dataset:small_boxes",
            _clip(
                "Topic: suspicious small bounding boxes, tiny box outliers, smallest annotations and examples. "
                "Suspiciously small bounding boxes flagged at ingest (smallest first): "
                + ("; ".join(small_lines) if small_lines else "none flagged.")
            ),
            {"source": "coco", "kind": "small_boxes"},
        )
    )

    per_image: dict[int, list[dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        per_image.setdefault(a["image_id"], []).append(a)
    licenses = {lic["id"]: lic.get("name", "unknown") for lic in coco.get("licenses", [])}
    for im in coco.get("images", []):
        parts = []
        for a in per_image.get(im["id"], []):
            x, y, w, h = a["bbox"]
            norm = (w * h) / float(im["width"] * im["height"])
            flag = (
                " [SUSPICIOUSLY SMALL BOX]"
                if (w * h < ctx.small_abs_area_px or norm < ctx.small_norm_area or min(w, h) < ctx.min_side_px)
                else ""
            )
            parts.append(
                f"ann {a['id']}: {names.get(a['category_id'], '?')} "
                f"bbox[x,y,w,h]=[{_f(x, 5)},{_f(y, 5)},{_f(w, 5)},{_f(h, 5)}] "
                f"area {_f(a.get('area'), 6)} px^2 ({norm * 100:.3f}% of image){flag}"
            )
        docs.append(
            Document(
                f"image:{im['id']}",
                _clip(
                    f"Topic: per-image annotations. Image {im['id']} file {sanitize(im['file_name'], 80)} size "
                    f"{im['width']}x{im['height']} px, licence "
                    f"{sanitize(licenses.get(im.get('license'), 'unspecified'), 40)}, "
                    f'author "{sanitize(im.get("author", ""), 60)}". '
                    f"{len(parts)} annotation(s): " + ("; ".join(parts) if parts else "none")
                ),
                {"source": "coco", "kind": "image", "image_id": im["id"]},
            )
        )

    if dataset_meta:
        col, ann = dataset_meta.get("collection", {}), dataset_meta.get("annotation", {})
        docs.append(
            Document(
                "collection:provenance",
                _clip(
                    "Topic: data collection, scraping, images collected accepted and rejected, download failures, "
                    "annotation failures, licences, provenance. Collection and annotation statistics. Source: "
                    f"{sanitize(dataset_meta.get('collection_source', 'unknown'))}. "
                    f"Collection: attempted {col.get('attempted')}, accepted {col.get('accepted')}, rejected by reason "
                    f"{json.dumps(col.get('rejected_by_reason', {}))}, accepted by licence "
                    f"{json.dumps({sanitize(k): v for k, v in col.get('accepted_by_license', {}).items()})}. "
                    f"Annotation: images processed {ann.get('images_processed')}, with detections "
                    f"{ann.get('images_with_detections')}, "
                    f"without detections {ann.get('images_without_detections')}, failed {ann.get('images_failed')}, "
                    f"annotations {ann.get('annotations')} "
                    f"(with masks {ann.get('annotations_with_masks')}), detections dropped by reason "
                    f"{json.dumps(ann.get('detections_dropped', {}))}. "
                    f"{sanitize(dataset_meta.get('note', ''), 200)}"
                ),
                {"source": "collection", "kind": "provenance"},
            )
        )

    if metrics:
        cfg = metrics.get("config", {})
        train_cfg, data_cfg = cfg.get("train", {}), cfg.get("data", {})
        docs.append(
            Document(
                "training:config",
                _clip(
                    "Topic: training configuration, hyperparameters, learning rate, batch size, optimizer, seed, "
                    f"model size, dataset splits. Training run {sanitize(metrics.get('run_id'))}. Task: "
                    f"{sanitize(metrics.get('task'), 200)}. Model {metrics.get('model', {}).get('name')} with "
                    f"{metrics.get('model', {}).get('trainable_parameters')} trainable parameters. Split sizes "
                    f"{json.dumps(metrics.get('dataset', {}).get('split_sizes'))}, "
                    f"split seed {data_cfg.get('split_seed')}. Hyper-parameters: epochs {train_cfg.get('epochs')}, "
                    f"batch size {train_cfg.get('batch_size')}, "
                    f"lr {train_cfg.get('lr')}, weight decay {train_cfg.get('weight_decay')}, warmup epochs "
                    f"{train_cfg.get('warmup_epochs')}, "
                    f"patience {train_cfg.get('patience')}, monitor {train_cfg.get('monitor')}, seed "
                    f"{train_cfg.get('seed')}, image size {data_cfg.get('image_size')}. "
                    f"Environment {json.dumps(metrics.get('environment'))}. "
                    f"{sanitize(metrics.get('metric_notes'), 400)}"
                ),
                {"source": "training", "kind": "config"},
            )
        )
        best, es = metrics.get("best", {}), metrics.get("early_stopping", {})
        docs.append(
            Document(
                "training:summary",
                _clip(
                    "Topic: training summary, best checkpoint, best epoch, early stopping, number of epochs. "
                    f"Training summary: {metrics.get('epochs_completed')} of {metrics.get('epochs_requested')} "
                    "epochs completed. Best checkpoint at epoch "
                    f"{best.get('epoch')} selected by {best.get('monitor')} ({best.get('mode')}) with value "
                    f"{_f(best.get('value'))}. Early stopping triggered: "
                    f"{es.get('triggered')} (patience {es.get('patience')}, stopped epoch {es.get('stopped_epoch')})."
                ),
                {"source": "training", "kind": "summary"},
            )
        )
        for rec in metrics.get("history", []):
            vals = ", ".join(f"{k}={_f(v)}" for k, v in rec.items() if k not in ("epoch", "is_best"))
            if rec["epoch"] == best.get("epoch"):
                tag = " (FINAL best checkpoint, saved as best.pt)"
            elif rec.get("is_best"):
                tag = " (new best so far at this epoch, later superseded)"
            else:
                tag = ""
            docs.append(
                Document(
                    f"training:epoch:{rec['epoch']}",
                    _clip(
                        "Topic: per-epoch training log, train loss, validation loss, learning rate, AP. Training "
                        f"log epoch {rec['epoch']}{tag}: {vals}."
                    ),
                    {"source": "training", "kind": "epoch", "epoch": rec["epoch"]},
                )
            )
        if metrics.get("test"):
            t = metrics["test"]
            docs.append(
                Document(
                    "training:test",
                    _clip(
                        "Topic: held-out test results, test AP and test loss. Held-out test metrics for the best "
                        f"checkpoint: loss {_f(t.get('loss'))}, AP50 {_f(t.get('ap50'))}, AP {_f(t.get('ap'))}, "
                        f"per-class AP50 {json.dumps(t.get('per_class_ap50'))}."
                    ),
                    {"source": "training", "kind": "test"},
                )
            )
    return docs


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    """Maps texts to L2-normalised float32 vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "which",
        "what",
        "how",
        "do",
        "does",
    ]
)
_TOKEN = re.compile(r"[a-z0-9_]+")
# Small domain vocabulary so that paraphrases land in the same hash bucket (lexical embedder only).
_CANON = {
    "bboxes": "box",
    "bbox": "box",
    "boxes": "box",
    "bounding": "box",
    "tiny": "small",
    "smallest": "small",
    "suspiciously": "suspicious",
    "imbalanced": "imbalance",
    "imbalances": "imbalance",
    "unbalanced": "imbalance",
    "overfitting": "overfit",
    "overfitted": "overfit",
    "epochs": "epoch",
    "checkpoints": "checkpoint",
    "classes": "class",
    "categories": "category",
    "annotations": "annotation",
    "images": "image",
}


class HashingEmbedder:
    """Dependency-free signed feature-hashing of word uni/bi-grams with optional IDF weighting.

    Hashing uses BLAKE2b so vectors are stable across processes (unlike Python's ``hash``). Call
    :meth:`fit` on the corpus to learn per-bucket IDF weights; they are persisted with the index.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim, self.name = dim, f"hashing-{dim}"
        self.idf = np.ones(dim, dtype=np.float32)

    @staticmethod
    def _normalise(piece: str) -> str:
        if piece in _CANON:
            return _CANON[piece]
        return piece[:-1] if len(piece) > 4 and piece.endswith("s") and not piece.endswith("ss") else piece

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        base: list[str] = []
        for tok in _TOKEN.findall(text.lower()):
            parts = [p for p in tok.split("_") if p]
            for piece in [tok] + parts if len(parts) > 1 else parts:
                if piece not in _STOP:
                    base.append(cls._normalise(piece))
        return base + [f"{a}_{b}" for a, b in zip(base, base[1:])]

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def fit(self, texts: Sequence[str]) -> None:
        """Learn smoothed IDF weights per hash bucket from a corpus."""
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for b in {self._bucket(t)[0] for t in self._tokens(text)}:
                df[b] += 1
        self.idf = (np.log((len(texts) + 1) / (df + 1)) + 1.0).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in self._tokens(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                idx, sign = self._bucket(tok)
                out[row, idx] += sign * (1.0 + math.log(c)) * self.idf[idx]
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class OpenAICompatEmbedder:
    """``POST {base_url}/embeddings`` (OpenAI, Azure-compatible gateways, Ollama /v1, vLLM, ...)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, batch: int = 64) -> None:
        self.model, self.batch = model, batch
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.name, self.dim = f"openai:{model}", 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": list(texts[i : i + self.batch])},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
        arr = np.asarray(vectors, dtype=np.float32)
        arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        self.dim = arr.shape[1]
        return arr


class SentenceTransformerEmbedder:
    """Local ``sentence-transformers`` model (optional dependency; weights are downloaded on first use)."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.name, self.dim = f"sentence-transformers:{model}", int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def make_embedder(spec: str, hashing_dim: int = 1024) -> Embedder:
    """``hashing`` | ``openai:<model>`` | ``sentence-transformers:<model>``."""
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(hashing_dim)
    if kind == "openai" and arg:
        return OpenAICompatEmbedder(arg)
    if kind == "sentence-transformers" and arg:
        return SentenceTransformerEmbedder(arg)
    raise ValueError(f"unknown embedder '{spec}' (use hashing | openai:<model> | sentence-transformers:<model>)")


# --------------------------------------------------------------------------- #
# Vector stores
# --------------------------------------------------------------------------- #
class VectorStore(Protocol):
    """Minimal persistent vector-store interface."""

    def add(self, docs: Sequence[Document], vectors: np.ndarray) -> None: ...
    def query(self, vector: np.ndarray, k: int, kinds: Sequence[str] | None = None) -> list[Hit]: ...
    def count(self) -> int: ...
    def save(self, directory: Path) -> None: ...


class NumpyVectorStore:
    """Exact cosine-similarity flat index persisted as ``vectors.npy`` + ``documents.jsonl``."""

    def __init__(self, dim: int) -> None:
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.docs: list[Document] = []

    def add(self, docs: Sequence[Document], vectors: np.ndarray) -> None:
        if len(docs) != len(vectors):
            raise ValueError("docs and vectors must have the same length")
        if len(vectors) and vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"vector dim {vectors.shape[1]} != index dim {self.vectors.shape[1]}")
        self.vectors = np.concatenate([self.vectors, vectors.astype(np.float32)], axis=0)
        self.docs.extend(docs)

    def query(self, vector: np.ndarray, k: int, kinds: Sequence[str] | None = None) -> list[Hit]:
        if not self.docs:
            return []
        q = vector.astype(np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.vectors @ q
        order = np.argsort(-scores, kind="stable")
        hits: list[Hit] = []
        for i in order:
            d = self.docs[int(i)]
            if kinds and d.metadata.get("kind") not in kinds:
                continue
            hits.append(Hit(d.chunk_id, float(scores[i]), d.text, d.metadata))
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        return len(self.docs)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "documents.jsonl").open("w", encoding="utf-8") as fh:
            for d in self.docs:
                fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorStore:
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        store = cls(vectors.shape[1])
        docs = [
            Document(**json.loads(line))
            for line in (directory / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        store.add(docs, vectors)
        return store


class ChromaVectorStore:
    """Optional ChromaDB adapter (``pip install chromadb``); embeddings are supplied by us, never by Chroma."""

    def __init__(self, directory: Path, dim: int) -> None:
        import chromadb

        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(directory / "chroma"))
        with contextlib.suppress(Exception):  # collection may not exist yet
            self._client.delete_collection("audit")
        self._col = self._client.create_collection("audit", metadata={"hnsw:space": "cosine"})
        self.dim = dim

    def add(self, docs: Sequence[Document], vectors: np.ndarray) -> None:
        for i in range(0, len(docs), 128):
            part = docs[i : i + 128]
            self._col.add(
                ids=[d.chunk_id for d in part],
                embeddings=vectors[i : i + 128].tolist(),
                documents=[d.text for d in part],
                metadatas=[{**d.metadata} for d in part],
            )

    def query(self, vector: np.ndarray, k: int, kinds: Sequence[str] | None = None) -> list[Hit]:
        where = {"kind": {"$in": list(kinds)}} if kinds else None
        res = self._col.query(
            query_embeddings=[vector.reshape(-1).tolist()], n_results=min(k, max(self._col.count(), 1)), where=where
        )
        return [
            Hit(i, 1.0 - float(dist), doc, meta)
            for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        return int(self._col.count())

    def save(self, directory: Path) -> None:  # persistence is handled by the PersistentClient
        pass


# --------------------------------------------------------------------------- #
# Index lifecycle + retriever
# --------------------------------------------------------------------------- #
@dataclass
class IndexInfo:
    """Metadata persisted next to the index so staleness can be detected."""

    embedder: str
    dim: int
    store: str
    num_documents: int
    input_fingerprint: str
    chunker_version: str
    created_utc: str
    rebuilt: bool = False


class Retriever:
    """Embeds a query and returns the top-k chunks with scores and provenance."""

    def __init__(self, store: VectorStore, embedder: Embedder, info: IndexInfo) -> None:
        self.store, self.embedder, self.info = store, embedder, info

    def retrieve(self, query: str, k: int = 6, kinds: Sequence[str] | None = None) -> list[Hit]:
        """Top-``k`` chunks for ``query`` (optionally restricted to metadata ``kind`` values)."""
        if not query.strip():
            return []
        return self.store.query(self.embedder.embed([query])[0], k, kinds)


def input_fingerprint(paths: Sequence[Path | None], embedder_name: str, store: str) -> str:
    """Hash of input file contents + embedder + store + chunker version."""
    h = hashlib.sha256(f"{CHUNKER_VERSION}|{embedder_name}|{store}".encode())
    for p in paths:
        h.update(file_sha256(p).encode() if p is not None and Path(p).is_file() else b"-")
    return h.hexdigest()


def build_or_load_index(
    ctx: AuditContext,
    index_dir: str | Path,
    *,
    dataset_meta_path: str | Path | None = None,
    embedder_spec: str = "hashing",
    hashing_dim: int = 1024,
    store_kind: str = "numpy",
    force_rebuild: bool = False,
) -> Retriever:
    """Load the persisted index when its inputs are unchanged, otherwise (re)build it."""
    if store_kind not in ("numpy", "chroma"):
        raise ValueError(f"unknown store '{store_kind}' (use numpy | chroma)")
    index_dir = Path(index_dir)
    embedder = make_embedder(embedder_spec, hashing_dim)
    meta_path = Path(dataset_meta_path) if dataset_meta_path else None
    fingerprint = input_fingerprint([ctx.coco_path, ctx.metrics_path, meta_path], embedder.name, store_kind)
    info_path = index_dir / "index_info.json"
    if not force_rebuild and info_path.is_file():
        info = IndexInfo(**json.loads(info_path.read_text(encoding="utf-8")))
        info.rebuilt = False
        if info.input_fingerprint == fingerprint and info.store == store_kind:
            try:
                if isinstance(embedder, HashingEmbedder):
                    embedder.idf = np.load(index_dir / "embedder_idf.npy", allow_pickle=False)
                if store_kind == "numpy":
                    return Retriever(NumpyVectorStore.load(index_dir), embedder, info)
                store = ChromaVectorStore.__new__(ChromaVectorStore)
                import chromadb

                store._client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
                store._col, store.dim = store._client.get_collection("audit"), info.dim
                return Retriever(store, embedder, info)
            except Exception:  # corrupt/partial index -> rebuild below
                pass
    metrics = (
        json.loads(Path(ctx.metrics_path).read_text(encoding="utf-8"))
        if ctx.metrics_path and Path(ctx.metrics_path).is_file()
        else None
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path and meta_path.is_file() else None
    docs = build_documents(ctx, metrics, meta)
    if isinstance(embedder, HashingEmbedder):
        embedder.fit([d.text for d in docs])
    vectors = embedder.embed([d.text for d in docs])
    store: VectorStore = (
        NumpyVectorStore(vectors.shape[1]) if store_kind == "numpy" else ChromaVectorStore(index_dir, vectors.shape[1])
    )
    store.add(docs, vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(index_dir)
    if isinstance(embedder, HashingEmbedder):
        np.save(index_dir / "embedder_idf.npy", embedder.idf)
    info = IndexInfo(
        embedder.name,
        int(vectors.shape[1]),
        store_kind,
        len(docs),
        fingerprint,
        CHUNKER_VERSION,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rebuilt=True,
    )
    save_json_atomic(asdict(info), info_path)
    return Retriever(store, embedder, info)
