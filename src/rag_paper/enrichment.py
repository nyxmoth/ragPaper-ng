from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from rag_paper.config import AppConfig, MetadataEnrichmentConfig, MetadataProviderName
from rag_paper.enrichment_cache import MetadataEnrichmentCache
from rag_paper.logging import logger
from rag_paper.metadata import (
    load_paper_metadata,
    metadata_for_pdf,
    metadata_key_for_pdf,
    save_paper_metadata,
)
from rag_paper.models import PaperChunk
from rag_paper.title_quality import (
    best_title,
    is_trusted_title,
    parse_filename_title_and_year,
    pick_title_line,
)

if TYPE_CHECKING:
    from rag_paper.store import ChromaPaperStore

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
TRAILING_DOI_CHARS = ".,;:)]}>\"'"


@dataclass(frozen=True)
class EnrichmentSummary:
    checked_files: int
    updated_files: int
    skipped_files: int
    failed_files: int
    cleared_files: int = 0
    low_confidence_files: int = 0
    review_path: str = ""


@dataclass(frozen=True)
class EnrichmentVerdict:
    """Outcome of validating a looked-up work against the local document.

    ``accepted`` is the basic gate (title similarity + year); ``confidence``
    grades an accepted match so dubious near-matches can be routed to a human
    review queue instead of silently polluting the metadata. ``high`` is written
    automatically; ``low`` is queued and not written.
    """

    accepted: bool
    confidence: str  # "high" | "low"
    reason: str
    title_similarity: float
    reference_title: str = ""


@dataclass(frozen=True)
class EnrichmentTarget:
    path: Path
    metadata: dict[str, Any]
    text: str = ""


@dataclass(frozen=True)
class MetadataWork:
    source: MetadataProviderName
    doi: str
    title: str
    authors: list[str]
    year: int | None
    container_title: str
    publisher: str
    url: str
    score: float | None
    openalex_id: str = ""
    abstract: str = ""
    referenced_work_ids: list[str] | None = None
    referenced_dois: list[str] | None = None
    related_work_ids: list[str] | None = None
    cited_by_count: int | None = None


CrossRefWork = MetadataWork


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval_seconds = 1.0 / requests_per_second
        self.last_request_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self.interval_seconds - (now - self.last_request_at)
        if delay > 0:
            time.sleep(delay)
        self.last_request_at = time.monotonic()


class CrossRefClient:
    def __init__(self, config: MetadataEnrichmentConfig) -> None:
        self.config = config
        self.limiter = RateLimiter(config.requests_per_second)
        headers = {"User-Agent": config.user_agent}
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_seconds,
            proxy=self._proxy_url(),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def lookup_by_doi(self, doi: str) -> MetadataWork | None:
        self.limiter.wait()
        response = self.client.get(
            f"/works/{quote(doi, safe='')}",
            params=self._common_params(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        message = response.json().get("message")
        if not isinstance(message, dict):
            return None
        return _work_from_crossref_message(message)

    def query_by_title(self, title: str) -> MetadataWork | None:
        title = title.strip()
        if not title:
            return None

        params: dict[str, object] = {
            "query.bibliographic": title[: self.config.max_query_chars],
            "rows": 1,
        }
        params.update(self._common_params())
        self.limiter.wait()
        response = self.client.get("/works", params=params)
        response.raise_for_status()
        message = response.json().get("message")
        if not isinstance(message, dict):
            return None
        items = message.get("items")
        if not isinstance(items, list) or not items:
            return None
        work = _work_from_crossref_message(items[0])
        if work is None:
            return None
        if work.score is not None and work.score < self.config.min_title_score:
            return None
        return work

    def _common_params(self) -> dict[str, str]:
        if self.config.mailto:
            return {"mailto": self.config.mailto}
        return {}

    def _proxy_url(self) -> str | None:
        if self.config.socks5_proxy:
            return self.config.socks5_proxy
        if self.config.https_proxy:
            return self.config.https_proxy
        if self.config.http_proxy:
            return self.config.http_proxy
        return None


class OpenAlexClient:
    def __init__(self, config: MetadataEnrichmentConfig) -> None:
        self.config = config
        self.limiter = RateLimiter(config.requests_per_second)
        headers = {"User-Agent": config.user_agent}
        self.client = httpx.Client(
            base_url=config.openalex_base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_seconds,
            proxy=self._proxy_url(),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def lookup_by_doi(self, doi: str) -> MetadataWork | None:
        self.limiter.wait()
        response = self.client.get(
            "/works",
            params={
                **self._common_params(),
                "filter": f"doi:{_normalize_doi(doi)}",
                "per-page": 1,
            },
        )
        response.raise_for_status()
        return self._first_work(response.json())

    def query_by_title(self, title: str) -> MetadataWork | None:
        title = title.strip()
        if not title:
            return None
        self.limiter.wait()
        response = self.client.get(
            "/works",
            params={
                **self._common_params(),
                "search": title[: self.config.max_query_chars],
                "per-page": 1,
            },
        )
        response.raise_for_status()
        work = self._first_work(response.json())
        if work is None:
            return None
        if work.score is not None and work.score < self.config.min_openalex_score:
            return None
        return work

    def _first_work(self, payload: dict[str, Any]) -> MetadataWork | None:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        return _work_from_openalex_message(results[0])

    def _common_params(self) -> dict[str, str]:
        email = self.config.openalex_email or self.config.mailto
        if email:
            return {"mailto": email}
        return {}

    def _proxy_url(self) -> str | None:
        if self.config.socks5_proxy:
            return self.config.socks5_proxy
        if self.config.https_proxy:
            return self.config.https_proxy
        if self.config.http_proxy:
            return self.config.http_proxy
        return None


def enrich_metadata(
    config: AppConfig,
    *,
    force: bool = False,
    file_path: str | None = None,
    max_files: int | None = None,
    dry_run: bool = False,
) -> EnrichmentSummary:
    if not config.metadata_enrichment.enabled:
        logger.info("metadata.enrichment_disabled")
        return EnrichmentSummary(checked_files=0, updated_files=0, skipped_files=0, failed_files=0)

    metadata_map = load_paper_metadata(config.metadata_path)
    targets = discover_enrichment_targets(config, metadata_map, file_path=file_path)
    if max_files is not None:
        targets = targets[:max_files]

    store: ChromaPaperStore | None = None
    if not dry_run:
        # Build a store so enrichment can index each paper's abstract as a
        # searchable chunk. Imported lazily to keep this module decoupled from the
        # indexing stack. Skipped in dry-run (no Chroma writes).
        from rag_paper.embeddings import build_embedding_provider
        from rag_paper.store import ChromaPaperStore

        store = ChromaPaperStore(
            config.chroma_dir, config.chroma.collection, build_embedding_provider(config)
        )
    try:
        summary = enrich_targets(
            config,
            targets,
            metadata_map=metadata_map,
            force=force,
            store=store,
            dry_run=dry_run,
        )
    finally:
        if store is not None:
            store.client.close()
    if not dry_run:
        save_paper_metadata(config.metadata_path, metadata_map)
    return summary


def enrich_targets(
    config: AppConfig,
    targets: list[EnrichmentTarget],
    *,
    metadata_map: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
    reverify: bool = False,
    store: ChromaPaperStore | None = None,
    dry_run: bool = False,
) -> EnrichmentSummary:
    if not config.metadata_enrichment.enabled:
        return EnrichmentSummary(checked_files=0, updated_files=0, skipped_files=0, failed_files=0)
    if metadata_map is None:
        metadata_map = load_paper_metadata(config.metadata_path)

    clients = _build_provider_clients(config.metadata_enrichment)
    cache = MetadataEnrichmentCache(config.metadata_cache_path)
    checked_files = 0
    updated_files = 0
    skipped_files = 0
    failed_files = 0
    cleared_files = 0
    low_confidence_files = 0

    try:
        try:
            for target in targets:
                checked_files += 1
                existing = metadata_for_pdf(metadata_map, target.path)
                # In normal runs a file that already has a DOI is left alone. In
                # reverify mode we re-check it: a DOI that fails the title gate is
                # dropped and replaced by a fresh title search, repairing bad
                # associations written before verification existed.
                if existing.get("doi") and not force and not reverify:
                    skipped_files += 1
                    logger.info("metadata.skip_existing_doi", file=str(target.path), doi=existing["doi"])
                    continue

                merged = dict(existing)
                merged.update({key: value for key, value in target.metadata.items() if value})

                graded, provider_failed = _lookup_work_with_fallback(
                    clients,
                    config.metadata_enrichment,
                    cache,
                    target,
                    merged,
                    force=force,
                    reverify=reverify,
                )

                if graded is None:
                    if provider_failed:
                        failed_files += 1
                        logger.warning("metadata.all_providers_failed", file=str(target.path))
                    elif (force or reverify) and existing.get("doi"):
                        # The stored DOI was rejected and no replacement was found.
                        # Rather than keep the now-untrusted enrichment data, clear
                        # it so the wrong DOI/title/authors stop polluting search &
                        # citations. (Skipped on provider failure: that may be
                        # transient, so existing data is left intact.)
                        previous_doi = existing["doi"]
                        if not dry_run:
                            key = metadata_key_for_pdf(metadata_map, target.path)
                            metadata_map[key] = _clear_enrichment_fields(existing)
                            save_paper_metadata(config.metadata_path, metadata_map)
                        cleared_files += 1
                        logger.warning(
                            "metadata.cleared_unmatched",
                            file=str(target.path),
                            previous_doi=previous_doi,
                        )
                    else:
                        skipped_files += 1
                        logger.info("metadata.no_provider_match", file=str(target.path))
                    continue

                work, verdict = graded
                if verdict.confidence != "high":
                    # Passed the basic gate but not confidently (weak title or
                    # author disagreement). Do NOT write plausible-but-wrong data;
                    # record it for human review instead.
                    low_confidence_files += 1
                    _record_review(config, target.path, work, verdict)
                    logger.info(
                        "metadata.low_confidence",
                        file=str(target.path),
                        doi=work.doi,
                        confidence=verdict.confidence,
                        reason=verdict.reason,
                        similarity=round(verdict.title_similarity, 3),
                    )
                    continue

                if not dry_run:
                    key = metadata_key_for_pdf(metadata_map, target.path)
                    metadata_map[key] = _merge_work_metadata(merged, work, force=force or reverify)
                    if (
                        reverify
                        and existing.get("doi")
                        and _normalize_doi(existing["doi"]) != work.doi
                    ):
                        logger.warning(
                            "metadata.reverify_corrected",
                            file=str(target.path),
                            previous_doi=existing["doi"],
                            doi=work.doi,
                            title=work.title,
                            source=work.source,
                        )
                    _maybe_index_abstract(store, metadata_map[key], target.path)
                    save_paper_metadata(config.metadata_path, metadata_map)
                updated_files += 1
                logger.info(
                    "metadata.enriched",
                    file=str(target.path),
                    doi=work.doi,
                    title=work.title,
                    source=work.source,
                    dry_run=dry_run,
                )
        except KeyboardInterrupt:
            logger.warning("metadata.interrupted", updated_files=updated_files)
    finally:
        for client in clients.values():
            client.close()
        cache.close()

    if not dry_run:
        save_paper_metadata(config.metadata_path, metadata_map)
    return EnrichmentSummary(
        checked_files=checked_files,
        updated_files=updated_files,
        skipped_files=skipped_files,
        failed_files=failed_files,
        cleared_files=cleared_files,
        low_confidence_files=low_confidence_files,
        review_path=str(config.enrichment_review_path) if low_confidence_files else "",
    )


def target_from_document(path: Path, metadata: dict[str, Any], text: str) -> EnrichmentTarget:
    return EnrichmentTarget(path=path.resolve(), metadata=dict(metadata), text=text)


def extract_doi(value: str) -> str | None:
    match = DOI_RE.search(value)
    if not match:
        return None
    return match.group(0).rstrip(TRAILING_DOI_CHARS)


def infer_title(metadata: dict[str, Any], text: str) -> str:
    title = metadata.get("title")
    if is_trusted_title(title):
        return title.strip()

    return pick_title_line(
        text.splitlines(), file_name=str(metadata.get("file_name") or "")
    )


def discover_enrichment_targets(
    config: AppConfig,
    metadata_map: dict[str, dict[str, Any]],
    *,
    file_path: str | None,
) -> list[EnrichmentTarget]:
    # Imported lazily so that metadata enrichment does not drag the full indexing
    # stack (chromadb / llama-index / pymupdf) into every caller. Mirrors the
    # lazy import of this module from indexer.py.
    from rag_paper.indexer import discover_configured_pdfs
    from rag_paper.pdf import extract_pdf_text

    if file_path:
        pdf_path = Path(file_path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files can be enriched: {pdf_path}")
        candidates = [pdf_path]
    else:
        candidates = [candidate.path for candidate in discover_configured_pdfs(config)]

    targets: list[EnrichmentTarget] = []
    for pdf_path in candidates:
        document = extract_pdf_text(pdf_path, extra_metadata=metadata_for_pdf(metadata_map, pdf_path))
        targets.append(target_from_document(pdf_path, document.metadata, document.text))
    return targets


def _build_provider_clients(
    config: MetadataEnrichmentConfig,
) -> dict[MetadataProviderName, CrossRefClient | OpenAlexClient]:
    clients: dict[MetadataProviderName, CrossRefClient | OpenAlexClient] = {}
    for provider in _provider_order(config):
        if provider == "crossref":
            clients[provider] = CrossRefClient(config)
        elif provider == "openalex":
            clients[provider] = OpenAlexClient(config)
    return clients


def _provider_order(config: MetadataEnrichmentConfig) -> list[MetadataProviderName]:
    providers = config.providers or [config.provider]
    ordered: list[MetadataProviderName] = []
    for provider in providers:
        if provider not in ordered:
            ordered.append(provider)
    if not ordered:
        ordered.append(config.provider)
    return ordered


def _lookup_work_with_fallback(
    clients: dict[MetadataProviderName, CrossRefClient | OpenAlexClient],
    config: MetadataEnrichmentConfig,
    cache: MetadataEnrichmentCache,
    target: EnrichmentTarget,
    metadata: dict[str, Any],
    *,
    force: bool,
    reverify: bool = False,
) -> tuple[tuple[MetadataWork, EnrichmentVerdict] | None, bool]:
    provider_failed = False
    for provider in _provider_order(config):
        client = clients.get(provider)
        if client is None:
            continue
        try:
            graded = _lookup_work_graded(
                client, provider, cache, target, metadata, force=force, reverify=reverify
            )
        except httpx.HTTPError as exc:
            provider_failed = True
            logger.warning(
                "metadata.provider_failed",
                provider=provider,
                file=str(target.path),
                error=str(exc),
            )
            continue
        if graded is not None:
            return graded, provider_failed
        logger.info("metadata.provider_no_match", provider=provider, file=str(target.path))
    return None, provider_failed


def _lookup_work_graded(
    client: CrossRefClient | OpenAlexClient,
    provider: MetadataProviderName,
    cache: MetadataEnrichmentCache,
    target: EnrichmentTarget,
    metadata: dict[str, Any],
    *,
    force: bool,
    reverify: bool = False,
) -> tuple[MetadataWork, EnrichmentVerdict] | None:
    config = client.config
    min_similarity = config.min_title_similarity
    min_similarity_high = config.min_title_similarity_high
    reference_authors = _reference_authors(metadata)

    file_name = str(metadata.get("file_name") or target.path.name)
    filename_title, filename_year = parse_filename_title_and_year(file_name)
    if force or reverify:
        # We are re-checking data that may already be wrong. The filename
        # ("Authors - Year - Title.pdf" for Zotero libraries) is the most reliable
        # reference — body-text inference can pick boilerplate like "Published as
        # a conference paper at ICLR ...". Prefer the filename title/year and fall
        # back to the text-derived title only when the filename doesn't parse.
        reference_title = filename_title or pick_title_line(
            target.text.splitlines(), file_name=file_name
        )
    else:
        reference_title = infer_title(metadata, target.text)
    # Cross-check the year whenever the filename provides one (Zotero naming is
    # reliable) — not only in force/reverify mode. This rejects same-title
    # different-year near-namesakes in normal runs too. None when no filename year.
    expected_year = filename_year

    # A DOI mined out of the PDF body is often a *citation*, not this paper's own
    # DOI. Treat user-supplied DOIs (metadata.doi/url/external_url) as trusted;
    # treat anything pulled from full text as speculative and require it to pass
    # the title-similarity gate. In force/reverify mode the already-stored DOI is
    # also treated as speculative so it actually gets re-derived and re-checked.
    trusted_doi, speculative_doi = _first_doi(
        metadata, target.text, trust_existing=not (force or reverify)
    )
    doi_candidates: list[tuple[str, bool]] = []
    if trusted_doi:
        doi_candidates.append((trusted_doi, True))
    elif speculative_doi:
        doi_candidates.append((speculative_doi, False))

    for possible_doi, is_trusted in doi_candidates:
        cached = None if force else _cache_get_work(cache, provider, "doi", possible_doi)
        work = cached if cached is not None else client.lookup_by_doi(possible_doi)
        if work is None:
            continue
        verdict = _evaluate_work(
            work,
            reference_title,
            is_trusted=is_trusted,
            min_similarity=min_similarity,
            min_similarity_high=min_similarity_high,
            expected_year=expected_year,
            reference_authors=reference_authors,
        )
        if verdict.accepted:
            if cached is None:
                cache.set(provider, "doi", possible_doi, asdict(work))
            return work, verdict
        title_matched = _work_matches_title(work, reference_title, min_similarity)
        logger.warning(
            "metadata.doi_year_mismatch" if title_matched else "metadata.doi_title_mismatch",
            provider=provider,
            file=str(target.path),
            doi=possible_doi,
            trusted=is_trusted,
            returned_title=work.title,
            returned_year=work.year,
            reference_title=reference_title,
            expected_year=expected_year,
        )

    if reference_title:
        title_key = _normalize_cache_text(reference_title)
        cached = None if force else _cache_get_work(cache, provider, "title", title_key)
        work = cached if cached is not None else client.query_by_title(reference_title)
        if work is None:
            return None
        verdict = _evaluate_work(
            work,
            reference_title,
            is_trusted=False,
            min_similarity=min_similarity,
            min_similarity_high=min_similarity_high,
            expected_year=expected_year,
            reference_authors=reference_authors,
        )
        if verdict.accepted:
            if cached is None:
                cache.set(provider, "title", title_key, asdict(work))
            return work, verdict
        title_matched = _work_matches_title(work, reference_title, min_similarity)
        logger.info(
            "metadata.year_mismatch" if title_matched else "metadata.title_similarity_below_threshold",
            provider=provider,
            file=str(target.path),
            similarity=round(_title_similarity(reference_title, work.title), 3),
            threshold=min_similarity,
            returned_year=work.year,
            expected_year=expected_year,
        )
    return None


def _lookup_work(
    client: CrossRefClient | OpenAlexClient,
    provider: MetadataProviderName,
    cache: MetadataEnrichmentCache,
    target: EnrichmentTarget,
    metadata: dict[str, Any],
    *,
    force: bool,
    reverify: bool = False,
) -> MetadataWork | None:
    """Boolean-view wrapper over ``_lookup_work_graded`` for callers/tests that
    only need the matched work (accept/reject), not the confidence grade."""
    graded = _lookup_work_graded(
        client, provider, cache, target, metadata, force=force, reverify=reverify
    )
    return graded[0] if graded is not None else None


def _cache_get_work(
    cache: MetadataEnrichmentCache,
    provider: MetadataProviderName,
    query_type: str,
    query_value: str,
) -> MetadataWork | None:
    payload = cache.get(provider, query_type, query_value)
    if payload is None:
        return None
    return MetadataWork(**payload)


def _normalize_cache_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _abstract_chunk_id(source_path: str) -> str:
    # Content-independent: a paper has at most one abstract slot, so re-enrichment
    # upserts the same record instead of orphaning the previous one when the
    # abstract text changes.
    return hashlib.sha256(f"{source_path}:abstract".encode("utf-8")).hexdigest()[:24]


def _build_abstract_chunk(
    metadata: dict[str, Any], pdf_path: Path
) -> PaperChunk | None:
    abstract = str(metadata.get("abstract") or "").strip()
    if not abstract:
        return None
    source_path = str(pdf_path.resolve())
    title = str(metadata.get("title") or metadata.get("file_name") or "").strip()
    text = f"{title}\n\n{abstract}".strip() if title else abstract
    chunk_metadata = dict(metadata)
    chunk_metadata["source_path"] = source_path
    chunk_metadata["chunk_type"] = "abstract"
    return PaperChunk(id=_abstract_chunk_id(source_path), text=text, metadata=chunk_metadata)


def _maybe_index_abstract(
    store: ChromaPaperStore | None, metadata: dict[str, Any], pdf_path: Path
) -> None:
    """Index a paper's abstract as its own searchable chunk when a store is wired.

    Best-effort: a failure to embed/upsert the abstract must not break enrichment.
    """
    if store is None:
        return
    chunk = _build_abstract_chunk(metadata, pdf_path)
    if chunk is None:
        return
    try:
        store.upsert_chunks([chunk])
    except Exception as exc:  # noqa: BLE001 - abstract indexing is optional
        logger.warning("metadata.abstract_chunk_failed", file=str(pdf_path), error=str(exc))


def _record_review(
    config: AppConfig, pdf_path: Path, work: MetadataWork, verdict: EnrichmentVerdict
) -> None:
    """Append a low-confidence match to the review queue (JSONL) for human action.

    These are matches that cleared the basic gate but not confidently (weak title
    similarity or author disagreement). Writing them automatically risks a wrong
    cross-field DOI; queueing keeps the library clean while surfacing the
    candidate with enough context to accept or reject by hand.
    """
    path = config.enrichment_review_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": str(pdf_path),
        "doi": work.doi,
        "provider": work.source,
        "returned_title": work.title,
        "returned_authors": work.authors,
        "returned_year": work.year,
        "reference_title": verdict.reference_title,
        "title_similarity": round(verdict.title_similarity, 4),
        "confidence": verdict.confidence,
        "reason": verdict.reason,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_title_for_compare(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (value or "").lower()).split())


def _title_similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity score between two titles.

    Takes the better of a character-sequence ratio and a token Jaccard overlap,
    so it tolerates rewording, subtitle stripping, and CJK titles (``\\w`` is
    Unicode-aware for ``str``).
    """
    normalized_a = _normalize_title_for_compare(a)
    normalized_b = _normalize_title_for_compare(b)
    if not normalized_a or not normalized_b:
        return 0.0
    ratio = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    tokens_a = set(normalized_a.split())
    tokens_b = set(normalized_b.split())
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        ratio = max(ratio, jaccard)
    return ratio


# When re-checking with a filename-derived year, allow this many years of slack
# (preprint vs published, etc.) before treating a year mismatch as a wrong paper.
_YEAR_TOLERANCE = 2


def _work_matches_title(
    work: MetadataWork, inferred_title: str, min_similarity: float
) -> bool:
    if not inferred_title or not work.title:
        return False
    return _title_similarity(inferred_title, work.title) >= min_similarity


def _year_matches(work: MetadataWork, expected_year: int | None) -> bool:
    """Cross-check the work's year against the filename-derived year.

    Used to reject near-namesakes (e.g. BERT vs "Spectrum-BERT") whose titles are
    intentionally similar but which were published years apart. Missing year on
    either side means we cannot judge, so we do not reject on year alone.
    """
    if expected_year is None or work.year is None:
        return True
    return abs(work.year - expected_year) <= _YEAR_TOLERANCE


def _basic_gate(
    work: MetadataWork,
    inferred_title: str,
    *,
    is_trusted: bool,
    min_similarity: float,
    expected_year: int | None = None,
) -> tuple[bool, str]:
    """Hard acceptance gate: trusted DOI passes; otherwise title similarity AND,
    when a filename year is available, year agreement. Returns (accepted, reason)."""
    if is_trusted:
        return True, "trusted_doi"
    if not _work_matches_title(work, inferred_title, min_similarity):
        return False, "title_similarity_below_threshold"
    if not _year_matches(work, expected_year):
        return False, "year_mismatch"
    return True, "title_and_year_match"


def _accept_work(
    work: MetadataWork,
    inferred_title: str,
    *,
    is_trusted: bool,
    min_similarity: float,
    expected_year: int | None = None,
) -> bool:
    """Boolean view of the basic gate (kept for callers/tests that only need
    accept/reject). Use ``_evaluate_work`` when the confidence grade matters."""
    return _basic_gate(
        work,
        inferred_title,
        is_trusted=is_trusted,
        min_similarity=min_similarity,
        expected_year=expected_year,
    )[0]


def _reference_authors(metadata: dict[str, Any]) -> list[str]:
    """Authors we can cross-check against, from user/PDF metadata. Best-effort:
    often empty in normal mode, in which case author agreement cannot be judged."""
    authors = metadata.get("authors")
    if isinstance(authors, list):
        return [str(item).strip() for item in authors if str(item).strip()]
    if isinstance(authors, str) and authors.strip():
        # Chroma flattens lists to ", "-joined strings; tolerate that.
        return [part.strip() for part in authors.split(",") if part.strip()]
    return []


def _author_surnames(authors: list[str]) -> set[str]:
    surnames: set[str] = set()
    for author in authors:
        tokens = re.sub(r"[^\w\s]", " ", author.lower()).split()
        if tokens:
            surnames.add(tokens[-1])
    return surnames


def _authors_overlap(reference: list[str], work_authors: list[str]) -> bool | None:
    """True/False when both sides have authors and a surname can be compared;
    None when either side is empty (cannot judge — do not penalize)."""
    if not reference or not work_authors:
        return None
    return bool(_author_surnames(reference) & _author_surnames(work_authors))


def _evaluate_work(
    work: MetadataWork,
    inferred_title: str,
    *,
    is_trusted: bool,
    min_similarity: float,
    min_similarity_high: float,
    expected_year: int | None = None,
    reference_authors: list[str] | None = None,
) -> EnrichmentVerdict:
    """Grade a looked-up work: hard gate first, then confidence.

    - rejected (title too dissimilar, or year mismatch) -> accepted=False
    - trusted DOI -> high
    - accepted + strong title similarity + (author agrees or unknowable) -> high
    - accepted but weak title, or author disagreement when comparable -> low

    Low-confidence matches are NOT written; they go to the review queue so a
    human decides, rather than a plausible-but-wrong cross-field DOI polluting
    search and citations.
    """
    title_similarity = (
        _title_similarity(inferred_title, work.title)
        if inferred_title and work.title
        else 0.0
    )
    if is_trusted:
        return EnrichmentVerdict(
            True, "high", "trusted_doi", title_similarity, inferred_title
        )

    accepted, reason = _basic_gate(
        work,
        inferred_title,
        is_trusted=is_trusted,
        min_similarity=min_similarity,
        expected_year=expected_year,
    )
    if not accepted:
        return EnrichmentVerdict(False, "low", reason, title_similarity, inferred_title)

    title_strong = title_similarity >= min_similarity_high
    author_overlap = _authors_overlap(reference_authors or [], work.authors)
    detail = "strong_title" if title_strong else "weak_title"
    if author_overlap is True:
        detail += "+author_match"
    elif author_overlap is False:
        detail += "+author_mismatch"
    confidence = "high" if (title_strong and author_overlap is not False) else "low"
    return EnrichmentVerdict(True, confidence, detail, title_similarity, inferred_title)


def _first_doi(
    metadata: dict[str, Any],
    text: str,
    *,
    trust_existing: bool = True,
) -> tuple[str | None, str | None]:
    """Return ``(trusted_doi, speculative_doi)`` for a document.

    A DOI is *trusted* when the user supplied it explicitly via ``metadata.doi``
    or a ``url``/``external_url`` field. A DOI is *speculative* when it is mined
    out of the PDF full text (where it is frequently a citation), or — in
    ``reverify`` mode (``trust_existing=False``) — when it is an already-stored
    DOI being re-checked against the title rather than taken on faith.

    Speculative DOIs are only produced when no trusted DOI is available, and
    callers must verify them against the inferred title before use.
    """
    existing: str | None = None

    doi = metadata.get("doi")
    if isinstance(doi, str) and doi.strip():
        existing = _normalize_doi(doi)

    if not existing:
        for key in ("url", "external_url"):
            value = metadata.get(key)
            if isinstance(value, str):
                extracted = extract_doi(value)
                if extracted:
                    existing = _normalize_doi(extracted)
                    break

    trusted: str | None = None
    speculative: str | None = None

    if existing:
        if trust_existing:
            trusted = existing
        else:
            # reverify mode: re-check an already-stored DOI against the title
            # instead of trusting it blindly.
            speculative = existing
    elif trust_existing:
        # Only mine the body when there is no stored/url DOI at all, and never in
        # reverify mode — a stored DOI that fails verification should fall back to
        # a title search, not to an even-riskier text-mined DOI.
        extracted = extract_doi(text[:4000])
        if extracted:
            speculative = _normalize_doi(extracted)

    return trusted, speculative


# Fields written by `_merge_work_metadata` (the enrichment output). On
# clear-on-reject these are dropped; user/file fields (file_name, source_path,
# tags, venue, ...) are preserved.
_ENRICHMENT_FIELDS = (
    "doi",
    "title",
    "authors",
    "year",
    "container_title",
    "publisher",
    "url",
    "openalex_id",
    "abstract",
    "referenced_dois",
    "referenced_work_ids",
    "related_work_ids",
    "cited_by_count",
    "metadata_source",
    "metadata_enriched_at",
)


def _clear_enrichment_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Strip enrichment-output fields, keeping user/file-derived fields."""
    return {
        key: value
        for key, value in metadata.items()
        if key not in _ENRICHMENT_FIELDS and value not in ("", None, [])
    }


def _merge_work_metadata(
    metadata: dict[str, Any],
    work: CrossRefWork,
    *,
    force: bool,
) -> dict[str, Any]:
    enriched = dict(metadata)
    if "title" in enriched and not is_trusted_title(enriched.get("title")):
        enriched.pop("title", None)
    _set_metadata(enriched, "doi", work.doi, force=force)
    _set_title_metadata(enriched, work.title, force=force)
    if not is_trusted_title(enriched.get("title")) and metadata.get("file_name"):
        fallback_title = best_title(file_name=str(metadata.get("file_name")))
        if fallback_title:
            enriched["title"] = fallback_title
    _set_metadata(enriched, "authors", work.authors, force=force)
    _set_metadata(enriched, "year", work.year, force=force)
    _set_metadata(enriched, "container_title", work.container_title, force=force)
    _set_metadata(enriched, "publisher", work.publisher, force=force)
    _set_metadata(enriched, "url", work.url, force=force)
    _set_metadata(enriched, "openalex_id", work.openalex_id, force=force)
    _set_metadata(enriched, "abstract", work.abstract, force=force)
    _set_metadata(enriched, "referenced_work_ids", work.referenced_work_ids or [], force=force)
    _set_metadata(enriched, "referenced_dois", work.referenced_dois or [], force=force)
    _set_metadata(enriched, "related_work_ids", work.related_work_ids or [], force=force)
    _set_metadata(enriched, "cited_by_count", work.cited_by_count, force=force)
    enriched["metadata_source"] = work.source
    enriched["metadata_enriched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {key: value for key, value in enriched.items() if value not in ("", [], None)}


def _set_metadata(metadata: dict[str, Any], key: str, value: Any, *, force: bool) -> None:
    if value in ("", [], None):
        return
    if force or not metadata.get(key):
        metadata[key] = value


def _set_title_metadata(metadata: dict[str, Any], value: Any, *, force: bool) -> None:
    if not is_trusted_title(value):
        return
    _set_metadata(metadata, "title", value, force=force)


def _work_from_crossref_message(message: dict[str, Any]) -> CrossRefWork | None:
    doi = _first_string(message.get("DOI"))
    if not doi:
        return None

    title = _first_string(message.get("title"))
    authors = _authors_from_crossref(message.get("author"))
    year = _year_from_crossref(message)
    container_title = _first_string(message.get("container-title"))
    publisher = _first_string(message.get("publisher"))
    url = _first_string(message.get("URL"))
    score = message.get("score")
    referenced_dois = _referenced_dois_from_crossref(message.get("reference"))

    return CrossRefWork(
        source="crossref",
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        container_title=container_title,
        publisher=publisher,
        url=url,
        score=float(score) if isinstance(score, (int, float)) else None,
        referenced_dois=referenced_dois,
    )


def _work_from_openalex_message(message: dict[str, Any]) -> MetadataWork | None:
    doi = _normalize_doi(_first_string(message.get("doi")))
    if not doi:
        return None

    primary_location = message.get("primary_location")
    source = primary_location.get("source") if isinstance(primary_location, dict) else None
    container_title = ""
    if isinstance(source, dict):
        container_title = _first_string(source.get("display_name"))

    score = message.get("relevance_score")
    cited_by_count = message.get("cited_by_count")
    return MetadataWork(
        source="openalex",
        doi=doi,
        title=_first_string(message.get("display_name")),
        authors=_authors_from_openalex(message.get("authorships")),
        year=_int_or_none(message.get("publication_year")),
        container_title=container_title,
        publisher=_first_string(message.get("publisher")),
        url=_first_string(message.get("doi")) or _first_string(message.get("id")),
        score=float(score) if isinstance(score, (int, float)) else None,
        openalex_id=_first_string(message.get("id")),
        abstract=_abstract_from_openalex(message.get("abstract_inverted_index")),
        referenced_work_ids=_strings_from_list(message.get("referenced_works")),
        related_work_ids=_strings_from_list(message.get("related_works")),
        cited_by_count=cited_by_count if isinstance(cited_by_count, int) else None,
    )


def _first_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _authors_from_crossref(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        given = _first_string(item.get("given"))
        family = _first_string(item.get("family"))
        name = " ".join(part for part in [given, family] if part)
        if name:
            authors.append(name)
    return authors


def _authors_from_openalex(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if isinstance(author, dict):
            name = _first_string(author.get("display_name"))
            if name:
                authors.append(name)
    return authors


def _referenced_dois_from_crossref(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    dois: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        doi = _normalize_doi(_first_string(item.get("DOI")) or _first_string(item.get("doi")))
        if doi:
            dois.append(doi)
    return sorted(set(dois))


def _abstract_from_openalex(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positions: dict[int, str] = {}
    for word, raw_indexes in value.items():
        if not isinstance(word, str) or not isinstance(raw_indexes, list):
            continue
        for index in raw_indexes:
            if isinstance(index, int):
                positions[index] = word
    return " ".join(positions[index] for index in sorted(positions))


def _strings_from_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _normalize_doi(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:", "", value, flags=re.IGNORECASE)
    return value.rstrip(TRAILING_DOI_CHARS)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _year_from_crossref(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
            and isinstance(date_parts[0][0], int)
        ):
            return date_parts[0][0]
    return None
