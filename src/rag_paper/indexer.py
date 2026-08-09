from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from traceback import format_exception_only
from typing import Any

import chromadb

from rag_paper.chunking import chunk_document
from rag_paper.config import AppConfig
from rag_paper.embeddings import build_embedding_provider
from rag_paper.logging import logger
from rag_paper.manifest import IndexManifest, file_sha256
from rag_paper.metadata import load_paper_metadata, metadata_for_pdf
from rag_paper.pdf import extract_pdf_text
from rag_paper.store import ChromaPaperStore, normalize_metadata


@dataclass(frozen=True)
class IndexSummary:
    indexed_files: int
    skipped_files: int
    chunks: int
    enriched_metadata_files: int = 0
    failed_metadata_files: int = 0
    duplicate_files: int = 0
    # Per-paper identifiers (file_name/title/doi/year/chunks) for the papers this
    # run actually indexed, so an AI assistant reading the index status can look
    # up the affected papers. Capped at _STATUS_PAPERS_CAP; updated_papers_total
    # holds the true count when truncated.
    updated_papers: list[dict[str, Any]] = field(default_factory=list)
    updated_papers_total: int = 0


@dataclass(frozen=True)
class RefreshSummary:
    papers: int
    chunks_updated: int
    skipped_papers: int
    updated_papers: list[dict[str, Any]] = field(default_factory=list)
    updated_papers_total: int = 0


# Cap the per-paper list stored in each index-status record so a large --force
# re-index cannot bloat the status history. The true count is kept separately.
_STATUS_PAPERS_CAP = 50


def _index_status_paper_entry(
    pdf_path: Path, chunks: int, metadata_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Build a compact identifier record for a paper touched by an update.

    Includes the lookup keys an AI assistant needs (file_name / title / doi /
    year) plus the chunk count, drawn from the (possibly enriched) metadata.
    """
    metadata = metadata_for_pdf(metadata_map, pdf_path)
    return {
        "file_name": pdf_path.name,
        "title": str(metadata.get("title") or ""),
        "doi": str(metadata.get("doi") or ""),
        "year": metadata.get("year"),
        "chunks": chunks,
    }


@dataclass(frozen=True)
class PdfCandidate:
    path: Path
    root_path: Path
    tags: tuple[str, ...]


@dataclass(frozen=True)
class IndexTarget:
    candidate: PdfCandidate
    digest: str | None


@dataclass(frozen=True)
class RootPlanSummary:
    root_path: str
    files: int


@dataclass(frozen=True)
class IndexPlan:
    total_files: int
    roots: tuple[RootPlanSummary, ...]


@dataclass(frozen=True)
class SkipMarkerHit:
    directory: Path
    marker_path: Path
    marker: str
    root_path: Path | None = None


def discover_pdfs(
    pdf_dir: Path,
    skip_marker_file: str = "",
    *,
    skip_marker_hits: list[SkipMarkerHit] | None = None,
) -> list[Path]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    marker = skip_marker_file.strip()
    pdfs: list[Path] = []

    def walk(directory: Path) -> None:
        if marker and (directory / marker).exists():
            marker_path = directory / marker
            if skip_marker_hits is not None:
                skip_marker_hits.append(
                    SkipMarkerHit(
                        directory=directory,
                        marker_path=marker_path,
                        marker=marker,
                        root_path=pdf_dir,
                    )
                )
            logger.info("index.skip_marked_dir", dir=str(directory), marker=marker)
            return

        for path in sorted(directory.iterdir()):
            if path.is_dir():
                walk(path)
            elif path.is_file() and path.suffix.lower() == ".pdf":
                pdfs.append(path)

    walk(pdf_dir)
    return pdfs


def discover_configured_pdfs(
    config: AppConfig,
    *,
    skip_marker_hits: list[SkipMarkerHit] | None = None,
) -> list[PdfCandidate]:
    pdfs: dict[str, PdfCandidate] = {}
    roots = [
        (root, Path(root_path).expanduser().resolve())
        for root in config.papers
        for root_path in root.root_paths
    ]
    root_count = len(roots)
    for root in config.papers:
        for raw_root_path in root.root_paths:
            root_path = Path(raw_root_path).expanduser().resolve()
            if not root_path.exists() and root_count > 1:
                logger.warning("index.skip_missing_paper_root", root=str(root_path))
                continue
            for pdf_path in discover_pdfs(
                root_path,
                skip_marker_file=root.skip_marker_file,
                skip_marker_hits=skip_marker_hits,
            ):
                resolved = pdf_path.resolve()
                key = str(resolved)
                if key not in pdfs:
                    pdfs[key] = PdfCandidate(
                        path=resolved,
                        root_path=root_path,
                        tags=tuple(root.tags),
                    )
    return list(pdfs.values())


def find_skip_markers_for_file(config: AppConfig, pdf_path: Path) -> list[SkipMarkerHit]:
    pdf_path = pdf_path.expanduser().resolve()
    hits: list[SkipMarkerHit] = []
    seen: set[tuple[str, str]] = set()

    for root in config.papers:
        marker = root.skip_marker_file.strip()
        if not marker:
            continue
        configured_roots = [
            Path(root_path).expanduser().resolve() for root_path in root.root_paths
        ]
        matching_roots = [
            root_path for root_path in configured_roots if _is_relative_to(pdf_path, root_path)
        ]
        search_roots = matching_roots or [None]
        for search_root in search_roots:
            for directory in _candidate_parent_dirs(pdf_path.parent, search_root):
                marker_path = directory / marker
                if not marker_path.exists():
                    continue
                key = (str(marker_path), marker)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    SkipMarkerHit(
                        directory=directory,
                        marker_path=marker_path,
                        marker=marker,
                        root_path=search_root,
                    )
                )
    return hits


def _candidate_parent_dirs(start: Path, stop_at: Path | None) -> list[Path]:
    directories: list[Path] = []
    current = start
    while True:
        directories.append(current)
        if stop_at is not None and current == stop_at:
            break
        if current.parent == current:
            break
        current = current.parent
    return directories


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _build_index_plan(targets: list[IndexTarget]) -> IndexPlan:
    root_counts: dict[str, int] = defaultdict(int)
    for target in targets:
        root_counts[str(target.candidate.root_path)] += 1
    roots = tuple(
        RootPlanSummary(root_path=root_path, files=files)
        for root_path, files in root_counts.items()
    )
    return IndexPlan(total_files=len(targets), roots=roots)


def _prepare_index_targets(
    candidates: list[PdfCandidate],
    manifest: IndexManifest,
    *,
    force: bool,
    only_new: bool,
    max_files: int | None,
) -> tuple[list[IndexTarget], int]:
    targets: list[IndexTarget] = []
    skipped_files = 0

    for index, candidate in enumerate(candidates):
        if max_files is not None and len(targets) >= max_files:
            skipped_files += len(candidates) - index
            break

        if only_new and manifest.get(candidate.path):
            skipped_files += 1
            logger.info("index.skip_existing", file=str(candidate.path))
            continue

        digest: str | None = None
        if not force:
            if manifest.has_same_stat(candidate.path):
                skipped_files += 1
                logger.info("index.skip_current_stat", file=str(candidate.path))
                continue

            digest = file_sha256(candidate.path)
            if manifest.is_current(candidate.path, digest):
                skipped_files += 1
                logger.info("index.skip_current_sha256", file=str(candidate.path))
                continue

        targets.append(IndexTarget(candidate=candidate, digest=digest))

    return targets, skipped_files


def load_failed_index_paths(path: Path) -> list[Path]:
    if not path.exists():
        return []
    paths: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_path = item.get("file")
            if isinstance(file_path, str):
                paths.append(Path(file_path).expanduser().resolve())
    return paths


def record_index_failure(path: Path, pdf_path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": str(pdf_path),
        "error_type": type(exc).__name__,
        "error": "".join(format_exception_only(type(exc), exc)).strip(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clear_index_failure(path: Path, pdf_path: Path) -> None:
    if not path.exists():
        return
    resolved = str(pdf_path.expanduser().resolve())
    remaining: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                remaining.append(line)
                continue
            if item.get("file") != resolved:
                remaining.append(line)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(remaining)


def run_indexing(
    config: AppConfig,
    *,
    force: bool = False,
    file_path: str | None = None,
    only_new: bool = False,
    retry_failed: bool = False,
    max_files: int | None = None,
    confirm_plan: Callable[[IndexPlan], None] | None = None,
    confirm_skip_markers: Callable[[tuple[SkipMarkerHit, ...]], None] | None = None,
) -> IndexSummary:
    manifest = IndexManifest(config.chroma_dir / "index_manifest.json")
    metadata_map = load_paper_metadata(config.metadata_path)

    if retry_failed:
        candidates = [
            PdfCandidate(path=path, root_path=path.parent, tags=())
            for path in load_failed_index_paths(config.index_failed_path)
            if path.exists() and path.suffix.lower() == ".pdf"
        ]
        skip_marker_hits = []
    elif file_path:
        pdf_path = Path(file_path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files can be indexed: {pdf_path}")
        skip_marker_hits = find_skip_markers_for_file(config, pdf_path)
        candidates = [PdfCandidate(path=pdf_path, root_path=pdf_path.parent, tags=())]
    else:
        skip_marker_hits = []
        candidates = discover_configured_pdfs(config, skip_marker_hits=skip_marker_hits)

    if skip_marker_hits and confirm_skip_markers is not None:
        confirm_skip_markers(tuple(skip_marker_hits))

    embeddings = None
    duplicate_files = 0
    if config.dedup.enabled:
        from rag_paper.dedup import filter_duplicate_candidates

        if config.dedup.semantic_enabled:
            embeddings = build_embedding_provider(config)
        original_candidate_count = len(candidates)
        candidates, duplicate_pairs = filter_duplicate_candidates(
            config,
            candidates,
            metadata_map=metadata_map,
            embeddings=embeddings,
        )
        duplicate_files = len({pair.duplicate_path for pair in duplicate_pairs})
        if config.dedup.action == "skip":
            duplicate_files = original_candidate_count - len(candidates)

    effective_max_files = config.indexing.max_files if max_files is None else max_files
    targets, skipped_files = _prepare_index_targets(
        candidates,
        manifest,
        force=force,
        only_new=only_new,
        max_files=effective_max_files,
    )
    if config.dedup.enabled and config.dedup.action == "skip":
        skipped_files += duplicate_files
    plan = _build_index_plan(targets)
    if confirm_plan is not None:
        confirm_plan(plan)
    if not targets:
        return IndexSummary(
            indexed_files=0,
            skipped_files=skipped_files,
            chunks=0,
            duplicate_files=duplicate_files,
        )

    if embeddings is None:
        embeddings = build_embedding_provider(config)
    store = ChromaPaperStore(config.chroma_dir, config.chroma.collection, embeddings)

    indexed_files = 0
    indexed_chunks = 0
    enriched_metadata_files = 0
    failed_metadata_files = 0
    post_index_enrichment_targets = []
    indexed_paper_tuples: list[tuple[Path, int]] = []

    try:
        for target in targets:
            pdf_path = target.candidate.path
            try:
                old_chunk_ids = manifest.remove(pdf_path)
                if old_chunk_ids:
                    store.delete_chunk_ids(old_chunk_ids)

                extra_metadata = metadata_for_pdf(metadata_map, pdf_path)
                if target.candidate.tags and not extra_metadata.get("tags"):
                    extra_metadata["tags"] = list(target.candidate.tags)

                document = extract_pdf_text(pdf_path, extra_metadata=extra_metadata)
                if not document.text:
                    skipped_files += 1
                    logger.warning("index.empty_pdf_text", file=str(pdf_path))
                    continue

                chunks = chunk_document(
                    document,
                    chunk_size=config.indexing.chunk_size,
                    chunk_overlap=config.indexing.chunk_overlap,
                )
                store.upsert_chunks(chunks)
                digest = target.digest or file_sha256(pdf_path)
                manifest.update(pdf_path, digest, [chunk.id for chunk in chunks])
                manifest.save()
                clear_index_failure(config.index_failed_path, pdf_path)
            except Exception as exc:
                skipped_files += 1
                record_index_failure(config.index_failed_path, pdf_path, exc)
                logger.warning("index.file_failed", file=str(pdf_path), error=str(exc))
                continue

            indexed_files += 1
            indexed_chunks += len(chunks)
            indexed_paper_tuples.append((pdf_path, len(chunks)))
            if config.metadata_enrichment.enabled:
                from rag_paper.enrichment import enrich_targets, target_from_document

                enrichment_target = target_from_document(pdf_path, document.metadata, document.text)
                if config.metadata_enrichment.timing == "per_file":
                    enrichment_summary = enrich_targets(
                        config,
                        [enrichment_target],
                        metadata_map=metadata_map,
                        store=store,
                    )
                    enriched_metadata_files += enrichment_summary.updated_files
                    failed_metadata_files += enrichment_summary.failed_files
                    if enrichment_summary.updated_files:
                        store.update_chunks_metadata(
                            chunks,
                            metadata_for_pdf(metadata_map, pdf_path),
                        )
                elif config.metadata_enrichment.timing == "after_index":
                    post_index_enrichment_targets.append((enrichment_target, pdf_path, chunks))

            logger.info(
                "index.file_complete",
                file=str(pdf_path),
                chunks=len(chunks),
                force=force,
                only_new=only_new,
            )
    except KeyboardInterrupt:
        manifest.save()
        logger.warning("index.interrupted", indexed_files=indexed_files, chunks=indexed_chunks)

    if config.metadata_enrichment.enabled and post_index_enrichment_targets:
        from rag_paper.enrichment import enrich_targets

        enrichment_summary = enrich_targets(
            config,
            [target for target, _, _ in post_index_enrichment_targets],
            metadata_map=metadata_map,
            store=store,
        )
        enriched_metadata_files += enrichment_summary.updated_files
        failed_metadata_files += enrichment_summary.failed_files
        for _, pdf_path, chunks in post_index_enrichment_targets:
            if metadata_for_pdf(metadata_map, pdf_path).get("doi"):
                store.update_chunks_metadata(chunks, metadata_for_pdf(metadata_map, pdf_path))

    manifest.save()
    updated_papers_all = [
        _index_status_paper_entry(pdf_path, chunk_count, metadata_map)
        for pdf_path, chunk_count in indexed_paper_tuples
    ]
    return IndexSummary(
        indexed_files=indexed_files,
        skipped_files=skipped_files,
        chunks=indexed_chunks,
        enriched_metadata_files=enriched_metadata_files,
        failed_metadata_files=failed_metadata_files,
        duplicate_files=duplicate_files,
        updated_papers=updated_papers_all[:_STATUS_PAPERS_CAP],
        updated_papers_total=len(updated_papers_all),
    )


def refresh_chunk_metadata(
    config: AppConfig,
    *,
    metadata_map: dict[str, dict[str, Any]] | None = None,
) -> RefreshSummary:
    """Update indexed chunks' metadata from paper_metadata.json without re-embedding.

    Run this after ``enrich-metadata`` has corrected titles/DOIs/etc.: for every
    paper already in Chroma, merge its corrected metadata into the chunks'
    existing metadata (preserving source_path / chunk_index / chunk_type). No text
    is re-extracted and no embeddings are recomputed, so it is fast. Chunk text
    (including abstract-chunk text) is not changed by this.
    """
    if metadata_map is None:
        metadata_map = load_paper_metadata(config.metadata_path)

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_or_create_collection(
        name=config.chroma.collection, metadata={"hnsw:space": "cosine"}
    )
    payload = collection.get(include=["metadatas"])

    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for chunk_id, metadata in zip(payload.get("ids", []), payload.get("metadatas", [])):
        if not isinstance(metadata, dict):
            continue
        source_path = str(
            metadata.get("source_path") or metadata.get("file_name") or chunk_id
        )
        groups[source_path].append((str(chunk_id), metadata))

    papers = 0
    chunks_updated = 0
    skipped_papers = 0
    updated_papers: list[dict[str, Any]] = []
    for source_path, items in groups.items():
        corrected = {
            key: value
            for key, value in metadata_for_pdf(metadata_map, Path(source_path)).items()
            if value not in ("", None, [])
        }
        if not corrected:
            skipped_papers += 1
            continue
        ids = [chunk_id for chunk_id, _ in items]
        metadatas = [normalize_metadata({**meta, **corrected}) for _, meta in items]
        collection.update(ids=ids, metadatas=metadatas)
        papers += 1
        chunks_updated += len(ids)
        updated_papers.append(
            {
                "file_name": Path(source_path).name,
                "title": str(corrected.get("title") or ""),
                "doi": str(corrected.get("doi") or ""),
                "year": corrected.get("year"),
                "chunks": len(ids),
            }
        )

    logger.info(
        "index.metadata_refresh",
        papers=papers,
        chunks_updated=chunks_updated,
        skipped_papers=skipped_papers,
    )
    # Metadata refresh does not change chunk count, so a long-lived MCP server's
    # count-based cache would never notice it. Rewrite the manifest to bump its
    # mtime — the retriever treats that as a "library changed" reload signal.
    IndexManifest(config.chroma_dir / "index_manifest.json").save()
    return RefreshSummary(
        papers=papers,
        chunks_updated=chunks_updated,
        skipped_papers=skipped_papers,
        updated_papers=updated_papers[:_STATUS_PAPERS_CAP],
        updated_papers_total=len(updated_papers),
    )


def count_total_chunks(config: AppConfig) -> int | None:
    """Best-effort total chunk count in the collection (None if Chroma unavailable)."""
    try:
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        collection = client.get_or_create_collection(
            name=config.chroma.collection, metadata={"hnsw:space": "cosine"}
        )
        return int(collection.count())
    except Exception as exc:  # noqa: BLE001 - status recording must never break indexing
        logger.warning("index.status_count_failed", error=str(exc))
        return None


def record_index_status(
    config: AppConfig,
    *,
    command: str,
    duration_seconds: float,
    total_chunks: int | None,
    mode: dict[str, Any],
    result: dict[str, Any],
    max_history: int = 25,
) -> dict[str, Any]:
    """Prepend a finished update's outcome to the index-status history (newest first).

    The MCP service reads this file so an AI assistant can see when the library
    last changed and what the update did, without restart or in-memory coupling.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "duration_seconds": round(duration_seconds, 3),
        "total_chunks": total_chunks,
        "mode": mode,
        "result": result,
    }

    path = config.index_status_path
    history: list[dict[str, Any]] = []
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                history = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                history = [data]
    except (OSError, json.JSONDecodeError):
        history = []

    history.insert(0, record)
    del history[max_history:]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return record


def load_index_status(config: AppConfig, *, limit: int = 1) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` index-status records (newest first).

    Empty list when no update has been recorded yet or the file is unreadable.
    """
    path = config.index_status_path
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)][:limit]
