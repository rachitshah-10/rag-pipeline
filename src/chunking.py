"""Document loading and section-aware overlapping chunker for policy files."""

from dataclasses import dataclass
from pathlib import Path

# all-MiniLM-L6-v2 truncates inputs beyond 256 wordpiece tokens (~180-200 English words).
# Using 150-word chunks with 30-word overlap avoids silent truncation while preserving boundary context.
DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 30


@dataclass(frozen=True)
class DocumentChunk:
    """Single text chunk with source document and section metadata."""

    chunk_id: str
    text: str
    doc_name: str
    section: str
    chunk_index: int


def load_policy_documents(policies_dir: Path) -> dict[str, str]:
    """Load all Markdown policy files sorted by filename from policies_dir."""
    if not policies_dir.exists():
        raise FileNotFoundError(f"Policies directory does not exist: {policies_dir}")

    files = sorted(policies_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown policy files found in: {policies_dir}")

    return {path.name: path.read_text(encoding="utf-8") for path in files}


def _split_markdown_sections(content: str) -> list[tuple[str, str]]:
    """Split a Markdown document into (section_title, section_text) pairs on '##' headings."""
    sections: list[tuple[str, str]] = []
    current_title = "Document Header"
    buffer: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            # Flush the preceding section; skip bare title-only preambles before the first '##'.
            non_empty = [ln for ln in buffer if ln.strip()]
            has_body = any(not ln.lstrip().startswith("#") for ln in non_empty)
            if non_empty and (current_title != "Document Header" or has_body):
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title = line.lstrip("#").strip()
            buffer = [line]
        else:
            buffer.append(line)

    trailing = "\n".join(buffer).strip()
    if trailing:
        sections.append((current_title, trailing))

    return sections


def _window_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Slice a list of words into overlapping windows capped at chunk_size."""
    if len(words) <= chunk_size:
        return [" ".join(words)]

    windows: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        end = start + chunk_size
        windows.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return windows


def chunk_markdown_document(
    doc_name: str,
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split one Markdown policy document into section-aware overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive (got {chunk_size})")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size (got {overlap})")

    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for section_title, section_text in _split_markdown_sections(content):
        words = section_text.split()
        if not words:
            continue
        for window in _window_words(words, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}::chunk-{chunk_idx}",
                    text=window,
                    doc_name=doc_name,
                    section=section_title,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

    return chunks


def chunk_corpus(
    policies_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Load and chunk every Markdown policy document in policies_dir."""
    docs = load_policy_documents(policies_dir)
    all_chunks: list[DocumentChunk] = []
    for doc_name, content in docs.items():
        all_chunks.extend(
            chunk_markdown_document(
                doc_name=doc_name,
                content=content,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return all_chunks
