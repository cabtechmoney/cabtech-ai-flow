
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    heading: str
    text: str


def load_markdown_files(
    knowledge_base_dir: Path,
) -> list[tuple[str, str]]:
    """Load non-empty Markdown files."""
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {knowledge_base_dir}"
        )

    documents: list[tuple[str, str]] = []

    for path in sorted(knowledge_base_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()

        if content:
            documents.append((path.name, content))

    return documents


def split_into_chunks(
    source: str,
    content: str,
    max_chars: int = 1200,
) -> list[DocumentChunk]:
    """Split Markdown into chunks while preserving headings."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100.")

    chunks: list[DocumentChunk] = []
    current_heading = "General"
    paragraphs: list[str] = []
    chunk_number = 0

    def save_paragraphs() -> None:
        nonlocal chunk_number

        if not paragraphs:
            return

        text = "\n\n".join(paragraphs).strip()

        if text:
            chunk_number += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}:{chunk_number}",
                    source=source,
                    heading=current_heading,
                    text=f"Section: {current_heading}\n\n{text}",
                )
            )

        paragraphs.clear()

    for line in content.splitlines():
        match = re.match(
            r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$",
            line,
        )

        if match:
            save_paragraphs()
            current_heading = match.group(1).strip()
            continue

        line = line.strip()

        if not line:
            save_paragraphs()
            continue

        # Preserve long paragraphs without silently dropping text.
        while len(line) > max_chars:
            if paragraphs:
                save_paragraphs()

            part = line[:max_chars]
            line = line[max_chars:]

            paragraphs.append(part)
            save_paragraphs()

        if not paragraphs:
            paragraphs.append(line)
        else:
            candidate = "\n\n".join([*paragraphs, line])

            if len(candidate) > max_chars:
                save_paragraphs()

            paragraphs.append(line)

    save_paragraphs()
    return chunks


def build_chunks(
    knowledge_base_dir: Path,
    max_chars: int = 1200,
) -> list[DocumentChunk]:
    """Load and chunk every non-empty Markdown file."""
    all_chunks: list[DocumentChunk] = []

    for source, content in load_markdown_files(knowledge_base_dir):
        chunks = split_into_chunks(
            source=source,
            content=content,
            max_chars=max_chars,
        )
        all_chunks.extend(chunks)

    return all_chunks
