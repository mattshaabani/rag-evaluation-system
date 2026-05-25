"""
src/data/loader.py

Document loaders for the RAG pipeline.
Supports: PDF, plain text (.txt), Markdown (.md), web URLs.

All loaders return a unified List[Document] structure so everything
downstream is format-agnostic.

Usage:
    from src.data.loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load("data/raw/paper.pdf")
    docs = loader.load("data/raw/notes.txt")
    docs = loader.load("https://example.com/article")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. The Document dataclass
# ─────────────────────────────────────────────

@dataclass
class Document:
    """
    The universal unit of text in this pipeline.
    Every loader produces these. Every chunker consumes these.

    Attributes:
        content:  The cleaned text content.
        metadata: Source information that travels with the text forever.
                  Never lost, never separated from content.
    """
    content:  str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(chars={len(self.content)}, preview='{preview}...')"


# ─────────────────────────────────────────────
# 2. Text cleaner
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw extracted text before it enters the pipeline.

    Why cleaning matters:
        PDFs often extract with artifacts like:
        - Multiple consecutive blank lines
        - Hyphenated words broken across lines ("transfor-\nmer")
        - Weird unicode characters
        - Leading/trailing whitespace on every line

    This function fixes all of that.
    """
    # Fix hyphenated line breaks (common in PDFs)
    # "transfor-\nmer" → "transformer"
    text = re.sub(r"-\n", "", text)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove non-printable characters (except newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

    return text.strip()


# ─────────────────────────────────────────────
# 3. Individual loaders
# ─────────────────────────────────────────────

def load_txt(path: Path) -> list[Document]:
    """
    Load a plain text or markdown file.
    Simple: read the file, clean it, wrap in Document.
    """
    logger.info(f"Loading text file", extra={"path": str(path)})

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback encoding for files with special characters
        text = path.read_text(encoding="latin-1")

    cleaned = clean_text(text)

    doc = Document(
        content=cleaned,
        metadata={
            "source":    str(path),
            "file_name": path.name,
            "file_type": path.suffix.lstrip("."),  # "txt" or "md"
        }
    )

    logger.info(f"Loaded text file", extra={"chars": len(cleaned), "path": str(path)})
    return [doc]


def load_pdf(path: Path) -> list[Document]:
    """
    Load a PDF file, one Document per page.

    Why one Document per page?
        PDFs can be hundreds of pages. Keeping page boundaries in
        metadata lets us cite exact page numbers in answers.
        The chunker will later break pages into smaller pieces.

    Requires: pip install pypdf
    """
    logger.info(f"Loading PDF", extra={"path": str(path)})

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF loading. "
            "Run: pip install pypdf"
        )

    reader = PdfReader(str(path))
    docs = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = clean_text(text)

        # Skip empty pages (cover pages, blank pages)
        if len(cleaned.strip()) < 50:
            logger.debug(f"Skipping near-empty page", extra={"page": page_num})
            continue

        doc = Document(
            content=cleaned,
            metadata={
                "source":      str(path),
                "file_name":   path.name,
                "file_type":   "pdf",
                "page":        page_num,
                "total_pages": len(reader.pages),
            }
        )
        docs.append(doc)

    logger.info(f"Loaded PDF", extra={"pages": len(docs), "path": str(path)})
    return docs


def load_url(url: str) -> list[Document]:
    """
    Load a web page by URL, stripping all HTML tags.

    Why strip HTML?
        HTML tags like <div>, <span>, <nav> are structure, not content.
        We only want the readable text a human would see.

    Requires: pip install requests beautifulsoup4
    """
    logger.info(f"Loading URL", extra={"url": url})

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "requests and beautifulsoup4 are required for URL loading. "
            "Run: pip install requests beautifulsoup4"
        )

    headers = {"User-Agent": "Mozilla/5.0 (RAG pipeline document loader)"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # raises exception for 4xx/5xx errors

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove navigation, scripts, styles — we only want body content
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned = clean_text(text)

    doc = Document(
        content=cleaned,
        metadata={
            "source":      url,
            "file_name":   url.split("/")[-1] or "webpage",
            "file_type":   "url",
            "status_code": response.status_code,
        }
    )

    logger.info(f"Loaded URL", extra={"chars": len(cleaned), "url": url})
    return [doc]


# ─────────────────────────────────────────────
# 4. Main DocumentLoader class
# ─────────────────────────────────────────────

class DocumentLoader:
    """
    Unified loader that detects format and routes to the right loader.

    This is the only class other modules should import.
    They don't need to know about load_pdf, load_txt etc.

    Usage:
        loader = DocumentLoader()
        docs = loader.load("data/raw/paper.pdf")      # PDF
        docs = loader.load("data/raw/notes.txt")      # text
        docs = loader.load("data/raw/guide.md")       # markdown
        docs = loader.load("https://arxiv.org/...")   # URL
        docs = loader.load_directory("data/raw/")     # entire folder
    """

    # Map file extensions to loader functions
    SUPPORTED_EXTENSIONS = {
        ".pdf": load_pdf,
        ".txt": load_txt,
        ".md":  load_txt,   # markdown is just text
    }

    def load(self, source: str) -> list[Document]:
        """
        Load from a file path or URL.
        Auto-detects the format from the extension or URL prefix.
        """
        # URL detection
        if source.startswith("http://") or source.startswith("https://"):
            return load_url(source)

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        loader_fn = self.SUPPORTED_EXTENSIONS[suffix]
        return loader_fn(path)

    def load_directory(
        self,
        directory: str,
        extensions: Optional[list[str]] = None,
        recursive: bool = True,
    ) -> list[Document]:
        """
        Load all supported files from a directory.

        Args:
            directory:  Path to folder containing documents.
            extensions: Filter to specific extensions e.g. [".pdf"]
                        Defaults to all supported types.
            recursive:  Whether to search subdirectories too.

        Returns:
            All documents from all files, as a flat list.
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Which extensions to look for
        target_extensions = extensions or list(self.SUPPORTED_EXTENSIONS.keys())

        # Collect all matching files
        all_files = []
        for ext in target_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            all_files.extend(dir_path.glob(pattern))

        if not all_files:
            logger.warning(f"No supported files found", extra={"directory": str(dir_path)})
            return []

        logger.info(f"Loading directory", extra={
            "directory": str(dir_path),
            "file_count": len(all_files)
        })

        # Load each file and collect all documents
        all_docs = []
        for file_path in sorted(all_files):
            try:
                docs = self.load(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                # Log the error but keep going — one bad file
                # shouldn't stop the entire pipeline
                logger.error(f"Failed to load file", extra={
                    "path": str(file_path),
                    "error": str(e)
                })

        logger.info(f"Directory loaded", extra={
            "total_documents": len(all_docs),
            "directory": str(dir_path)
        })

        return all_docs