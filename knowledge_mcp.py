"""
knowledge_mcp.py - the KNOWLEDGE MCP server: shared "plugin memory" for every agent.

What lives here
---------------
Read-only reference knowledge: the documents in the project_brain/ folder (PDF, .md, .txt).
Every agent can search it; nobody can write to it through MCP. To teach the agents something
new, drop another file into the folder - it is picked up on the next search.

How search works (no AI, no extra services)
-------------------------------------------
1. Each document is split into small chunks, and every chunk remembers its document + page.
2. Lines repeated on most pages (running headers / footers) are dropped, so they don't
   match every query.
3. A query is scored against every chunk with BM25 - the classic keyword-ranking formula that
   search engines are built on. This follows the design doc's advice: "exact search first;
   semantic search where useful".
4. Results come back with their source (document + page), so an agent can cite them.

Tools exposed:
    list_documents()                       -> what is in the folder (and what was skipped, and why)
    search_knowledge(query, top_k=3)       -> best-matching chunks with document + page
    read_page(document, page)              -> the full cleaned text of one page

The folder comes from the AGENTFORGE_KNOWLEDGE_DIR environment variable (AgentForge sets it),
default: the project_brain/ folder next to this file.
"""

import math                                       # log() for BM25
import os                                         # folder paths, environment variable
import re                                         # splitting text into words
from collections import Counter                   # word counts per chunk

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("agentforge-knowledge", instructions="Search the shared project documents.",
                   log_level="WARNING")

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("AGENTFORGE_KNOWLEDGE_DIR") or os.path.join(HERE, "project_brain")
SUPPORTED = (".pdf", ".md", ".txt")
CHUNK_CHARS = 700                                 # aim for chunks about this long

# Common words that carry no meaning for search.
STOPWORDS = set("""a an and are as at be by for from has have how in is it its of on or that the
this to was were will with what when where which who why do does can should not no""".split())

# ---- index state (rebuilt whenever the folder changes) ----
_signature = None                                 # (file, size, mtime) of every file at last index
_pages: dict = {}                                 # (document, page) -> cleaned page text
_chunks: list = []                                # [{"document", "page", "text", "tf": Counter, "len": n}]
_df: Counter = Counter()                          # word -> number of chunks containing it
_avg_len = 0.0                                    # average chunk length in words
_documents: list = []                             # [{"name", "pages", "chunks"}]
_skipped: list = []                               # [{"name", "reason"}]


def _words(text: str) -> list:
    """'Retries are bounded' -> ['retrie', 'bounded'] (lowercase, no stopwords, crude plural strip)."""
    out = []
    for w in re.findall(r"[^\W_]+", text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                            # tickets -> ticket, agents -> agent
        out.append(w)
    return out


def _read_file(path: str) -> list:
    """Return the file as a list of page texts (a .md / .txt file counts as one page)."""
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader           # imported only if there is a PDF to read
        except ImportError:
            raise RuntimeError("install pypdf to read PDF files (pip install pypdf)")
        return [page.extract_text() or "" for page in PdfReader(path).pages]
    with open(path, encoding="utf-8", errors="replace") as f:
        return [f.read()]


def _strip_repeated_lines(pages: list) -> list:
    """Drop lines that appear on more than half of the pages (headers, footers) and bare page numbers."""
    if len(pages) < 3:
        return pages
    counts = Counter(line.strip() for page in pages for line in set(page.splitlines()) if line.strip())
    repeated = {line for line, n in counts.items() if n > len(pages) / 2}
    cleaned = []
    for page in pages:
        keep = [re.sub(r"(\s*\.){4,}", " ", l)      # "Summary . . . . . 3" -> "Summary 3" (contents page)
                for l in page.splitlines()
                if l.strip() and l.strip() not in repeated and not l.strip().isdigit()]
        cleaned.append("\n".join(l for l in keep if l.strip()))
    return cleaned


def _split(text: str) -> list:
    """Group lines into chunks of roughly CHUNK_CHARS characters."""
    chunks, current = [], ""
    for line in text.splitlines():
        current = f"{current}\n{line}" if current else line
        if len(current) >= CHUNK_CHARS:
            chunks.append(current)
            current = ""
    if current.strip():
        chunks.append(current)
    return chunks


def _folder_signature() -> tuple:
    if not os.path.isdir(FOLDER):
        return ()
    sig = []
    for name in sorted(os.listdir(FOLDER)):
        path = os.path.join(FOLDER, name)
        if name.lower().endswith(SUPPORTED) and os.path.isfile(path):
            st = os.stat(path)
            sig.append((name, st.st_size, st.st_mtime))
    return tuple(sig)


def _ensure_index() -> None:
    """(Re)build the index if a file was added, removed or changed since last time."""
    global _signature, _pages, _chunks, _df, _avg_len, _documents, _skipped
    sig = _folder_signature()
    if sig == _signature:
        return                                    # nothing changed - reuse the index
    _signature, _pages, _chunks, _df, _documents, _skipped = sig, {}, [], Counter(), [], []
    for name, _size, _mtime in sig:
        try:
            pages = _strip_repeated_lines(_read_file(os.path.join(FOLDER, name)))
        except Exception as exc:                  # unreadable file: report it, don't crash
            _skipped.append({"name": name, "reason": str(exc)})
            continue
        n_chunks = 0
        for page_no, text in enumerate(pages, start=1):
            _pages[(name, page_no)] = text
            for chunk in _split(text):
                words = _words(chunk)
                if not words:
                    continue
                _chunks.append({"document": name, "page": page_no, "text": chunk,
                                "tf": Counter(words), "len": len(words)})
                n_chunks += 1
        _documents.append({"name": name, "pages": len(pages), "chunks": n_chunks})
    for c in _chunks:
        _df.update(c["tf"].keys())                # each chunk counts once per word
    _avg_len = sum(c["len"] for c in _chunks) / len(_chunks) if _chunks else 0.0


def _bm25(query_words: list, chunk: dict, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 score: rare words count more; repeated words help, with diminishing returns;
    long chunks are penalised a little so they don't win just by being long."""
    n = len(_chunks)
    score = 0.0
    for w in set(query_words):
        tf = chunk["tf"].get(w, 0)
        if not tf:
            continue
        idf = math.log(1 + (n - _df[w] + 0.5) / (_df[w] + 0.5))
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunk["len"] / (_avg_len or 1)))
    return score


# ---------------------------------------------------------------------------
@server.tool()
def list_documents() -> dict:
    """List the documents in the shared knowledge folder, and any files that could not be read."""
    _ensure_index()
    return {"folder": FOLDER, "documents": _documents, "skipped": _skipped}


@server.tool()
def search_knowledge(query: str, top_k: int = 3) -> dict:
    """Search the shared project documents. Returns the best-matching passages with document and page."""
    _ensure_index()
    words = _words(query)
    if not words or not _chunks:
        return {"query": query, "results": []}
    scored = [(s, c) for c in _chunks if (s := _bm25(words, c)) > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [{"document": c["document"], "page": c["page"], "text": c["text"], "score": round(s, 3)}
               for s, c in scored[:max(0, min(top_k, 10))]]
    return {"query": query, "results": results}


@server.tool()
def read_page(document: str, page: int) -> dict:
    """Return the full text of one page of a document (a .md / .txt file has only page 1)."""
    _ensure_index()
    if (document, page) not in _pages:
        names = [d["name"] for d in _documents]
        if document not in names:
            raise ToolError(f"NotFound: Document '{document}' is not in the knowledge folder. Have: {names}")
        raise ToolError(f"NotFound: '{document}' has no page {page}")
    return {"document": document, "page": page, "text": _pages[(document, page)]}


if __name__ == "__main__":
    server.run()
