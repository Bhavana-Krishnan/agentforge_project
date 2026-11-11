"""
memory_mcp.py - the MEMORY MCP server: each agent's own personal memory.

What lives here
---------------
Short notes an agent decided were worth keeping ("reviewer wants input validation on every
form"). Unlike chat, memory is SAVED TO DISK - one JSON file per agent in agent_memory/ -
so an agent still knows it in the next run.

Memory is private: an agent only reads and writes its own file. Shared knowledge for everyone
lives in knowledge_mcp.py instead.

"Only when needed" - three rules keep memory small and useful
------------------------------------------------------------
1. Nothing is stored automatically. An agent calls remember() when something happened that
   is worth keeping (the agent's node code or its LLM decides when).
2. Near-duplicates are refused, so remembering the same lesson twice changes nothing.
3. Old memories are never deleted, only marked "stale" with a reason, so they stop showing up
   but the history stays (as the design doc's section 6.4 recommends).

Every memory has a kind, following the design doc's trust policy (section 6.3):
    fact         - verified: backed by code, test output or project state
    decision     - approved by the lead / a human
    hypothesis   - a guess that is NOT verified yet
    observation  - something that happened once (a review comment, a failed attempt)

Tools exposed:
    register_agent(name, team)
    remember(agent, text, kind="observation", source="")   -> stored? (or why not)
    recall(agent, query="", limit=5)                        -> most relevant active memories
    forget(agent, memory_id, reason)                        -> mark one memory stale
    list_memories(agent, include_stale=False)               -> everything the agent remembers

Folder: AGENTFORGE_MEMORY_DIR environment variable (AgentForge sets it), default agent_memory/.
"""

import contextlib
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("agentforge-memory", instructions="Personal long-term memory for each agent.",
                   log_level="WARNING")

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("AGENTFORGE_MEMORY_DIR") or os.path.join(HERE, "agent_memory")
KINDS = ("fact", "decision", "hypothesis", "observation")
MAX_CHARS = 500                                    # memories are notes, not documents
DUPLICATE_OVERLAP = 0.8                            # 80% of the same words = same memory

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this
to was were will with""".split())

_agents: dict = {}                                 # agent key -> {"name", "team"}
_cache: dict = {}                                  # agent key -> list of memory dicts (loaded from disk)


def _key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ToolError("AgentForgeError: Names must be non-empty strings")
    return name.strip().lower()


def _agent(name: str) -> str:
    k = _key(name)
    if k not in _agents:
        raise ToolError(f"NotFound: Agent '{name}' does not exist")
    return k


def _path(agent_key: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]", "_", agent_key)  # "dev one" -> "dev_one", safe as a file name
    if safe != agent_key:                          # something was replaced: add a hash so that
        digest = hashlib.sha1(agent_key.encode("utf-8")).hexdigest()[:8]   # "dev one" != "dev_one"
        safe = f"{safe}-{digest}"
    return os.path.join(FOLDER, f"{safe}.json")


def _load(agent_key: str) -> list:
    """Read the agent's memory file fresh from disk (other processes may have changed it)."""
    path = _path(agent_key)
    try:
        with open(path, encoding="utf-8") as f:
            memories = json.load(f)["memories"]
        for m in memories:                         # shape check: every note needs these fields
            m["id"], m["text"], m["status"]
        if not isinstance(memories, list):
            raise TypeError("memories is not a list")
    except FileNotFoundError:
        memories = []                              # new agent - nothing remembered yet
    except (ValueError, KeyError, TypeError):      # corrupt file: keep it as .bak, start empty
        with contextlib.suppress(OSError):
            os.replace(path, path + ".bak")
        memories = []
    _cache[agent_key] = memories
    return memories


@contextlib.contextmanager
def _locked(agent_key: str):
    """Hold an exclusive file lock while one agent's memory is read-modified-written."""
    os.makedirs(FOLDER, exist_ok=True)
    try:
        import fcntl
    except ImportError:                            # e.g. Windows: no locking available
        yield
        return
    with open(_path(agent_key) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save(agent_key: str) -> None:
    """Write to a temporary file, then swap it in, so a crash never leaves half a file."""
    os.makedirs(FOLDER, exist_ok=True)
    path = _path(agent_key)
    fd, tmp = tempfile.mkstemp(dir=FOLDER, suffix=".tmp")      # unique name per writer
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"agent": _agents[agent_key]["name"], "memories": _cache[agent_key]}, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _words(text: str) -> set:
    return {w for w in re.findall(r"[^\W_]+", text.lower()) if w not in STOPWORDS}


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------------------
@server.tool()
def register_agent(name: str, team: str) -> dict:
    """Tell the server an agent exists and which team it belongs to."""
    _agents[_key(name)] = {"name": name.strip(), "team": team.strip()}
    return {"ok": True}


@server.tool()
def remember(agent: str, text: str, kind: str = "observation", source: str = "") -> dict:
    """Save a short note to the agent's personal memory. Near-duplicates are not stored again.
    kind: fact (verified), decision (approved), hypothesis (unverified guess), observation (happened once).
    source: where it came from, e.g. 'ticket #3'."""
    k = _agent(agent)
    text = (text or "").strip()
    if not text:
        raise ToolError("AgentForgeError: Nothing to remember")
    if len(text) > MAX_CHARS:
        raise ToolError(f"AgentForgeError: Keep memories under {MAX_CHARS} characters")
    if kind not in KINDS:
        raise ToolError(f"AgentForgeError: kind must be one of {KINDS}")
    new_words = _words(text)
    with _locked(k):
        memories = _load(k)
        for m in memories:                         # rule 2: refuse near-duplicates
            if m["status"] == "active" and (
                    m["text"] == text or _overlap(new_words, _words(m["text"])) >= DUPLICATE_OVERLAP):
                return {"stored": False, "reason": "already remembered", "id": m["id"]}
        entry = {"id": max((m["id"] for m in memories), default=0) + 1, "text": text, "kind": kind,
                 "source": source, "created": datetime.now().isoformat(timespec="seconds"),
                 "status": "active"}
        memories.append(entry)
        _save(k)
    return {"stored": True, "id": entry["id"]}


@server.tool()
def recall(agent: str, query: str = "", limit: int = 5) -> dict:
    """Return the agent's active memories. With a query: only those sharing words with it, best first.
    Without a query: the newest ones."""
    k = _agent(agent)
    active = [m for m in _load(k) if m["status"] == "active"]
    if query.strip():
        q = _words(query)
        scored = [(len(q & _words(m["text"])), m) for m in active]
        active = [m for s, m in sorted(scored, key=lambda p: (p[0], p[1]["id"]), reverse=True) if s > 0]
    else:
        active = sorted(active, key=lambda m: m["id"], reverse=True)
    return {"memories": active[:max(0, limit)]}


@server.tool()
def forget(agent: str, memory_id: int, reason: str) -> dict:
    """Mark one memory as stale (no longer true). It stays in the file as history but is not recalled."""
    k = _agent(agent)
    with _locked(k):
        for m in _load(k):
            if m["id"] == memory_id:
                if m["status"] != "stale":         # keep the original reason if already stale
                    m["status"], m["stale_reason"] = "stale", reason
                    _save(k)
                return {"ok": True}
    raise ToolError(f"NotFound: {agent} has no memory #{memory_id}")


@server.tool()
def list_memories(agent: str, include_stale: bool = False) -> dict:
    """Everything the agent remembers (stale ones too, if include_stale)."""
    k = _agent(agent)
    return {"memories": [m for m in _load(k) if include_stale or m["status"] == "active"]}


if __name__ == "__main__":
    server.run()
