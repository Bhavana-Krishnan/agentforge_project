"""
chat_mcp.py - the CHAT MCP server: personal 1:1 (solo) chats between agents.

What lives here
---------------
- private conversations between exactly two agents (any team to any team)
- an unread inbox per agent

How it runs
-----------
Same as system_mcp.py: AgentForge never imports this file. mcp_client.py starts it as its
own process the first time an agent uses a solo-chat tool, then talks to it over MCP.

Tools exposed:
    register_agent(name, team)        -> tell the server an agent exists
    send_dm(sender, to, text)         -> send a private message
    read_dm(reader, other)            -> full conversation between two agents
    inbox(agent)                      -> unread messages (reading them marks them read)
    get_context(reader, other)        -> saved summary of the conversation + messages after it
    save_summary(reader, other, summary, upto)  -> store an updated summary

Run by hand:   python chat_mcp.py
"""

from datetime import datetime
from itertools import count

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("agentforge-chat", instructions="Private 1:1 chats between AgentForge agents.",
                   log_level="WARNING")

# ---- state, private to this process ----
_agents: dict[str, dict] = {}                          # agent key -> {"name": ..., "team": ...}
_conversations: dict[frozenset, list[dict]] = {}       # {a, b} -> messages (same key both directions)
_unread: dict[str, list[dict]] = {}                    # agent key -> messages not read yet
_summaries: dict[frozenset, dict] = {}                 # {a, b} -> {"summary": ..., "upto": last message id}
_ids = count(1)


def _key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ToolError("AgentForgeError: Names must be non-empty strings")
    return name.strip().lower()


def _agent(name: str) -> dict:
    k = _key(name)
    if k not in _agents:
        raise ToolError(f"NotFound: Agent '{name}' does not exist")
    return _agents[k]


@server.tool()
def register_agent(name: str, team: str) -> dict:
    """Tell the server an agent exists and which team it belongs to."""
    _agents[_key(name)] = {"name": name.strip(), "team": _key(team)}
    return {"ok": True}


@server.tool()
def send_dm(sender: str, to: str, text: str) -> dict:
    """Send a private message from one agent to another."""
    a, b = _agent(sender), _agent(to)
    if _key(a["name"]) == _key(b["name"]):
        raise ToolError("AgentForgeError: An agent cannot DM itself")
    msg = {"id": next(_ids), "sender": a["name"], "to": b["name"], "kind": "solo", "text": text,
           "ts": datetime.now().isoformat(timespec="seconds")}
    # frozenset({a, b}) == frozenset({b, a}) so both directions share one conversation
    _conversations.setdefault(frozenset({_key(a["name"]), _key(b["name"])}), []).append(msg)
    _unread.setdefault(_key(b["name"]), []).append(msg)       # into the receiver's inbox
    return msg


@server.tool()
def read_dm(reader: str, other: str) -> dict:
    """Read the whole private conversation between the reader and one other agent."""
    a, b = _agent(reader), _agent(other)
    return {"messages": list(_conversations.get(frozenset({_key(a["name"]), _key(b["name"])}), []))}


@server.tool()
def inbox(agent: str) -> dict:
    """Return the agent's unread private messages and mark them as read."""
    a = _agent(agent)
    return {"messages": _unread.pop(_key(a["name"]), [])}     # pop = return AND remove


# ---- context: rolling summary of one conversation (same flow as system_mcp.py) ----
def _pair(reader: str, other: str) -> frozenset:
    a, b = _agent(reader), _agent(other)
    return frozenset({_key(a["name"]), _key(b["name"])})      # only the two people in it can name it


@server.tool()
def get_context(reader: str, other: str) -> dict:
    """Get the saved summary of the reader's conversation with `other`, plus messages after it."""
    pair = _pair(reader, other)
    saved = _summaries.get(pair, {"summary": "", "upto": 0})
    new = [m for m in _conversations.get(pair, []) if m["id"] > saved["upto"]]
    return {"summary": saved["summary"], "upto": saved["upto"], "new_messages": new}


@server.tool()
def save_summary(reader: str, other: str, summary: str, upto: int) -> dict:
    """Store an updated summary of the conversation, covering every message up to id `upto`."""
    pair = _pair(reader, other)
    current = _summaries.get(pair, {"upto": 0})["upto"]
    msgs = _conversations.get(pair, [])
    upto = min(upto, msgs[-1]["id"] if msgs else 0)           # can't "cover" messages that don't exist yet
    if upto > current:                                        # ignore stale saves
        _summaries[pair] = {"summary": summary, "upto": upto}
    return {"ok": True, "upto": max(upto, current)}


if __name__ == "__main__":
    server.run()
