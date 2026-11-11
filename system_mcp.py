"""
system_mcp.py - the SYSTEM MCP server: group chat channels.

What lives here
---------------
- one "general" channel that every agent belongs to
- one channel per team (frontend, backend, ...), only that team's members can read / post

How it runs
-----------
AgentForge does NOT import this file. When an agent first uses a group-chat tool,
mcp_client.py starts this file as a separate Python process and talks to it over
stdin/stdout using the MCP protocol (JSON-RPC messages).
All the channel data lives inside THIS process, so it is gone when the process stops.

Tools exposed (an LLM or the forge can call these):
    register_team(team)                      -> create the team's channel
    register_agent(name, team)               -> tell the server an agent exists
    list_channels()                          -> all channel names
    channel_members(channel)                 -> who can see a channel
    post_message(sender, channel, text)      -> post to a channel
    read_channel(reader, channel)            -> read a channel's messages
    get_context(reader, channel)             -> saved summary + messages posted after it
    save_summary(reader, channel, summary, upto)  -> store an updated summary

Errors are raised as ToolError("<Kind>: message"). The Kind (NotFound, PermissionDenied,
AgentForgeError) lets the forge turn it back into the right Python exception.

You can also run it by hand to try it with any MCP client:   python system_mcp.py
"""

from datetime import datetime                          # timestamps on messages
from itertools import count                            # message ids 1, 2, 3 ...

from mcp.server.mcpserver import MCPServer             # the MCP SDK's server class (v2 name of FastMCP)
from mcp.server.mcpserver.exceptions import ToolError  # an "expected" error whose message reaches the client

# Create the server. The name is what clients see; log_level WARNING keeps the console quiet.
server = MCPServer("agentforge-system", instructions="Group chat channels for AgentForge teams.",
                   log_level="WARNING")

GENERAL = "general"                                    # the all-hands channel

# ---- state, private to this process ----
_agents: dict[str, dict] = {}                          # agent key -> {"name": "Rahul", "team": "frontend"}
_channels: dict[str, list[dict]] = {GENERAL: []}       # channel key -> list of message dicts
_ids = count(1)                                        # next(_ids) gives the next message id
# channel key -> {"summary": "...", "upto": <id of the last message the summary covers>}
# One summary per channel, shared by all its members (they all see the same messages).
_summaries: dict[str, dict] = {}


def _key(name: str) -> str:
    """Case-insensitive lookup key: '  Rahul ' -> 'rahul'."""
    if not isinstance(name, str) or not name.strip():
        raise ToolError("AgentForgeError: Names must be non-empty strings")
    return name.strip().lower()


def _agent(name: str) -> dict:
    """Find a registered agent or fail."""
    k = _key(name)
    if k not in _agents:
        raise ToolError(f"NotFound: Agent '{name}' does not exist")
    return _agents[k]


def _member_channel(agent: dict, channel: str) -> str:
    """Return the channel key if the agent may use it, otherwise fail."""
    ch = _key(channel)
    if ch not in _channels:
        raise ToolError(f"NotFound: Channel '{channel}' does not exist")
    if ch != GENERAL and _key(agent["team"]) != ch:    # team channels are members-only
        raise ToolError(f"PermissionDenied: {agent['name']} is not a member of channel '{channel}'")
    return ch


# ---------------------------------------------------------------------------
# Tools. @server.tool() registers the function; its docstring becomes the tool
# description and its type hints become the JSON input schema an LLM would see.
# ---------------------------------------------------------------------------
@server.tool()
def register_team(team: str) -> dict:
    """Create a team's group channel. Called by AgentForge when the team exists."""
    _channels.setdefault(_key(team), [])               # no-op if it already exists
    return {"ok": True, "channel": _key(team)}


@server.tool()
def register_agent(name: str, team: str) -> dict:
    """Tell the server an agent exists and which team it belongs to."""
    _agents[_key(name)] = {"name": name.strip(), "team": team.strip()}
    _channels.setdefault(_key(team), [])               # make sure their team channel exists
    return {"ok": True}


@server.tool()
def list_channels() -> dict:
    """List every group channel."""
    return {"channels": list(_channels)}


@server.tool()
def channel_members(channel: str) -> dict:
    """List the agents who can read and post in a channel."""
    ch = _key(channel)
    if ch not in _channels:
        raise ToolError(f"NotFound: Channel '{channel}' does not exist")
    if ch == GENERAL:
        return {"members": [a["name"] for a in _agents.values()]}             # everyone
    return {"members": [a["name"] for a in _agents.values() if _key(a["team"]) == ch]}


@server.tool()
def post_message(sender: str, channel: str, text: str) -> dict:
    """Post a message to a group channel the sender belongs to."""
    agent = _agent(sender)
    ch = _member_channel(agent, channel)
    msg = {"id": next(_ids), "sender": agent["name"], "to": ch, "kind": "group", "text": text,
           "ts": datetime.now().isoformat(timespec="seconds")}
    _channels[ch].append(msg)
    return msg                                         # the stored message goes back to the caller


@server.tool()
def read_channel(reader: str, channel: str) -> dict:
    """Read every message in a group channel the reader belongs to."""
    agent = _agent(reader)
    ch = _member_channel(agent, channel)
    return {"messages": list(_channels[ch])}


# ---- context: a rolling summary so agents don't start from zero every time ----
# The server only STORES the summary. Writing it needs an LLM (or another summarizer),
# which lives on the agent's side, so the flow is:
#   1. agent calls get_context  -> old summary + only the messages newer than it
#   2. agent merges them into a new summary with its summarizer
#   3. agent calls save_summary -> next time, only messages after this point are "new"
@server.tool()
def get_context(reader: str, channel: str) -> dict:
    """Get a channel's saved summary plus the messages posted after that summary was made."""
    agent = _agent(reader)
    ch = _member_channel(agent, channel)                              # members only, like read_channel
    saved = _summaries.get(ch, {"summary": "", "upto": 0})            # nothing summarized yet -> empty
    new = [m for m in _channels[ch] if m["id"] > saved["upto"]]       # what the summary doesn't cover
    return {"summary": saved["summary"], "upto": saved["upto"], "new_messages": new}


@server.tool()
def save_summary(reader: str, channel: str, summary: str, upto: int) -> dict:
    """Store an updated channel summary that covers every message up to id `upto`."""
    agent = _agent(reader)
    ch = _member_channel(agent, channel)
    current = _summaries.get(ch, {"upto": 0})["upto"]
    last = _channels[ch][-1]["id"] if _channels[ch] else 0
    upto = min(upto, last)                            # can't "cover" messages that don't exist yet
    if upto > current:                                # never replace a newer summary with an older one
        _summaries[ch] = {"summary": summary, "upto": upto}
    return {"ok": True, "upto": max(upto, current)}


if __name__ == "__main__":
    server.run()                                       # default transport: stdio (stdin/stdout)
