"""
AgentForge - a small multi-agent orchestration framework built from scratch.

Concepts
--------
- team(name)                          -> create a team (frontend, backend, anything)
- teamlead(agent_name, team_name)     -> add a TeamLead node to a team
- developer(agent_name, team_name)    -> add a Developer node to a team
- move_to(src, dst)                   -> fixed hop from one agent to the next
- branch_to(src, router)              -> decide the next hop at run time: router(state) -> agent name or END
- run(state, start)                   -> execute the graph

Every agent name is unique across the whole forge, so the same person can never
hold two roles (e.g. rahul cannot be a TeamLead and a Developer).

Core (always in memory, inside this file):
- AgentForge.KanbanBoard   -> tickets with TODO / IN_PROGRESS / REVIEW / DONE and role-checked moves
                              (create, assign, take, move, submit, approve, reject)
- A Developer can only take tickets the TeamLead has ASSIGNED to them.

Tools (outside this file, started only when first used):
- system_mcp.py            -> group chat channels (one per team + "general")
- chat_mcp.py              -> personal 1:1 chats + inbox
- AgentForge.MCPHub        -> knows those servers and starts each one lazily
- agent.channel_context() / agent.dm_context(other) -> rolling summary of a chat, so an LLM
                              agent knows what was already said (summarizer is pluggable)
- knowledge_mcp.py         -> shared documents in project_brain/ (agent.lookup(query))
- memory_mcp.py            -> each agent's personal memory, saved in agent_memory/
                              (agent.remember / recall / forget)
- agent.system_prompt      -> rebuilt automatically when a ticket that concerns the agent changes:
                              role + open tickets + matching knowledge + memories + recent changes

Node functions have the signature   fn(agent, state) -> dict
`agent` is the TeamLead / Developer object, so inside a node you can call
agent.create_ticket(...), agent.take(...), agent.submit(...), agent.say(...), agent.dm(...), etc.

Call forge.close() (or use `with AgentForge() as forge:`) to stop any MCP servers it started.
"""

# Lets us write type hints that mention classes defined further down the file (e.g. "AgentForge").
from __future__ import annotations

import itertools                                  # count() for auto-increment ids, cycle() for round-robin
import json                                       # to export the board as JSON text
import os                                         # to find system_mcp.py / chat_mcp.py next to this file
from dataclasses import asdict, dataclass, field  # @dataclass writes __init__ etc. for simple record classes
from collections import deque                     # a list that keeps only the last N items
from datetime import datetime                     # timestamps on messages and ticket history
from typing import Callable, Dict, List, Optional # type hints only - they don't change behaviour

END = "END"   # special name meaning "stop the graph". Used by move_to / branch_to.


# --------------------------------------------------------------------------- #
# Errors - our own exception types so callers can catch exactly what they want
# --------------------------------------------------------------------------- #
class AgentForgeError(Exception):          # parent of all errors below; `except AgentForgeError` catches all
    """Base error for everything raised by AgentForge."""


class RoleConflictError(AgentForgeError):  # e.g. adding rahul as Developer when he is already a TeamLead
    """An agent name is already used by another role or team."""


class PermissionDenied(AgentForgeError):   # e.g. a Developer trying to mark a ticket DONE
    """An agent tried to do something its role does not allow."""


class NotFound(AgentForgeError):           # unknown team / agent / ticket / channel
    """Unknown team, agent, ticket or channel."""


# --------------------------------------------------------------------------- #
# Data records
# --------------------------------------------------------------------------- #
class Status:                                # just a namespace of constants, so we never mistype "IN_PROGRES"
    TODO = "TODO"                            # created, nobody working on it
    IN_PROGRESS = "IN_PROGRESS"              # a developer is coding it
    REVIEW = "REVIEW"                        # code submitted, waiting for the lead
    DONE = "DONE"                            # lead approved it - only a lead can set this
    ALL = (TODO, IN_PROGRESS, REVIEW, DONE)  # handy for validation and for building board columns


def _now() -> str:
    """Current time as text, e.g. '2026-10-29T16:18:05'."""
    return datetime.now().isoformat(timespec="seconds")


def _shorten(text: str, limit: int) -> str:
    """Collapse whitespace and cut to `limit` characters, ending with '...' if cut."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _folder_signature(folder: str) -> tuple:
    """(name, size, mtime) of every file in a folder - changes whenever a document is added/edited/removed."""
    try:
        sig = []
        for name in sorted(os.listdir(folder)):
            try:
                st = os.stat(os.path.join(folder, name))
            except OSError:
                continue
            sig.append((name, st.st_size, st.st_mtime))
        return tuple(sig)
    except OSError:
        return ()


MAX_PROMPT_TICKETS = 20                      # a prompt lists at most this many tickets (newest first)
MAX_TICKET_TEXT = 300                        # description / feedback are cut to this many characters
MAX_MEMORY_TEXT = 300


def _key(name: str) -> str:
    """Names are compared case-insensitively so 'Rahul' and 'rahul' are the same person."""
    if not isinstance(name, str) or not name.strip():         # reject None, numbers, "" and "   "
        raise AgentForgeError("Names must be non-empty strings")
    return name.strip().lower()                                # "  Rahul " -> "rahul"


@dataclass                                   # auto-generates __init__, __repr__, __eq__ from the fields below
class Ticket:
    id: int                                  # unique number, 1, 2, 3 ...
    title: str                               # short name, e.g. "Login page"
    description: str                         # details for the developer
    team: str                                # which team owns it
    created_by: str                          # the lead who created it
    assignee: Optional[str] = None           # developer it is assigned to (None = waiting for the lead to assign)
    status: str = Status.TODO                # every ticket starts in TODO
    code: str = ""                           # code the developer submitted
    feedback: str = ""                       # lead's latest rejection reason
    rejections: int = 0                      # how many times the lead rejected it
    # default_factory makes a NEW empty list per ticket (a plain `= []` would be shared by all tickets)
    history: List[dict] = field(default_factory=list)
    # passages from project_brain/ that match this ticket; looked up once, the first time an
    # agent's system prompt needs them (None = not looked up yet)
    knowledge: Optional[List[dict]] = None

    def to_dict(self) -> dict:               # plain dict version - easy to send to a UI / API as JSON
        return asdict(self)


@dataclass
class Team:
    name: str                                         # e.g. "frontend"
    description: str = ""                             # optional free text
    lead: Optional[str] = None                        # name of the TeamLead (max one)
    developers: List[str] = field(default_factory=list)   # names of the Developers

    @property                                         # lets you write team.members (no brackets)
    def members(self) -> List[str]:
        # lead first (if there is one), then all developers
        return ([self.lead] if self.lead else []) + list(self.developers)


# --------------------------------------------------------------------------- #
# Agents (the nodes of the graph)
# --------------------------------------------------------------------------- #
class Agent:
    """Common base for TeamLead and Developer: identity + chat tools + read-only board view."""
    role = "Agent"                           # overridden by subclasses; used in error messages

    duties = ""                              # one paragraph per role, used at the top of the system prompt

    def __init__(self, forge: "AgentForge", name: str, team: str, fn: Callable):
        self.forge = forge                   # back-reference so the agent can reach the board and chat
        self.name = name                     # e.g. "rahul"
        self.team = team                     # e.g. "frontend"
        self.fn = fn                         # the node function run() will call: fn(agent, state)
        # ---- self-updating system prompt state ----
        self._prompt: Optional[str] = None   # last built prompt
        self._prompt_stale = True            # True = something changed, rebuild on next read
        self.prompt_version = 0              # goes up every time the prompt text actually changes
        self.recent_changes = deque(maxlen=5)  # last 5 ticket events that concern this agent

    # ---- communication shortcuts ----
    # The chat logic and data live in the MCP servers, not here. Each method below is a
    # one-line forward: forge.mcp.call(<server>, <tool>, <arguments>). The first call to a
    # server starts it; servers that are never used are never started.
    # Messages come back as plain dicts: {"id", "sender", "to", "kind", "text", "ts"}.

    # group chat  ->  system_mcp.py
    def say(self, text: str, channel: Optional[str] = None) -> dict:
        """Post to a group channel. Defaults to the agent's own team channel."""
        return self.forge.mcp.call("system", "post_message",
                                   {"sender": self.name, "channel": self.team if channel is None else channel, "text": text})

    def read_channel(self, channel: Optional[str] = None) -> List[dict]:
        """All messages in a group channel (default: own team)."""
        return self.forge.mcp.call("system", "read_channel",
                                   {"reader": self.name, "channel": self.team if channel is None else channel})["messages"]

    # solo chat  ->  chat_mcp.py
    def dm(self, to: str, text: str) -> dict:
        """Send a solo (1:1) message."""
        return self.forge.mcp.call("chat", "send_dm", {"sender": self.name, "to": to, "text": text})

    def read_dm(self, other: str) -> List[dict]:
        """Full solo-chat history with one other agent."""
        return self.forge.mcp.call("chat", "read_dm", {"reader": self.name, "other": other})["messages"]

    def inbox(self) -> List[dict]:
        """Unread solo messages; reading them marks them as read."""
        return self.forge.mcp.call("chat", "inbox", {"agent": self.name})["messages"]

    # ---- context: "what has been said so far?" ----
    # An LLM agent remembers nothing between calls. These give it a short summary of a chat
    # to put in its prompt. Only messages newer than the saved summary get summarized, so
    # asking again is cheap, and a summary made by one member is reused by the others.
    def channel_context(self, channel: Optional[str] = None) -> str:
        """Summary of a group channel (default: own team). '' if nothing was said yet."""
        return self._context("system", {"reader": self.name, "channel": self.team if channel is None else channel})

    def dm_context(self, other: str) -> str:
        """Summary of my solo chat with `other`. '' if we never talked."""
        return self._context("chat", {"reader": self.name, "other": other})

    def _context(self, server: str, args: dict) -> str:
        ctx = self.forge.mcp.call(server, "get_context", args)    # old summary + newer messages
        new = ctx["new_messages"]
        if not new:                                               # summary is up to date
            return ctx["summary"]
        summary = self.forge.summarizer(ctx["summary"], new)      # merge old summary + new messages
        self.forge.mcp.call(server, "save_summary",               # store it for everyone next time
                            {**args, "summary": summary, "upto": new[-1]["id"]})
        return summary

    # ---- shared knowledge (plugin memory)  ->  knowledge_mcp.py ----
    def lookup(self, query: str, top_k: int = 3) -> List[dict]:
        """Search the documents in project_brain/. Each hit: {"document", "page", "text", "score"}."""
        return self.forge.mcp.call("knowledge", "search_knowledge",
                                   {"query": query, "top_k": top_k})["results"]

    # ---- personal memory  ->  memory_mcp.py ----
    def remember(self, text: str, kind: str = "observation", source: str = "") -> dict:
        """Save a note to MY memory. Returns {"stored": True/False, ...}; duplicates are not stored."""
        result = self.forge.mcp.call("memory", "remember",
                                     {"agent": self.name, "text": text, "kind": kind, "source": source})
        if result["stored"]:
            self._prompt_stale = True        # the "what you remember" part of my prompt changed
        return result

    def recall(self, query: str = "", limit: int = 5) -> List[dict]:
        """My active memories - matching `query` if given, otherwise the newest."""
        return self.forge.mcp.call("memory", "recall",
                                   {"agent": self.name, "query": query, "limit": limit})["memories"]

    def forget(self, memory_id: int, reason: str) -> None:
        """Mark one of my memories as stale (kept as history, no longer recalled)."""
        self.forge.mcp.call("memory", "forget", {"agent": self.name, "memory_id": memory_id, "reason": reason})
        self._prompt_stale = True

    def memories(self, include_stale: bool = False) -> List[dict]:
        """Everything I remember."""
        return self.forge.mcp.call("memory", "list_memories",
                                   {"agent": self.name, "include_stale": include_stale})["memories"]

    # ---- kanban (read) ----
    def board(self, status: Optional[str] = None) -> List[Ticket]:
        """Tickets of my team, optionally filtered by status."""
        return self.forge.board.view(self.name, status=status)

    # ---- self-updating system prompt ----
    # The board calls _note_change() whenever a ticket that concerns this agent changes
    # (created, assigned, moved, rejected, approved). That only marks the prompt as stale -
    # cheap, no MCP call. The prompt is rebuilt the next time someone reads
    # agent.system_prompt, so it is always current, and an agent whose prompt is never read
    # (e.g. in kanban-only code) never starts the knowledge / memory servers.
    def _note_change(self, message: str) -> None:
        self.recent_changes.append(_shorten(message, 400))
        self._prompt_stale = True

    @property
    def system_prompt(self) -> str:
        """Who I am + my open tickets + matching project knowledge + my memories + recent changes."""
        kn_sig = _folder_signature(self.forge.knowledge_dir)
        if kn_sig != self.__dict__.get("_prompt_kn_sig"):               # a document was added/edited/removed
            self._prompt_kn_sig = kn_sig
            self._prompt_stale = True
        if self._prompt is None or self._prompt_stale:
            new = self._build_system_prompt()
            if new != self._prompt:
                self._prompt = new
                self.prompt_version += 1
            self._prompt_stale = False
        return self._prompt

    def open_tickets(self) -> List[Ticket]:
        """Tickets this agent should be thinking about (overridden per role)."""
        return []

    def _build_system_prompt(self) -> str:
        tickets = self.open_tickets()
        parts = [f"You are {self.name}, the {self.role} of the {self.team} team in AgentForge.",
                 self.duties]

        # 1. tickets
        if tickets:
            lines = []
            shown = tickets if len(tickets) <= MAX_PROMPT_TICKETS else tickets[-MAX_PROMPT_TICKETS:]
            if len(shown) < len(tickets):
                lines.append(f"({len(tickets) - len(shown)} older open tickets not shown)")
            for t in shown:
                who = f" (assignee: {t.assignee or 'unassigned'})" if isinstance(self, TeamLead) else ""
                desc = _shorten(t.description or "", MAX_TICKET_TEXT)
                lines.append(f"- #{t.id} [{t.status}] {t.title}{who}: {desc}".rstrip(": "))
                if t.feedback and t.status == Status.IN_PROGRESS:
                    lines.append(f"  Reviewer feedback to address: {_shorten(t.feedback, MAX_TICKET_TEXT)}")
            parts.append("## Your tickets\n" + "\n".join(lines))
        else:
            parts.append("## Your tickets\nNone right now.")

        # 2. shared knowledge for those tickets (looked up once per ticket, then cached on it)
        hits, seen = [], set()
        sigs = self.__dict__.setdefault("_knowledge_sigs", {})          # ticket id -> folder signature
        sig = _folder_signature(self.forge.knowledge_dir)
        for t in tickets[:3]:
            if t.knowledge is None or sigs.get(t.id) != sig:           # first time, or documents changed
                sigs[t.id] = sig
                t.knowledge = [{"document": h["document"], "page": h["page"], "text": h["text"]}
                               for h in self.lookup(f"{t.title} {t.description}", top_k=2)]
            for h in t.knowledge:
                if (h["document"], h["page"]) not in seen:
                    seen.add((h["document"], h["page"]))
                    hits.append(h)
        if hits:
            parts.append("## Relevant project knowledge (cite the source when you use it)\n" + "\n".join(
                f"- [{h['document']}, p.{h['page']}] {_shorten(h['text'], 350)}" for h in hits[:4]))

        # 3. my own memories that relate to these tickets (newest ones if I have no tickets)
        query = " ".join(f"{t.title} {t.description} {t.feedback}" for t in tickets)
        mems = self.recall(query, limit=5)
        if mems:
            parts.append("## What you remember\n" + "\n".join(
                f"- ({m['kind']}{', ' + m['source'] if m['source'] else ''}) {_shorten(m['text'], MAX_MEMORY_TEXT)}"
                for m in mems))

        # 4. what just happened
        if self.recent_changes:
            parts.append("## Recent changes\n" + "\n".join(f"- {c}" for c in self.recent_changes))
        return "\n\n".join(p for p in parts if p)

    def __repr__(self) -> str:               # how the agent prints, e.g. TeamLead('rahul', team='frontend')
        return f"{self.role}({self.name!r}, team={self.team!r})"


class TeamLead(Agent):
    """Only a TeamLead can create, assign, approve (DONE) and reject tickets."""
    role = "TeamLead"
    duties = ("You break work into tickets, assign every ticket to a developer of your team, review "
              "submitted code, and you are the only one who can mark a ticket DONE.")

    def open_tickets(self) -> List[Ticket]:
        """Every ticket of my team that is not DONE yet."""
        return [t for t in self.forge.board.tickets(team=self.team) if t.status != Status.DONE]

    # Each method just forwards to the board, passing self.name so the board can check permissions.
    def create_ticket(self, title: str, description: str = "", assignee: Optional[str] = None) -> Ticket:
        return self.forge.board.create_ticket(self.name, title, description, assignee)

    def assign(self, ticket_id: int, developer: str) -> Ticket:
        return self.forge.board.assign(self.name, ticket_id, developer)

    def approve(self, ticket_id: int) -> Ticket:
        """Only a TeamLead can mark a ticket DONE."""
        return self.forge.board.approve(self.name, ticket_id)

    def reject(self, ticket_id: int, feedback: str) -> Ticket:
        """Send a ticket in REVIEW back to IN_PROGRESS with feedback."""
        return self.forge.board.reject(self.name, ticket_id, feedback)

    def review_queue(self) -> List[Ticket]:
        """Tickets of my team waiting for my review."""
        return self.forge.board.tickets(team=self.team, status=Status.REVIEW)


class Developer(Agent):
    """A Developer can take tickets, move them to IN_PROGRESS / REVIEW, and submit code."""
    role = "Developer"
    duties = ("You work only on tickets your TeamLead assigned to you: take them, write the code, and "
              "submit it for review. You cannot mark tickets DONE.")

    def open_tickets(self) -> List[Ticket]:
        """Tickets assigned to me that are not DONE yet."""
        return [t for t in self.my_tickets() if t.status != Status.DONE]

    def available_tickets(self) -> List[Ticket]:
        """TODO tickets the TeamLead assigned to me. Unassigned tickets are NOT mine to take."""
        return self.forge.board.tickets(team=self.team, status=Status.TODO, assignee=self.name)

    def my_tickets(self, status: Optional[str] = None) -> List[Ticket]:
        """Tickets assigned to me, optionally filtered by status."""
        return self.forge.board.tickets(team=self.team, status=status, assignee=self.name)

    def take(self, ticket_id: int) -> Ticket:
        """Pick up a ticket -> IN_PROGRESS."""
        return self.forge.board.take(self.name, ticket_id)

    def move(self, ticket_id: int, status: str) -> Ticket:
        """Change status (the board decides whether this move is allowed)."""
        return self.forge.board.move(self.name, ticket_id, status)

    def submit(self, ticket_id: int, code: str) -> Ticket:
        """Attach code and move the ticket to REVIEW."""
        return self.forge.board.submit(self.name, ticket_id, code)


# --------------------------------------------------------------------------- #
# Default summarizer (used when AgentForge(summarizer=...) is not given) - no AI involved
# --------------------------------------------------------------------------- #
def default_summarizer(previous: str, new_messages: List[dict], max_lines: int = 20) -> str:
    """Not a real summary: keeps the latest `max_lines` lines of 'sender: text'.
    Pass an LLM-based summarizer to AgentForge for real summaries (see test_with_ai.py)."""
    lines = previous.splitlines() if previous else []               # what we had before
    lines += [f"{m['sender']}: {m['text']}" for m in new_messages]  # add the new messages
    return "\n".join(lines[-max_lines:])                            # keep only the most recent lines


# --------------------------------------------------------------------------- #
# Default node behaviours (used when no fn is passed) - no AI involved
# --------------------------------------------------------------------------- #
def default_teamlead(agent: TeamLead, state: dict) -> dict:
    # ---- 1. review anything waiting ----
    for t in agent.review_queue():
        if (t.code or "").strip():                                 # some code was submitted -> accept
            agent.approve(t.id)
            agent.say(f"Approved #{t.id} '{t.title}'. Nice work {t.assignee}!")
        else:                                              # empty submission -> send back
            agent.reject(t.id, "No code was submitted.")

    # ---- 2. turn requirements into tickets (once each) ----
    # Accept either state["requirements"] (a list) or state["task"] (one string, like your original code).
    reqs = state.get("requirements") or ([state["task"]] if state.get("task") else [])
    existing = {t.title.strip() for t in agent.forge.board.tickets(team=agent.team)}   # titles already ticketed
    devs = agent.forge.teams[_key(agent.team)].developers                     # developers to assign to
    rr = itertools.cycle(devs) if devs else None           # endless iterator: priya, arjun, priya, arjun ...
    for req in reqs:
        if req.strip() not in existing:                    # don't create the same ticket twice on later visits
            t = agent.create_ticket(req, f"Implement: {req}", assignee=next(rr) if rr else None)
            existing.add(t.title)
            agent.say(f"New ticket #{t.id} '{t.title}' -> {t.assignee or 'unassigned'}")

    board = agent.forge.board
    # Returned dict is merged into the shared state by run().
    return {
        "open_tickets": len([t for t in board.tickets(team=agent.team) if t.status != Status.DONE]),
        "done_tickets": [t.id for t in board.tickets(team=agent.team, status=Status.DONE)],
    }


def default_developer(agent: Developer, state: dict) -> dict:
    for t in agent.available_tickets():                    # pick up everything free or assigned to me
        agent.take(t.id)

    code = dict(state.get("code") or {})                   # copy so we add to, not replace, earlier code
    for t in agent.my_tickets(status=Status.IN_PROGRESS):  # new tickets AND rejected ones come back here
        snippet = f"# code for {t.title}\n# by {agent.name}"     # placeholder "code" (no AI)
        if t.feedback:
            snippet += f"\n# addressed feedback: {t.feedback}"
            # A rejection is worth remembering for next time - this is the "only when needed" moment.
            # (memory_mcp.py ignores it if the same lesson is already stored.)
            agent.remember(f"Reviewer feedback on '{t.title}': {t.feedback}",
                           kind="observation", source=f"ticket #{t.id}")
        agent.submit(t.id, snippet)                        # IN_PROGRESS -> REVIEW
        code[t.id] = snippet
        lead = agent.forge.teams[_key(agent.team)].lead
        if lead:
            agent.dm(lead, f"#{t.id} is ready for review.")   # solo chat to the lead
    return {"code": code}


# --------------------------------------------------------------------------- #
# The framework
# --------------------------------------------------------------------------- #
class AgentForge:
    """Graph of TeamLead / Developer agents with a built-in Kanban board and lazily-loaded MCP chat tools."""

    # ======================= Kanban board (nested class) ======================= #
    class KanbanBoard:
        # Allowed (from_status, to_status) pairs per role. Anything not listed is refused.
        DEV_MOVES = {
            (Status.TODO, Status.IN_PROGRESS),     # start work
            (Status.IN_PROGRESS, Status.REVIEW),   # hand in work
            (Status.REVIEW, Status.IN_PROGRESS),   # pull back own work
        }
        LEAD_MOVES = {
            (Status.REVIEW, Status.DONE),          # only the lead can close
            (Status.REVIEW, Status.IN_PROGRESS),   # request changes
        }

        def __init__(self, forge: "AgentForge"):
            self.forge = forge                         # needed to look up agents and their roles
            self._tickets: Dict[int, Ticket] = {}      # ticket id -> Ticket
            self._ids = itertools.count(1)             # next(self._ids) gives 1, 2, 3 ...

        # ---------- read API ----------
        def get(self, ticket_id: int) -> Ticket:
            """One ticket by id, or NotFound."""
            if (isinstance(ticket_id, bool) or not isinstance(ticket_id, int)
                    or ticket_id not in self._tickets):
                raise NotFound(f"Ticket #{ticket_id} does not exist")
            return self._tickets[ticket_id]

        def tickets(self, team: Optional[str] = None, status: Optional[str] = None,
                    assignee: Optional[str] = None) -> List[Ticket]:
            """All tickets, narrowed by any filters you pass (None = don't filter on that)."""
            out = list(self._tickets.values())
            if team is not None:
                out = [t for t in out if _key(t.team) == _key(team)]
            if status is not None:
                self._check_status(status)                     # catches typos like "Done"
                out = [t for t in out if t.status == status]
            if assignee is not None:
                out = [t for t in out if t.assignee and _key(t.assignee) == _key(assignee)]
            return out

        def view(self, agent_name: str, status: Optional[str] = None) -> List[Ticket]:
            """What an agent can see: every ticket of their own team."""
            agent = self.forge.get_agent(agent_name)
            return self.tickets(team=agent.team, status=status)

        def snapshot(self, team: Optional[str] = None) -> Dict[str, List[dict]]:
            """Board as columns -> ticket dicts. Handy for a UI / REST API."""
            cols = {s: [] for s in Status.ALL}                 # {"TODO": [], "IN_PROGRESS": [], ...}
            for t in self.tickets(team=team):
                cols[t.status].append(t.to_dict())             # drop each ticket into its column
            return cols

        def to_json(self, team: Optional[str] = None) -> str:
            """snapshot() as a JSON string."""
            return json.dumps(self.snapshot(team), indent=2)

        def all_done(self, team: str) -> bool:
            """True only if the team has at least one ticket and every ticket is DONE."""
            ts = self.tickets(team=team)
            return bool(ts) and all(t.status == Status.DONE for t in ts)

        def render(self, team: Optional[str] = None) -> str:
            """Human-readable board for printing."""
            lines = []
            for s, items in self.snapshot(team).items():
                lines.append(f"[{s}]")
                for t in items:
                    lines.append(f"  #{t['id']} {t['title']}  ({t['assignee'] or 'unassigned'})")
            return "\n".join(lines)

        # ---------- write API (every method checks who is calling) ----------
        def create_ticket(self, lead_name: str, title: str, description: str = "",
                          assignee: Optional[str] = None) -> Ticket:
            lead = self._require(lead_name, TeamLead, "create tickets")      # must be a TeamLead
            if not isinstance(title, str) or not title.strip():
                raise AgentForgeError("Ticket title cannot be empty")
            if assignee is not None:
                assignee = self._dev_in_team(assignee, lead.team).name       # must be a dev of the lead's team
            t = Ticket(id=next(self._ids), title=title.strip(), description=description,
                       team=lead.team, created_by=lead.name, assignee=assignee)
            self._log(t, lead.name, "created", f"assignee={assignee}")       # first history entry
            self._tickets[t.id] = t                                          # store it
            return t

        def assign(self, lead_name: str, ticket_id: int, developer: str) -> Ticket:
            lead = self._require(lead_name, TeamLead, "assign tickets")
            t = self.get(ticket_id)
            self._same_team(lead, t)                                         # can't assign another team's ticket
            if t.status == Status.DONE:
                raise AgentForgeError(f"Ticket #{t.id} is already DONE")
            previous = t.assignee
            t.assignee = self._dev_in_team(developer, lead.team).name
            self._log(t, lead.name, "assigned", t.assignee)
            if previous and _key(previous) != _key(t.assignee):          # it left someone's list
                self.forge.get_agent(previous)._note_change(
                    f"ticket #{t.id} '{t.title}' was reassigned to {t.assignee}")
            return t

        def take(self, dev_name: str, ticket_id: int) -> Ticket:
            dev = self._require(dev_name, Developer, "take tickets")         # must be a Developer
            t = self.get(ticket_id)
            self._same_team(dev, t)                                          # only own team's tickets
            if t.status != Status.TODO:
                raise AgentForgeError(f"Ticket #{t.id} is {t.status}, only TODO tickets can be taken")
            if t.assignee is None:                                           # nobody assigned yet
                raise PermissionDenied(
                    f"Ticket #{t.id} is unassigned; the TeamLead must assign it before {dev.name} can take it")
            if _key(t.assignee) != _key(dev.name):                           # assigned to someone else
                raise PermissionDenied(f"Ticket #{t.id} is assigned to {t.assignee}, not {dev.name}")
            return self._transition(t, dev, Status.IN_PROGRESS)              # TODO -> IN_PROGRESS

        def move(self, agent_name: str, ticket_id: int, new_status: str) -> Ticket:
            """Generic status change. Role rules are enforced here and in _transition."""
            self._check_status(new_status)
            agent = self.forge.get_agent(agent_name)
            t = self.get(ticket_id)
            self._same_team(agent, t)
            if isinstance(agent, Developer):
                if t.status == Status.TODO and new_status == Status.IN_PROGRESS:
                    return self.take(agent.name, ticket_id)   # same rules as picking it up
                if not t.assignee or _key(t.assignee) != _key(agent.name):   # must be my ticket
                    raise PermissionDenied(f"{agent.name} is not the assignee of ticket #{t.id}")
                if new_status == Status.DONE:                                # the key rule
                    raise PermissionDenied("Only a TeamLead can mark a ticket DONE")
            return self._transition(t, agent, new_status)

        def submit(self, dev_name: str, ticket_id: int, code: str) -> Ticket:
            dev = self._require(dev_name, Developer, "submit code")
            t = self.get(ticket_id)
            if not t.assignee or _key(t.assignee) != _key(dev.name):
                raise PermissionDenied(f"{dev.name} is not the assignee of ticket #{t.id}")
            if not isinstance(code, str):
                raise AgentForgeError("Submitted code must be a string")
            old_code = t.code
            t.code = code                                                    # attach the code
            try:
                return self.move(dev.name, ticket_id, Status.REVIEW)          # IN_PROGRESS -> REVIEW
            except Exception:
                t.code = old_code                                            # refused -> change nothing
                raise

        def approve(self, lead_name: str, ticket_id: int) -> Ticket:
            self._require(lead_name, TeamLead, "mark tickets DONE")
            return self.move(lead_name, ticket_id, Status.DONE)              # REVIEW -> DONE

        def reject(self, lead_name: str, ticket_id: int, feedback: str) -> Ticket:
            self._require(lead_name, TeamLead, "reject tickets")
            t = self.get(ticket_id)
            old_feedback, old_rejections = t.feedback, t.rejections
            t.feedback = feedback                                            # tell the dev what to fix
            t.rejections += 1
            try:
                return self.move(lead_name, ticket_id, Status.IN_PROGRESS)   # REVIEW -> IN_PROGRESS
            except Exception:
                t.feedback, t.rejections = old_feedback, old_rejections      # refused -> change nothing
                raise

        # ---------- internals (leading underscore = not meant to be called from outside) ----------
        def _transition(self, t: Ticket, agent: Agent, new_status: str) -> Ticket:
            """The single place where a status actually changes."""
            allowed = self.LEAD_MOVES if isinstance(agent, TeamLead) else self.DEV_MOVES
            if (t.status, new_status) not in allowed:
                raise PermissionDenied(
                    f"{agent.role} {agent.name} cannot move ticket #{t.id} from {t.status} to {new_status}")
            old, t.status = t.status, new_status                            # remember old, set new
            self._log(t, agent.name, "status", f"{old} -> {new_status}")
            return t

        def _require(self, name: str, role: type, action: str) -> Agent:
            """Look up the agent and make sure it has the required role."""
            agent = self.forge.get_agent(name)
            if not isinstance(agent, role):
                raise PermissionDenied(f"{agent.name} is a {agent.role}; only a {role.role} can {action}")
            return agent

        def _dev_in_team(self, name: str, team: str) -> Developer:
            """Make sure `name` is a Developer belonging to `team`."""
            dev = self.forge.get_agent(name)
            if not isinstance(dev, Developer):
                raise PermissionDenied(f"{dev.name} is not a Developer")
            if _key(dev.team) != _key(team):
                raise PermissionDenied(f"{dev.name} is in team '{dev.team}', not '{team}'")
            return dev

        @staticmethod                                   # doesn't need `self`
        def _same_team(agent: Agent, t: Ticket) -> None:
            if _key(agent.team) != _key(t.team):
                raise PermissionDenied(f"{agent.name} ({agent.team}) cannot touch ticket #{t.id} of team '{t.team}'")

        @staticmethod
        def _check_status(status: str) -> None:
            if status not in Status.ALL:
                raise AgentForgeError(f"Unknown status '{status}'. Use one of {Status.ALL}")

        def _log(self, t: Ticket, by: str, action: str, detail: str = "") -> None:
            """Append an audit entry: who did what, when - and tell the agents involved.
            Every ticket change passes through here, so this is the one place that keeps
            the agents' system prompts up to date."""
            t.history.append({"ts": _now(), "by": by, "action": action, "detail": detail})
            if action == "created":
                msg = f"ticket #{t.id} '{t.title}' was created" + (f" and assigned to {t.assignee}"
                                                                   if t.assignee else "")
            elif action == "assigned":
                msg = f"ticket #{t.id} '{t.title}' was assigned to {t.assignee}"
            else:                                                   # a status change
                msg = f"ticket #{t.id} '{t.title}' moved {detail}"
                if detail.endswith(Status.IN_PROGRESS) and t.feedback and by == self._lead_of(t):
                    msg += f" - changes requested: {t.feedback}"
            for name in {self._lead_of(t), t.assignee} - {None}:    # the team's lead + the assignee
                self.forge.get_agent(name)._note_change(msg)

        def _lead_of(self, t: Ticket) -> Optional[str]:
            return self.forge.teams[_key(t.team)].lead

    # ======================= MCP hub (nested class) ======================= #
    class MCPHub:
        """
        Knows which MCP servers exist and starts each one only when it is first needed.

            forge.mcp.call("system", "post_message", {...})
                 |  first time? -> start system_mcp.py as a process, connect, sync the roster
                 v
            system_mcp.py runs the tool and sends the result back
        """
        _HERE = os.path.dirname(os.path.abspath(__file__))   # folder this file is in

        # name -> server script. Add a new MCP server by adding one line here.
        REGISTRY = {
            "system": os.path.join(_HERE, "system_mcp.py"),   # group chat channels
            "chat": os.path.join(_HERE, "chat_mcp.py"),       # personal 1:1 chats
            "knowledge": os.path.join(_HERE, "knowledge_mcp.py"),  # shared documents (project_brain/)
            "memory": os.path.join(_HERE, "memory_mcp.py"),   # each agent's personal memory
        }

        # server error prefix -> our exception class
        _ERRORS = {"NotFound": NotFound, "PermissionDenied": PermissionDenied,
                   "RoleConflictError": RoleConflictError, "AgentForgeError": AgentForgeError}

        def __init__(self, forge: "AgentForge"):
            self.forge = forge
            self._connections: Dict[str, object] = {}         # server name -> MCPConnection (only loaded ones)

        def loaded(self) -> List[str]:
            """Names of the servers currently running, e.g. ['system']."""
            return list(self._connections)

        def server(self, name: str):
            """Return the connection for `name`, starting the server on first use."""
            if name not in self.REGISTRY:
                raise NotFound(f"Unknown MCP server '{name}'. Known: {list(self.REGISTRY)}")
            if name not in self._connections:
                from mcp_client import MCPConnection          # lazy import: `mcp` loads only now
                self.forge._say(f"loading MCP server '{name}'")
                conn = MCPConnection(name, self.REGISTRY[name], env={   # tell servers where the folders are
                    "AGENTFORGE_KNOWLEDGE_DIR": self.forge.knowledge_dir,
                    "AGENTFORGE_MEMORY_DIR": self.forge.memory_dir,
                })
                self._connections[name] = conn
                self._sync(conn)                              # tell it about existing teams/agents
            return self._connections[name]

        def call(self, server: str, tool: str, arguments: Optional[dict] = None):
            """Call one tool on one server; server-side errors become our exception classes."""
            from mcp_client import MCPConnectionError, MCPToolError
            for attempt in (1, 2):
                try:
                    return self.server(server).call(tool, arguments or {})
                except MCPToolError as e:
                    raise self._ERRORS.get(e.kind, AgentForgeError)(e.message) from None
                except MCPConnectionError as e:
                    self._evict(server)                   # dead server: forget it, restart on next use
                    if attempt == 2:
                        raise AgentForgeError(f"MCP server '{server}' is unavailable: {e}") from None

        def _evict(self, name: str) -> None:
            """Drop a broken connection so the next call starts a fresh server."""
            conn = self._connections.pop(name, None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        def tools(self, server: str) -> Dict[str, object]:
            """The tools a server offers (name -> MCP Tool with description + input_schema)."""
            return self.server(server).tools

        # ---- keeping the servers' roster in step with the forge ----
        def _sync(self, conn) -> None:
            """Send every existing team and agent to a server that just started."""
            for team in self.forge.teams.values():
                self._push_team(conn, team.name)
            for agent in self.forge.agents.values():
                self._push_agent(conn, agent)

        def team_added(self, team_name: str) -> None:        # called by AgentForge.team()
            for conn in self._connections.values():           # only servers already running
                self._push_team(conn, team_name)

        def agent_added(self, agent: "Agent") -> None:        # called by teamlead() / developer()
            for conn in self._connections.values():
                self._push_agent(conn, agent)

        @staticmethod
        def _push_team(conn, team_name: str) -> None:
            if "register_team" in conn.tools:                 # only servers that care about teams
                conn.call("register_team", {"team": team_name})

        @staticmethod
        def _push_agent(conn, agent: "Agent") -> None:
            if "register_agent" in conn.tools:
                conn.call("register_agent", {"name": agent.name, "team": agent.team})

        def close(self) -> None:
            """Stop every running server."""
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()

    # ======================= Graph ======================= #
    def __init__(self, name: str = "AgentForge", max_steps: int = 50, verbose: bool = True,
                 summarizer: Optional[Callable[[str, List[dict]], str]] = None,
                 knowledge_dir: Optional[str] = None, memory_dir: Optional[str] = None):
        self.name = name                         # shown in printed logs
        # summarizer(previous_summary, new_messages) -> new summary; used by channel_context / dm_context
        self.summarizer = summarizer or default_summarizer
        here = os.path.dirname(os.path.abspath(__file__))
        # shared documents every agent can search (read-only)
        self.knowledge_dir = os.path.abspath(knowledge_dir or os.path.join(here, "project_brain"))
        # one memory file per agent, kept between runs
        self.memory_dir = os.path.abspath(memory_dir or os.path.join(here, "agent_memory"))
        self.max_steps = max_steps               # loop guard for run()
        self.verbose = verbose                   # print each step or stay quiet
        self.teams: Dict[str, Team] = {}         # team key  -> Team
        self.agents: Dict[str, Agent] = {}       # agent key -> Agent (the nodes)
        self.edges: Dict[str, str] = {}          # move_to:   agent key -> next agent key / END
        self.branches: Dict[str, Callable] = {}  # branch_to: agent key -> router(state)
        self.trace: List[str] = []               # every printed step, kept for inspection
        self.board = AgentForge.KanbanBoard(self)   # core: one board per forge (always in memory)
        self.mcp = AgentForge.MCPHub(self)          # chat tools: servers start only when first used

    # ---- cleanup: stop MCP servers ----
    def close(self) -> None:
        self.mcp.close()

    def __enter__(self) -> "AgentForge":            # lets you write:  with AgentForge() as forge:
        return self

    def __exit__(self, *exc) -> None:               # ... and servers are stopped when the block ends
        self.close()

    # ---- building ----
    def team(self, team_name: str, description: str = "") -> Team:
        k = _key(team_name)
        if k == "general":                                   # "general" is the all-hands chat channel
            raise AgentForgeError(f"'{team_name}' is reserved")
        if k in self.teams:
            raise AgentForgeError(f"Team '{team_name}' already exists")
        self.teams[k] = Team(name=team_name.strip(), description=description)
        self.mcp.team_added(self.teams[k].name)              # running servers create its channel
        return self.teams[k]

    def teamlead(self, agent_name: str, team_name: str, fn: Optional[Callable] = None) -> TeamLead:
        team = self._get_team(team_name)                     # team must exist
        self._check_new_agent(agent_name, "TeamLead")        # name not used by anyone yet
        if team.lead:                                        # one lead per team
            raise RoleConflictError(f"Team '{team.name}' already has a TeamLead: {team.lead}")
        agent = TeamLead(self, agent_name.strip(), team.name, fn or default_teamlead)
        team.lead = agent.name
        self.agents[_key(agent_name)] = agent                # register as a node
        self.mcp.agent_added(agent)                          # running servers learn about the new agent
        return agent

    def developer(self, agent_name: str, team_name: str, fn: Optional[Callable] = None) -> Developer:
        team = self._get_team(team_name)
        self._check_new_agent(agent_name, "Developer")       # this is what blocks rahul-as-developer
        agent = Developer(self, agent_name.strip(), team.name, fn or default_developer)
        team.developers.append(agent.name)
        self.agents[_key(agent_name)] = agent
        self.mcp.agent_added(agent)
        return agent

    def move_to(self, src: str, dst: str) -> None:
        """Always go from src to dst (dst can be END)."""
        s = _key(self.get_agent(src).name)                   # src must already exist
        if s in self.branches:
            raise AgentForgeError(f"'{src}' already has a branch_to; use one or the other")
        self.edges[s] = END if dst == END else _key(dst)     # dst is checked later in validate()

    def branch_to(self, src: str, router: Callable[[dict], str]) -> None:
        """Pick the next agent at run time: router(state) -> agent name or END."""
        s = _key(self.get_agent(src).name)
        if s in self.edges:
            raise AgentForgeError(f"'{src}' already has a move_to; use one or the other")
        self.branches[s] = router

    # ---- lookup ----
    def get_agent(self, name: str) -> Agent:
        k = _key(name)
        if k not in self.agents:
            raise NotFound(f"Agent '{name}' does not exist")
        return self.agents[k]

    def validate(self) -> None:
        """Make sure every move_to points at a real agent (or END) before running."""
        for src, dst in self.edges.items():
            if dst != END and dst not in self.agents:
                raise NotFound(f"move_to target '{dst}' (from '{src}') is not an agent")

    # ---- running ----
    def run(self, state: Optional[dict] = None, start: Optional[str] = None,
            max_steps: Optional[int] = None) -> dict:
        if start is None:
            raise AgentForgeError("run() needs a start agent")
        self.validate()
        state = dict(state or {})                       # copy, so the caller's dict is never changed
        limit = max_steps if max_steps is not None else self.max_steps
        current = _key(self.get_agent(start).name)      # the agent we're about to run
        steps = 0

        while current != END:                           # same loop as your original Graph.run
            steps += 1
            if steps > limit:
                raise AgentForgeError(f"Stopped after {limit} steps - possible infinite loop")
            agent = self.agents[current]
            self._say(f"step {steps}: {agent}")
            update = agent.fn(agent, state) or {}       # call the node; None counts as "no changes"
            if not isinstance(update, dict):
                raise AgentForgeError(f"{agent.name} must return a dict, got {type(update).__name__}")
            state.update(update)                        # save its changes into shared state
            current = self._next(current, state)        # decide where to go next

        self._say("reached END")
        return state

    def _next(self, current: str, state: dict) -> str:
        if current in self.branches:                    # dynamic: ask the router
            nxt = self.branches[current](state)
            if nxt == END:
                return END
            return _key(self.get_agent(nxt).name)       # also checks the router returned a real agent
        if current in self.edges:                       # fixed hop
            return self.edges[current]
        raise AgentForgeError(f"No move_to / branch_to defined after '{self.agents[current].name}'")

    # ---- internals ----
    def _get_team(self, team_name: str) -> Team:
        k = _key(team_name)
        if k not in self.teams:
            raise NotFound(f"Team '{team_name}' does not exist. Create it with forge.team('{team_name}')")
        return self.teams[k]

    def _check_new_agent(self, agent_name: str, role: str) -> None:
        """The one-name-one-role rule."""
        k = _key(agent_name)
        if k == _key(END):                                   # an agent called END would break routing
            raise AgentForgeError("'END' is reserved")
        if k in self.agents:
            existing = self.agents[k]
            if existing.role == role:
                raise RoleConflictError(f"'{existing.name}' is already a {role} of team '{existing.team}'")
            raise RoleConflictError(
                f"'{existing.name}' is already a {existing.role} of team '{existing.team}' "
                f"and cannot also be added as {role}")

    def _say(self, line: str) -> None:
        self.trace.append(line)
        if self.verbose:
            print(f"[{self.name}] {line}")


# --------------------------------------------------------------------------- #
# Quick demo - mirrors your original example. Runs only with `python agentforge.py`.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    forge = AgentForge("demo")
    forge.team("frontend")
    forge.teamlead("rahul", "frontend")
    forge.developer("priya", "frontend")

    try:
        forge.developer("rahul", "frontend")            # should be blocked
    except RoleConflictError as e:
        print("blocked:", e)

    # lead -> developer -> lead -> ... until every frontend ticket is DONE
    forge.move_to("priya", "rahul")
    forge.branch_to("rahul", lambda s: END if forge.board.all_done("frontend") else "priya")

    print("MCP servers running before run():", forge.mcp.loaded())      # [] - nothing loaded yet
    final = forge.run({"task": "Build login page"}, start="rahul")
    print("MCP servers running after run(): ", forge.mcp.loaded())      # both, because agents chatted
    print(forge.board.render("frontend"))
    print("frontend channel:", [m["text"] for m in forge.get_agent("priya").read_channel()])
    print(final)
    forge.close()                                       # stop the MCP server processes
