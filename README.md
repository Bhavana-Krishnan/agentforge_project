# AgentForge

A small, from-scratch multi-agent framework in Python. A **TeamLead** creates tickets, a
**Developer** takes the tickets assigned to them and submits code, and only the TeamLead can
mark a ticket **DONE**. The agents coordinate through a Kanban board, team and private chat,
a shared knowledge folder, and personal memory. Each agent's system prompt rebuilds itself
whenever one of its tickets changes.

The framework runs with or without an LLM. The non-AI tests cover every rule; the AI test
uses OpenAI's GPT-6 Luna as the brain of each agent.

## Features

- **Teams and roles.** Each team has one TeamLead and any number of Developers. A name can
  hold only one role across the whole forge, so the same person can't be both.
- **Graph flow.** Wire agents together with `move_to` (fixed hop) and `branch_to` (decide at
  run time), then call `run()`. A step limit stops accidental infinite loops.
- **Kanban board** with the statuses `TODO → IN_PROGRESS → REVIEW → DONE`, and strict rules
  about who may move a ticket where (see [Ticket rules](#ticket-rules)).
- **Chat through MCP servers.** Group channels per team plus a `general` channel, and private
  one-to-one chats with an unread inbox.
- **Rolling chat summaries.** Agents summarize what has been said so far and only summarize
  new messages on later calls.
- **Shared knowledge ("plugin memory").** Drop PDF, Markdown or text files into
  `project_brain/`; agents search them with keyword ranking and cite the page they used.
- **Personal memory.** Each agent keeps private notes in its own file. Notes are saved only
  when something is worth keeping, duplicates are refused, and outdated notes are marked stale
  instead of deleted.
- **Self-updating system prompt.** When a ticket is created, assigned, moved, rejected or
  approved, the affected agents' prompts are refreshed with their open tickets, matching
  project knowledge, relevant memories and recent changes.
- **Lazy loading.** Each MCP server starts the first time something needs it. Code that only
  moves tickets starts no servers at all.

## Project layout

```
agentforge_project/
├── agentforge.py        # core: teams, agents, graph, Kanban board, system prompts
├── system_mcp.py        # MCP server: group chat channels and group summaries
├── chat_mcp.py          # MCP server: one-to-one chats, inbox and summaries
├── knowledge_mcp.py     # MCP server: searches the documents in project_brain/
├── memory_mcp.py        # MCP server: each agent's personal memory (saved to disk)
├── mcp_client.py        # bridge that lets synchronous code call the async MCP SDK
├── project_brain/       # shared knowledge folder (put your .pdf / .md / .txt files here)
├── agent_memory/        # created automatically; one JSON file per agent
├── test_without_ai.py   # 94 tests, no LLM needed
├── test_with_ai.py      # end-to-end test with GPT-6 Luna (skipped without an API key)
├── requirements.txt
└── README.md
```

Where new code belongs:
- **Ticket or graph logic** goes in `agentforge.py`.
- **Group chat** goes in `system_mcp.py`; **one-to-one chat** goes in `chat_mcp.py`.
- **Knowledge everyone shares** goes in `project_brain/` as a file.
- **What one agent learned** is saved through `memory_mcp.py`.

## Requirements

- Python 3.10 or newer (tested on 3.13)
- The packages in `requirements.txt`: `mcp>=2`, `openai` (only for the AI test) and `pypdf`
  (for reading PDFs)

## Installation

Use a virtual environment so the packages stay inside this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `python3 -m venv` fails on Ubuntu with "ensurepip is not available", run
`sudo apt install python3-venv` and try again.

## Quick start

```python
from agentforge import AgentForge, END

def lead_node(agent, state):
    # The first time through, create tickets for the goal.
    if not agent.forge.board.tickets(team=agent.team):
        agent.create_ticket("Login page", "Email and password form", assignee="priya")
        agent.say("Created the login ticket for priya.")
    # Approve anything waiting for review.
    for ticket in agent.review_queue():
        agent.approve(ticket.id)
    return {}

def developer_node(agent, state):
    # Take the tickets assigned to me, write the code, and submit it.
    for ticket in agent.available_tickets():
        agent.take(ticket.id)
    for ticket in agent.my_tickets(status="IN_PROGRESS"):
        agent.submit(ticket.id, "def login(): ...")
    return {}

with AgentForge("demo") as forge:            # stops the MCP servers when the block ends
    forge.team("frontend")
    forge.teamlead("rahul", "frontend", fn=lead_node)
    forge.developer("priya", "frontend", fn=developer_node)

    forge.move_to("priya", "rahul")          # developer hands over to the lead
    forge.branch_to("rahul", lambda s: END if forge.board.all_done("frontend") else "priya")

    forge.run({}, start="rahul")
    print(forge.board.render("frontend"))
    print("servers started:", forge.mcp.loaded())
```

Run `python demo.py` (or paste the code into a file) to see the flow. Nodes are plain
functions with the signature `fn(agent, state) -> dict`. The dictionary you return is merged
into the shared `state`.

## Ticket rules

These are enforced by the Kanban board, so they hold no matter which node calls them.

| Action | Who can do it | Notes |
|---|---|---|
| Create a ticket | TeamLead | Optionally assign it to a developer on the same team |
| Assign or reassign | TeamLead | Only to a developer on the lead's team |
| Take a ticket (`TODO → IN_PROGRESS`) | Developer | **Only if the lead assigned it to them.** Unassigned tickets are refused |
| Submit code (`IN_PROGRESS → REVIEW`) | Assigned developer | Attaches the code to the ticket |
| Pull back own work (`REVIEW → IN_PROGRESS`) | Assigned developer | |
| Reject with feedback (`REVIEW → IN_PROGRESS`) | TeamLead | Feedback is stored on the ticket |
| Approve (`REVIEW → DONE`) | TeamLead only | A developer can never mark a ticket DONE |

Every change is recorded in the ticket's `history`.

## Chat

```python
agent.say("Standup at 10")                  # group channel of my team
agent.say("Launch freeze", channel="general")
agent.read_channel()                        # messages in my team channel
agent.dm("rahul", "Blocked on the API")     # private one-to-one message
agent.read_dm("rahul")                      # full private conversation
agent.inbox()                               # unread private messages (marks them read)
```

Only team members can read or post in a team channel. Only the two people in a one-to-one
chat can read it.

### Chat summaries

An LLM agent forgets everything between calls, so it needs a summary of earlier chat:

```python
agent.channel_context()        # summary of my team channel
agent.dm_context("rahul")      # summary of my chat with rahul
```

Only messages newer than the saved summary are summarized, so asking again is cheap. One
summary per conversation is shared by everyone in it. The summarizer is set when you create the
forge:

```python
AgentForge("demo", summarizer=my_summarizer)   # my_summarizer(previous_summary, new_messages) -> str
```

Without one, a simple default keeps the latest 20 lines.

## Shared knowledge

Put any `.pdf`, `.md` or `.txt` file in `project_brain/`. It is indexed automatically, and
changes are picked up on the next search without restarting anything.

```python
hits = agent.lookup("how long do JWT tokens last", top_k=3)
hits[0]["document"], hits[0]["page"], hits[0]["text"]   # e.g. "api_notes.md", 1, "..."
```

- Search is keyword ranking (BM25). Lines that repeat on most pages, such as running headers,
  are removed first.
- Results always include the document and page, so agents can cite them.
- Use a different folder by passing `knowledge_dir=` to `AgentForge(...)`.

## Personal memory

Each agent has a private memory, saved to `agent_memory/<name>.json` so it survives between runs.

```python
agent.remember("Reviewer wants email validation on every form",
               kind="decision", source="ticket #3")       # {"stored": True, "id": 1}
agent.remember("reviewer wants email validation on every form!")   # {"stored": False, ...}
agent.recall("form validation")                           # best matches first
agent.forget(1, reason="Requirement changed")             # marked stale, kept as history
agent.memories(include_stale=True)                        # everything, including stale notes
```

Rules that keep memory small:
- **Nothing is saved automatically.** Your node code decides when to remember something.
- **Near-duplicates are refused.**
- **Stale notes are kept**, not deleted, so the history remains.
- **Kinds:** `fact` (verified), `decision` (approved), `hypothesis` (unverified), `observation`
  (happened once).

## The self-updating system prompt

```python
print(agent.system_prompt)
```

The prompt contains, in order:
1. who the agent is and its duties for its role
2. its open tickets, including any reviewer feedback
3. matching passages from `project_brain/`, with their source
4. its own relevant memories
5. the last five changes to its tickets

It is rebuilt only when something changed, and `agent.prompt_version` goes up each time the
text really changes. Pass `agent.system_prompt` as the `system` message to an LLM. The AI test
does this.

## Using an LLM (GPT-6 Luna)

Set your key in the environment, and optionally the model:

```bash
export OPENAI_API_KEY=sk-...
export AGENTFORGE_MODEL=gpt-6-luna          # default used by test_with_ai.py
```

If you use an OpenRouter key instead, also set:

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export AGENTFORGE_MODEL=openai/gpt-6-luna
```

Never commit your API key. Keep it in environment variables, not in the code.

## Running the tests

```bash
python -m unittest test_without_ai -v      # 94 tests, about one minute (starts real MCP servers)
python -m unittest test_with_ai -v         # needs OPENAI_API_KEY; skipped otherwise
```

The non-AI tests write to temporary folders, so they never touch your real `agent_memory/`.

## Extending the project

**Add a tool to an existing server.** Write a function with `@server.tool()`, give it type hints
and a docstring (the docstring becomes the tool's description), and raise
`ToolError("NotFound: ...")` for expected failures. Then add a one-line method on `Agent` that
calls it through `self.forge.mcp.call(...)`.

**Add a new MCP server.** Copy `chat_mcp.py` as a template, then add one line to
`AgentForge.MCPHub.REGISTRY`:

```python
"git": os.path.join(_HERE, "git_mcp.py"),
```

Call it with `forge.mcp.call("git", "<tool>", {...})`. It starts on first use.

**Try a server by hand.** Run `python system_mcp.py` to start it, or open it with the MCP
Inspector: `npx @modelcontextprotocol/inspector python system_mcp.py`.

## Design notes

- **Why MCP servers?** Chat, knowledge and memory are tools that can later be used by any MCP
  client, not only by this framework. Each one is also loaded only when needed.
- **Why is the core synchronous?** Agents take turns, so plain sequential code is easier to
  read and test. `mcp_client.py` runs the MCP SDK's async client on a background thread, so the
  rest of the code can call it like an ordinary function.
- **Why keyword search?** It needs no extra service and gives exact, citable matches. Meaning-
  based search can be added later as another server.

## Limitations

- The Kanban board and chat history live in memory. They are lost when the program ends.
  Personal memory is the only part saved to disk.
- Knowledge search is keyword-based. A question that uses different words from the document may
  not find it.
- The MCP servers run on your machine with no authentication. Use this for local experiments,
  not as a production service.

## License

Add a license before publishing, for example MIT. Make sure you have the right to publish
`project_brain/AgentForge.pdf`, or replace it with your own document.
