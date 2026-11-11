# AgentForge

A from-scratch multi-agent framework: TeamLead and Developer agents work through a Kanban
board. Their tools live in four MCP servers, each started only when it is first needed:
- group chat
- personal chat
- shared knowledge
- personal memory

Each agent also has a system prompt that rebuilds itself whenever one of its tickets changes.

## 1. What MCP is, in one paragraph

MCP (Model Context Protocol) is a standard way for a program to offer **tools** to another
program, and especially to an LLM. An **MCP server** is a separate process that says "here are
my tools, with names, descriptions and input schemas". It then runs a tool when asked. An
**MCP client** connects to the server, asks for the tool list (`list_tools`), and calls tools
(`call_tool`). Messages are JSON-RPC sent over stdin/stdout (the "stdio transport"). Because the
format is standard, the same server can be used by your framework, by Claude, by GPT-6 Luna
(once its tools are converted to the OpenAI format), or by any other MCP client.

## 2. What goes where

| File | What it holds | When it is loaded |
|---|---|---|
| `agentforge.py` | **Skeleton / core**: teams, TeamLead and Developer nodes, the graph (`move_to`, `branch_to`, `run`), the Kanban board (create, assign, take, move, submit, approve, reject), and `MCPHub` | Always (you import it) |
| `system_mcp.py` | **System MCP server**: group chat. A `general` channel plus one channel per team | Started the first time an agent calls `say()` or `read_channel()` |
| `chat_mcp.py` | **Chat MCP server**: personal 1:1 chats and the inbox | Started the first time an agent calls `dm()`, `read_dm()` or `inbox()` |
| `knowledge_mcp.py` | **Knowledge MCP server**: searches the shared documents in `project_brain/` | Started the first time an agent calls `lookup()` or reads its `system_prompt` |
| `memory_mcp.py` | **Memory MCP server**: each agent's personal memory, saved in `agent_memory/` | Started the first time an agent calls `remember()`, `recall()` or reads its `system_prompt` |
| `project_brain/` | **Shared knowledge folder** ("plugin memory"). Holds `AgentForge.pdf`; add any `.pdf`, `.md` or `.txt` | Read by `knowledge_mcp.py` |
| `agent_memory/` | One JSON file per agent (e.g. `kiran.json`). Created automatically; kept between runs | Read and written by `memory_mcp.py` |
| `mcp_client.py` | A generic bridge that lets plain synchronous code talk to the async MCP SDK | Imported only when the first server starts |
| `test_without_ai.py` | 94 tests, no LLM | — |
| `test_with_ai.py` | End-to-end test with GPT-6 Luna (skipped when there's no `OPENAI_API_KEY`) | — |

The rule for deciding where new code goes:
- **Core work goes in `agentforge.py`.** That means anything that changes tickets or the graph, including permissions.
- **Chat goes in an MCP server.** Group features go in `system_mcp.py`; one-to-one features go in `chat_mcp.py`.
- **Knowledge everyone shares** goes in `project_brain/` as a document.
- **What one agent learned** goes in that agent's memory, through `memory_mcp.py`.

## 2b. Knowledge, memory and the self-updating system prompt

These three pieces follow section 6 ("Project Brain") of `AgentForge.pdf`.

**Shared knowledge: `project_brain/` + `knowledge_mcp.py`**
- **What it is.** A folder of reference documents that every agent can search. It's read-only: agents can't change it.
- **Adding to it.** Drop another `.pdf`, `.md` or `.txt` file into the folder. It's picked up on the next search, with no restart needed.
- **How search works.** It's keyword search (BM25), following the document's advice to start with exact search before semantic search.
- **Sources.** Every result says which document and page it came from, so agents can cite it. Headers repeated on every page are stripped out first.
- **Usage:** `agent.lookup("ticket state machine")` returns `[{"document": "AgentForge.pdf", "page": 8, "text": ..., "score": ...}]`.

**Personal memory: `memory_mcp.py` + `agent_memory/`**
- **What it is.** Notes one agent decided to keep. They're private to that agent and saved to disk, so they survive into the next run.
- **Usage:**
  - `agent.remember(text, kind, source)` saves a note.
  - `agent.recall(query)` gets matching notes back.
  - `agent.forget(id, reason)` marks a note as stale.
- **The "only when needed" rules:**
  1. Nothing is saved automatically. The node code decides when, or the LLM does. With AI, `ai_reflect` asks GPT-6 Luna after a rejection whether there's a lesson for future tickets.
  2. Near-duplicates are refused. Remembering the same lesson twice stores it once.
  3. Old memories are marked stale, never deleted, so the history stays (section 6.4).
- **Kinds** follow the document's trust policy (section 6.3):
  - `fact`: verified
  - `decision`: approved
  - `hypothesis`: not verified
  - `observation`: happened once

**Self-updating system prompt: `agent.system_prompt`**
- **What triggers an update.** Every ticket change (created, assigned, reassigned, moved, rejected, approved) passes through `KanbanBoard._log`. That method tells the ticket's lead and assignee that something changed.
- **When it rebuilds.** The prompt is marked stale and rebuilt the next time it's read. It's always current, and code that never reads it never starts the knowledge or memory servers.
- **What it contains:**
  1. who the agent is and its role's duties
  2. its open tickets, with any reviewer feedback
  3. matching passages from `project_brain/`, with their source. These are looked up once per ticket and cached on the ticket. This is the "context packet on a new ticket" from section 6.1.
  4. the agent's own memories that relate to those tickets
  5. the last 5 ticket changes that concern it
- **Version counter.** `agent.prompt_version` goes up each time the text actually changes.
- **Using it with GPT-6 Luna.** Pass it as the `system` message, as `test_with_ai.py` does.

## 3. What happens when an agent posts a message

```
priya.say("PR is up")                                  agentforge.py  (Agent.say)
  -> forge.mcp.call("system", "post_message", {...})   agentforge.py  (MCPHub.call)
       first time? start system_mcp.py, list its tools,
                   send it every team + agent (register_team / register_agent)
  -> MCPConnection.call("post_message", {...})         mcp_client.py  (sync -> async bridge)
  -> JSON-RPC over stdin/stdout
  -> post_message(sender, channel, text)               system_mcp.py  (checks membership, stores message)
  <- message dict, or an error "PermissionDenied: ..."
  <- MCPHub turns the error prefix back into PermissionDenied / NotFound / AgentForgeError
```

Chat data lives **inside the server process**. The forge keeps no copy of it. When you call
`forge.close()`, the processes stop and the chat history is gone. This is the same as before,
when everything was in memory.

`forge.mcp.loaded()` tells you which servers are running at any moment.
- A run that never chats starts no servers.
- A run that only posts in group channels starts only `system`.

## 3b. Chat context: why agents need summaries

The servers keep every message for the life of the forge, so the chat itself never resets.
The **agent** is what forgets: each LLM call starts with no memory. Unless you put earlier
messages into the prompt, the agent starts from zero every time. Sending the whole history
gets long and expensive, so agents ask for a **rolling summary** instead:

```python
agent.channel_context()          # summary of my team channel  (system_mcp.py)
agent.channel_context("general") # summary of the general channel
agent.dm_context("rahul")        # summary of my 1:1 chat with rahul  (chat_mcp.py)
```

How it works:
1. The agent calls the server's `get_context` tool. The server returns the saved summary and only the messages posted **after** that summary was made.
2. If there are no new messages, the saved summary is returned straight away and no LLM call is made.
3. Otherwise the forge's `summarizer(previous_summary, new_messages)` merges the two.
4. The agent saves the result with `save_summary`. Next time, only newer messages count as new.

A few properties follow from this:
- **One summary per conversation.** There is one summary per channel and one per 1:1 conversation, shared by everyone who is in it. When one member builds it, the others reuse it.
- **Same privacy rules as messages.** Only members can get a team channel's context, and only the two people in a 1:1 chat can get that chat's context.
- **Where the summarizer runs.** The servers only *store* summaries. The summarizer runs on the agent side because it needs the LLM.
  - `AgentForge(summarizer=...)` sets it.
  - Without AI, the default one simply keeps the latest 20 lines.
  - `test_with_ai.py` passes `ai_summarizer`, which uses GPT-6 Luna.
- **How the AI nodes use it.** The lead includes the team channel summary when reviewing. The developer includes the summary of their chat with the lead when fixing a rejected ticket.
- **Lifetime.** Summaries live in the server processes, like messages, so they are gone after `forge.close()`. To keep context between separate runs, the servers would need to save their data to a file.

## 4. Step by step: using it

1. **Install** inside a virtual environment: `python3 -m venv .venv`, then `source .venv/bin/activate`, then `pip install -r requirements.txt` (installs `mcp>=2`, `openai`, `pypdf`). The code uses MCP SDK 2.x, where `FastMCP` was renamed `MCPServer`.
2. **Create teams and agents.**
   ```python
   forge = AgentForge("my-run")
   forge.team("frontend")
   forge.teamlead("rahul", "frontend")
   forge.developer("priya", "frontend")
   ```
   Adding `rahul` as a developer now raises `RoleConflictError`.
3. **Wire the graph.** Use `move_to(a, b)` for a fixed next hop, or `branch_to(a, router)`, where `router(state)` returns the next agent's name or `END`.
4. **Write the node functions.** Each one has the form `fn(agent, state) -> dict`, and you pass it as `fn=` when you create the agent. Inside it, use:
   - the lead's tools: `create_ticket`, `assign`, `review_queue`, `approve`, `reject`
   - the developer's tools: `available_tickets`, `take`, `submit`, `my_tickets`
   - chat (either role): `say`, `read_channel`, `dm`, `read_dm`, `inbox`
5. **Run:** `forge.run(state, start="rahul")`.
6. **Close:** call `forge.close()`, or use `with AgentForge() as forge:`. Either way the MCP processes are stopped.

## 5. The ticket rules (all enforced in `KanbanBoard`)

- **Creating, assigning and approving.** Only a TeamLead can create, assign, approve (move to DONE) or reject tickets, and only for their own team.
- **Taking a ticket.** A Developer can take a ticket **only if the lead assigned it to them**.
  - Taking an unassigned ticket raises `PermissionDenied`.
  - `available_tickets()` shows only the developer's own assigned TODO tickets.
- **Allowed moves.**
  - Developer: TODO → IN_PROGRESS, IN_PROGRESS → REVIEW, and pulling their own work back from REVIEW → IN_PROGRESS.
  - Lead: REVIEW → DONE, and REVIEW → IN_PROGRESS (reject with feedback).

## 6. Step by step: maintaining it

**Adding a new tool to an existing server** (for example, `delete_message` in `system_mcp.py`):
1. Write a function decorated with `@server.tool()`. Give it type hints and a docstring. The hints become the input schema and the docstring becomes the tool's description.
2. For an expected failure, raise `ToolError("PermissionDenied: ...")`. Use one of the prefixes `NotFound`, `PermissionDenied` or `AgentForgeError`.
3. If agents should have a shortcut for it, add a one-line method on `Agent` that calls `self.forge.mcp.call("system", "delete_message", {...})`.
4. Add a test in `test_without_ai.py`.

**Adding a whole new MCP server** (for example, `git_mcp.py`):
1. Copy `chat_mcp.py` as a template, then rename the server and replace its tools.
2. If the server needs to know who the agents are, give it a `register_agent` tool. If it needs the teams, also give it `register_team`. The hub automatically calls whichever of the two the server has.
3. Add one line to `MCPHub.REGISTRY` in `agentforge.py`: `"git": os.path.join(_HERE, "git_mcp.py")`.
4. Call it with `forge.mcp.call("git", "<tool>", {...})`. It starts on first use.

**Trying a server by itself:**
- `python system_mcp.py` starts the server and waits for MCP messages on stdin.
- To browse its tools and call them by hand, use any MCP client, such as the MCP Inspector (`npx @modelcontextprotocol/inspector python system_mcp.py`).

## 7. Running the tests

```
python test_without_ai.py          # ~60 s: chat, knowledge and memory tests start real server processes
export OPENAI_API_KEY=sk-...
python test_with_ai.py             # uses GPT-6 Luna
```

## 8. A natural next step

Right now the node code decides when an agent chats. Each server can already describe its
tools (`forge.mcp.tools("chat")` returns each tool's name, description and `input_schema`). You
could convert that list into the OpenAI `tools=[...]` format and pass it to GPT-6 Luna. The
model would then decide for itself when to post or DM, and your loop would run whatever tool
calls it returns through `forge.mcp.call(...)`. This is where MCP really pays off.
