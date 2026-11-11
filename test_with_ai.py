"""
End-to-end test for AgentForge with real LLM agents (OpenAI GPT-6 Luna).

The TeamLead uses the LLM to split a goal into tickets and to review code.
The Developer uses the LLM to write the code for each ticket.

Setup:
    pip install openai "mcp>=2"
    export OPENAI_API_KEY=sk-...
    export AGENTFORGE_MODEL=gpt-6-luna        # optional, this is the default

    # If your key is an OpenRouter key instead of an OpenAI key:
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export AGENTFORGE_MODEL=openai/gpt-6-luna

Run:
    python -m unittest test_with_ai.py -v
Tests are skipped automatically when the key or the SDK is missing.
"""

import json          # turn the LLM's JSON text into Python lists / dicts
import os            # read environment variables (API key, model name)
import re            # regular expressions, used to dig JSON / code out of LLM replies
import shutil        # delete the temporary memory folder after a test
import tempfile      # make a temporary memory folder for a test
import unittest      # Python's built-in test framework

from agentforge import END, AgentForge, Status   # our framework

# Which model to call. Read from the environment so you can switch without editing code.
MODEL = os.getenv("AGENTFORGE_MODEL") or "gpt-6-luna"

# Safety valve: after a ticket is rejected this many times, the lead approves it anyway.
# Without this, a picky LLM reviewer could reject forever and the test would never finish.
MAX_REJECTIONS = 2

# Try to create the OpenAI client. If the SDK isn't installed or there is no key,
# _client stays None and the test class below is skipped instead of crashing.
try:
    from openai import OpenAI                                   # official OpenAI SDK
    # OpenAI() reads OPENAI_API_KEY (and OPENAI_BASE_URL, if set) from the environment.
    _client = OpenAI(timeout=60) if os.getenv("OPENAI_API_KEY") else None   # 60 s: never hang
except ImportError:                                             # `pip install openai` not done
    _client = None


# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #
class LLMTruncated(ValueError):
    """The model hit max_completion_tokens before finishing its reply."""


def llm(system: str, prompt: str, max_tokens: int = 4000) -> str:
    """Send one system + user message to GPT-6 Luna and return the reply text."""
    resp = _client.chat.completions.create(       # Chat Completions endpoint
        model=MODEL,                              # e.g. "gpt-6-luna"
        max_completion_tokens=max_tokens,         # cap on reply length (includes any hidden reasoning)
        messages=[
            {"role": "system", "content": system},   # who the agent is
            {"role": "user", "content": prompt},     # what we want it to do now
        ],
    )
    if resp.choices[0].finish_reason == "length":  # cut off by max_tokens: don't trust a half reply
        raise LLMTruncated(f"Reply was truncated at {max_tokens} tokens")
    # The reply lives at choices[0].message.content; `or ""` guards against None.
    return (resp.choices[0].message.content or "").strip()


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (handles ``` fences)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):                             # try every "[" or "{" as a start point
        if ch in "[{":
            try:
                value, _end = decoder.raw_decode(text, i)     # parses ONE balanced JSON value
                return value
            except ValueError:
                continue                                      # not JSON here (e.g. "[1]" in prose)
    raise ValueError(f"No JSON in reply: {text[:200]}")


def strip_fences(text: str) -> str:
    """If the reply is wrapped in ```python ... ```, return only the code inside."""
    m = re.search(r"```[\w+-]*[ \t]*\n(.*?)```", text, re.S)  # a closed fenced block (first one)
    if m:
        return m.group(1).strip()
    m = re.search(r"```[\w+-]*[ \t]*\n(.*)\Z", text, re.S)     # unclosed fence (truncated reply)
    if m:
        return m.group(1).strip()
    m = re.fullmatch(r"```[\w+-]*[ \t]+(.*?)```", text.strip(), re.S)   # one-line fence
    return (m.group(1) if m else text).strip()                # inside of the fence, or the raw text


# --------------------------------------------------------------------------- #
# AI summarizer  -  AgentForge calls this as summarizer(previous_summary, new_messages)
# --------------------------------------------------------------------------- #
def ai_summarizer(previous: str, new_messages: list) -> str:
    """Fold new chat messages into the running summary with GPT-6 Luna."""
    new_text = "\n".join(f"{m['sender']}: {m['text']}" for m in new_messages)
    summary = llm(
        system="You keep short running summaries of a software team's chat.",
        prompt=(
            f"Summary so far:\n{previous or '(nothing yet)'}\n\n"
            f"New messages:\n{new_text}\n\n"
            "Write the updated summary in at most 6 bullet points: decisions made, "
            "open questions, who is doing what. Reply with the bullets only."
        ),
        max_tokens=1000,
    )
    return summary or previous or new_text[-500:]             # never return an empty summary


# --------------------------------------------------------------------------- #
# AI node functions  -  AgentForge calls these as fn(agent, state)
# --------------------------------------------------------------------------- #
def ai_teamlead(agent, state):
    board = agent.forge.board            # shortcut to the kanban board

    # ---- 1. Plan: turn the goal into tickets (only the first time the lead runs) ----
    if not board.tickets(team=agent.team):              # team has no tickets yet
        # Go and get knowledge when needed: check the shared documents for anything about this goal.
        docs = agent.lookup(state["goal"], top_k=2)
        reference = "\n".join(f"[{d['document']}, p.{d['page']}] {d['text'][:400]}" for d in docs)
        reply = llm(
            system=agent.system_prompt,                 # self-updating: role, tickets, knowledge, memory
            prompt=(
                f"Goal: {state['goal']}\n\n"
                + (f"Project documents that may help:\n{reference}\n\n" if reference else "")
                + f"Split this into {state.get('max_tickets', 2)} small, independent coding tickets. "
                'Reply with ONLY a JSON array: [{"title": "...", "description": "..."}]'
            ),
        )
        devs = agent.forge.teams[agent.team.lower()].developers   # names of developers in this team
        # Take at most max_tickets items; enumerate gives i = 0, 1, 2 ... for round-robin.
        plan = extract_json(reply)
        if isinstance(plan, dict):                              # {"tickets": [...]} -> the list
            plan = plan.get("tickets", [])
        items = [it for it in plan if isinstance(it, dict)
                 and isinstance(it.get("title"), str) and it["title"].strip()] if isinstance(plan, list) else []
        if not items:                                           # fail fast instead of re-planning forever
            raise ValueError(f"The plan has no usable tickets: {reply[:200]}")
        for i, item in enumerate(items[: state.get("max_tickets", 2)]):
            t = agent.create_ticket(item["title"], str(item.get("description") or ""),
                                    assignee=devs[i % len(devs)])  # 0->kiran, 1->asha, 2->kiran ...
            agent.say(f"Created #{t.id} '{t.title}' for {t.assignee}")   # post in team channel

    # ---- 2. Review: look at every ticket waiting in REVIEW ----
    queue = agent.review_queue()
    team_chat = agent.channel_context() if queue else ""      # what the team said so far (summary)
    for t in queue:
        if t.rejections >= MAX_REJECTIONS:                     # rejected enough - accept it
            agent.approve(t.id)                                # REVIEW -> DONE
            agent.say(f"Approved #{t.id} after {t.rejections} revisions.")
            continue                                           # next ticket, skip the LLM call
        reply = llm(
            system=agent.system_prompt + "\n\nAs a reviewer be pragmatic: approve working code; "
                   "reject only for real bugs or missing requirements.",
            prompt=(
                f"Ticket: {t.title}\n{t.description}\n\n"
                + (f"Team chat so far:\n{team_chat}\n\n" if team_chat else "")
                + f"Code:\n{t.code}\n\n"
                'Reply with ONLY JSON: {"approve": true/false, "feedback": "..."}'
            ),
            max_tokens=1500,
        )
        verdict = extract_json(reply)                          # {"approve": ..., "feedback": ...}
        if not isinstance(verdict, dict):
            raise ValueError(f"Reviewer reply is not a JSON object: {reply[:200]}")
        if verdict.get("approve") is True:
            agent.approve(t.id)                                # REVIEW -> DONE
            agent.say(f"Approved #{t.id} '{t.title}'.")
        else:
            agent.reject(t.id, verdict.get("feedback") or "Please revise.")    # REVIEW -> IN_PROGRESS
            agent.dm(t.assignee, f"#{t.id} needs changes: {t.feedback}")      # solo chat to the dev

    # Whatever we return is merged into the shared state. Here: count how often the lead ran.
    return {"reviews": state.get("reviews", 0) + 1}


def ai_developer(agent, state):
    # ---- 1. Pick up every TODO ticket the lead assigned to me -> IN_PROGRESS ----
    for t in agent.available_tickets():
        agent.take(t.id)

    code_by_ticket = dict(state.get("code") or {})      # copy existing code so we don't lose it
    # ---- 2. Write code for every ticket I have IN_PROGRESS (new ones and rejected ones) ----
    lead = agent.forge.teams[agent.team.lower()].lead
    for t in agent.my_tickets(Status.IN_PROGRESS):
        prompt = f"Ticket: {t.title}\n{t.description}\n"
        if t.feedback:                                  # it was rejected - show old code + feedback
            ai_reflect(agent, t)                        # worth remembering for next time? (LLM decides)
            prompt += f"\nPrevious code:\n{t.code}\n\nReviewer feedback: {t.feedback}\n"
            history = agent.dm_context(lead)            # summary of my chat with the lead so far
            if history:
                prompt += f"\nYour chat with {lead} so far:\n{history}\n"
        prompt += f"\nWrite the {state.get('language', 'Python')} code. Reply with one code block only."
        # agent.system_prompt already holds this ticket, matching knowledge and my memories
        code = strip_fences(llm(system=agent.system_prompt, prompt=prompt))
        if not code.strip():
            raise ValueError(f"The model returned no code for ticket #{t.id}")
        agent.submit(t.id, code)                        # save code + move IN_PROGRESS -> REVIEW
        code_by_ticket[t.id] = code
        agent.dm(lead, f"#{t.id} ready for review.")    # tell the lead in solo chat
    return {"code": code_by_ticket}                     # merged into state["code"]


def ai_reflect(agent, ticket):
    """After a rejection, let the LLM decide whether there is a lasting lesson to save.
    Most of the time there isn't - then nothing is written. That is the 'only when needed' rule."""
    try:
        reply = _reflect_llm(agent, ticket)
        decision = extract_json(reply)
    except ValueError:                                  # truncated / not JSON - save nothing
        return None
    return _save_lesson(agent, ticket, decision)


def _reflect_llm(agent, ticket):
    return llm(
        system=agent.system_prompt,
        prompt=(
            f"Your code for ticket #{ticket.id} '{ticket.title}' was rejected.\n"
            f"Reviewer feedback: {ticket.feedback}\n\n"
            "Is there a lesson that will matter on FUTURE tickets too (not just this one)? "
            "Do not repeat anything already under 'What you remember'.\n"
            'Reply with ONLY JSON: {"remember": "<one short sentence>" or null, '
            '"kind": "observation" | "decision" | "fact" | "hypothesis"}'
        ),
        max_tokens=500,
    )


def _save_lesson(agent, ticket, decision):
    if not isinstance(decision, dict) or not isinstance(decision.get("remember"), str) \
            or not decision["remember"].strip():
        return None                                     # nothing worth saving (or an odd reply)
    kind = decision.get("kind")
    if kind not in ("observation", "decision", "fact", "hypothesis"):
        kind = "observation"
    return agent.remember(decision["remember"], kind=kind, source=f"ticket #{ticket.id}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_client, "Set OPENAI_API_KEY and `pip install openai` to run AI tests")
class TestAgentForgeWithAI(unittest.TestCase):

    def build(self) -> AgentForge:
        """One backend team: lead meera, developers kiran and asha, all LLM-driven."""
        memory_dir = tempfile.mkdtemp()                 # test memories go to a throwaway folder
        self.addCleanup(shutil.rmtree, memory_dir, True)
        f = AgentForge("ai-test", max_steps=20, summarizer=ai_summarizer,   # real LLM summaries
                       memory_dir=memory_dir)           # knowledge_dir defaults to project_brain/
        self.addCleanup(f.close)                       # stop the MCP servers after the test
        f.team("backend")
        f.teamlead("meera", "backend", fn=ai_teamlead)
        f.developer("kiran", "backend", fn=ai_developer)
        f.developer("asha", "backend", fn=ai_developer)
        f.move_to("kiran", "asha")                     # kiran always hands over to asha
        f.move_to("asha", "meera")                     # asha always hands over to the lead
        # After the lead: stop if every ticket is DONE, otherwise send work back to kiran.
        f.branch_to("meera", lambda s: END if f.board.all_done("backend") else "kiran")
        return f

    def test_goal_to_done_tickets(self):
        f = self.build()
        out = f.run({"goal": "A tiny in-memory user store with add_user(name, email) "
                             "and get_user(email), plus email validation.",
                     "max_tickets": 2, "language": "Python"},
                    start="meera")

        tickets = f.board.tickets(team="backend")
        print("\n" + f.board.render("backend"))       # show the final board in the test output

        self.assertGreaterEqual(len(tickets), 1)                    # lead made tickets
        self.assertTrue(f.board.all_done("backend"))                # all of them finished
        for t in tickets:
            self.assertTrue(t.code.strip(), f"ticket #{t.id} has no code")   # devs wrote code
            self.assertIn(t.assignee, {"kiran", "asha"})                   # assigned to real devs
            self.assertEqual(t.history[-1]["by"], "meera")                 # only the lead closed it
        self.assertTrue(f.get_agent("kiran").read_channel())  # lead posted in the group channel
        self.assertTrue(f.get_agent("meera").read_dm("kiran"))  # solo chat (chat_mcp.py) was used
        self.assertEqual(set(f.mcp.loaded()), {"system", "chat", "knowledge", "memory"})
        for t in tickets:
            self.assertIsNotNone(t.knowledge)                 # each ticket was matched against project_brain/
        print("\nkiran's memories:", [m["text"] for m in f.get_agent("kiran").memories()])
        summary = f.get_agent("kiran").channel_context()             # LLM summary of the team channel
        print("\nTeam channel summary:\n" + summary)
        self.assertTrue(summary.strip())
        self.assertGreaterEqual(out["reviews"], 2)             # lead ran at least plan + review

    def test_ai_developer_still_cannot_close(self):
        f = self.build()
        lead = f.get_agent("meera")
        t = lead.create_ticket("Add function", "def add(a, b) returning a + b", assignee="kiran")
        ai_developer(f.get_agent("kiran"), {})          # call the node directly, outside run()
        self.assertEqual(t.status, Status.REVIEW)       # dev submitted it
        self.assertIn("def add", t.code)                # and the LLM wrote the function
        with self.assertRaises(Exception):              # but the dev still can't mark it DONE
            f.get_agent("kiran").move(t.id, Status.DONE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
