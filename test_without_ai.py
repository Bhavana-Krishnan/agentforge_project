"""
Tests for AgentForge with no AI involved.
Run:  python -m unittest test_without_ai.py -v      (or: python test_without_ai.py)

Chat tests start the real MCP servers (system_mcp.py / chat_mcp.py) as child processes,
so they take about a second each. Every forge is closed after its test, which stops them.
"""

import os
import shutil
import tempfile
import unittest

from agentforge import (
    END, AgentForge, AgentForgeError, Developer, NotFound, PermissionDenied,
    RoleConflictError, Status, TeamLead,
)


class ForgeTest(unittest.TestCase):
    """Base class: makes forges and makes sure their MCP servers are stopped afterwards."""

    def new_forge(self, name: str = "test", **kwargs) -> AgentForge:
        if "memory_dir" not in kwargs:            # tests never write into the real agent_memory/
            kwargs["memory_dir"] = self.temp_dir()
        f = AgentForge(name, verbose=False, **kwargs)
        self.addCleanup(f.close)                  # runs after the test, pass or fail
        return f

    def temp_dir(self, files: dict | None = None) -> str:
        """A throwaway folder (deleted after the test), optionally with some files in it."""
        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path, True)
        for name, text in (files or {}).items():
            with open(os.path.join(path, name), "w", encoding="utf-8") as f:
                f.write(text)
        return path

    def build(self) -> AgentForge:
        """Two teams: frontend (rahul -> priya, arjun) and backend (meera -> kiran)."""
        f = self.new_forge()
        f.team("frontend")
        f.team("backend")
        f.teamlead("rahul", "frontend")
        f.developer("priya", "frontend")
        f.developer("arjun", "frontend")
        f.teamlead("meera", "backend")
        f.developer("kiran", "backend")
        return f


# --------------------------------------------------------------------------- #
class TestTeamsAndRoles(ForgeTest):
    def setUp(self):
        self.f = self.build()

    def test_agents_have_correct_roles(self):
        self.assertIsInstance(self.f.get_agent("rahul"), TeamLead)
        self.assertIsInstance(self.f.get_agent("priya"), Developer)
        self.assertEqual(self.f.get_agent("kiran").team, "backend")

    def test_team_membership(self):
        t = self.f.teams["frontend"]
        self.assertEqual(t.lead, "rahul")
        self.assertEqual(t.developers, ["priya", "arjun"])

    def test_teamlead_cannot_be_added_as_developer(self):
        with self.assertRaises(RoleConflictError):
            self.f.developer("rahul", "frontend")
        with self.assertRaises(RoleConflictError):
            self.f.developer("rahul", "backend")

    def test_developer_cannot_be_added_as_teamlead(self):
        self.f.team("mobile")
        with self.assertRaises(RoleConflictError):
            self.f.teamlead("priya", "mobile")

    def test_names_are_case_insensitive(self):
        with self.assertRaises(RoleConflictError):
            self.f.developer("Rahul", "frontend")
        with self.assertRaises(RoleConflictError):
            self.f.developer("  PRIYA ", "backend")

    def test_one_teamlead_per_team(self):
        with self.assertRaises(RoleConflictError):
            self.f.teamlead("vikram", "frontend")

    def test_same_lead_cannot_lead_two_teams(self):
        self.f.team("data")
        with self.assertRaises(RoleConflictError):
            self.f.teamlead("rahul", "data")

    def test_unknown_team(self):
        with self.assertRaises(NotFound):
            self.f.developer("sam", "devops")

    def test_duplicate_team(self):
        with self.assertRaises(AgentForgeError):
            self.f.team("Frontend")

    def test_reserved_names(self):
        with self.assertRaises(AgentForgeError):
            self.f.team("general")
        with self.assertRaises(AgentForgeError):
            self.f.developer("END", "frontend")


# --------------------------------------------------------------------------- #
class TestKanban(ForgeTest):
    def setUp(self):
        self.f = self.build()
        self.rahul = self.f.get_agent("rahul")
        self.priya = self.f.get_agent("priya")
        self.arjun = self.f.get_agent("arjun")
        self.kiran = self.f.get_agent("kiran")
        self.meera = self.f.get_agent("meera")

    def test_lead_creates_ticket(self):
        t = self.rahul.create_ticket("Login page", "email + password")
        self.assertEqual(t.status, Status.TODO)
        self.assertEqual(t.team, "frontend")
        self.assertEqual(t.created_by, "rahul")
        self.assertIsNone(t.assignee)

    def test_developer_cannot_create_ticket(self):
        with self.assertRaises(PermissionDenied):
            self.f.board.create_ticket("priya", "Sneaky ticket")

    def test_empty_title_rejected(self):
        with self.assertRaises(AgentForgeError):
            self.rahul.create_ticket("   ")

    def test_cannot_assign_to_other_team_or_to_a_lead(self):
        with self.assertRaises(PermissionDenied):
            self.rahul.create_ticket("X", assignee="kiran")
        with self.assertRaises(PermissionDenied):
            self.rahul.create_ticket("X", assignee="meera")

    def test_developer_sees_only_own_team_tickets(self):
        self.rahul.create_ticket("Login page")
        self.meera.create_ticket("Auth API")
        self.assertEqual([t.title for t in self.priya.board()], ["Login page"])
        self.assertEqual([t.title for t in self.kiran.board()], ["Auth API"])

    # ---- the "assigned only" rule ----
    def test_developer_cannot_take_unassigned_ticket(self):
        t = self.rahul.create_ticket("Login page")
        with self.assertRaises(PermissionDenied):
            self.priya.take(t.id)
        with self.assertRaises(PermissionDenied):          # same rule through move()
            self.priya.move(t.id, Status.IN_PROGRESS)
        self.assertEqual(t.status, Status.TODO)
        self.assertIsNone(t.assignee)                       # taking did not claim it

    def test_available_tickets_only_shows_my_assigned_ones(self):
        self.rahul.create_ticket("Unassigned")
        self.rahul.create_ticket("For arjun", assignee="arjun")
        mine = self.rahul.create_ticket("For priya", assignee="priya")
        self.assertEqual([t.id for t in self.priya.available_tickets()], [mine.id])

    def test_assign_then_take(self):
        t = self.rahul.create_ticket("Cards")
        self.rahul.assign(t.id, "priya")
        self.priya.take(t.id)
        self.assertEqual(t.status, Status.IN_PROGRESS)

    def test_cannot_take_ticket_assigned_to_someone_else(self):
        t = self.rahul.create_ticket("Sidebar", assignee="priya")
        with self.assertRaises(PermissionDenied):
            self.arjun.take(t.id)
        self.priya.take(t.id)

    # ---- status flow ----
    def test_full_happy_path(self):
        t = self.rahul.create_ticket("Login page", assignee="priya")
        self.priya.take(t.id)
        self.assertEqual(t.status, Status.IN_PROGRESS)
        self.priya.submit(t.id, "<form>...</form>")
        self.assertEqual(t.status, Status.REVIEW)
        self.rahul.approve(t.id)
        self.assertEqual(t.status, Status.DONE)
        self.assertEqual(len(t.history), 4)  # created + 3 status changes

    def test_developer_can_move_inprogress_and_review(self):
        t = self.rahul.create_ticket("Navbar", assignee="priya")
        self.priya.move(t.id, Status.IN_PROGRESS)
        self.priya.move(t.id, Status.REVIEW)
        self.priya.move(t.id, Status.IN_PROGRESS)  # pull back
        self.assertEqual(t.status, Status.IN_PROGRESS)

    def test_developer_cannot_mark_done(self):
        t = self.rahul.create_ticket("Navbar", assignee="priya")
        self.priya.take(t.id)
        self.priya.submit(t.id, "code")
        with self.assertRaises(PermissionDenied):
            self.priya.move(t.id, Status.DONE)
        with self.assertRaises(PermissionDenied):
            self.f.board.approve("priya", t.id)
        self.assertEqual(t.status, Status.REVIEW)

    def test_lead_can_only_close_from_review(self):
        t = self.rahul.create_ticket("Footer", assignee="priya")
        with self.assertRaises(PermissionDenied):
            self.rahul.approve(t.id)  # still TODO
        self.priya.take(t.id)
        with self.assertRaises(PermissionDenied):
            self.rahul.approve(t.id)  # IN_PROGRESS

    def test_other_teams_lead_cannot_close(self):
        t = self.rahul.create_ticket("Footer", assignee="priya")
        self.priya.take(t.id)
        self.priya.submit(t.id, "code")
        with self.assertRaises(PermissionDenied):
            self.meera.approve(t.id)

    def test_other_team_developer_cannot_take(self):
        t = self.rahul.create_ticket("Sidebar", assignee="priya")
        with self.assertRaises(PermissionDenied):
            self.kiran.take(t.id)

    def test_only_assignee_can_move_or_submit(self):
        t = self.rahul.create_ticket("Sidebar", assignee="priya")
        self.priya.take(t.id)
        with self.assertRaises(PermissionDenied):
            self.arjun.move(t.id, Status.REVIEW)
        with self.assertRaises(PermissionDenied):
            self.arjun.submit(t.id, "code")

    def test_cannot_take_twice(self):
        t = self.rahul.create_ticket("Sidebar", assignee="priya")
        self.priya.take(t.id)
        with self.assertRaises(AgentForgeError):
            self.priya.take(t.id)

    def test_reject_sends_back_with_feedback(self):
        t = self.rahul.create_ticket("Login", assignee="priya")
        self.priya.take(t.id)
        self.priya.submit(t.id, "v1")
        self.rahul.reject(t.id, "add validation")
        self.assertEqual(t.status, Status.IN_PROGRESS)
        self.assertEqual(t.feedback, "add validation")
        self.assertEqual(t.rejections, 1)
        self.priya.submit(t.id, "v2")
        self.rahul.approve(t.id)
        self.assertEqual(t.code, "v2")

    def test_assign_and_reassign(self):
        t = self.rahul.create_ticket("Cards")
        self.rahul.assign(t.id, "arjun")
        self.assertEqual(t.assignee, "arjun")
        self.rahul.assign(t.id, "priya")
        self.assertEqual(t.assignee, "priya")
        with self.assertRaises(PermissionDenied):
            self.f.board.assign("priya", t.id, "arjun")    # developers can't assign

    def test_unknown_ticket_and_status(self):
        with self.assertRaises(NotFound):
            self.f.board.get(999)
        t = self.rahul.create_ticket("X", assignee="priya")
        with self.assertRaises(AgentForgeError):
            self.priya.move(t.id, "BLOCKED")

    def test_snapshot_and_json(self):
        a = self.rahul.create_ticket("A", assignee="priya")
        self.rahul.create_ticket("B")
        self.priya.take(a.id)
        snap = self.f.board.snapshot("frontend")
        self.assertEqual(len(snap[Status.TODO]), 1)
        self.assertEqual(snap[Status.IN_PROGRESS][0]["title"], "A")
        self.assertIn('"IN_PROGRESS"', self.f.board.to_json())

    def test_all_done(self):
        self.assertFalse(self.f.board.all_done("frontend"))  # no tickets yet
        t = self.rahul.create_ticket("A", assignee="priya")
        self.priya.take(t.id)
        self.priya.submit(t.id, "c")
        self.assertFalse(self.f.board.all_done("frontend"))
        self.rahul.approve(t.id)
        self.assertTrue(self.f.board.all_done("frontend"))

    def test_kanban_never_loads_mcp(self):
        t = self.rahul.create_ticket("A", assignee="priya")
        self.priya.take(t.id)
        self.priya.submit(t.id, "c")
        self.rahul.approve(t.id)
        self.assertEqual(self.f.mcp.loaded(), [])            # core work needs no MCP server


# --------------------------------------------------------------------------- #
class TestLazyMCP(ForgeTest):
    def setUp(self):
        self.f = self.build()

    def test_nothing_loaded_at_start(self):
        self.assertEqual(self.f.mcp.loaded(), [])

    def test_group_chat_loads_only_system(self):
        self.f.get_agent("rahul").say("Standup at 10")
        self.assertEqual(self.f.mcp.loaded(), ["system"])

    def test_solo_chat_loads_only_chat(self):
        self.f.get_agent("priya").dm("rahul", "hi")
        self.assertEqual(self.f.mcp.loaded(), ["chat"])

    def test_each_server_has_only_its_own_tools(self):
        system = set(self.f.mcp.tools("system"))
        chat = set(self.f.mcp.tools("chat"))
        self.assertEqual(system, {"register_team", "register_agent", "list_channels",
                                  "channel_members", "post_message", "read_channel",
                                  "get_context", "save_summary"})
        self.assertEqual(chat, {"register_agent", "send_dm", "read_dm", "inbox",
                                "get_context", "save_summary"})

    def test_tools_describe_themselves(self):
        tool = self.f.mcp.tools("chat")["send_dm"]
        self.assertTrue(tool.description)                                # docstring -> description
        self.assertEqual(set(tool.input_schema["required"]), {"sender", "to", "text"})

    def test_agents_and_teams_added_after_loading_are_synced(self):
        self.f.get_agent("rahul").say("first")                          # system server now running
        self.f.team("mobile")
        self.f.teamlead("vikram", "mobile")
        self.f.get_agent("vikram").say("mobile team here")              # new team + agent known
        self.assertIn("mobile", self.f.mcp.call("system", "list_channels")["channels"])

    def test_unknown_server(self):
        with self.assertRaises(NotFound):
            self.f.mcp.call("weather", "forecast")

    def test_close_stops_servers(self):
        self.f.get_agent("rahul").say("x")
        self.f.close()
        self.assertEqual(self.f.mcp.loaded(), [])


# --------------------------------------------------------------------------- #
class TestComms(ForgeTest):
    def setUp(self):
        self.f = self.build()
        self.rahul = self.f.get_agent("rahul")
        self.priya = self.f.get_agent("priya")
        self.kiran = self.f.get_agent("kiran")

    # group chat (system_mcp.py)
    def test_team_group_channel(self):
        self.rahul.say("Standup at 10")
        msgs = self.priya.read_channel()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["sender"], "rahul")
        self.assertEqual(msgs[0]["kind"], "group")

    def test_outsider_cannot_post_or_read_team_channel(self):
        with self.assertRaises(PermissionDenied):
            self.kiran.say("hi", channel="frontend")
        with self.assertRaises(PermissionDenied):
            self.kiran.read_channel("frontend")

    def test_general_channel_is_for_everyone(self):
        self.kiran.say("Deploy freeze Friday", channel="general")
        self.assertEqual(self.priya.read_channel("general")[0]["text"], "Deploy freeze Friday")
        members = self.f.mcp.call("system", "channel_members", {"channel": "general"})["members"]
        self.assertEqual(set(members), {"rahul", "priya", "arjun", "meera", "kiran"})

    def test_channels_exist_per_team(self):
        channels = self.f.mcp.call("system", "list_channels")["channels"]
        self.assertEqual(set(channels), {"general", "frontend", "backend"})
        with self.assertRaises(NotFound):
            self.rahul.say("x", channel="random")

    # solo chat (chat_mcp.py)
    def test_solo_chat_both_directions(self):
        self.priya.dm("rahul", "Blocked on API spec")
        self.rahul.dm("priya", "Ask meera")
        convo = self.priya.read_dm("rahul")
        self.assertEqual([m["text"] for m in convo], ["Blocked on API spec", "Ask meera"])
        self.assertEqual(convo, self.rahul.read_dm("priya"))

    def test_solo_chat_across_teams(self):
        self.priya.dm("kiran", "What's the login endpoint?")
        self.assertEqual(len(self.kiran.read_dm("priya")), 1)

    def test_solo_chat_is_private(self):
        self.priya.dm("rahul", "secret")
        self.assertEqual(self.kiran.read_dm("rahul"), [])

    def test_inbox_marks_read(self):
        self.priya.dm("rahul", "one")
        self.priya.dm("rahul", "two")
        self.assertEqual([m["text"] for m in self.rahul.inbox()], ["one", "two"])
        self.assertEqual(self.rahul.inbox(), [])

    def test_cannot_dm_self_or_unknown(self):
        with self.assertRaises(AgentForgeError):
            self.priya.dm("priya", "hi")
        with self.assertRaises(NotFound):
            self.priya.dm("ghost", "hi")


# --------------------------------------------------------------------------- #
class TestContext(ForgeTest):
    """channel_context / dm_context: a rolling summary so agents don't start from zero."""

    def setUp(self):
        self.calls = []                                       # every (previous, new_texts) the summarizer saw

        def recording_summarizer(previous, new_messages):
            self.calls.append((previous, [m["text"] for m in new_messages]))
            return (previous + " | " if previous else "") + " + ".join(m["text"] for m in new_messages)

        self.f = self.new_forge(summarizer=recording_summarizer)
        for team, lead, devs in [("frontend", "rahul", ["priya", "arjun"]), ("backend", "meera", ["kiran"])]:
            self.f.team(team)
            self.f.teamlead(lead, team)
            for d in devs:
                self.f.developer(d, team)
        self.rahul, self.priya, self.kiran = (self.f.get_agent(n) for n in ("rahul", "priya", "kiran"))

    def test_empty_chat_has_empty_context(self):
        self.assertEqual(self.priya.channel_context(), "")
        self.assertEqual(self.priya.dm_context("rahul"), "")
        self.assertEqual(self.calls, [])                      # nothing to summarize -> no summarizer call

    def test_channel_context_summarizes_messages(self):
        self.rahul.say("Login uses OTP")
        self.priya.say("I'll take the OTP screen")
        self.assertEqual(self.priya.channel_context(), "Login uses OTP + I'll take the OTP screen")

    def test_only_new_messages_are_summarized(self):
        self.rahul.say("one")
        self.priya.channel_context()
        self.rahul.say("two")
        summary = self.priya.channel_context()
        self.assertEqual(self.calls, [("", ["one"]), ("one", ["two"])])   # 2nd call got only "two"
        self.assertEqual(summary, "one | two")

    def test_no_new_messages_reuses_summary(self):
        self.rahul.say("one")
        first = self.priya.channel_context()
        second = self.priya.channel_context()
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)                  # second ask cost nothing

    def test_summary_is_shared_by_channel_members(self):
        self.rahul.say("Deploy on Friday")
        self.priya.channel_context()                          # priya builds it
        self.assertEqual(self.f.get_agent("arjun").channel_context(), "Deploy on Friday")
        self.assertEqual(len(self.calls), 1)                  # arjun reused it

    def test_outsider_cannot_get_team_channel_context(self):
        self.rahul.say("frontend secret")
        with self.assertRaises(PermissionDenied):
            self.kiran.channel_context("frontend")

    def test_general_channel_context(self):
        self.kiran.say("Freeze on Friday", channel="general")
        self.assertEqual(self.priya.channel_context("general"), "Freeze on Friday")

    def test_dm_context_both_sides_and_private(self):
        self.priya.dm("rahul", "Blocked on API")
        self.rahul.dm("priya", "Ask meera")
        self.assertEqual(self.priya.dm_context("rahul"), "Blocked on API + Ask meera")
        self.assertEqual(self.rahul.dm_context("priya"), "Blocked on API + Ask meera")  # same conversation
        self.assertEqual(self.kiran.dm_context("rahul"), "")  # kiran's own (empty) chat with rahul

    def test_channel_and_dm_contexts_are_separate(self):
        self.rahul.say("group news")
        self.priya.dm("rahul", "private note")
        self.assertEqual(self.priya.channel_context(), "group news")
        self.assertEqual(self.priya.dm_context("rahul"), "private note")

    def test_context_loads_only_the_server_it_needs(self):
        self.priya.dm_context("rahul")
        self.assertEqual(self.f.mcp.loaded(), ["chat"])

    def test_default_summarizer_keeps_latest_lines(self):
        f = self.new_forge()                                  # no summarizer given -> default one
        f.team("t")
        f.teamlead("lead", "t")
        lead = f.get_agent("lead")
        for i in range(25):
            lead.say(f"msg {i}")
        lines = lead.channel_context().splitlines()
        self.assertEqual(len(lines), 20)
        self.assertEqual(lines[0], "lead: msg 5")
        self.assertEqual(lines[-1], "lead: msg 24")


# --------------------------------------------------------------------------- #
GUIDE = {
    "style_guide.md": "# Frontend style guide\n\nEvery form must validate email input before submit.\n"
                      "Buttons use the primary colour.\n",
    "api_notes.txt": "The auth API returns a JWT token. Tokens expire after 15 minutes.\n",
}
PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_brain", "AgentForge.pdf")


class TestKnowledge(ForgeTest):
    """Shared plugin memory: documents in a folder, searched through knowledge_mcp.py."""

    def setUp(self):
        self.folder = self.temp_dir(GUIDE)
        self.f = self.new_forge(knowledge_dir=self.folder)
        self.f.team("frontend")
        self.f.teamlead("rahul", "frontend")
        self.f.developer("priya", "frontend")
        self.priya = self.f.get_agent("priya")

    def test_lists_documents(self):
        names = {d["name"] for d in self.f.mcp.call("knowledge", "list_documents")["documents"]}
        self.assertEqual(names, {"style_guide.md", "api_notes.txt"})

    def test_search_finds_right_document_with_source(self):
        hits = self.priya.lookup("how long until the jwt token expires")
        self.assertEqual(hits[0]["document"], "api_notes.txt")
        self.assertEqual(hits[0]["page"], 1)
        self.assertIn("15 minutes", hits[0]["text"])

    def test_no_match_returns_empty(self):
        self.assertEqual(self.priya.lookup("kubernetes helm chart"), [])

    def test_new_file_is_picked_up_without_restart(self):
        self.priya.lookup("anything")                         # server running, index built
        with open(os.path.join(self.folder, "deploy.md"), "w") as f:
            f.write("Deploys happen every Friday at noon.")
        self.assertEqual(self.priya.lookup("when do deploys happen")[0]["document"], "deploy.md")

    def test_read_page_and_unknown_document(self):
        page = self.f.mcp.call("knowledge", "read_page", {"document": "api_notes.txt", "page": 1})
        self.assertIn("JWT", page["text"])
        with self.assertRaises(NotFound):
            self.f.mcp.call("knowledge", "read_page", {"document": "nope.md", "page": 1})

    def test_empty_folder(self):
        f = self.new_forge(knowledge_dir=self.temp_dir())
        f.team("t")
        f.teamlead("a", "t")
        self.assertEqual(f.get_agent("a").lookup("anything"), [])

    @unittest.skipUnless(os.path.exists(PDF), "project_brain/AgentForge.pdf not found")
    def test_real_pdf(self):
        f = self.new_forge()                                  # default folder: project_brain/
        f.team("t")
        f.teamlead("a", "t")
        a = f.get_agent("a")
        docs = f.mcp.call("knowledge", "list_documents")
        if docs["skipped"]:
            self.skipTest(docs["skipped"][0]["reason"])       # e.g. pypdf not installed
        self.assertEqual(a.lookup("ticket state machine")[0]["page"], 8)
        self.assertEqual(a.lookup("memory write trust policy hypothesis")[0]["page"], 11)
        top = a.lookup("idempotency key retry duplicate payment")[0]
        self.assertEqual((top["document"], top["page"]), ("AgentForge.pdf", 10))
        self.assertNotIn("The AI Software Company", top["text"])   # running header was stripped


class TestMemory(ForgeTest):
    """Personal memory through memory_mcp.py - private, saved to disk, only when needed."""

    def setUp(self):
        self.mem_dir = self.temp_dir()
        self.f = self.new_forge(memory_dir=self.mem_dir)
        self.f.team("frontend")
        self.f.teamlead("rahul", "frontend")
        self.f.developer("priya", "frontend")
        self.f.developer("arjun", "frontend")
        self.priya = self.f.get_agent("priya")

    def test_remember_and_recall(self):
        self.assertTrue(self.priya.remember("Rahul wants email validation on every form")["stored"])
        self.assertEqual(self.priya.recall("form validation")[0]["text"],
                         "Rahul wants email validation on every form")

    def test_duplicate_is_not_stored(self):
        self.priya.remember("Rahul wants email validation on every form")
        again = self.priya.remember("rahul wants email validation on every form!")
        self.assertFalse(again["stored"])
        self.assertEqual(len(self.priya.memories()), 1)

    def test_memories_are_private(self):
        self.priya.remember("secret lesson")
        self.assertEqual(self.f.get_agent("arjun").memories(), [])

    def test_memory_survives_a_new_run(self):
        self.priya.remember("Use the primary colour for buttons", kind="decision", source="ticket #4")
        self.f.close()                                        # end of run 1
        f2 = self.new_forge(memory_dir=self.mem_dir)          # run 2, same folder
        f2.team("frontend")
        f2.developer("priya", "frontend")
        mem = f2.get_agent("priya").memories()[0]
        self.assertEqual((mem["text"], mem["kind"], mem["source"]),
                         ("Use the primary colour for buttons", "decision", "ticket #4"))
        self.assertTrue(os.path.exists(os.path.join(self.mem_dir, "priya.json")))

    def test_forget_marks_stale_but_keeps_history(self):
        mid = self.priya.remember("API is on port 8000")["id"]
        self.priya.forget(mid, "moved to port 9000")
        self.assertEqual(self.priya.recall(), [])
        old = self.priya.memories(include_stale=True)[0]
        self.assertEqual((old["status"], old["stale_reason"]), ("stale", "moved to port 9000"))

    def test_bad_input(self):
        with self.assertRaises(AgentForgeError):
            self.priya.remember("x", kind="gossip")
        with self.assertRaises(AgentForgeError):
            self.priya.remember("   ")
        with self.assertRaises(AgentForgeError):
            self.priya.remember("x" * 501)

    def test_default_developer_remembers_rejections_once(self):
        f = self.new_forge(memory_dir=self.temp_dir())
        f.team("backend")
        state = {"rejected": 0}

        def picky_lead(agent, s):
            if not agent.forge.board.tickets(team=agent.team):
                agent.create_ticket("Auth API", assignee="kiran")
            for t in agent.review_queue():
                if state["rejected"] < 2:                     # reject twice with the same feedback
                    state["rejected"] += 1
                    agent.reject(t.id, "add input validation")
                else:
                    agent.approve(t.id)
            return {}

        f.teamlead("meera", "backend", fn=picky_lead)
        f.developer("kiran", "backend")                      # default developer node
        f.move_to("kiran", "meera")
        f.branch_to("meera", lambda s: END if f.board.all_done("backend") else "kiran")
        f.run({}, start="meera")
        mems = f.get_agent("kiran").memories()
        self.assertEqual(len(mems), 1)                        # 2 rejections, 1 lesson (duplicate refused)
        self.assertIn("add input validation", mems[0]["text"])
        self.assertEqual(mems[0]["source"], "ticket #1")


class TestSystemPrompt(ForgeTest):
    """agent.system_prompt rebuilds itself when a ticket that concerns the agent changes."""

    def setUp(self):
        self.f = self.new_forge(knowledge_dir=self.temp_dir(GUIDE))
        self.f.team("frontend")
        self.f.teamlead("rahul", "frontend")
        self.f.developer("priya", "frontend")
        self.f.developer("arjun", "frontend")
        self.rahul, self.priya, self.arjun = (self.f.get_agent(n) for n in ("rahul", "priya", "arjun"))

    def test_role_and_no_tickets(self):
        p = self.priya.system_prompt
        self.assertIn("You are priya, the Developer of the frontend team", p)
        self.assertIn("None right now.", p)

    def test_raised_ticket_appears_in_assignee_prompt(self):
        self.priya.system_prompt
        v = self.priya.prompt_version
        self.rahul.create_ticket("Signup form", "Validate the email field", assignee="priya")
        p = self.priya.system_prompt
        self.assertIn("#1 [TODO] Signup form: Validate the email field", p)
        self.assertIn("ticket #1 'Signup form' was created and assigned to priya", p)
        self.assertEqual(self.priya.prompt_version, v + 1)

    def test_prompt_brings_matching_knowledge_with_source(self):
        self.rahul.create_ticket("Signup form", "Validate the email input", assignee="priya")
        p = self.priya.system_prompt
        self.assertIn("[style_guide.md, p.1]", p)
        self.assertIn("validate email input", p)
        self.assertIsNotNone(self.f.board.get(1).knowledge)  # looked up once, cached on the ticket

    def test_other_developers_prompt_is_not_touched(self):
        self.arjun.system_prompt
        v = self.arjun.prompt_version
        self.rahul.create_ticket("Signup form", assignee="priya")
        self.assertNotIn("Signup form", self.arjun.system_prompt)
        self.assertEqual(self.arjun.prompt_version, v)

    def test_unassigned_ticket_shows_up_after_assign(self):
        t = self.rahul.create_ticket("Navbar")
        self.assertNotIn("Navbar", self.priya.system_prompt)
        self.rahul.assign(t.id, "priya")
        self.assertIn("Navbar", self.priya.system_prompt)
        self.rahul.assign(t.id, "arjun")                      # reassigned away from priya
        p = self.priya.system_prompt
        self.assertNotIn("#1 [TODO] Navbar", p)
        self.assertIn("reassigned to arjun", p)

    def test_feedback_and_done(self):
        t = self.rahul.create_ticket("Signup form", assignee="priya")
        self.priya.take(t.id)
        self.priya.submit(t.id, "v1")
        self.rahul.reject(t.id, "add a loading spinner")
        self.assertIn("Reviewer feedback to address: add a loading spinner", self.priya.system_prompt)
        self.priya.submit(t.id, "v2")
        self.rahul.approve(t.id)
        p = self.priya.system_prompt
        self.assertIn("None right now.", p)                   # DONE tickets leave the list
        self.assertIn("REVIEW -> DONE", p)                    # ...but the change is mentioned

    def test_lead_prompt_lists_team_tickets(self):
        self.rahul.create_ticket("Signup form", assignee="priya")
        self.rahul.create_ticket("Navbar")
        p = self.rahul.system_prompt
        self.assertIn("Signup form (assignee: priya)", p)
        self.assertIn("Navbar (assignee: unassigned)", p)
        self.assertIn("only one who can mark a ticket DONE", p)

    def test_memories_appear_in_prompt(self):
        self.rahul.create_ticket("Signup form", "email validation", assignee="priya")
        self.priya.remember("Rahul rejects forms without email validation", kind="observation",
                            source="ticket #0")
        self.assertIn("(observation, ticket #0) Rahul rejects forms without email validation",
                      self.priya.system_prompt)

    def test_no_change_no_rebuild(self):
        self.rahul.create_ticket("Signup form", assignee="priya")
        self.priya.system_prompt
        v = self.priya.prompt_version
        self.priya.system_prompt
        self.priya.system_prompt
        self.assertEqual(self.priya.prompt_version, v)

    def test_kanban_alone_starts_no_server(self):
        t = self.rahul.create_ticket("Signup form", assignee="priya")
        self.priya.take(t.id)
        self.assertEqual(self.f.mcp.loaded(), [])             # prompt marked stale, not rebuilt yet
        self.priya.system_prompt
        self.assertEqual(set(self.f.mcp.loaded()), {"knowledge", "memory"})


# --------------------------------------------------------------------------- #
class TestGraphRun(ForgeTest):
    def test_original_example_single_task(self):
        f = self.new_forge()
        f.team("frontend")
        f.teamlead("team_lead", "frontend")
        f.developer("developer", "frontend")
        f.move_to("team_lead", "developer")
        f.move_to("developer", END)
        out = f.run({"task": "Build login page"}, start="team_lead")
        t = f.board.tickets()[0]
        self.assertEqual(t.assignee, "developer")   # default lead assigns every ticket
        self.assertEqual(t.status, Status.REVIEW)   # lead hasn't reviewed yet
        self.assertIn("Build login page", out["code"][t.id])

    def test_loop_until_all_done_two_teams(self):
        f = self.build()
        # frontend lead -> priya -> arjun -> frontend lead (loop) ; then backend
        f.move_to("priya", "arjun")
        f.move_to("arjun", "rahul")
        f.branch_to("rahul", lambda s: "meera" if f.board.all_done("frontend") else "priya")
        f.move_to("kiran", "meera")
        f.branch_to("meera", lambda s: END if f.board.all_done("backend") else "kiran")

        f.run({"requirements": ["Login page", "Signup page", "Dashboard"]}, start="rahul",
              max_steps=20)
        self.assertTrue(f.board.all_done("frontend"))
        self.assertTrue(f.board.all_done("backend"))  # backend made tickets for the same reqs
        fe = f.board.tickets(team="frontend")
        self.assertEqual({t.assignee for t in fe}, {"priya", "arjun"})  # round robin
        self.assertTrue(all(t.code for t in fe))
        dms = f.get_agent("rahul").read_dm("priya")
        self.assertTrue(any("ready for review" in m["text"] for m in dms))
        self.assertTrue(f.get_agent("priya").read_channel())
        self.assertEqual(set(f.mcp.loaded()), {"system", "chat"})

    def test_unassigned_tickets_wait_for_the_lead(self):
        f = self.new_forge()
        f.team("backend")

        def lead(agent, state):
            board = agent.forge.board
            if not board.tickets(team=agent.team):
                agent.create_ticket("Auth API")                 # created WITHOUT an assignee
                return {"phase": "created"}
            if state["phase"] == "created":
                agent.assign(1, "kiran")                        # second visit: assign it
                return {"phase": "assigned"}
            for t in agent.review_queue():
                agent.approve(t.id)
            return {}

        seen = []

        def dev(agent, state):
            seen.append([t.id for t in agent.available_tickets()])
            for t in agent.available_tickets():
                agent.take(t.id)
                agent.submit(t.id, "def auth(): ...")
            return {}

        f.teamlead("meera", "backend", fn=lead)
        f.developer("kiran", "backend", fn=dev)
        f.move_to("kiran", "meera")
        f.branch_to("meera", lambda s: END if f.board.all_done("backend") else "kiran")
        f.run({}, start="meera")
        self.assertEqual(seen, [[], [1]])        # 1st visit: nothing to take; 2nd: the assigned ticket
        self.assertEqual(f.board.get(1).status, Status.DONE)

    def test_custom_node_functions(self):
        f = self.new_forge()
        f.team("backend")

        def lead(agent, state):
            if not agent.forge.board.tickets(team=agent.team):
                agent.create_ticket("Auth API", assignee="kiran")
            for t in agent.review_queue():
                if "def login" in t.code:
                    agent.approve(t.id)
                else:
                    agent.reject(t.id, "missing login function")
            return {"reviews": state.get("reviews", 0) + 1}

        attempts = {"n": 0}

        def dev(agent, state):
            for t in agent.available_tickets():
                agent.take(t.id)
            for t in agent.my_tickets(Status.IN_PROGRESS):
                attempts["n"] += 1
                code = "def login(): ..." if t.feedback else "def logout(): ..."
                agent.submit(t.id, code)
            return {}

        f.teamlead("meera", "backend", fn=lead)
        f.developer("kiran", "backend", fn=dev)
        f.move_to("kiran", "meera")
        f.branch_to("meera", lambda s: END if f.board.all_done("backend") else "kiran")
        out = f.run({}, start="meera")
        t = f.board.get(1)
        self.assertEqual(t.status, Status.DONE)
        self.assertEqual(t.rejections, 1)
        self.assertEqual(attempts["n"], 2)
        self.assertEqual(out["reviews"], 3)
        self.assertEqual(f.mcp.loaded(), [])     # these nodes never chat -> no MCP server started

    def test_infinite_loop_guard(self):
        f = self.new_forge(max_steps=5)
        f.team("t")
        f.teamlead("a", "t", fn=lambda ag, s: {})
        f.developer("b", "t", fn=lambda ag, s: {})
        f.move_to("a", "b")
        f.move_to("b", "a")
        with self.assertRaises(AgentForgeError):
            f.run({}, start="a")

    def test_missing_edge_and_bad_target(self):
        f = self.new_forge()
        f.team("t")
        f.teamlead("a", "t", fn=lambda ag, s: {})
        with self.assertRaises(AgentForgeError):
            f.run({}, start="a")  # no move_to after a
        f.move_to("a", "ghost")
        with self.assertRaises(NotFound):
            f.run({}, start="a")

    def test_node_must_return_dict(self):
        f = self.new_forge()
        f.team("t")
        f.teamlead("a", "t", fn=lambda ag, s: "oops")
        f.move_to("a", END)
        with self.assertRaises(AgentForgeError):
            f.run({}, start="a")

    def test_cannot_mix_move_to_and_branch_to(self):
        f = self.build()
        f.move_to("rahul", "priya")
        with self.assertRaises(AgentForgeError):
            f.branch_to("rahul", lambda s: END)

    def test_run_does_not_mutate_input_state(self):
        f = self.new_forge()
        f.team("t")
        f.teamlead("a", "t", fn=lambda ag, s: {"x": 1})
        f.move_to("a", END)
        original = {"y": 2}
        out = f.run(original, start="a")
        self.assertEqual(original, {"y": 2})
        self.assertEqual(out, {"y": 2, "x": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
