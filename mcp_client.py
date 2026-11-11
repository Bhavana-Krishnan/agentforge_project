"""
mcp_client.py - a small synchronous bridge to an MCP server.

Why this file exists
--------------------
The MCP Python SDK is async (async/await). AgentForge is plain synchronous code.
MCPConnection hides the async part: it runs an event loop in a background thread,
starts the server script as a child process, and gives you a normal blocking

    conn.call("post_message", {"sender": "rahul", "channel": "frontend", "text": "hi"})

This file knows nothing about AgentForge - it works with any MCP server script.
AgentForge imports it only when the first MCP tool is needed, so the `mcp` package itself
is not even loaded into memory until then.
"""

import asyncio                    # the MCP SDK runs on an asyncio event loop
import atexit                     # close any servers still running when Python exits
import concurrent.futures         # a thread-safe "result will arrive later" box
import json                       # tool results come back as JSON text
import sys                        # sys.executable = the Python that is running us
import threading                  # the event loop lives in its own thread

from mcp import Client, StdioServerParameters   # official MCP SDK client


class MCPConnectionError(Exception):
    """The server process is gone or the connection broke (not a tool-level error)."""


class MCPToolError(Exception):
    """A tool call failed. `kind` is the prefix the server put on the message, e.g. 'PermissionDenied'."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


_open_connections: list = []      # every live connection, so atexit can close them


class MCPConnection:
    """One running MCP server process + one client session talking to it."""

    def __init__(self, name: str, script_path: str, timeout: float = 30.0, env: dict | None = None):
        self.name = name                       # e.g. "system"
        self.script_path = script_path         # e.g. ".../system_mcp.py"
        self.env = env or {}                   # extra environment variables for the server (e.g. folder paths)
        self.timeout = timeout                 # seconds to wait for start-up or a tool call
        self.tools: dict = {}                  # tool name -> MCP Tool (name, description, input_schema)

        # 1. A private event loop in a daemon thread (daemon = won't keep Python alive).
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=f"mcp-{name}", daemon=True)
        self._thread.start()

        # 2. Start _main() on that loop. It launches the server and keeps the session open.
        self._ready = concurrent.futures.Future()     # set once the session is connected
        self._main_task = asyncio.run_coroutine_threadsafe(self._main(), self._loop)

        # 3. Block here until connected (or re-raise whatever went wrong).
        try:
            self._ready.result(timeout=self.timeout)
        except BaseException:
            self._main_task.cancel()
            self._shutdown_loop()                         # don't leak the loop thread on a failed start
            raise
        _open_connections.append(self)

    async def _main(self):
        """Runs for the whole life of the connection, inside the background loop."""
        self._stop = asyncio.Event()           # set by close() to end the session
        # StdioServerParameters = "run this command and talk to it on stdin/stdout"
        # env is merged over a safe default environment (PATH, HOME, ...) by the SDK
        params = StdioServerParameters(command=sys.executable, args=[self.script_path], env=self.env)
        try:
            async with Client(params) as client:          # starts the process + MCP handshake
                self._client = client
                listing = await client.list_tools()       # ask the server what it can do
                self.tools = {t.name: t for t in listing.tools}
                self._ready.set_result(True)              # unblock __init__
                await self._stop.wait()                   # stay connected until close()
        except BaseException as exc:
            if not self._ready.done():                    # failed while starting
                self._ready.set_exception(exc)
            raise
        # leaving `async with` closes the session and stops the server process

    def call(self, tool: str, arguments: dict | None = None):
        """Call one tool and return its result as plain Python data (dict / list)."""
        if self._main_task.done() or self._loop.is_closed():
            raise MCPConnectionError(f"MCP server '{self.name}' is not running")
        future = asyncio.run_coroutine_threadsafe(        # hand the coroutine to the background loop
            self._client.call_tool(tool, arguments or {}), self._loop)
        try:
            result = future.result(timeout=self.timeout)  # wait for the answer
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise MCPConnectionError(f"MCP server '{self.name}' timed out on '{tool}'") from None
        except Exception as exc:                          # e.g. "Connection closed" from the SDK
            raise MCPConnectionError(f"MCP server '{self.name}' failed on '{tool}': {exc}") from None

        text = "".join(getattr(c, "text", "") for c in result.content)   # human/LLM-readable part
        if result.is_error:
            raise _parse_error(text)
        if result.structured_content is not None:         # typed JSON, when the server provides it
            data = result.structured_content
            # tools returning a list get wrapped as {"result": [...]}; unwrap that
            return data["result"] if set(data) == {"result"} else data
        return json.loads(text) if text else None         # otherwise parse the JSON text

    def close(self):
        """Stop the session and the server process, then the background loop."""
        if self not in _open_connections:
            return                                        # already closed
        _open_connections.remove(self)
        if not self._main_task.done():
            self._loop.call_soon_threadsafe(self._stop.set)   # wake _main() so it exits `async with`
        try:
            self._main_task.result(timeout=10)
        except BaseException:
            pass                                          # shutting down - nothing useful to do
        self._shutdown_loop()

    def _shutdown_loop(self):
        """Stop the background loop and thread."""
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)  # end run_forever() in the thread
        self._thread.join(timeout=5)                      # wait for the thread to finish
        if not self._thread.is_alive():
            self._loop.close()                            # free the loop's resources

    def __repr__(self):
        return f"MCPConnection({self.name!r}, tools={sorted(self.tools)})"


def _parse_error(text: str) -> MCPToolError:
    """'Error executing tool post_message: PermissionDenied: x is not ...' -> MCPToolError."""
    prefix = "Error executing tool "
    if text.startswith(prefix) and ": " in text:
        text = text.split(": ", 1)[1]                     # drop "Error executing tool <name>: "
    kind, sep, message = text.partition(": ")
    if sep and kind.isidentifier():                       # looks like "Kind: message"
        return MCPToolError(kind, message)
    return MCPToolError("ToolError", text)                # crash or argument error - keep full text


@atexit.register
def _close_all():
    for conn in list(_open_connections):
        conn.close()
