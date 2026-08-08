import json
from pathlib import Path

import mcp.types as mt

from server.butler import Butler
from server.vault_mcp import VAULT_TOOL_NAMES, build_vault_server


# --- lightweight duck-typed fakes: the Butler reads .content/.text/.subtype/.session_id ---
class _Text:
    def __init__(self, text): self.text = text


class _Assistant:
    def __init__(self, text): self.content = [_Text(text)]


class _Result:
    def __init__(self, session_id): self.subtype = "success"; self.session_id = session_id


class FakeClient:
    last = None

    def __init__(self, options):
        self.options = options
        self.queries = []
        self.connected = False
        FakeClient.last = self

    async def connect(self): self.connected = True
    async def disconnect(self): self.connected = False
    async def query(self, text, session_id="default"): self.queries.append(text)

    async def receive_response(self):
        yield _Assistant('{"spoken": "Session 2.", '
                         '"display": "You left off at [[Tibet Session 2]].", '
                         '"citations": ["Tibet Session 2"]}')
        yield _Result("sess-123")


def make_butler(tmp_path, resume=None):
    def opts(resume_id):
        return {"resume": resume_id}  # a plain marker; FakeClient just stores it
    return Butler(options_builder=opts,
                  state_path=tmp_path / "butler.json",
                  client_factory=FakeClient)


async def test_ask_returns_parsed_output(tmp_path):
    b = make_butler(tmp_path)
    out = await b.ask("where did I leave the tibet study?")
    assert out["spoken"] == "Session 2."
    assert out["citations"] == ["Tibet Session 2"]
    assert FakeClient.last.queries == ["where did I leave the tibet study?"]


async def test_ask_persists_session_id(tmp_path):
    b = make_butler(tmp_path)
    await b.ask("hi")
    saved = json.loads((tmp_path / "butler.json").read_text())
    assert saved["session_id"] == "sess-123"


async def test_resume_id_loaded_and_passed_to_options(tmp_path):
    (tmp_path / "butler.json").write_text(json.dumps({"session_id": "prev-999"}))
    b = make_butler(tmp_path)
    await b.ask("hi")
    # options_builder was called with the resumed id from disk
    assert FakeClient.last.options == {"resume": "prev-999"}


async def test_vault_server_exposes_four_named_tools(tmp_path):
    server = build_vault_server(tmp_path)
    assert server is not None
    assert VAULT_TOOL_NAMES == [
        "mcp__vault__vault_search", "mcp__vault__vault_read",
        "mcp__vault__vault_capture", "mcp__vault__vault_log"]
    # ...and the server really registers those four, under exactly those names.
    # VAULT_TOOL_NAMES is the allowlist: a wrapper whose @tool name drifts from
    # it is still registered but never permitted, so the butler loses the
    # capability silently at runtime. Assert the real list_tools reply, not a
    # second copy of the constant.
    listed = await server["instance"].request_handlers[mt.ListToolsRequest](
        mt.ListToolsRequest(method="tools/list"))
    assert [f"mcp__vault__{t.name}" for t in listed.root.tools] == VAULT_TOOL_NAMES
