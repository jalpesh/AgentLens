from agentlens.adapters import GeminiCliAdapter


def test_antigravity_and_config_files_are_not_gemini_cli_chats(tmp_path, monkeypatch):
    home = tmp_path / ".gemini"
    for rel in ("antigravity/brain/x/transcript.jsonl", "antigravity-ide/a.json",
                "config/settings.json", "tmp/abc/chats/session-1.json"):
        f = home / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('{"messages": []}')
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_HOME", raising=False)
    found = [str(s.path.relative_to(home)) for s in GeminiCliAdapter().discover()]
    assert found == ["tmp/abc/chats/session-1.json"]
