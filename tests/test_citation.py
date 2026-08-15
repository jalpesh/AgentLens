"""Loop citation tests — the narrow, opt-in disk read.

This is the one feature in AgentLens that reads file *content* off disk
rather than session-log metadata. It must:

1. never touch the filesystem unless this exact endpoint is called;
2. degrade gracefully (never crash) when the file is missing, relocated, or
   this is a different machine than the one the session ran on; and
3. never write anything, and never cache the content it read to SQLite.

See SECURITY.md's "Loop citations" section for the user-facing contract this
file is checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.web.server import make_handler  # noqa: E402


@pytest.fixture()
def citation():
    return make_handler(None)._citation


def test_missing_file_degrades_gracefully(citation):
    result = citation("/definitely/not/a/real/path/on/this/machine.py")
    assert result == {
        "available": False,
        "reason": "file not available on this machine",
    }


def test_empty_path_degrades_gracefully(citation):
    result = citation("")
    assert result["available"] is False


def test_windows_style_absolute_path_is_recognized_as_absolute(citation):
    """A session logged on Windows records paths like `C:\\Users\\...`. The
    citation endpoint must recognize that shape as absolute even when the
    dashboard itself is running on Linux/macOS (where `pathlib.Path` would
    otherwise judge it by POSIX rules) — the whole feature exists to show
    files from a session that may not be running on this machine."""
    result = citation(r"C:\Users\dev\project\definitely-not-here.py")
    assert result["reason"] != "not an absolute path"


def test_unc_path_is_recognized_as_absolute(citation):
    result = citation(r"\\myhost\share\definitely-not-here.py")
    assert result["reason"] != "not an absolute path"


def test_relative_path_is_refused(citation):
    """A relative path is ambiguous about *which* file it means (relative to
    what — the server's cwd? the browser's?) and accepting one would be an
    easy way to read something unintended. Refuse rather than guess."""
    result = citation("relative/path.py")
    assert result["available"] is False
    assert "absolute" in result["reason"]


def test_directory_is_refused_not_crashed(citation, tmp_path):
    result = citation(str(tmp_path))
    assert result["available"] is False
    assert "reason" in result


def test_existing_file_is_read(citation, tmp_path):
    p = tmp_path / "cicd" / "azure-pipelines.yml"
    p.parent.mkdir()
    p.write_text("stages:\n  - build\n  - test\n")
    result = citation(str(p))
    assert result["available"] is True
    assert "stages:" in result["content"]
    assert result["truncated"] is False


def test_large_file_is_truncated_not_dumped_whole(citation, tmp_path):
    from agentlens.web.server import _CITATION_MAX_LINES

    p = tmp_path / "huge.log"
    p.write_text("\n".join(f"line {i}" for i in range(_CITATION_MAX_LINES + 50)))
    result = citation(str(p))
    assert result["available"] is True
    assert result["truncated"] is True
    assert result["content"].count("\n") + 1 <= _CITATION_MAX_LINES


def test_unicode_file_is_not_falsely_flagged_truncated(citation, tmp_path):
    """A byte-size vs character-count mismatch on multi-byte UTF-8 content
    (emoji, non-ASCII text) must not make a fully-read small file look
    truncated."""
    p = tmp_path / "notes.md"
    p.write_text("done ✅ ⚠️ café — all good\n" * 5, encoding="utf-8")
    result = citation(str(p))
    assert result["available"] is True
    assert result["truncated"] is False


def test_binary_file_does_not_crash_the_endpoint(citation, tmp_path):
    p = tmp_path / "binary.bin"
    p.write_bytes(bytes(range(256)) * 4)
    result = citation(str(p))
    # Whatever it decides ("available" with replacement chars, or a graceful
    # refusal), it must never raise.
    assert "available" in result


def test_default_payload_build_never_reads_files_off_disk(monkeypatch, tmp_path):
    """The one hard rule: reading file *content* is opt-in per loop, via
    `/api/citation` only — never part of the default `/api/data` payload
    build. `Store` talks to SQLite directly (not `Path.open`), so a
    functioning `build_payload` call should never touch `Path.open` at all;
    if it ever does, that's this feature's no-filesystem-scanning guarantee
    quietly breaking."""
    from agentlens.store import Store

    db = tmp_path / "empty.db"
    with Store(db):
        pass  # just create an empty, schema'd database

    calls = []
    real_open = Path.open

    def spy_open(self, *a, **k):
        calls.append(str(self))
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", spy_open)

    from agentlens.web.server import build_payload

    build_payload(str(db))

    assert calls == [], f"build_payload opened file(s) it should never touch: {calls}"
