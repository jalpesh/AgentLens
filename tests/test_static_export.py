"""Static HTML export tests.

The exporter reuses `web/static/index.html` verbatim — see `export_html`'s
docstring for why a fork would be a mistake. These tests check the
*functional* claim ("this opens offline with zero network calls"), not a
literal absence of the substring "fetch(" — the template legitimately keeps
that code path (for the live server) as dead code behind an `if`, and a naive
grep would flag correct behaviour as a bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.store import Store  # noqa: E402
from agentlens.web.server import _MARKER, export_html  # noqa: E402
from generate import build  # noqa: E402


@pytest.fixture()
def seeded_db(tmp_path) -> Path:
    paths = build(tmp_path / "projects")
    adapter = ClaudeCodeAdapter()
    db = tmp_path / "t.db"
    with Store(db) as s:
        for p in paths:
            s.add_events(adapter.parse(SourceFile.of(p)))
    return db


def _extract_payload(html: str) -> dict:
    marker = 'const AGENTLENS_STATIC_DATA = {"totals"'
    i = html.index(marker) + len("const AGENTLENS_STATIC_DATA = ")
    data, _ = json.JSONDecoder().raw_decode(html[i:])
    return data


def test_export_produces_one_self_contained_file(seeded_db, tmp_path):
    out = export_html(str(tmp_path / "report.html"), db=str(seeded_db))
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert _MARKER not in html, "the live marker must be replaced, not left in the output"
    payload = _extract_payload(html)
    assert payload["totals"]["sessions"] > 0
    assert payload["findings"]


def test_export_never_forks_the_template(seeded_db, tmp_path):
    """If the exporter ever started rendering its own HTML instead of
    substituting into the shared template, this is the test that would catch
    the drift — the served and exported pages must literally be the same
    file except for one line."""
    from agentlens.web.server import STATIC

    out = export_html(str(tmp_path / "report.html"), db=str(seeded_db))
    exported = out.read_text(encoding="utf-8")
    template = (STATIC / "index.html").read_text(encoding="utf-8")

    exported_before_marker = exported.split("const AGENTLENS_STATIC_DATA =")[0]
    template_before_marker = template.split("const AGENTLENS_STATIC_DATA =")[0]
    assert exported_before_marker == template_before_marker


def test_export_is_redacted_by_default(seeded_db, tmp_path):
    out = export_html(str(tmp_path / "report.html"), db=str(seeded_db))
    payload = _extract_payload(out.read_text(encoding="utf-8"))
    assert payload["redacted"] is True
    blob = json.dumps(payload)
    assert "/home/dev" not in blob, "raw repo paths must not survive redaction"


def test_export_raw_opts_out_of_redaction(seeded_db, tmp_path):
    out = export_html(str(tmp_path / "report.html"), db=str(seeded_db), redact=False)
    payload = _extract_payload(out.read_text(encoding="utf-8"))
    assert payload["redacted"] is False


def test_export_raises_clearly_if_marker_removed(seeded_db, tmp_path, monkeypatch):
    """If a future edit to index.html accidentally deletes the marker line,
    this must fail loudly at export time, not silently ship a page with no
    data."""
    from agentlens.web import server

    static_dir = tmp_path / "fakestatic"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>no marker here</html>", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC", static_dir)

    with pytest.raises(RuntimeError, match="marker"):
        export_html(str(tmp_path / "report.html"), db=str(seeded_db))


def test_static_page_hides_prompt_lab_and_fires_no_network_requests(seeded_db, tmp_path):
    """The actual claim of this feature, verified in a real browser with the
    network cut, not inferred from source text."""
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    out = export_html(str(tmp_path / "report.html"), db=str(seeded_db))

    requests_seen = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(offline=True)
        page = ctx.new_page()
        page.on("request", lambda r: requests_seen.append(r.url))
        page.goto(out.as_uri())
        page.wait_for_timeout(500)
        lab_tab = page.query_selector('nav button[data-p="lab"]')
        badge_text = page.inner_text(".badge")
        ctx.close()
        browser.close()

    assert requests_seen == [out.as_uri()], "no request besides loading the file itself"
    assert lab_tab is None, "Prompt Lab must be hidden in a static export"
    assert "static" in badge_text.lower()
