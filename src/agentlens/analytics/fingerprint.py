"""Project fingerprinting — inferred from session history, not the filesystem.

Everything here comes from events already in the database: file extensions the
agent touched, commands it ran, manifest files it opened. Nothing scans the
user's disk.

That is a deliberate constraint, and it buys three things:

* the offline/privacy story stays intact — no new filesystem access to justify;
* it works on a machine where the project is no longer checked out, which is
  exactly when you're reviewing last quarter's spend; and
* it describes what the agent *actually did*, not what a config file claims.
  A repo with a `pyproject.toml` and no Python edits isn't a Python project as
  far as your agent spend is concerned.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Any

from ..schema import Event, ToolKind
from .rewrite import detect_test_command

_EXT_LANG = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".java": "java", ".kt": "kotlin", ".swift": "swift", ".cs": "csharp",
    ".sql": "sql", ".sh": "shell", ".yml": "yaml", ".yaml": "yaml",
}

_MANIFESTS = {
    "pyproject.toml": ("python", "uv/pip"),
    "requirements.txt": ("python", "pip"),
    "setup.py": ("python", "pip"),
    "package.json": ("javascript", "npm"),
    "pnpm-lock.yaml": ("javascript", "pnpm"),
    "yarn.lock": ("javascript", "yarn"),
    "go.mod": ("go", "go"),
    "cargo.toml": ("rust", "cargo"),
    "gemfile": ("ruby", "bundler"),
    "pom.xml": ("java", "maven"),
    "build.gradle": ("java", "gradle"),
}

_FRAMEWORK_HINTS = {
    "react": ("react", ".tsx", ".jsx"),
    "django": ("django",),
    "flask": ("flask",),
    "fastapi": ("fastapi",),
    "next": ("next.config",),
    "vue": (".vue",),
}

_LINTERS = {
    "python": ["ruff check", "mypy"],
    "typescript": ["eslint", "tsc --noEmit"],
    "javascript": ["eslint"],
    "go": ["go vet"],
    "rust": ["cargo clippy"],
}


@dataclass(slots=True)
class ProjectProfile:
    repo_id: str | None = None
    repo_label: str | None = None
    languages: list[str] = field(default_factory=list)
    package_manager: str | None = None
    test_command: str | None = None
    frameworks: list[str] = field(default_factory=list)
    #: How much evidence this is based on. A profile built from four events is
    #: a guess; one built from four hundred is a description. Surfaced so the
    #: UI never presents a thin inference with the same confidence as a solid
    #: one.
    sample_events: int = 0

    @property
    def primary_language(self) -> str | None:
        return self.languages[0] if self.languages else None

    @property
    def confident(self) -> bool:
        return self.sample_events >= 20 and bool(self.languages)

    def linters(self) -> list[str]:
        return _LINTERS.get(self.primary_language or "", [])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["primary_language"] = self.primary_language
        d["confident"] = self.confident
        d["linters"] = self.linters()
        return d


def fingerprint(events: list[Event]) -> ProjectProfile:
    """Infer a project profile from one project's events."""
    if not events:
        return ProjectProfile()

    langs: Counter[str] = Counter()
    frameworks: set[str] = set()
    package_manager: str | None = None

    for e in events:
        path = (e.tool.target_path if e.tool else None) or ""
        low = path.lower()
        name = low.rsplit("/", 1)[-1]

        if name in _MANIFESTS:
            lang, pm = _MANIFESTS[name]
            langs[lang] += 3  # a manifest is stronger evidence than one edit
            package_manager = package_manager or pm

        for ext, lang in _EXT_LANG.items():
            if low.endswith(ext):
                langs[lang] += 1
                break

        haystack = f"{low} {(e.text or '').lower()}"
        for fw, hints in _FRAMEWORK_HINTS.items():
            if any(h in haystack for h in hints):
                frameworks.add(fw)

        if not package_manager and e.tool and e.tool.kind is ToolKind.BASH and e.text:
            cmd = e.text.lower().split()
            head = cmd[0] if cmd else ""
            # Match the *command being run*, not a substring anywhere in the
            # line — "npm" appears inside plenty of text that isn't an npm
            # invocation.
            for tool_name in ("pnpm", "yarn", "npm", "uv", "pip", "poetry",
                              "cargo", "go", "bundler", "mvn", "gradle"):
                if head == tool_name:
                    package_manager = tool_name
                    break

    first = events[0]
    return ProjectProfile(
        repo_id=first.repo_id,
        repo_label=first.repo_label,
        languages=[lang for lang, _ in langs.most_common(3)],
        package_manager=package_manager,
        test_command=detect_test_command(events),
        frameworks=sorted(frameworks),
        sample_events=len(events),
    )


def fingerprint_all(sessions: list[list[Event]]) -> ProjectProfile:
    """One profile across every session in the current scope.

    When the scope is a single project this is that project's profile; when it
    spans several it is a blended one, which is why the UI shows it only
    alongside a project filter.
    """
    flat = [e for evs in sessions for e in evs]
    return fingerprint(flat)
