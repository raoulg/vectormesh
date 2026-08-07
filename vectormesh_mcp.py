#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp[cli]>=1.2.0"]
# ///
"""MCP server that exposes the VectorMesh course as design/search coaching tools.

The course is a set of markdown chapters in `docs/`, indexed by the reading-order table in
`docs/README.md`. Unlike a code-review server, this one is forward-looking: it coaches a student
through two activities the docs describe but a chat window cannot enforce on its own —

- **designing** a new component or architecture from the library's own principles, worked through
  as a five-stage conversation (tensor shapes -> possible enrichments -> possible flows -> search
  space -> search practices) rather than handed over as one document to skim;
- **searching** an architecture with Ray Tune once there is more than one plausible design and no
  principled way to pick between them by inspection.

The design conversation (`vectormesh_design_checklist`) is deliberately stateful *within one
server process* (i.e. for the current session): it returns one stage's checklist at a time, and
only advances once the assistant reports back what the student actually said. The point is to make
the assistant cooperate and think *with* the student against this library's own framework, not
hand them a finished pipeline.

It also exposes the chapters themselves for lookup and keyword search, the same way the
`codestyle` MCP server exposes its guidelines. It reads `docs/` from disk when they sit next to
this script; when launched standalone (e.g. via `uv run` straight from a URL, with no course files
nearby) it fetches the same content from the public repo over HTTP instead. No build step, no
embeddings, no database.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Paths and course model
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_README = DOCS_DIR / "README.md"

# If the course files aren't next to this script, fetch them from the public repo over HTTP.
# This is what makes the zero-clone one-liner work:
#   uv run --no-project <RAW_BASE>/vectormesh_mcp.py
#
# VECTORMESH_REF selects which git ref (branch, tag, or commit) to fetch content from; it
# defaults to "main". Point the launch URL at the same ref and both the script and the docs it
# fetches stay pinned together — see MCP_SERVER.md.
VECTORMESH_REF = os.environ.get("VECTORMESH_REF", "main")
RAW_BASE = f"https://raw.githubusercontent.com/raoulg/vectormesh/{VECTORMESH_REF}"
LOCAL = DOCS_README.is_file() and DOCS_DIR.is_dir()

# Cache for content fetched over HTTP (unused in local mode).
_remote_cache: dict[str, str] = {}


def _load_text(rel_path: str) -> str:
    """Load a course file's text, relative to the repo root.

    e.g. `_load_text("docs/README.md")` or `_load_text("docs/01-core-concepts.md")`.

    Local files are read fresh on each call, so edits show up without a restart. Remote files are
    fetched once and cached.
    """
    if LOCAL:
        return (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    if rel_path not in _remote_cache:
        with urllib.request.urlopen(f"{RAW_BASE}/{rel_path}", timeout=30) as resp:
            _remote_cache[rel_path] = resp.read().decode("utf-8")
    return _remote_cache[rel_path]


mcp = FastMCP("vectormesh_mcp")


# --------------------------------------------------------------------------- #
# Loading the chapter index from docs/README.md's reading-order table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chapter:
    """One row of the docs/README.md reading-order table."""

    number: int  # 1..9
    title: str  # human-readable chapter title, e.g. "Core concepts"
    file: str  # doc filename, e.g. "01-core-concepts.md"
    summary: str  # one-line "what it covers"


# Matches a reading-order table row:
#   | 1 | [Core concepts](01-core-concepts.md) | The embed-once economics, ... |
_ROW = re.compile(
    r"^\|\s*(?P<num>\d+)\s*\|\s*\[(?P<title>.+?)\]\((?P<file>.+?\.md)\)\s*\|\s*(?P<summary>.+?)\s*\|\s*$"
)


def _load_chapters() -> list[Chapter]:
    """Parse docs/README.md's reading-order table into an ordered list of Chapters."""
    chapters: list[Chapter] = []
    for line in _load_text("docs/README.md").splitlines():
        row = _ROW.match(line)
        if not row:
            continue
        chapters.append(
            Chapter(
                number=int(row.group("num")),
                title=row.group("title"),
                file=row.group("file"),
                summary=row.group("summary"),
            )
        )
    return chapters


# Loaded once at startup; the corpus is tiny and static per process.
CHAPTERS: list[Chapter] = _load_chapters()
# Map a few forms of a name/filename/number to its Chapter for forgiving lookups.
_BY_KEY: dict[str, Chapter] = {}
for _c in CHAPTERS:
    _BY_KEY[str(_c.number)] = _c
    _BY_KEY[f"{_c.number:02d}"] = _c
    _BY_KEY[_c.file.lower()] = _c
    _BY_KEY[_c.file[:-3].lower()] = _c  # without ".md"
    _BY_KEY[_c.title.lower()] = _c


def _find_chapter(query: str) -> Optional[Chapter]:
    """Resolve a chapter by number, filename, stem, or title (case-insensitive)."""
    key = query.strip().lower()
    if key in _BY_KEY:
        return _BY_KEY[key]
    # Fall back to a partial match on the title.
    for chapter in CHAPTERS:
        if key in chapter.title.lower():
            return chapter
    return None


def _must_find_chapter(query: str) -> Chapter:
    """Like _find_chapter, for internal callers that name a chapter known to exist.

    Used only for the fixed chapter numbers this module cites in its own coaching text (e.g.
    "01", "09") — never for user-supplied input, which goes through _find_chapter and handles
    None explicitly.
    """
    chapter = _find_chapter(query)
    assert chapter is not None, f"expected chapter '{query}' to exist in docs/README.md"
    return chapter


def _doc_url(filename: str, anchor: str = "") -> str:
    """Browsable GitHub URL for a chapter (rendered markdown, for humans), optionally to an anchor.

    Uses the same ref the server is pinned to, so a link always points at the exact text the
    reader is being coached on.
    """
    suffix = f"#{anchor}" if anchor else ""
    return f"https://github.com/raoulg/vectormesh/blob/{VECTORMESH_REF}/docs/{filename}{suffix}"


def _chapter_ref(chapter: Chapter) -> str:
    """A markdown link to a chapter's online text."""
    return f"[{chapter.file}]({_doc_url(chapter.file)})"


def _read_doc(filename: str) -> str:
    """Read a chapter's text, raising a clear error if it is missing.

    Works whether the docs are local or fetched from the public repo; any failure to obtain the
    file is normalised to FileNotFoundError.
    """
    try:
        return _load_text(f"docs/{filename}")
    except (OSError, urllib.error.URLError) as exc:
        raise FileNotFoundError(filename) from exc


# --------------------------------------------------------------------------- #
# Input models — chapter lookup
# --------------------------------------------------------------------------- #


class ListConceptsInput(BaseModel):
    """Input for listing course chapters."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    keyword: Optional[str] = Field(
        default=None,
        description=(
            "Optional substring filter, matched case-insensitively against each chapter's "
            "title and one-line summary. Omit to list all chapters."
        ),
        max_length=200,
    )


class GetConceptInput(BaseModel):
    """Input for fetching a single chapter."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chapter: str = Field(
        ...,
        description=(
            "Chapter to fetch. Accepts the number ('4' or '04'), the doc filename "
            "('04-components.md'), its stem ('04-components'), or the title "
            "('Components')."
        ),
        min_length=1,
        max_length=200,
    )


class SearchInput(BaseModel):
    """Input for searching across all chapters."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Keyword(s) to search for across every chapter's text.",
        min_length=2,
        max_length=200,
    )
    limit: int = Field(
        default=5,
        description="Maximum number of matching chapters to return.",
        ge=1,
        le=20,
    )


# --------------------------------------------------------------------------- #
# Tools — chapter lookup (same shape as codestyle's guideline tools)
# --------------------------------------------------------------------------- #


@mcp.tool(
    name="vectormesh_list_concepts",
    annotations=ToolAnnotations(
        title="List VectorMesh course chapters",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def vectormesh_list_concepts(params: ListConceptsInput) -> str:
    """List the VectorMesh course chapters, in reading order.

    Use this to discover what the course covers before fetching a chapter with
    vectormesh_get_concept, or before calling vectormesh_design_checklist / vectormesh_search_method
    to see which chapters they draw on.

    Args:
        params (ListConceptsInput):
            - keyword (Optional[str]): if set, keep only chapters whose title or summary
              contains this substring (case-insensitive).

    Returns:
        str: Markdown numbered list. Each line has the chapter number, title, filename link,
        and one-line summary.
    """
    chapters = CHAPTERS
    header = "# VectorMesh course chapters"
    if params.keyword:
        key = params.keyword.lower()
        chapters = [
            c for c in chapters if key in c.title.lower() or key in c.summary.lower()
        ]
        header = f"# VectorMesh chapters matching '{params.keyword}'"

    lines = [header, ""]
    for c in chapters:
        lines.append(f"{c.number}. **{c.title}** — {_chapter_ref(c)} — {c.summary}")
    if not chapters:
        lines.append("(no chapters matched)")
    return "\n".join(lines)


@mcp.tool(
    name="vectormesh_get_concept",
    annotations=ToolAnnotations(
        title="Get a VectorMesh course chapter",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def vectormesh_get_concept(params: GetConceptInput) -> str:
    """Fetch the full markdown of one VectorMesh course chapter.

    Args:
        params (GetConceptInput):
            - chapter (str): number, filename, stem, or title of the chapter.

    Returns:
        str: The chapter's full markdown, prefixed with a link back to it online. If the
        chapter is not found, returns an error listing valid chapter names.
    """
    chapter = _find_chapter(params.chapter)
    if chapter is None:
        names = ", ".join(f"{c.number}:{c.title}" for c in CHAPTERS)
        return (
            f"Error: no chapter matching '{params.chapter}'. "
            f"Call vectormesh_list_concepts first. Known chapters: {names}"
        )
    try:
        body = _read_doc(chapter.file)
    except FileNotFoundError:
        return f"Error: doc file '{chapter.file}' is listed in docs/README.md but missing on disk."

    header = f"_Further reading (share this link): {_doc_url(chapter.file)}_\n\n"
    return header + body


@mcp.tool(
    name="vectormesh_search",
    annotations=ToolAnnotations(
        title="Search VectorMesh course chapters",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def vectormesh_search(params: SearchInput) -> str:
    """Search every chapter for keyword(s) and return the best matches.

    Ranking is simple term-frequency: chapters are scored by how many times the query terms
    appear. Good for "where do the docs talk about X?" — e.g. "masked mean", "beartype", "ray
    tune", "chunk alignment".

    Args:
        params (SearchInput):
            - query (str): one or more keywords.
            - limit (int): max chapters to return (1-20, default 5).

    Returns:
        str: Markdown list of matching chapters, each with a match count and a short snippet
        around the first hit. "No matches" if nothing is found.
    """
    terms = [t for t in params.query.lower().split() if t]
    scored: list[tuple[int, Chapter, str]] = []

    for chapter in CHAPTERS:
        try:
            text = _read_doc(chapter.file)
        except FileNotFoundError:
            continue
        lower = text.lower()
        score = sum(lower.count(term) for term in terms)
        if score == 0:
            continue
        first = min(
            (lower.find(term) for term in terms if lower.find(term) != -1),
            default=-1,
        )
        start = max(0, first - 80)
        snippet = text[start : first + 120].replace("\n", " ").strip()
        scored.append((score, chapter, snippet))

    if not scored:
        return f"No matches for '{params.query}'."

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = [f"# Search results for '{params.query}'", ""]
    for score, chapter, snippet in scored[: params.limit]:
        lines.append(f"## {chapter.title} — {score} hit(s)")
        lines.append(f"Further reading: {_doc_url(chapter.file)}")
        lines.append(f"…{snippet}…")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The design conversation — five stages, walked through one at a time
# --------------------------------------------------------------------------- #


class DesignStage(str, Enum):
    """A stage in the pre-design conversation, in the order it must happen."""

    tensors = "tensors"
    enrichments = "enrichments"
    flows = "flows"
    search_space = "search_space"
    search_practices = "search_practices"


STAGE_ORDER: list[DesignStage] = [
    DesignStage.tensors,
    DesignStage.enrichments,
    DesignStage.flows,
    DesignStage.search_space,
    DesignStage.search_practices,
]

STAGE_TITLE: dict[DesignStage, str] = {
    DesignStage.tensors: "1. What do you actually have? (tensor shapes)",
    DesignStage.enrichments: "2. What else could you add? (enrichments)",
    DesignStage.flows: "3. Which compositions are even on the table? (flows)",
    DesignStage.search_space: "4. What are you actually uncertain about? (search space)",
    DesignStage.search_practices: "5. Before you trust any result (search practices)",
}


def _core() -> str:
    return _chapter_ref(_must_find_chapter("01"))


def _contracts() -> str:
    return _chapter_ref(_must_find_chapter("02"))


def _data() -> str:
    return _chapter_ref(_must_find_chapter("03"))


def _teaching() -> str:
    return _chapter_ref(_must_find_chapter("08"))


def _search_ch() -> str:
    return _chapter_ref(_must_find_chapter("09"))


STAGE_INTRO: dict[DesignStage, str] = {
    DesignStage.tensors: (
        "Before anything gets designed, pin down what shape the data is actually in. Most "
        f"downstream confusion — and most `BeartypeCallHintParamViolation`s ({_contracts()}) — "
        "traces back to skipping this."
    ),
    DesignStage.enrichments: (
        "Architecture is not the only lever. Before reaching for a fancier composition, ask "
        "what extra signal could be added at the data layer — often cheaper, and sometimes it "
        "makes the architecture question moot."
    ),
    DesignStage.flows: (
        "Only now — with the shapes and the streams named — does it make sense to ask which "
        f"compositions ({_core()} §1.6) are even possible. Picking a mechanism before this point "
        "is picking a hammer before knowing what the object is."
    ),
    DesignStage.search_space: (
        "Not everything from stages 1-3 needs to be searched — only what is genuinely "
        "undecided. This stage turns open questions into a search space, not the other way "
        "around."
    ),
    DesignStage.search_practices: (
        "Before any result from that search gets trusted, a few habits separate a real finding "
        f"from a rounding error. Full detail: {_search_ch()}."
    ),
}

# Each row: (question, why it matters, what a good answer names)
StageRow = tuple[str, str, str]

STAGE_ROWS: dict[DesignStage, list[StageRow]] = {
    DesignStage.tensors: [
        (
            "What is your raw input, per item, and roughly how big is one item?",
            "Determines which `Vectorizer` you need and whether chunking is even a concept for "
            "your data.",
            "the modality (text / image / timeseries window / tabular row) and a rough size",
        ),
        (
            f"After vectorizing ONE item, what shape do you get? ({_core()} §1.3)",
            "This is your starting rung on the tensor-flow ladder — everything else is a move "
            "from here.",
            "`(D,)` a single vector, or `(C, D)` a variable-length set of chunks",
        ),
        (
            "If it's `(C, D)`, how much does `C` vary across items?",
            "Decides whether `FixedPadding` or `DynamicPadding` is right, and whether "
            "truncation would lose real signal.",
            "a rough distribution, e.g. 'mostly under 20, a tail past 150'",
        ),
        (
            "What shape do you need going INTO the loss function?",
            "Names the far end of the ladder you're building toward, so you can check the "
            "whole path rather than only the start.",
            "`(B, K)` logits for classification, `(B, 1)` for regression, etc.",
        ),
        (
            "How many separate representations do you have of the same item?",
            "One stream stays a `Serial` chain. More than one is a fusion problem before it's "
            "an architecture problem (see stage 3).",
            "a count, and which encoder/feature-extractor produced each one",
        ),
    ],
    DesignStage.enrichments: [
        (
            "Is there a hand-designed, interpretable signal specific to your domain that a "
            "frozen encoder wouldn't see on its own?",
            "Interpretable isn't automatically worse — a hand-written regex vector beats a "
            "dense embedding on small data in this course's own notebooks.",
            "a candidate feature, and why it isn't redundant with the embedding",
        ),
        (
            "If your modality is sequential or time-based, have you considered spectral or "
            "decomposition features (FFT magnitude, trend/seasonality, lag or rolling stats)?",
            "VectorMesh doesn't ship these out of the box — only `RegexVectorizer` and "
            "`ImageVectorizer` exist today — so this is a build-it-yourself decision, the same "
            "shape as an existing `Vectorizer`, not a missing feature to wait for.",
            "the specific transform, and the shape it produces per item",
        ),
        (
            "Is a second pretrained encoder available that might see something different?",
            "Only worth the extra cost if it's likely to make *different* mistakes than the "
            "first — not just more of the same signal at a different resolution.",
            "the second encoder, and one concrete reason it might disagree with the first",
        ),
        (
            f"For each enrichment: is it a new column on an EXISTING cache, or a new cache "
            f"entirely? ({_core()} §1.7, {_data()})",
            "Extending a cache adds the cheap feature without recomputing the expensive "
            "embedding — that's the whole reason `metadata.json` records what's already there.",
            "which cache gets extended, or why a fresh one is actually needed",
        ),
        (
            "Is this enrichment something that should eventually be LEARNED instead of "
            f"hand-designed? ({_teaching()}, cross-cutting thread #2)",
            "Moving a decision across the learned-vs-designed line — e.g. mean pooling to "
            "attention pooling — is the core creative move this course is built to teach.",
            "which side of the line it's on today, and whether that's a temporary choice",
        ),
    ],
    DesignStage.flows: [
        (
            "Given stage 1: do you have one stream or several, and which?",
            "One stream -> a `Serial` chain is probably enough. Several -> there's a fusion "
            "decision before there's an architecture decision.",
            "the count and identity of each stream",
        ),
        (
            "If several: do they fuse per-chunk (before aggregation) or per-document (after)?",
            "Fusing before aggregation lets cross-stream interaction happen at the token/chunk "
            "level; fusing after is cheaper and usually the safer default.",
            "the choice, and the connector it implies (`Concatenate3D` before vs "
            "`Concatenate2D`/`Stack2D` after)",
        ),
        (
            f"Where exactly does reduction happen? ({_core()} §1.3 — aggregation is the ONLY "
            "rank-reducing step)",
            "This is the single most consequential structural decision in a pipeline — get it "
            "vague and everything downstream is guesswork.",
            "which aggregator, at which point in the pipeline",
        ),
        (
            "Is there a real reason some inputs need MORE computation than others (conditional "
            "computation), or is a uniform transform enough?",
            f"This is the question that justifies climbing the `Skip -> Gate -> Highway -> MoE` "
            f"ladder ({_teaching()}, cross-cutting thread #3) at all — not 'MoE sounds "
            "powerful'.",
            "plain transform, or which rung and the concrete reason",
        ),
        (
            "Can you sketch the whole thing as nested `Serial`/`Parallel` calls, using only "
            f"existing components? ({_core()} §1.6)",
            "If you can't, either the idea needs a genuinely new component, or it isn't concrete "
            "enough yet to build.",
            "a few lines of pseudocode naming real component classes",
        ),
    ],
    DesignStage.search_space: [
        (
            "Which decisions from stages 1-3 are you genuinely UNSURE about, versus already "
            "decided?",
            "A search space should only contain what you don't already know the answer to — "
            "searching a decided axis just wastes trials.",
            "the 2-5 axes actually in question",
        ),
        (
            f"For each axis: is it independent, or conditional on another axis's value? "
            f"({_search_ch()} §9.4)",
            "Conditional axes (e.g. `n_experts` only matters if `mechanism == 'moe'`) get "
            "wastefully sampled by most search algorithms — name them so the waste isn't a "
            "surprise later.",
            "which axes are conditional on which",
        ),
        (
            "Which axes could CONFOUND the comparison you actually care about, if left fixed?",
            "If `lr` is fixed while searching `mechanism`, a win might just mean the fixed "
            "learning rate happened to suit one mechanism.",
            "at least one confounder being included for this reason",
        ),
        (
            "Roughly how many trials can you afford, given how long one run takes?",
            "Sets `num_samples` honestly instead of picking a round number that sounds fine.",
            "a number, and the arithmetic behind it (trials x seconds-per-trial)",
        ),
        (
            "What is the ONE metric the search optimizes for?",
            "`tune.TuneConfig` needs exactly one metric and mode — decide before writing the "
            "search space, not after reading confusing results.",
            "the metric name and whether it's maximised or minimised",
        ),
    ],
    DesignStage.search_practices: [
        (
            f"What's the cheapest possible baseline, with NO training at all? ({_search_ch()} "
            "§9.2)",
            "A cosine-similarity/nearest-centroid classifier on the frozen embeddings gives you "
            "the floor before a single epoch runs. If a trained model can't clear it, the "
            "architecture usually isn't the problem.",
            "the floor score and how it was computed",
        ),
        (
            f"What's your TRAINED baseline, and over how many seeds? ({_search_ch()} §9.3)",
            "This is the number every later search result gets measured against.",
            "mean +/- std over at least 3 seeds",
        ),
        (
            f"How are you deciding when to stop training a trial? ({_search_ch()} §9.5)",
            "A fixed, guessed epoch count silently under-trains some configs and over-trains "
            "others — which then looks like an architecture effect. Default to a generous "
            "epoch CEILING plus an adaptive rule (plateau scheduler + early stop) instead.",
            "the stopping rule, named explicitly — not just an epoch number",
        ),
        (
            f"How will a search result become a CLAIM, not just a number? ({_search_ch()} §9.7)",
            "Effect size in units of the trained baseline's own sigma, compared against the "
            "no-training floor for context — not the raw difference.",
            "the formula that will be applied",
        ),
        (
            f"Where and how often will you log? ({_search_ch()} §9.8)",
            "Once, after the whole search finishes — never per-trial into a shared sqlite "
            "backend, which concurrent trial processes will corrupt.",
            "the logging plan",
        ),
    ],
}

# In-memory record of what the student said at each stage, for THIS server process (i.e. this
# session). Resets when the server restarts — the design conversation is meant to happen in one
# sitting, not to persist silently across days.
_progress: dict[str, str] = {}

MIN_ANSWER_CHARS = 20


def _next_stage(stage: DesignStage) -> Optional[DesignStage]:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


def _roadmap() -> str:
    lines = [
        "# Designing something new with VectorMesh — a five-stage conversation",
        "",
        "This is deliberately not a single document to skim. Work through these five stages with "
        "the student, in order, one call to this tool per stage — each later stage assumes the "
        "earlier ones were actually reasoned through together, not skipped.",
        "",
    ]
    for i, stage in enumerate(STAGE_ORDER, start=1):
        mark = "✅" if stage.value in _progress else "▢"
        lines.append(f"{mark} {i}. **{STAGE_TITLE[stage]}**")
    lines += [
        "",
        'Call this tool again with `stage="tensors"` to begin — always start there for a new '
        "idea, even if it feels obvious; stage 3 depends on what stage 1 actually established.",
    ]
    return "\n".join(lines)


def _stage_table(stage: DesignStage) -> str:
    rows = STAGE_ROWS[stage]
    lines = [
        f"# Stage: {STAGE_TITLE[stage]}",
        "",
        STAGE_INTRO[stage],
        "",
        "| Question | Why it matters | A good answer names |",
        "|---|---|---|",
    ]
    for q, why, names in rows:
        lines.append(f"| {q} | {why} | {names} |")
    lines += [
        "",
        "**How to use this table:** ask the student these questions yourself — together or one "
        "at a time, your judgement. Do NOT answer them yourself, and do not treat this as done "
        "until they've engaged in their own words. Once they have, call "
        f'`vectormesh_design_checklist` again with `stage="{stage.value}"` and '
        "`answers=<a summary of what they actually said>` to record it and receive the next "
        "stage.",
    ]
    return "\n".join(lines)


def _completion_summary() -> str:
    lines = [
        "# All five stages recorded",
        "",
        "What was reasoned through, in order:",
        "",
    ]
    for stage in STAGE_ORDER:
        lines.append(f"**{STAGE_TITLE[stage]}**")
        lines.append(f"> {_progress.get(stage.value, '(not recorded)')}")
        lines.append("")
    lines += [
        "The student now has enough to sketch a first version themselves. If the 'flows' or "
        "'search_space' answers named more than one plausible variant, call "
        "`vectormesh_search_method` (or run the `search` prompt) next, rather than picking one "
        "by feel.",
    ]
    return "\n".join(lines)


class DesignChecklistInput(BaseModel):
    """Input for the staged design-checklist tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    stage: Optional[DesignStage] = Field(
        default=None,
        description=(
            "Which stage to open. Omit on the very first call to get the roadmap and confirm "
            "where to start (always 'tensors' for a new idea). One of: tensors, enrichments, "
            "flows, search_space, search_practices."
        ),
    )
    answers: Optional[str] = Field(
        default=None,
        description=(
            "A summary, in your own words, of what the student actually said when asked this "
            "stage's questions. Omit to just receive the stage's checklist. Provide it on a "
            "second call for the SAME stage to record it and advance to the next stage. Do not "
            "fabricate this on the student's behalf — it must reflect their real answers."
        ),
        max_length=4000,
    )


@mcp.tool(
    name="vectormesh_design_checklist",
    annotations=ToolAnnotations(
        title="Walk through the pre-design checklist with a student, one stage at a time",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def vectormesh_design_checklist(params: DesignChecklistInput) -> str:
    """Coach a student through five stages of thinking BEFORE they design anything.

    This is a conversation, not a document dump: each call returns exactly ONE stage's checklist
    table. Ask the student those questions yourself, in the chat, and wait for real answers — do
    not answer on their behalf. Once they've responded, call this tool again with the SAME
    `stage` and their answers summarised in `answers`; that records the stage and returns the
    next one. Progress is held in memory for this server process only (this session) and is not
    persisted across restarts.

    The five stages, in the order they must happen:
        1. tensors            — what shape is the data actually in, right now?
        2. enrichments        — what extra signal could be added before touching architecture?
        3. flows              — given 1 and 2, which compositions (Serial/Parallel/gating) are
                                 even on the table?
        4. search_space       — of the open questions from 1-3, which are worth searching?
        5. search_practices   — how will a search result be trusted, not just read?

    Args:
        params (DesignChecklistInput):
            - stage (Optional[DesignStage]): which stage to open; omit for the roadmap.
            - answers (Optional[str]): the student's answers to the CURRENT stage, in your
              words; omit to just receive the stage's checklist.

    Returns:
        str: The roadmap (no stage given), a stage's checklist table (stage given, no answers
        yet), or a recorded confirmation plus the next stage's checklist (stage + answers given).
        After the last stage, returns a recap of everything recorded.
    """
    if params.stage is None:
        return _roadmap()

    stage = params.stage
    if not params.answers or len(params.answers.strip()) < MIN_ANSWER_CHARS:
        return _stage_table(stage)

    _progress[stage.value] = params.answers.strip()
    nxt = _next_stage(stage)
    if nxt is None:
        return _completion_summary()

    recap = f"✅ Recorded for **{STAGE_TITLE[stage]}**: {params.answers.strip()}\n\n"
    return recap + _stage_table(nxt)


# --------------------------------------------------------------------------- #
# The search method — an ordered, linked checklist for executing a Ray Tune search
# --------------------------------------------------------------------------- #


class SearchMethodInput(BaseModel):
    """Input for the search-method tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    idea: str = Field(
        default="my architecture idea",
        description=(
            "One-sentence description of what the student wants to search over, e.g. "
            "'gate vs highway vs MoE on the aggregated embedding'. Purely for framing the "
            "returned method's header; the method itself is generic."
        ),
        max_length=300,
    )


@mcp.tool(
    name="vectormesh_search_method",
    annotations=ToolAnnotations(
        title="Get the method for a Ray Tune architecture search",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def vectormesh_search_method(params: SearchMethodInput) -> str:
    """Return the method for setting up a Ray Tune architecture search over VectorMesh components.

    Call this once `vectormesh_design_checklist` has been worked through (particularly the
    search_space and search_practices stages) and there is more than one plausible design with
    no principled way to pick between them by inspection. This returns the *order of operations*
    and links to the full chapter, which has a runnable `trainable()` template built from this
    library's own gating components; fetch it with `vectormesh_get_concept('09')` when ready to
    write code.

    Args:
        params (SearchMethodInput):
            - idea (str): one-sentence description of what is being searched, used only to
              frame the header.

    Returns:
        str: Markdown checklist, in the order the steps must happen, each linked to the section
        of the architecture-search chapter it comes from.
    """
    ch = _must_find_chapter("09")
    url = _doc_url(ch.file)

    lines = [
        f"# Searching: {params.idea}",
        "",
        "Do these in order — reversing the order (e.g. reading results before establishing a "
        f"baseline) is the most common way a search stops being trustworthy. Full chapter: {url}",
        "",
        "1. **Get the no-training floor first.** A cosine-similarity/nearest-centroid "
        "classifier on the frozen embeddings, no gradient step anywhere. If a trained pipeline "
        f"can't clear it, the architecture is probably not the problem. ({url}"
        "#92-the-cheapest-baseline-no-training-at-all)",
        "",
        "2. **Then a trained baseline, with a seed spread.** Run the simplest plausible "
        "pipeline over 3-5 seeds *before* writing any search space. Its `std` is the number "
        f"every later claim gets measured against. ({url}#93-a-trained-baseline-with-a-seed-spread)",
        "",
        "3. **Design the search space as one dict.** Categorical axes with `tune.choice`, "
        "continuous ones with `tune.uniform`/`tune.loguniform`. Include any axis that could "
        "confound the comparison you actually care about (e.g. `lr` and `hidden` alongside "
        f"`mechanism`, if the question is about `mechanism`). ({url}#94-the-search-space)",
        "",
        "4. **Write the trainable assuming an empty process, with an epoch CEILING plus an "
        "adaptive stop.** A Ray trial runs in its own process — rebuild the cache load, the "
        "pipeline, and the optimizer from scratch every call. Set `epochs` generously and let a "
        "plateau scheduler + patience-based early stop decide the real per-trial budget, so no "
        f"config is silently under- or over-trained. ({url}#95-the-trainable-function)",
        "",
        "5. **Read the result one axis at a time**, not just the top row. Group by each axis and "
        "check whether its groups actually separate before treating that axis's 'best' value as "
        f"meaningful. ({url}#96-reading-a-search-one-axis-at-a-time)",
        "",
        "6. **Turn a difference into a claim by dividing by the baseline's own sigma**, not by "
        "comparing to zero — and sanity-check against the no-training floor from step 1. Rerun "
        f"the winning config over several seeds before writing anything down as a conclusion. "
        f"({url}#97-turning-a-difference-into-a-claim)",
        "",
        "7. **Log once, after `tuner.fit()` returns** — never per-trial if the logging backend "
        f"uses local sqlite (e.g. mlflow's default); concurrent trial processes will corrupt it. "
        f"({url}#98-logging-discipline-one-summary-per-search)",
        "",
        "8. **If launching under `uv run`**, set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before "
        f"`import ray`, or Ray's workers will look for a `pyproject.toml` in the wrong place. "
        f"({url}#99-a-uv-run-gotcha-worth-knowing)",
        "",
        "Call `vectormesh_get_concept('09')` for the runnable `trainable()` template — it "
        "searches over this library's own `Skip -> Gate -> Highway -> MoE` ladder as the "
        "worked example, including the plateau/early-stop loop from step 4.",
    ]
    return "\n".join(lines)


@mcp.prompt(
    name="design",
    description="Start a five-stage design conversation for a new VectorMesh component or architecture.",
)
def design_prompt(idea: str = "a new component or architecture idea") -> str:
    """A one-command design coaching session.

    Args:
        idea: what the student wants to design (a short description). Defaults to a generic
            framing if the student has not narrowed it down yet.
    """
    return (
        f"I want to design {idea} using VectorMesh. Coach me through it as a mentor helping me "
        "think, not as someone writing the code for me.\n\n"
        "Step 1: call the `vectormesh_design_checklist` tool with no stage to get the roadmap.\n"
        'Step 2: call it again with `stage="tensors"` and ask me those questions yourself, in '
        "the chat. Wait for my real answers — don't answer for me.\n"
        "Step 3: once I've answered, call the tool again with the same stage and a summary of "
        "what I said in `answers`, to record it and get the next stage. Repeat this for every "
        "stage, in order — never skip ahead, and never fetch two stages before I've answered the "
        "first.\n"
        "Step 4: after the last stage, summarise what we worked out together and tell me whether "
        "I should run the `search` prompt next (if there's more than one plausible variant) or "
        "just start building."
    )


@mcp.prompt(
    name="search",
    description="Coach me through setting up a Ray Tune search over a VectorMesh architecture.",
)
def search_prompt(idea: str = "my architecture idea") -> str:
    """A one-command search coaching session.

    Args:
        idea: what the student wants to search over (a short description).
    """
    return (
        f"I want to search over {idea} using Ray Tune, against VectorMesh components. Coach me "
        "through it in the right order.\n\n"
        "Step 1: call the `vectormesh_search_method` tool with my idea to get the ordered "
        "checklist.\n"
        "Step 2: follow it with me in order — starting with the no-training floor and the "
        "trained baseline's seed spread, never skipping ahead to the search space or the "
        "results.\n"
        "Step 3: when we get to writing the trainable function, call "
        "`vectormesh_get_concept('09')` for the runnable template and adapt it to my idea rather "
        "than writing one from scratch."
    )


# --------------------------------------------------------------------------- #
# Resources — each chapter is also addressable by URI
# --------------------------------------------------------------------------- #


@mcp.resource("vectormesh://concept/{name}")
def concept_resource(name: str) -> str:
    """Expose a chapter's raw markdown as an MCP resource.

    `name` may be the chapter number, the doc filename ('04-components.md'), its stem, or the
    chapter title — the same lookup as vectormesh_get_concept.
    """
    chapter = _find_chapter(name)
    if chapter is None:
        raise FileNotFoundError(f"No chapter named '{name}'")
    return _read_doc(chapter.file)


if __name__ == "__main__":
    mcp.run()
