#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastmcp>=2.14.2"]
# ///
"""MCP server that exposes the VectorMesh course as design/search coaching tools.

The course is a set of markdown chapters in `docs/`, indexed by the reading-order table in
`docs/README.md`. Unlike a code-review server, this one is forward-looking: it coaches a student
through two activities the docs describe but a chat window cannot enforce on its own —

- **designing** a deep learning architecture for a problem from the library's own principles,
  worked through as an eight-stage conversation (problem -> tensors -> pretrained models ->
  context -> architecture -> search space -> search practices -> results) rather than handed
  over as one document to skim, offering only the architectures of the student's lesson or
  earlier;
- **searching** an architecture with Ray Tune once there is more than one plausible design and no
  principled way to pick between them by inspection.

The design conversation (`vectormesh_design_checklist`) is deliberately stateful *within one
server process* (i.e. for the current session): it returns one stage's checklist at a time, and
only advances once the assistant reports back what the student actually said. Code comes
after the last stage. The point is to make
the assistant cooperate and think *with* the student against this library's own framework, not
hand them a finished pipeline.

It also exposes the chapters themselves for lookup and keyword search, the same way the
`codestyle` MCP server exposes its guidelines. It reads `docs/` from disk when they sit next to
this script; when launched standalone (e.g. via `uv run` straight from a URL, with no course files
nearby) it fetches the same content from the public repo over HTTP instead. No build step, no
embeddings, no database.
"""

from __future__ import annotations

import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
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

    number: int  # 1..10
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


def _find_chapter(query: str) -> Chapter | None:
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

    keyword: str | None = Field(
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
    annotations={
        "title": "List VectorMesh course chapters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def vectormesh_list_concepts(params: ListConceptsInput) -> str:
    """List the VectorMesh course chapters, in reading order.

    Use this to discover what the course covers before fetching a chapter with
    vectormesh_get_concept, or before calling vectormesh_design_checklist / vectormesh_search_method
    to see which chapters they draw on.

    Args:
        params (ListConceptsInput):
            - keyword (str | None): if set, keep only chapters whose title or summary
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
    annotations={
        "title": "Get a VectorMesh course chapter",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
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
    annotations={
        "title": "Search VectorMesh course chapters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
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
# The design conversation — eight stages, walked through one at a time
# --------------------------------------------------------------------------- #


class DesignStage(str, Enum):
    """A stage in the pre-design conversation, in the order it must happen."""

    problem = "problem"
    tensors = "tensors"
    pretrained = "pretrained"
    context = "context"
    architecture = "architecture"
    search_space = "search_space"
    search_practices = "search_practices"
    results = "results"


STAGE_ORDER: list[DesignStage] = [
    DesignStage.problem,
    DesignStage.tensors,
    DesignStage.pretrained,
    DesignStage.context,
    DesignStage.architecture,
    DesignStage.search_space,
    DesignStage.search_practices,
    DesignStage.results,
]

STAGE_TITLE: dict[DesignStage, str] = {
    DesignStage.problem: "1. What is the problem, and what is good enough? (problem)",
    DesignStage.tensors: "2. What do you actually have? (tensor shapes)",
    DesignStage.pretrained: "3. Who has already learned this? (pretrained models)",
    DesignStage.context: "4. What else could you add? (context)",
    DesignStage.architecture: "5. Which compositions are on the table? (architecture)",
    DesignStage.search_space: "6. What are you actually uncertain about? (search space)",
    DesignStage.search_practices: "7. Before you trust any result (search practices)",
    DesignStage.results: "8. Reading and sharing the runs (results)",
}

# What the student has met in each lesson of the course. `components` are names exported by
# vectormesh (tests/test_mcp.py imports every one, so the table cannot drift from the
# library); `ideas` are the techniques and tools of that lesson that are not a component.
LESSONS: dict[int, dict[str, tuple[str, ...]]] = {
    1: {
        "components": (
            "Serial",
            "Parallel",
            "Concatenate2D",
            "NeuralNet",
            "Projection",
            "GaussianNoise",
        ),
        "ideas": (
            "linear and MLP heads on frozen pretrained image encoders (caches)",
            "ray tune search",
            "mlflow",
        ),
    },
    2: {
        "components": (),
        "ideas": ("windowing", "1D convolution", "RNN / GRU / LSTM", "N-BEATS", "MASE"),
    },
    3: {
        "components": ("Gate", "Highway", "MoE", "SparseMoE", "Skip"),
        "ideas": ("scaling curves", "search algorithms"),
    },
    4: {
        "components": (
            "Attention",
            "TransformerBlock",
            "MeanAggregator",
            "MaskedMeanAggregator",
            "AttentionAggregator",
            "RNNAggregator",
            "FixedPadding",
            "DynamicPadding",
            "Concatenate3D",
            "RegexVectorizer",
        ),
        "ideas": ("tokenization", "chunk aggregators", "regex feature streams"),
    },
    5: {
        "components": ("Siamese", "JEPA", "PatchImageVectorizer"),
        "ideas": ("self-supervised learning", "patch tokens"),
    },
    6: {
        "components": (),
        "ideas": (
            "probabilistic heads",
            "conformal prediction",
            "cost-driven decisions",
        ),
    },
    7: {
        "components": (),
        "ideas": (
            "time-series foundation models",
            "LLMs as judge",
            "the cascade (router to head or LLM)",
        ),
    },
}

FIRST_LESSON = min(LESSONS)
LAST_LESSON = max(LESSONS)
# The lesson that introduces the gating ladder the chapter 9 template searches over.
GATING_LESSON = next(n for n in sorted(LESSONS) if "Gate" in LESSONS[n]["components"])


def _core() -> str:
    return _chapter_ref(_must_find_chapter("01"))


def _contracts() -> str:
    return _chapter_ref(_must_find_chapter("02"))


def _data() -> str:
    return _chapter_ref(_must_find_chapter("03"))


def _arch_ch() -> str:
    return _chapter_ref(_must_find_chapter("05"))


def _teaching() -> str:
    return _chapter_ref(_must_find_chapter("08"))


def _search_ch() -> str:
    return _chapter_ref(_must_find_chapter("09"))


def _reporters() -> str:
    return _chapter_ref(_must_find_chapter("10"))


def _patterns_ref(lesson: int) -> str:
    """Point at §5.7 only for the rows of its table the student has met."""
    if lesson >= 4:
        return f"{_arch_ch()} §5.7"
    rows = (
        "'Two document-level representations' and 'Small training set, frozen features'"
    )
    if lesson >= 3:
        rows = (
            "'Two document-level representations', 'Want depth without instability', 'Want "
            "the model to learn whether to transform', 'Want specialisation across inputs' "
            "and 'Small training set, frozen features'"
        )
    return f"{_arch_ch()} §5.7 — at this lesson only its rows {rows}"


STAGE_INTRO: dict[DesignStage, str] = {
    DesignStage.problem: (
        "Before any data gets touched, pin down what the model is for and how good it has to "
        "be. Every later stage is measured against the metric and the acceptance bound named "
        "here; without them, no model is ever finished and no simpler one is ever good enough."
    ),
    DesignStage.tensors: (
        "Now pin down what shape the data is actually in. Most downstream confusion — and most "
        f"`BeartypeCallHintParamViolation`s ({_contracts()}) — traces back to skipping this."
    ),
    DesignStage.pretrained: (
        f"Embed once, reuse many times ({_core()}): a pretrained encoder may already have "
        "learned most of what this task needs. You may search the HuggingFace Hub for the "
        "student, but the rationale stays theirs. The rule for this stage: the simplest model "
        "that meets the acceptance bound."
    ),
    DesignStage.context: (
        "Architecture is not the only lever. Before reaching for a fancier composition, ask "
        "what extra signal could be added at the data layer — often cheaper, and sometimes it "
        "makes the architecture question moot."
    ),
    DesignStage.architecture: (
        "Only now — with the shapes and the streams named — does it make sense to ask which "
        f"compositions ({_core()} §1.6, {{patterns}}) are possible. Start with the "
        "baseline, then go up in complexity only when the baseline falls short of the "
        "acceptance bound."
    ),
    DesignStage.search_space: (
        "Not everything from the earlier stages needs to be searched — only what is genuinely "
        "undecided. This stage turns open questions into a search space, not the other way "
        "around."
    ),
    DesignStage.search_practices: (
        "Before any result from that search gets trusted, a few habits separate a real finding "
        f"from a rounding error. Full detail: {_search_ch()}."
    ),
    DesignStage.results: (
        "A search is finished when someone else can read it: the runs answer the hypothesis, "
        f"and they are shared so that others can see who made them and what was explored "
        f"({_search_ch()} §9.6-9.7, {_reporters()})."
    ),
}


@dataclass(frozen=True)
class Row:
    """One row of a stage's table: question / why it matters / what a good answer names.

    `id` stays the same when the text is reworded. `since` is the first lesson this text
    applies to. A stage may hold several rows with one id: the one with the highest `since`
    at or below the student's lesson is shown, and an id with none is left out.
    """

    id: str
    question: str
    why: str
    names: str
    since: int = FIRST_LESSON


STAGE_ROWS: dict[DesignStage, list[Row]] = {
    DesignStage.problem: [
        Row(
            "target",
            "What is predicted, from what, and for whom?",
            "Names both ends of the pipeline and the person who acts on its output, which "
            "decides which mistakes are expensive.",
            "the target, the input, and who uses the prediction",
        ),
        Row(
            "need_dl",
            "Is deep learning needed at all? Could a simple algorithm or symbolic logic do it?",
            "A regex extracts an e-mail address better than an agentic model. A learned model "
            "earns its place only where a rule cannot be written down.",
            "the simplest non-learned approach, and why it does or does not suffice",
        ),
        Row(
            "metric",
            "Which metrics, and why those? (accuracy, precision, recall, F1; a task-specific "
            "one such as MASE for forecasting)",
            "The metric decides what 'better' means for every comparison that follows.",
            "the metric(s), tied to which mistake costs the most",
        ),
        Row(
            "bound",
            "What is the acceptance bound: the lowest score that is good enough to use?",
            "95%? 90%? In some settings 51% already pays, because slightly better than guessing "
            "beats guessing. The bound is what 'the simplest model that meets it' is measured "
            "against.",
            "a number on the chosen metric, and why that number",
        ),
        Row(
            "floor",
            f"What would a no-training floor score? ({_search_ch()} §9.2)",
            "If the floor already clears the bound, the rest of this conversation is about "
            "cost, not about accuracy.",
            "the floor method (e.g. cosine centroid) and its score or an estimate",
        ),
    ],
    DesignStage.tensors: [
        Row(
            "raw_input",
            "What is your raw input, per item, and roughly how big is one item?",
            "Determines which `Vectorizer` or frozen encoder turns it into a vector.",
            "the modality (e.g. an image for a frozen image encoder, a tabular row) and a "
            "rough size",
        ),
        Row(
            "raw_input",
            "What is your raw input, per item, and roughly how big is one item?",
            "Determines which `Vectorizer` or frozen encoder turns it into a tensor, and "
            "whether a window over time is part of the item.",
            "the modality (image / timeseries window / tabular row) and a rough size",
            since=2,
        ),
        Row(
            "raw_input",
            "What is your raw input, per item, and roughly how big is one item?",
            "Determines which `Vectorizer` you need and whether chunking is even a concept for "
            "your data.",
            "the modality (text / image / timeseries window / tabular row) and a rough size",
            since=4,
        ),
        Row(
            "preprocessor",
            "What preprocessor turns the raw data into a tensor?",
            "The preprocessor fixes the tensor every later stage works with.",
            "the `Vectorizer` (or other preprocessor) by name",
        ),
        Row(
            "shape",
            f"After vectorizing ONE item, what shape do you get, and what rank? ({_core()} §1.3)",
            "This is your starting rung on the tensor-flow ladder — everything else is a move "
            "from here.",
            "`(D,)`: one vector per item from the frozen encoder",
        ),
        Row(
            "shape",
            f"After vectorizing ONE item, what shape do you get, and what rank? ({_core()} §1.3)",
            "This is your starting rung on the tensor-flow ladder — everything else is a move "
            "from here.",
            "`(D,)` a single vector, or `(T, F)` a window of time steps",
            since=2,
        ),
        Row(
            "shape",
            f"After vectorizing ONE item, what shape do you get, and what rank? ({_core()} §1.3)",
            "This is your starting rung on the tensor-flow ladder — everything else is a move "
            "from here.",
            "`(D,)` a single vector, `(T, F)` a window, or `(C, D)` a variable-length set of "
            "chunks",
            since=4,
        ),
        Row(
            "axes",
            "What does each axis of that tensor mean?",
            "An axis without a meaning cannot be reduced, padded or fused on purpose.",
            "a name per axis (e.g. the encoder's features)",
        ),
        Row(
            "axes",
            "What does each axis of that tensor mean?",
            "An axis without a meaning cannot be reduced, padded or fused on purpose.",
            "a name per axis (e.g. time steps, features)",
            since=2,
        ),
        Row(
            "axes",
            "What does each axis of that tensor mean?",
            "An axis without a meaning cannot be reduced, padded or fused on purpose.",
            "a name per axis (e.g. chunks, time steps, features)",
            since=4,
        ),
        Row(
            "chunk_spread",
            "If it's `(C, D)`, how much does `C` vary across items?",
            "Decides whether `FixedPadding` or `DynamicPadding` is right, and whether "
            "truncation would lose real signal.",
            "a rough distribution, e.g. 'mostly under 20, a tail past 150'",
            since=4,
        ),
        Row(
            "loss_shape",
            "What shape do you need going INTO the loss function?",
            "Names the far end of the ladder you're building toward, so you can check the "
            "whole path rather than only the start.",
            "`(B, K)` logits for classification, `(B, 1)` for regression, etc.",
        ),
        Row(
            "representations",
            "How many separate representations do you have of the same item?",
            "One stream stays a `Serial` chain. More than one is a fusion problem before it's "
            "an architecture problem (see stage 5).",
            "a count, and which encoder/feature-extractor produced each one",
        ),
    ],
    DesignStage.pretrained: [
        Row(
            "candidate",
            "Is there a pretrained or foundation model for this modality?",
            "Training a head on frozen vectors costs seconds; training an encoder costs days. "
            "Search the HuggingFace Hub with the student if they ask.",
            "a candidate model, by name",
        ),
        Row(
            "trained_on",
            "What was it trained on, and does that resemble this data?",
            "An encoder only helps as far as its training data resembles yours.",
            "its training data, and one way this data differs",
        ),
        Row(
            "hardware",
            "Can this hardware run it?",
            "The encoder runs once, to fill the cache — but it has to run that once.",
            "the model size and where it will run",
        ),
        Row(
            "simplest",
            "Is the best model worth it, when a simpler one already clears the bound with "
            "cosine similarity alone?",
            "The simplest model that meets the acceptance bound wins; a larger one has to "
            "earn its cost.",
            "the cosine floor of the simpler model, against the bound",
        ),
    ],
    DesignStage.context: [
        Row(
            "context_source",
            "What contextual information could be added: other sensors, domain knowledge?",
            "Often absent for a handed-out dataset, often present for your own. Extra signal "
            "at the data layer is usually cheaper than extra architecture.",
            "a candidate source, or why there is none",
        ),
        Row(
            "domain_signal",
            "Is there a hand-designed, interpretable signal specific to your domain that a "
            "frozen encoder wouldn't see on its own?",
            "Interpretable isn't automatically worse: a hand-designed feature can carry signal "
            "the encoder misses, at a fraction of the cost.",
            "a candidate feature, and why it isn't redundant with the embedding",
        ),
        Row(
            "domain_signal",
            "Is there a hand-designed, interpretable signal specific to your domain that a "
            "frozen encoder wouldn't see on its own?",
            "Interpretable isn't automatically worse — a hand-written regex vector beats a "
            "dense embedding on small data in this course's own notebooks.",
            "a candidate feature, and why it isn't redundant with the embedding",
            since=4,
        ),
        Row(
            "spectral",
            "If your modality is sequential or time-based, have you considered spectral or "
            "decomposition features (FFT magnitude, trend/seasonality, lag or rolling stats)?",
            "The library ships no spectral or decomposition vectorizer. Building one is a "
            "build-it-yourself decision: write it in the shape of an existing vectorizer, "
            "so its output lands in a cache like any other.",
            "the specific transform, and the shape it produces per item",
            since=2,
        ),
        Row(
            "second_encoder",
            "Is a second pretrained encoder available that might see something different?",
            "Only worth the extra cost if it's likely to make *different* mistakes than the "
            "first — not just more of the same signal at a different resolution.",
            "the second encoder, and one concrete reason it might disagree with the first",
        ),
        Row(
            "multimodal",
            "Does the added context make this multimodal, and where would the streams meet?",
            "Two kinds of vector need a meeting point; naming it here is what stage 5 builds.",
            "the streams, and the point where they meet",
        ),
        Row(
            "cache_column",
            f"For each enrichment: is it a new column on an EXISTING cache, or a new cache "
            f"entirely? ({_core()} §1.7, {_data()})",
            "Extending a cache adds the cheap feature without recomputing the expensive "
            "embedding — that's the whole reason `metadata.json` records what's already there.",
            "which cache gets extended, or why a fresh one is actually needed",
        ),
        Row(
            "learned",
            "Is this enrichment something that should eventually be LEARNED instead of "
            f"hand-designed? ({_teaching()}, cross-cutting thread #2)",
            "Moving a decision across the learned-vs-designed line is the core creative move "
            "this course is built to teach.",
            "which side of the line it's on today, and whether that's a temporary choice",
        ),
        Row(
            "learned",
            "Is this enrichment something that should eventually be LEARNED instead of "
            f"hand-designed? ({_teaching()}, cross-cutting thread #2)",
            "Moving a decision across the learned-vs-designed line — e.g. mean pooling to "
            "attention pooling — is the core creative move this course is built to teach.",
            "which side of the line it's on today, and whether that's a temporary choice",
            since=4,
        ),
    ],
    DesignStage.architecture: [
        Row(
            "baseline",
            "What is the baseline: the simplest head on the tensor from stage 2?",
            "Every step up in complexity is measured against it; without it, a complex model "
            "has nothing to beat.",
            "the baseline pipeline, by component name",
        ),
        Row(
            "streams",
            "Given stages 2 and 4: do you have one stream or several, and which?",
            "One stream -> a `Serial` chain is probably enough. Several -> there's a fusion "
            "decision before there's an architecture decision.",
            "the count and identity of each stream",
        ),
        Row(
            "connect",
            "If several: how are vectors of different kinds connected?",
            "The meeting point decides which interactions between the streams the model can "
            "learn, and what that costs.",
            "the connector by component name, and the level at which the streams meet",
        ),
        Row(
            "fuse_level",
            "If several: do they fuse per-chunk (before aggregation) or per-document (after)?",
            "Fusing before aggregation lets cross-stream interaction happen at the token/chunk "
            "level; fusing after is cheaper and usually the safer default.",
            "the choice, and the connector it implies (`Concatenate3D` before vs "
            "`Concatenate2D`/`Stack2D` after)",
            since=4,
        ),
        Row(
            "reduction",
            f"Where exactly does reduction happen? ({_core()} §1.3 — aggregation is the ONLY "
            "rank-reducing step)",
            "This is the single most consequential structural decision in a pipeline — get it "
            "vague and everything downstream is guesswork.",
            "which aggregator, at which point in the pipeline",
            since=4,
        ),
        Row(
            "reshape",
            "May the tensor be reconsidered to fit an architecture (a time series reshaped so "
            "that a 1D or 2D convolution can scan it)?",
            "The tensor from stage 2 is the obvious one, not the only one.",
            "the reshape, and the architecture it opens up",
            since=2,
        ),
        Row(
            "conditional",
            "Is there a real reason some inputs need MORE computation than others (conditional "
            "computation), or is a uniform transform enough?",
            f"This is the question that justifies climbing the `Skip -> Gate -> Highway -> MoE` "
            f"ladder ({_teaching()}, cross-cutting thread #3) at all — not 'MoE sounds "
            "powerful'.",
            "plain transform, or which rung and the concrete reason",
            since=3,
        ),
        Row(
            "width",
            "Which range for the hidden width (`hidden_size`)?",
            "Much data and compute justify a wide model; little data and compute push the "
            "width down. Think in a range, not a single number.",
            "a range for `hidden_size`, and the data size behind it",
        ),
        Row(
            "width",
            "Which range for the hidden width (`hidden_size`)?",
            "Scaling laws: much data and compute justify a large model; little data and "
            "compute push it down. Think in a range, not a single number.",
            "a range for `hidden_size`, and the data size behind it",
            since=3,
        ),
        Row(
            "sketch",
            "Can you sketch the whole thing as nested `Serial`/`Parallel` calls, using only "
            f"existing components? ({_core()} §1.6)",
            "If you can't, either the idea needs a genuinely new component, or it isn't concrete "
            "enough yet to build.",
            "a few lines of pseudocode naming real component classes",
        ),
        Row(
            "sketch",
            "Can you sketch the whole thing as nested `Serial`/`Parallel` calls, using existing "
            "components or your own `nn.Module` (a 1D convolution, an RNN, an N-BEATS block) "
            f"composed through `Serial`? ({_core()} §1.6)",
            "A layer the library does not ship still composes, as long as its input and output "
            "shapes are stated.",
            "a few lines of pseudocode naming real classes, and the shapes of any own module",
            since=2,
        ),
    ],
    DesignStage.search_space: [
        Row(
            "open_axes",
            "Which decisions from the earlier stages are you genuinely UNSURE about, versus "
            "already decided?",
            "A search space should only contain what you don't already know the answer to — "
            "searching a decided axis just wastes trials.",
            "the 2-5 axes actually in question",
        ),
        Row(
            "ranges",
            "Which hyperparameters, and which ranges: width, depth, learning rate, scheduler, "
            "training length?",
            "A range too narrow hides the answer; one too wide spends trials where nothing "
            "works.",
            "a range per axis",
        ),
        Row(
            "conditional_axes",
            f"For each axis: is it independent, or conditional on another axis's value? "
            f"({_search_ch()} §9.4)",
            "Conditional axes (e.g. the hidden width only matters if the head has a hidden "
            "layer at all) get wastefully sampled by most search algorithms — name them so the "
            "waste isn't a surprise later.",
            "which axes are conditional on which",
        ),
        Row(
            "conditional_axes",
            f"For each axis: is it independent, or conditional on another axis's value? "
            f"({_search_ch()} §9.4)",
            "Conditional axes (e.g. `n_experts` only matters if `mechanism == 'moe'`) get "
            "wastefully sampled by most search algorithms — name them so the waste isn't a "
            "surprise later.",
            "which axes are conditional on which",
            since=3,
        ),
        Row(
            "confound",
            "Which axes could CONFOUND the comparison you actually care about, if left fixed?",
            "If `lr` is fixed while comparing a linear head with an MLP head, a win might just "
            "mean the fixed learning rate happened to suit one of them.",
            "at least one confounder being included for this reason",
        ),
        Row(
            "confound",
            "Which axes could CONFOUND the comparison you actually care about, if left fixed?",
            "If `lr` is fixed while searching `mechanism`, a win might just mean the fixed "
            "learning rate happened to suit one mechanism.",
            "at least one confounder being included for this reason",
            since=3,
        ),
        Row(
            "budget",
            "What is the budget (ten minutes, an hour), and how many trials fit in it? "
            "(`vectormesh_search_budget` does the arithmetic)",
            "Sets `num_samples` honestly instead of picking a round number that sounds fine, "
            "after reserving the baseline and the winner's rerun.",
            "the budget, seconds per trial, and the trial count that follows",
        ),
        Row(
            "wide_or_long",
            "Many short trials in a wide scan, or fewer long ones?",
            "A wide, short scan finds the region; fewer, longer trials compare candidates "
            "within it. One budget buys one or the other.",
            "the choice, and why it fits this question",
        ),
        Row(
            "one_metric",
            "What is the ONE metric the search optimizes for?",
            "`tune.TuneConfig` needs exactly one metric and mode — decide before writing the "
            "search space, not after reading confusing results.",
            "the metric name and whether it's maximised or minimised",
        ),
    ],
    DesignStage.search_practices: [
        Row(
            "floor",
            "Has the no-training floor from stage 1 now been computed, on the same split and "
            f"metric as the trained baseline? ({_search_ch()} §9.2)",
            "A floor on another split or metric cannot be compared with the baseline; if a "
            "trained model can't clear it, the architecture usually isn't the problem.",
            "the computed floor score, and the split it was computed on",
        ),
        Row(
            "trained_baseline",
            f"What's your TRAINED baseline, and over how many seeds? ({_search_ch()} §9.3)",
            "This is the number every later search result gets measured against.",
            "mean +/- std over at least 3 seeds",
        ),
        Row(
            "stopping",
            f"How are you deciding when to stop training a trial? ({_search_ch()} §9.5)",
            "A fixed, guessed epoch count silently under-trains some configs and over-trains "
            "others — which then looks like an architecture effect. Default to a generous "
            "epoch CEILING plus an adaptive rule (plateau scheduler + early stop) instead.",
            "the stopping rule, named explicitly — not just an epoch number",
        ),
        Row(
            "rerun",
            f"How many seeds is the winner rerun over? ({_search_ch()} §9.7)",
            "The best single trial is partly luck; its rerun over seeds is the result.",
            "the number of seeds",
        ),
        Row(
            "claim",
            f"How will a search result become a CLAIM, not just a number? ({_search_ch()} §9.7)",
            "Effect size in units of the trained baseline's own sigma, compared against the "
            "no-training floor for context — not the raw difference.",
            "the formula that will be applied",
        ),
        Row(
            "logging",
            f"Where and when will you log? ({_search_ch()} §9.8)",
            "Each trial runs in its own process; logging from inside the trainable makes "
            "concurrent writers corrupt or block a shared sqlite store.",
            "logging from the driver after `tuner.fit()` returns: one mlflow run per trial "
            "(role `trial`), plus the `baseline` and `final` runs",
        ),
    ],
    DesignStage.results: [
        Row(
            "views",
            "Which mlflow views will you read?",
            "Each view answers a different question; reading only the best row lets a lucky "
            "trial pass for a result.",
            "a run table filtered on the final runs, parameters against the score over all "
            "trials, and per-epoch curves of chosen trials",
        ),
        Row(
            "comparison",
            "Which comparison answers the hypothesis?",
            "A claim rests on a comparison chosen before reading the runs; one chosen afterwards "
            "lets noise pick the winner.",
            "the two runs or groups compared, on the stage-1 metric, and the spread the "
            "difference is measured against",
        ),
        Row(
            "bound_met",
            "Does the result clear the acceptance bound from stage 1?",
            "The bound was set before the search, so it cannot be moved to fit the result.",
            "yes or no, and the margin",
        ),
        Row(
            "sharing",
            "How are the runs shared? If the course has a run-sharing protocol "
            "(`docs/submit-runs.md` in the course repository), the runs go through it.",
            "A result nobody else can read cannot be checked or built on.",
            "the protocol or channel",
        ),
        Row(
            "tags",
            "Is each run named and tagged so that someone else can see who made it and what "
            "was explored?",
            "Untagged runs from several students cannot be told apart, nor a baseline from a "
            "trial.",
            "the run names and tags (who, and which role: baseline, trial, final)",
        ),
    ],
}


@dataclass
class DesignProgress:
    """What the student said at each stage, and which lesson they are in."""

    answers: dict[str, str] = field(default_factory=dict)
    lesson: int | None = None


# In-memory record for THIS server process (i.e. this session). Resets when the server
# restarts — the design conversation is meant to happen in one sitting, not to persist
# silently across days.
_state = DesignProgress()

MIN_ANSWER_CHARS = 20

# Which earlier stage's recorded answer a later stage's checklist is read against, and the
# framing that turns the quote into an instruction rather than a reminder.
_WIRED_FROM: dict[DesignStage, tuple[DesignStage, str]] = {
    DesignStage.pretrained: (
        DesignStage.problem,
        (
            "**The metric and acceptance bound from stage 1:** {answer}\n\nA pretrained model is "
            "good enough the moment it clears that bound; the questions below ask whether a "
            "simpler one already does."
        ),
    ),
    DesignStage.search_practices: (
        DesignStage.problem,
        (
            "**What stage 1 said about the metric, the bound and the no-training floor:** "
            "{answer}\n\nThe floor was named there; the questions below ask only what is new."
        ),
    ),
    DesignStage.search_space: (
        DesignStage.problem,
        (
            "**The metric and acceptance bound from stage 1:** {answer}\n\nThe search optimizes "
            "that metric. A search that cannot reach the bound within the budget is the wrong "
            "search, not a reason to lower the bound."
        ),
    ),
    DesignStage.results: (
        DesignStage.problem,
        (
            "**The metric and acceptance bound from stage 1:** {answer}\n\nRead every run against "
            "that metric and that bound — the ones set before the search, not new ones chosen "
            "after reading the results."
        ),
    ),
}

_ASK_LESSON = (
    "**First ask the student which lesson of the course they are in "
    f"({FIRST_LESSON} to {LAST_LESSON}).** Every stage offers only the material of that "
    "lesson or earlier. Someone outside the course passes "
    f"`lesson={LAST_LESSON}`. Pass it as `lesson=<N>` on the next call to "
    "`vectormesh_design_checklist`; it is kept with the progress.\n\n"
)


def _next_stage(stage: DesignStage) -> DesignStage | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


def _effective_lesson(lesson: int | None) -> int:
    """The lesson the tables are written for: the earliest one while the student's is unknown."""
    return FIRST_LESSON if lesson is None else lesson


def offered_components(lesson: int) -> list[str]:
    """The vectormesh components a student in `lesson` has met."""
    return [c for n in sorted(LESSONS) if n <= lesson for c in LESSONS[n]["components"]]


def _lesson_line(number: int) -> str:
    entry = LESSONS[number]
    parts = [f"`{c}`" for c in entry["components"]] + list(entry["ideas"])
    return ", ".join(parts)


def _lesson_block(lesson: int) -> str:
    """The lesson table for stage 5: what is offered, and what comes later."""
    lines = [f"**Offered at lesson {lesson}** (what the student has met so far):", ""]
    lines += [
        f"- lesson {n}: {_lesson_line(n)}" for n in sorted(LESSONS) if n <= lesson
    ]
    later = [n for n in sorted(LESSONS) if n > lesson]
    if later:
        lines += [
            "",
            (
                "**Not offered yet.** If the student reaches for one of these, name it as "
                '"comes in lesson N" and stay with what is offered:'
            ),
            "",
        ]
        lines += [f"- comes in lesson {n}: {_lesson_line(n)}" for n in later]
    return "\n".join(lines)


def rows_for(stage: DesignStage, lesson: int) -> list[Row]:
    """The rows of `stage` a student in `lesson` sees, one per id, in table order."""
    chosen: dict[str, Row] = {}
    for row in STAGE_ROWS[stage]:
        if row.since <= lesson and (
            row.id not in chosen or row.since > chosen[row.id].since
        ):
            chosen[row.id] = row
    return list(chosen.values())


def _roadmap() -> str:
    lines = [
        "# Designing something new with VectorMesh — an eight-stage conversation",
        "",
        (
            "This is deliberately not a single document to skim. Work through these eight stages "
            "with the student, in order, one call to this tool per stage — each later stage "
            "assumes the earlier ones were actually reasoned through together, not skipped. Code "
            "comes after the last stage, not before."
        ),
        "",
    ]
    if _state.lesson is not None:
        lines += [f"Lesson: **{_state.lesson}**", ""]
    for stage in STAGE_ORDER:
        mark = "✅" if stage.value in _state.answers else "▢"
        lines.append(f"{mark} **{STAGE_TITLE[stage]}** (`{stage.value}`)")
    lines += [
        "",
        (
            'Call this tool again with `stage="problem"` to begin — always start there for a new '
            "idea, even if it feels obvious; every later stage is measured against the metric and "
            "acceptance bound that stage 1 establishes."
        ),
    ]
    return "\n".join(lines)


def _stage_intro(stage: DesignStage, lesson: int) -> str:
    """The stage's fixed rationale, plus — where a later stage depends on an earlier one's
    actual answer — a quote of what was recorded, so the checklist reads against the
    student's own words instead of restating the method in the abstract."""
    parts = [STAGE_INTRO[stage].replace("{patterns}", _patterns_ref(lesson))]
    wired = _WIRED_FROM.get(stage)
    if wired is not None:
        source_stage, framing = wired
        answer = _state.answers.get(source_stage.value)
        if answer:
            parts.append(framing.format(answer=answer))
    if stage is DesignStage.architecture:
        parts.append(_lesson_block(lesson))
    return "\n\n".join(parts)


def _stage_table(stage: DesignStage) -> str:
    if stage is DesignStage.architecture and _state.lesson is None:
        return _ASK_LESSON + (
            "Stage 5 opens once the lesson is known: call `vectormesh_design_checklist` again "
            'with `stage="architecture"` and `lesson=<N>`.'
        )
    lesson = _effective_lesson(_state.lesson)
    lines = [
        f"# Stage: {STAGE_TITLE[stage]}",
        "",
        _stage_intro(stage, lesson),
        "",
        "| Question | Why it matters | A good answer names |",
        "|---|---|---|",
    ]
    for row in rows_for(stage, lesson):
        lines.append(f"| {row.question} | {row.why} | {row.names} |")
    lines += [
        "",
        (
            "**How to use this table:** ask the student these questions yourself — together or one "
            "at a time, your judgement. Do NOT answer them yourself, and do not treat this as done "
            "until they've engaged in their own words. Write no code yet: code comes after the "
            "last stage. Once they have answered, call "
            f'`vectormesh_design_checklist` again with `stage="{stage.value}"` and '
            "`answers=<a summary of what they actually said>` to record it and receive the next "
            "stage."
        ),
    ]
    return "\n".join(lines)


def _completion_summary() -> str:
    lines = [
        "# All eight stages recorded",
        "",
    ]
    if _state.lesson is not None:
        lines += [f"Lesson: **{_state.lesson}**", ""]
    lines += ["What was reasoned through, in order:", ""]
    for stage in STAGE_ORDER:
        lines.append(f"**{STAGE_TITLE[stage]}**")
        lines.append(f"> {_state.answers.get(stage.value, '(not recorded)')}")
        lines.append("")
    lines += [
        (
            "Now is the right time to write code, if the student wants it: a baseline first, "
            "built from the components stage 5 offered. If the 'architecture' or 'search_space' "
            "answers named more than one plausible variant, call `vectormesh_search_method` (or "
            "run the `search` prompt) next, rather than picking one by feel."
        ),
    ]
    return "\n".join(lines)


class DesignChecklistInput(BaseModel):
    """Input for the staged design-checklist tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    stage: DesignStage | None = Field(
        default=None,
        description=(
            "Which stage to open. Omit on the very first call to get the roadmap and confirm "
            "where to start (always 'problem' for a new idea). One of: problem, tensors, "
            "pretrained, context, architecture, search_space, search_practices, results."
        ),
    )
    answers: str | None = Field(
        default=None,
        description=(
            "A summary, in your own words, of what the student actually said when asked this "
            "stage's questions. Omit to just receive the stage's checklist. Provide it on a "
            "second call for the SAME stage to record it and advance to the next stage. Do not "
            "fabricate this on the student's behalf — it must reflect their real answers."
        ),
        max_length=4000,
    )
    lesson: int | None = Field(
        default=None,
        description=(
            "The lesson of the course the student is in (1 to 7), as the student said it; "
            "someone outside the course passes 7. Kept with the progress; every stage offers "
            "only material of this lesson or earlier. Omit when the student has not said it yet."
        ),
        ge=FIRST_LESSON,
        le=LAST_LESSON,
    )


@mcp.tool(
    name="vectormesh_design_checklist",
    annotations={
        "title": "Walk through the pre-design checklist with a student, one stage at a time",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def vectormesh_design_checklist(params: DesignChecklistInput) -> str:
    """Coach a student through eight stages of thinking BEFORE they design anything.

    This is a conversation, not a document dump: each call returns exactly ONE stage's checklist
    table. Ask the student those questions yourself, in the chat, and wait for real answers — do
    not answer on their behalf. Once they've responded, call this tool again with the SAME
    `stage` and their answers summarised in `answers`; that records the stage and returns the
    next one. Progress, including the student's lesson, is held in memory for this server
    process only (this session) and is not persisted across restarts. Code comes after the
    last stage.

    The eight stages, in the order they must happen:
        1. problem            — what is predicted, is deep learning needed, which metric, and
                                 what score is good enough?
        2. tensors            — what shape is the data actually in, right now?
        3. pretrained         — is there a pretrained model, and is the simplest one enough?
        4. context            — what extra signal could be added before touching architecture?
        5. architecture       — from the baseline up, which compositions are on the table at
                                 the student's lesson?
        6. search_space       — of the open questions, which are worth searching, in what
                                 budget?
        7. search_practices   — how will a search result be trusted, not just read?
        8. results            — how are the runs read in mlflow and shared?

    Every table is written for the student's lesson: rows and examples that need later
    material are left out. While the lesson is unknown, the tables are written for lesson 1
    and stage 5 returns only the request to ask for the lesson.

    Args:
        params (DesignChecklistInput):
            - stage (DesignStage | None): which stage to open; omit for the roadmap.
            - answers (str | None): the student's answers to the CURRENT stage, in your
              words; omit to just receive the stage's checklist.
            - lesson (int | None): the student's lesson (1 to 7); kept once given. While it
              is unknown, the response asks you to ask the student for it.

    Returns:
        str: The roadmap (no stage given), a stage's checklist table (stage given, no answers
        yet), or a recorded confirmation plus the next stage's checklist (stage + answers given).
        After the last stage, returns a recap of everything recorded.
    """
    if params.lesson is not None:
        _state.lesson = params.lesson
    ask = _ASK_LESSON if _state.lesson is None else ""

    if params.stage is None:
        return ask + _roadmap()

    stage = params.stage
    if not params.answers or len(params.answers.strip()) < MIN_ANSWER_CHARS:
        table = _stage_table(stage)
        return table if table.startswith(_ASK_LESSON) else ask + table

    _state.answers[stage.value] = params.answers.strip()
    nxt = _next_stage(stage)
    if nxt is None:
        return ask + _completion_summary()

    recap = f"✅ Recorded for **{STAGE_TITLE[stage]}**: {params.answers.strip()}\n\n"
    table = _stage_table(nxt)
    return recap + table if table.startswith(_ASK_LESSON) else ask + recap + table


# --------------------------------------------------------------------------- #
# The search budget — how many trials fit, as arithmetic
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SearchBudget:
    """How a time budget divides into rounds of parallel runs.

    One round runs up to `parallel_trials` runs side by side and takes `seconds_per_trial`.
    The baseline over seeds and the winner's rerun over seeds each take whole rounds; the
    search gets the rounds that are left.
    """

    total_rounds: int
    reserved_rounds: int
    search_rounds: int
    trials: int
    min_budget_minutes: (
        float  # the smallest budget that leaves room for one search round
    )


def search_budget(
    seconds_per_trial: float,
    budget_minutes: float,
    parallel_trials: int = 1,
    seeds_for_rerun: int = 3,
) -> SearchBudget:
    """Divide a budget into the baseline over seeds, the search, and the winner's rerun."""
    if seconds_per_trial <= 0 or budget_minutes <= 0:
        raise ValueError("seconds_per_trial and budget_minutes must be positive")
    if parallel_trials < 1 or seeds_for_rerun < 1:
        raise ValueError("parallel_trials and seeds_for_rerun must be at least 1")
    total_rounds = math.floor(budget_minutes * 60 / seconds_per_trial)
    rounds_per_seed_set = math.ceil(seeds_for_rerun / parallel_trials)
    reserved_rounds = 2 * rounds_per_seed_set
    search_rounds = max(0, total_rounds - reserved_rounds)
    return SearchBudget(
        total_rounds=total_rounds,
        reserved_rounds=reserved_rounds,
        search_rounds=search_rounds,
        trials=search_rounds * parallel_trials,
        min_budget_minutes=(reserved_rounds + 1) * seconds_per_trial / 60,
    )


@mcp.tool(
    name="vectormesh_search_budget",
    annotations={
        "title": "How many search trials fit in a time budget",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def vectormesh_search_budget(
    seconds_per_trial: Annotated[
        float, Field(gt=0, description="Wall-clock seconds one training run takes.")
    ],
    budget_minutes: Annotated[
        float, Field(gt=0, description="The whole budget, e.g. 10 or 60 minutes.")
    ],
    parallel_trials: Annotated[
        int, Field(ge=1, le=256, description="How many runs fit side by side.")
    ] = 1,
    seeds_for_rerun: Annotated[
        int,
        Field(
            ge=1,
            le=100,
            description="Seeds for the baseline, and again for the winner's rerun.",
        ),
    ] = 3,
) -> str:
    """Return how many search trials fit in a budget, and the trade it leaves.

    The baseline is run over `seeds_for_rerun` seeds before the search and the winner is
    rerun over as many seeds after it; both are reserved first, in rounds of
    `parallel_trials` runs, and the search gets what is left. The result ends with the trade
    between many short trials in a wide scan and fewer long ones, as a question for the
    student. Use it at the search_space stage of `vectormesh_design_checklist`.

    Args:
        seconds_per_trial (float): wall-clock seconds one training run takes.
        budget_minutes (float): the whole budget in minutes.
        parallel_trials (int): runs that fit side by side (default 1).
        seeds_for_rerun (int): seeds for the baseline and for the winner's rerun (default 3).

    Returns:
        str: Markdown with the arithmetic, the number of trials, and the trade as a question.
    """
    b = search_budget(
        seconds_per_trial, budget_minutes, parallel_trials, seeds_for_rerun
    )
    lines = [
        f"# Search budget: {budget_minutes:g} minutes",
        "",
        f"- One round: up to {parallel_trials} run(s) side by side, {seconds_per_trial:g} s.",
        f"- Rounds in the budget: {b.total_rounds}.",
        (
            f"- Reserved: {b.reserved_rounds} round(s) — the baseline over {seeds_for_rerun} "
            f"seed(s) before the search, and the winner's rerun over {seeds_for_rerun} seed(s) "
            "after it."
        ),
        f"- Left for the search: {b.search_rounds} round(s), so **{b.trials} trial(s)**.",
        "",
    ]
    if b.trials == 0:
        lines.append(
            "This budget does not leave room for a search after the baseline and the rerun. "
            f"The smallest budget that does is {b.min_budget_minutes:.1f} minutes; or shorten "
            "a run, or run more in parallel."
        )
        return "\n".join(lines)

    short = search_budget(
        seconds_per_trial / 2, budget_minutes, parallel_trials, seeds_for_rerun
    )
    long = search_budget(
        seconds_per_trial * 2, budget_minutes, parallel_trials, seeds_for_rerun
    )
    lines += [
        (
            "**The trade, for the student to decide:** wide and short, or narrow and long? Runs "
            f"half as long ({seconds_per_trial / 2:g} s) give {short.trials} trials to scan a wide "
            f"space; runs twice as long ({seconds_per_trial * 2:g} s) give {long.trials} trials "
            "that each train further. Which does this question need: finding the region where "
            "good configurations live, or comparing a few candidates within it?"
        ),
    ]
    return "\n".join(lines)


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
    annotations={
        "title": "Get the method for a Ray Tune architecture search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def vectormesh_search_method(params: SearchMethodInput) -> str:
    """Return the method for setting up a Ray Tune architecture search over VectorMesh components.

    Call this once `vectormesh_design_checklist` has been worked through (particularly the
    search_space and search_practices stages) and there is more than one plausible design with
    no principled way to pick between them by inspection. This returns the *order of operations*
    and links to the full chapter, which has a runnable `trainable()` template built from this
    library's own gating components; fetch it with `vectormesh_get_concept('09')` when ready to
    write code. Below lesson 3 (or while the student's lesson is unknown) the result says the
    template's gating ladder is lesson 3 material and the same steps apply to a plain head.

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
        (
            "Do these in order — reversing the order (e.g. reading results before establishing a "
            f"baseline) is the most common way a search stops being trustworthy. Full chapter: {url}"
        ),
        "",
        (
            "1. **Get the no-training floor first.** A cosine-similarity/nearest-centroid "
            "classifier on the frozen embeddings, no gradient step anywhere. If a trained pipeline "
            f"can't clear it, the architecture is probably not the problem. ({url}"
            "#92-the-cheapest-baseline-no-training-at-all)"
        ),
        "",
        (
            "2. **Then a trained baseline, with a seed spread.** Run the simplest plausible "
            "pipeline over 3-5 seeds *before* writing any search space. Its `std` is the number "
            f"every later claim gets measured against. ({url}#93-a-trained-baseline-with-a-seed-spread)"
        ),
        "",
        (
            "3. **Design the search space as one dict.** Categorical axes with `tune.choice`, "
            "continuous ones with `tune.uniform`/`tune.loguniform`. Include any axis that could "
            "confound the comparison you actually care about (e.g. `lr` and `hidden` alongside "
            f"`mechanism`, if the question is about `mechanism`). ({url}#94-the-search-space)"
        ),
        "",
        (
            "4. **Write the trainable assuming an empty process, with an epoch CEILING plus an "
            "adaptive stop.** A Ray trial runs in its own process — rebuild the cache load, the "
            "pipeline, and the optimizer from scratch every call. Set `epochs` generously and let a "
            "plateau scheduler + patience-based early stop decide the real per-trial budget, so no "
            f"config is silently under- or over-trained. ({url}#95-the-trainable-function)"
        ),
        "",
        (
            "5. **Read the result one axis at a time**, not just the top row. Group by each axis and "
            "check whether its groups actually separate before treating that axis's 'best' value as "
            f"meaningful. ({url}#96-reading-a-search-one-axis-at-a-time)"
        ),
        "",
        (
            "6. **Turn a difference into a claim by dividing by the baseline's own sigma**, not by "
            "comparing to zero — and sanity-check against the no-training floor from step 1. Rerun "
            f"the winning config over several seeds before writing anything down as a conclusion. "
            f"({url}#97-turning-a-difference-into-a-claim)"
        ),
        "",
        (
            "7. **Log from the driver, after `tuner.fit()` returns** — never from inside the "
            "trainable: each trial is its own process, and concurrent writers corrupt a local "
            "sqlite store (mlflow's default). One mlflow run per trial (role `trial`), plus the "
            "`baseline` and `final` runs. "
            f"({url}#98-logging-discipline-one-summary-per-search)"
        ),
        "",
        (
            "8. **If launching under `uv run`**, set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before "
            f"`import ray`, or Ray's workers will look for a `pyproject.toml` in the wrong place. "
            f"({url}#99-a-uv-run-gotcha-worth-knowing)"
        ),
        "",
        (
            "Call `vectormesh_get_concept('09')` for the runnable `trainable()` template — it "
            "searches over this library's own `Skip -> Gate -> Highway -> MoE` ladder as the "
            "worked example, including the plateau/early-stop loop from step 4."
        ),
    ]
    lesson = _state.lesson
    if lesson is None or lesson < GATING_LESSON:
        who = (
            "The student's lesson is not known yet. If it is below lesson "
            f"{GATING_LESSON}: the"
            if lesson is None
            else f"At lesson {lesson}, the"
        )
        lines += [
            "",
            (
                f"{who} gating ladder in the chapter 9 template (`Skip`, `Gate`, `Highway`, `MoE`) "
                f"is lesson {GATING_LESSON} material. The same steps apply to a plain head: search "
                "its width, depth and learning rate instead of the mechanism."
            ),
        ]
    return "\n".join(lines)


@mcp.prompt(
    name="design",
    description=(
        "Start an eight-stage design conversation (problem, tensors, pretrained, context, "
        "architecture, search_space, search_practices, results) for a deep learning "
        "architecture built with VectorMesh."
    ),
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
        "Step 1: call the `vectormesh_design_checklist` tool with no stage to get the roadmap, "
        "and ask me which lesson of the course I am in; pass it as `lesson` from then on.\n"
        'Step 2: call it again with `stage="problem"` and ask me those questions yourself, in '
        "the chat. Wait for my real answers — don't answer for me.\n"
        "Step 3: once I've answered, call the tool again with the same stage and a summary of "
        "what I said in `answers`, to record it and get the next stage. Repeat this for every "
        "stage, in order — never skip ahead, and never fetch two stages before I've answered the "
        "first. The eight stages are problem, tensors, pretrained, context, architecture, "
        "search_space, search_practices and results. Write no code before the last one.\n"
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
