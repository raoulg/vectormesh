"""Tests for the coaching MCP server, `vectormesh_mcp.py` at the repository root.

The script is not part of the package, so it is imported from its path. It reads `docs/` from
disk because the docs sit next to it.
"""

import asyncio
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import vectormesh
import vectormesh.components

SCRIPT = Path(__file__).resolve().parents[1] / "vectormesh_mcp.py"
LONG = "a summary of what the student said, well past the minimum length"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vectormesh_mcp", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vectormesh_mcp"] = module
    spec.loader.exec_module(module)
    return module


# Typed as Any: the script's names are only known once it has been executed.
mcp_script: Any = _load_script()


@pytest.fixture(autouse=True)
def fresh_progress():
    """Every test starts a new design conversation."""
    mcp_script._state = mcp_script.DesignProgress()
    yield


def call(tool: str, **arguments: Any) -> str:
    async def _call() -> str:
        async with Client(mcp_script.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return result.content[0].text

    return asyncio.run(_call())


def checklist(**params: Any) -> str:
    return call("vectormesh_design_checklist", params=params)


def test_stage_order():
    assert [s.value for s in mcp_script.STAGE_ORDER] == [
        "problem",
        "tensors",
        "pretrained",
        "context",
        "architecture",
        "search_space",
        "search_practices",
        "results",
    ]
    assert list(mcp_script.STAGE_ROWS) == mcp_script.STAGE_ORDER
    assert list(mcp_script.STAGE_TITLE) == mcp_script.STAGE_ORDER


def test_every_row_has_three_columns_and_a_valid_lesson():
    for rows in mcp_script.STAGE_ROWS.values():
        for row in rows:
            assert all(
                cell.strip() for cell in (row.id, row.question, row.why, row.names)
            )
            assert row.since in mcp_script.LESSONS


def test_row_ids_have_one_text_per_lesson():
    for rows in mcp_script.STAGE_ROWS.values():
        keys = [(row.id, row.since) for row in rows]
        assert len(keys) == len(set(keys))


def test_short_answer_does_not_advance():
    text = checklist(stage="problem", answers="too short", lesson=2)
    assert text.startswith("# Stage: 1.")
    assert "problem" not in mcp_script._state.answers


def test_long_answer_advances():
    text = checklist(stage="problem", answers=LONG, lesson=2)
    assert "Recorded for" in text
    assert "# Stage: 2." in text
    assert mcp_script._state.answers["problem"] == LONG


def test_every_table_tells_the_assistant_not_to_answer():
    for stage in mcp_script.STAGE_ORDER:
        text = checklist(stage=stage.value, lesson=3)
        assert "Do NOT answer them yourself" in text
        assert "Write no code yet" in text


def test_lesson_is_asked_for_until_given_and_then_kept():
    assert "which lesson" in checklist()
    assert "which lesson" in checklist(stage="problem")
    checklist(stage="problem", answers=LONG, lesson=4)
    assert mcp_script._state.lesson == 4
    assert "which lesson" not in checklist(stage="tensors")
    assert mcp_script._state.lesson == 4


def test_lesson_out_of_range_is_refused():
    with pytest.raises(ToolError):
        checklist(stage="problem", lesson=8)


def _offered_part(text: str) -> str:
    return text.split("**Not offered yet.**")[0]


def test_lesson_1_does_not_offer_attention():
    text = checklist(stage="architecture", lesson=1)
    assert "`Attention`" not in _offered_part(text)
    assert "comes in lesson 4" in text
    assert "Attention" not in mcp_script.offered_components(1)


def test_lesson_4_offers_attention():
    text = checklist(stage="architecture", lesson=4)
    assert "`Attention`" in _offered_part(text)
    assert "Attention" in mcp_script.offered_components(4)


def _row_ids(stage: str, lesson: int) -> set[str]:
    return {
        row.id for row in mcp_script.rows_for(mcp_script.DesignStage(stage), lesson)
    }


def _row(stage: str, row_id: str, lesson: int) -> Any:
    rows = mcp_script.rows_for(mcp_script.DesignStage(stage), lesson)
    return next(row for row in rows if row.id == row_id)


def test_gating_ladder_row_appears_from_lesson_3():
    ladder = _row("architecture", "conditional", 3).question
    assert ladder not in checklist(stage="architecture", lesson=1)
    assert ladder in checklist(stage="architecture", lesson=3)


def test_later_material_is_gated_in_every_stage():
    assert "chunk_spread" not in _row_ids("tensors", 3)
    assert "chunk_spread" in _row_ids("tensors", 4)
    assert "spectral" not in _row_ids("context", 1)
    assert "spectral" in _row_ids("context", 2)
    assert "(C, D)" not in _row("tensors", "shape", 3).names
    assert "(C, D)" in _row("tensors", "shape", 4).names
    assert "regex" not in _row("context", "domain_signal", 3).why
    assert "attention pooling" not in _row("context", "learned", 3).why
    assert "Scaling laws" not in _row("architecture", "width", 2).why
    assert "Scaling laws" in _row("architecture", "width", 3).why
    assert "n_experts" not in _row("search_space", "conditional_axes", 2).why
    assert "n_experts" in _row("search_space", "conditional_axes", 3).why


def test_lesson_1_tables_speak_of_frozen_image_vectors():
    assert "image" in _row("tensors", "raw_input", 1).names
    assert _row("tensors", "shape", 1).names.startswith("`(D,)`")
    assert "timeseries" not in _row("tensors", "raw_input", 1).names


def test_lesson_2_sketch_row_allows_an_own_module():
    assert "nn.Module" not in _row("architecture", "sketch", 1).question
    assert "nn.Module" in _row("architecture", "sketch", 2).question
    assert "convolution" in _row("architecture", "reshape", 2).question


def test_architecture_without_a_lesson_returns_only_the_ask():
    text = checklist(stage="architecture")
    assert "which lesson" in text
    assert "| Question |" not in text
    assert "`lesson=7`" in text


def test_tables_without_a_lesson_are_written_for_lesson_1():
    assert "(C, D)" not in checklist(stage="tensors")


def test_architecture_intro_names_only_the_patterns_of_the_lesson():
    assert "only its rows" in checklist(stage="architecture", lesson=1)
    assert "only its rows" not in checklist(stage="architecture", lesson=4)


def test_search_method_flags_the_gating_template_below_lesson_3():
    checklist(lesson=2)
    assert "lesson 3 material" in call("vectormesh_search_method", params={})
    checklist(lesson=3)
    assert "lesson 3 material" not in call("vectormesh_search_method", params={})


_CLASS_NAME = re.compile(r"`([A-Z][A-Za-z0-9]+)`")
# Backticked class names in the coaching text that are not vectormesh's own.
_NOT_VECTORMESH = {"BeartypeCallHintParamViolation"}


def _exists_in_vectormesh(name: str) -> bool:
    return hasattr(vectormesh, name) or hasattr(vectormesh.components, name)


def test_backticked_names_in_ideas_and_rows_exist_in_vectormesh():
    texts = [idea for entry in mcp_script.LESSONS.values() for idea in entry["ideas"]]
    texts += [
        cell
        for rows in mcp_script.STAGE_ROWS.values()
        for row in rows
        for cell in (row.question, row.why, row.names)
    ]
    names = {
        name for text in texts for name in _CLASS_NAME.findall(text)
    } - _NOT_VECTORMESH
    assert names
    missing = sorted(name for name in names if not _exists_in_vectormesh(name))
    assert not missing


def test_rows_name_only_components_the_lesson_offers():
    lesson_of = {
        c: n for n, entry in mcp_script.LESSONS.items() for c in entry["components"]
    }
    for lesson in mcp_script.LESSONS:
        for stage in mcp_script.STAGE_ORDER:
            for row in mcp_script.rows_for(stage, lesson):
                text = f"{row.question} {row.why} {row.names}"
                for name in _CLASS_NAME.findall(text):
                    assert lesson_of.get(name, 1) <= lesson, (
                        stage,
                        row.id,
                        name,
                        lesson,
                    )


def test_lesson_table_components_exist_in_vectormesh():
    for lesson in mcp_script.LESSONS.values():
        for name in lesson["components"]:
            assert hasattr(vectormesh, name) or hasattr(vectormesh.components, name), (
                name
            )


def test_metric_and_bound_are_quoted_later():
    bound = "macro F1, acceptance bound 0.80, cosine floor 0.62"
    checklist(stage="problem", answers=bound, lesson=1)
    for stage in ("search_space", "search_practices", "results"):
        assert bound in checklist(stage=stage)


def test_full_walk_returns_eight_different_tables():
    tables = [checklist(stage="problem", lesson=5)]
    for stage in mcp_script.STAGE_ORDER:
        tables.append(checklist(stage=stage.value, answers=f"{LONG} ({stage.value})"))
    stage_tables = [tables[0]] + [t.split("\n\n", 1)[1] for t in tables[1:-1]]
    assert len(stage_tables) == 8
    assert len(set(stage_tables)) == 8
    assert tables[-1].startswith("# All eight stages recorded")


def test_design_prompt_names_the_eight_stages():
    text = mcp_script.design_prompt()
    for stage in mcp_script.STAGE_ORDER:
        assert stage.value in text


def test_every_chapter_in_the_reading_order_loads():
    assert mcp_script.CHAPTERS
    for chapter in mcp_script.CHAPTERS:
        assert mcp_script._read_doc(chapter.file).strip()


def test_budget_reserves_baseline_and_rerun():
    # 10 minutes of 30 s runs is 20 rounds; 3 seeds twice, one at a time, reserve 6.
    b = mcp_script.search_budget(30, 10)
    assert (b.total_rounds, b.reserved_rounds, b.search_rounds, b.trials) == (
        20,
        6,
        14,
        14,
    )


def test_budget_with_parallel_trials():
    # 60 minutes of 60 s runs is 60 rounds of 4; 3 seeds fit in one round, twice.
    b = mcp_script.search_budget(60, 60, parallel_trials=4, seeds_for_rerun=3)
    assert (b.total_rounds, b.reserved_rounds, b.search_rounds, b.trials) == (
        60,
        2,
        58,
        232,
    )


def test_budget_too_small_leaves_no_trials():
    b = mcp_script.search_budget(120, 10, seeds_for_rerun=5)
    assert b.trials == 0
    assert b.min_budget_minutes == pytest.approx(22.0)
    assert "does not leave room" in call(
        "vectormesh_search_budget",
        seconds_per_trial=120,
        budget_minutes=10,
        seeds_for_rerun=5,
    )


def test_budget_refuses_impossible_input():
    with pytest.raises(ValueError):
        mcp_script.search_budget(0, 10)
    with pytest.raises(ValueError):
        mcp_script.search_budget(30, 10, parallel_trials=0)


def test_budget_tool_states_the_trade_as_a_question():
    text = call("vectormesh_search_budget", seconds_per_trial=30, budget_minutes=10)
    assert "**14 trial(s)**" in text
    # Half as long: 40 rounds - 6 = 34 trials; twice as long: 10 rounds - 6 = 4 trials.
    assert "give 34 trials" in text
    assert "give 4 trials" in text
    assert text.rstrip().endswith("?")
