"""Tests for the coaching MCP server, `vectormesh_mcp.py` at the repository root.

The script is not part of the package, so it is imported from its path. It reads `docs/` from
disk because the docs sit next to it.
"""

import asyncio
import importlib.util
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


def test_every_row_has_three_columns():
    for rows in mcp_script.STAGE_ROWS.values():
        for row in rows:
            assert len(row) == 3
            assert all(isinstance(cell, str) and cell for cell in row)


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


def test_gated_rows_belong_to_the_architecture_stage():
    questions = {
        row[0] for row in mcp_script.STAGE_ROWS[mcp_script.DesignStage.architecture]
    }
    assert set(mcp_script.ROW_LESSON) <= questions


def test_lesson_table_components_exist_in_vectormesh():
    for lesson in mcp_script.LESSONS.values():
        for name in lesson["components"]:
            assert hasattr(vectormesh, name) or hasattr(vectormesh.components, name), (
                name
            )


def test_metric_and_bound_are_quoted_later():
    bound = "macro F1, acceptance bound 0.80, cosine floor 0.62"
    checklist(stage="problem", answers=bound, lesson=1)
    for stage in ("search_space", "results"):
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
