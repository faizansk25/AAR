"""Execution-history identity and observation-shape regressions."""

from __future__ import annotations

import pytest

from aar.cost import ExecutionHistory
from aar.ir import Node, NodeType
from aar.runtime import Executor, NodeOutcome


def test_record_preserves_every_input_and_unknown_metrics_are_none():
    history = ExecutionHistory(target_key="target-a")

    rec = history.record(
        "aar-op-v1:join",
        "duckdb",
        (2_000_000_000, 8_000_000_000),
        (20_000_000, 80_000_000),
        125.0,
        output_bytes=900_000_000,
        output_rows=9_000_000,
    )

    assert rec.input_bytes == (2_000_000_000, 8_000_000_000)
    assert rec.input_rows == (20_000_000, 80_000_000)
    assert rec.total_input_bytes == 10_000_000_000
    assert rec.total_input_rows == 100_000_000
    assert rec.output_bytes == 900_000_000
    assert rec.output_rows == 9_000_000
    assert rec.peak_memory is None
    assert rec.bytes_transferred is None


def test_scalar_legacy_calls_are_stored_as_one_element_vectors():
    history = ExecutionHistory()
    rec = history.record("op", "duckdb", 1000, 10, 5.0)
    assert rec.input_bytes == (1000,)
    assert rec.input_rows == (10,)
    assert rec.nbytes == 1000
    assert rec.rows == 10


def test_input_vectors_must_describe_the_same_number_of_inputs():
    history = ExecutionHistory()
    with pytest.raises(ValueError, match="same inputs"):
        history.record("join", "duckdb", (100, 200), (10,), 5.0)


def test_semantic_key_reuses_history_across_separately_created_nodes():
    history = ExecutionHistory(
        operation_key_fn=lambda node: f"aar-op-v1:{node.type.value}",
        target_key="machine-a",
    )
    first = Node(NodeType.GROUPBY)
    second = Node(NodeType.GROUPBY)
    assert first.id != second.id

    history.record(
        history.operation_key(first), "duckdb", (1000,), (10,), 50.0,
        output_bytes=100, output_rows=1,
    )

    assert history.operation_key(first) == history.operation_key(second)
    assert history.predict(second, "duckdb", (1000,)) == pytest.approx(0.05)


def test_target_identity_prevents_cross_machine_training():
    history = ExecutionHistory(target_key="machine-a")
    history.record("op", "duckdb", (1000,), (10,), 10.0,
                   target_key="machine-b")

    assert history.records("op", "duckdb") == []
    assert history.predict("op", "duckdb", (1000,)) is None

    history.record("op", "duckdb", (1000,), (10,), 20.0,
                   target_key="machine-a")
    assert history.predict("op", "duckdb", (1000,)) == pytest.approx(0.02)


def test_regression_uses_total_bytes_without_discarding_the_vector():
    history = ExecutionHistory(min_samples_for_regression=3, target_key="t")
    for inputs, elapsed in (
        ((500, 500), 10.0),
        ((1000, 1000), 20.0),
        ((1500, 1500), 30.0),
        ((2000, 2000), 40.0),
    ):
        history.record("join", "duckdb", inputs, (1, 1), elapsed)

    small = history.predict("join", "duckdb", (1000, 1000))
    large = history.predict("join", "duckdb", (4000, 4000))
    assert small is not None and large is not None
    assert large > small
    assert history.records("join", "duckdb")[0].input_bytes == (500, 500)


def test_executor_observer_passes_the_full_join_shape_to_history():
    history = ExecutionHistory(
        operation_key_fn=lambda _node: "aar-op-v1:join",
        target_key="machine-a",
    )
    node = Node(NodeType.JOIN)
    outcome = NodeOutcome(
        node_id=node.id,
        node_type=str(node.type),
        engine_requested="duckdb",
        engine_used="duckdb",
        input_rows=(10, 80),
        input_bytes=(2_000, 8_000),
        rows_in=90,
        bytes_in=10_000,
        rows_out=7,
        bytes_out=700,
        elapsed_ms=12.5,
    )

    Executor._observe(history, outcome, node, success=True)

    rec = history.records("aar-op-v1:join", "duckdb")[0]
    assert rec.input_bytes == (2_000, 8_000)
    assert rec.input_rows == (10, 80)
    assert rec.output_bytes == 700
    assert rec.output_rows == 7
    assert rec.target_key == "machine-a"


def test_failed_observation_does_not_claim_an_output():
    history = ExecutionHistory(
        operation_key_fn=lambda _node: "aar-op-v1:join",
        target_key="machine-a",
    )
    node = Node(NodeType.JOIN)
    outcome = NodeOutcome(
        node_id=node.id,
        node_type=str(node.type),
        engine_requested="duckdb",
        engine_used="duckdb",
        input_rows=(10, 20),
        input_bytes=(100, 200),
        rows_in=30,
        bytes_in=300,
        elapsed_ms=1.0,
        error="boom",
    )

    Executor._observe(history, outcome, node, success=False)

    all_records = history.records(
        "aar-op-v1:join", "duckdb", successful_only=False)
    assert len(all_records) == 1
    rec = all_records[0]
    assert rec.success is False
    assert rec.output_bytes is None
    assert rec.output_rows is None
    assert history.predict(node, "duckdb", (100, 200)) is None
