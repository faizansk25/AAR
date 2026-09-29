"""Type system: mapping, loss detection, schema algebra, drift."""

from __future__ import annotations

import pytest

from aar import types as T
from aar.types import (DECIMAL, FLOAT64, INT32, INT64, INT8, TIMESTAMP, UINT8,
                       ConversionLog, Field, Schema, TypeKind)


class TestSourceMapping:
    """Every documented source type must map, or raise - never guess."""

    @pytest.mark.parametrize("name,expected", [
        ("int4", INT32), ("int8", INT64), ("float4", T.FLOAT32),
        ("float8", FLOAT64), ("text", T.UTF8), ("bytea", T.BINARY),
    ])
    def test_postgres(self, name, expected):
        assert T.from_source(name, "postgresql") == expected

    def test_postgres_timestamptz_is_utc_aware(self):
        got = T.from_source("timestamptz", "postgresql")
        assert got.kind is TypeKind.TIMESTAMP
        assert got.timezone == "UTC"
        assert got.unit == "us"

    def test_postgres_timestamp_is_naive(self):
        got = T.from_source("timestamp", "postgresql")
        assert got.kind is TypeKind.TIMESTAMP
        assert got.timezone is None

    def test_postgres_parameterised_numeric(self):
        assert T.from_source("numeric(12,2)", "postgresql") == DECIMAL(12, 2)
        assert T.from_source("numeric(5,0)", "postgresql") == DECIMAL(5, 0)

    def test_mongo_bson_names(self):
        assert T.from_source("int", "mongodb") == INT32
        assert T.from_source("long", "mongodb") == INT64
        assert T.from_source("double", "mongodb") == FLOAT64
        assert T.from_source("date", "mongodb") == TIMESTAMP("ms", "UTC")

    def test_excel_normalises_to_canonical(self):
        assert T.from_source("number", "excel") == FLOAT64
        assert T.from_source("text", "excel") == T.UTF8
        assert T.from_source("date", "excel") == TIMESTAMP("s", None)

    def test_sqlite_case_insensitive(self):
        assert T.from_source("INTEGER", "sqlite") == INT64
        assert T.from_source("integer", "sqlite") == INT64

    def test_arrow_preserves_resolution(self):
        assert T.from_source("timestamp[ms]", "arrow") == TIMESTAMP("ms", None)
        assert T.from_source("timestamp[ns]", "arrow") == TIMESTAMP("ns", None)

    def test_unknown_type_raises_rather_than_guessing(self):
        with pytest.raises(T.UnmappableType):
            T.from_source("quaternion", "postgresql")

    def test_unknown_system_raises(self):
        with pytest.raises(T.UnmappableType):
            T.from_source("int", "mainframe-db")

    def test_python_inference(self):
        assert T.from_python(1) == INT64
        assert T.from_python(True) is T.BOOLEAN
        assert T.from_python("x") is T.UTF8
        assert T.from_python(None) is T.NULL

    def test_python_datetime_keeps_timezone(self):
        import datetime as dt

        got = T.from_python(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
        assert got.kind is TypeKind.TIMESTAMP
        assert got.timezone == "UTC"


class TestLossDetection:
    """`lossy` is the single source of truth for conversion safety."""

    def test_identity_is_lossless(self):
        assert T.lossy(INT64, INT64) is None

    def test_widening_integer_is_lossless(self):
        assert T.lossy(INT8, INT32) is None
        assert T.lossy(INT8, INT64) is None
        assert T.lossy(INT32, INT64) is None

    def test_narrowing_integer_is_lossy(self):
        assert T.lossy(INT64, INT32) is not None
        assert T.lossy(INT32, INT8) is not None

    def test_signed_to_unsigned_boundary(self):
        # -128..127 fits in uint8's 0..255
        assert T.lossy(INT8, UINT8) is None
        # int32's range does not
        assert T.lossy(INT32, UINT8) is not None

    def test_unsigned_to_signed_boundary(self):
        # uint8 0..255 needs 9 bits; int8 has 8
        assert T.lossy(UINT8, T.INT8) is not None
        # uint8 does fit in int16
        assert T.lossy(UINT8, T.INT16) is None

    def test_int64_to_float64_accepted(self):
        assert T.lossy(INT64, FLOAT64) is None

    def test_float_to_int_is_lossy(self):
        assert T.lossy(FLOAT64, INT64) is not None

    def test_timestamp_resolution_narrowing_is_lossy(self):
        # ns -> us drops precision, so it is genuinely lossy.
        assert T.lossy(TIMESTAMP("ns"), TIMESTAMP("us")) is not None
        assert T.lossy(TIMESTAMP("us"), TIMESTAMP("ms")) is not None

    def test_timestamp_resolution_widening_is_lossless(self):
        # Widening a timestamp never loses a value: microseconds become
        # nanoseconds exactly. This is the asymmetry that matters, and the
        # reason "always store the finest resolution" is a safe default.
        assert T.lossy(TIMESTAMP("us"), TIMESTAMP("ns")) is None
        assert T.lossy(TIMESTAMP("ms"), TIMESTAMP("ns")) is None
        assert T.lossy(TIMESTAMP("s"), TIMESTAMP("ms")) is None

    def test_naive_to_aware_requires_conversion(self):
        assert T.lossy(TIMESTAMP("us"), TIMESTAMP("us", "UTC")) is not None
        assert T.lossy(TIMESTAMP("us", "UTC"), TIMESTAMP("us", "UTC")) is None

    def test_null_source_is_always_safe(self):
        assert T.lossy(T.NULL, INT64) is None
        assert T.lossy(T.NULL, T.UTF8) is None

    def test_text_to_numeric_is_lossy(self):
        assert T.lossy(T.UTF8, INT64) is not None


class TestCoercion:
    def test_widen_promotes_float32(self):
        assert T.widen(T.FLOAT32) is FLOAT64

    def test_widen_promotes_int8(self):
        assert T.widen(INT8) is INT8

    def test_common_type_integers(self):
        assert T.common_type([INT32, INT64]) is INT64

    def test_common_type_mixed_float_wins(self):
        assert T.common_type([INT32, FLOAT64]) is FLOAT64

    def test_common_type_all_null(self):
        assert T.common_type([T.NULL, T.NULL]) is T.NULL

    def test_common_type_empty(self):
        assert T.common_type([]) is T.NULL

    def test_arrow_name_round_trips_for_scalars(self):
        for dt in (INT64, FLOAT64, T.UTF8, T.BOOLEAN, T.DATE32):
            assert T.from_source(T.arrow_name(dt), "arrow") == dt


class TestSchemaAlgebra:
    def test_select_and_drop(self):
        s = Schema((Field("a", INT64), Field("b", INT32), Field("c", T.UTF8)))
        assert s.select(["a", "c"]).names == ("a", "c")
        assert s.drop(["b"]).names == ("a", "c")

    def test_rename(self):
        s = Schema((Field("a", INT64),))
        assert s.rename({"a": "z"}).names == ("z",)

    def test_duplicate_column_rejected(self):
        with pytest.raises(ValueError):
            Schema((Field("a", INT64), Field("a", INT32)))

    def test_get_missing_raises_with_help(self):
        s = Schema((Field("a", INT64), Field("b", INT32)))
        with pytest.raises(KeyError) as exc:
            s.get("zzz")
        assert "have:" in str(exc.value)

    def test_cast_preserves_classification(self):
        s = Schema((Field("salary", INT64,
                          classification=frozenset({"CONFIDENTIAL"}))),)
        got = s.cast({"salary": DECIMAL(12, 2)})
        assert got.get("salary").type == DECIMAL(12, 2)
        assert "CONFIDENTIAL" in got.get("salary").classification

    def test_inherit_classification_widens_tags(self):
        s = Schema((Field("id", INT64, classification=frozenset({"PII"})),
                    Field("amt", FLOAT64)))
        got = s.inherit_classification("id")
        assert "PII" in got.get("amt").classification

    def test_all_classifications(self):
        s = Schema((Field("a", INT64, classification=frozenset({"X"})),
                    Field("b", INT32, classification=frozenset({"Y"}))))
        assert s.all_classifications() == frozenset({"X", "Y"})


class TestSchemaDrift:
    def test_identical_schema_has_no_drift(self):
        s = Schema((Field("a", INT64),))
        assert s.diff(s).identical

    def test_added_column_detected(self):
        expected = Schema((Field("a", INT64),))
        actual = Schema((Field("a", INT64), Field("b", INT32)))
        d = expected.diff(actual)
        assert d.added == ("b",)
        assert not d.identical

    def test_removed_column_detected(self):
        expected = Schema((Field("a", INT64), Field("b", INT32)))
        actual = Schema((Field("a", INT64),))
        assert expected.diff(actual).removed == ("b",)

    def test_retyped_column_detected(self):
        expected = Schema((Field("a", INT64),))
        actual = Schema((Field("a", INT32),))
        assert expected.diff(actual).retyped == (("a", INT64, INT32),)

    def test_drift_describes_human_readably(self):
        expected = Schema((Field("a", INT64), Field("gone", INT32)))
        actual = Schema((Field("a", T.UTF8), Field("new", INT32)))
        text = expected.diff(actual).describe()
        assert "added" in text and "removed" in text and "retyped" in text


class TestConversionLog:
    def test_lossy_conversion_is_flagged(self):
        log = ConversionLog()
        entry = log.record("amount", "excel", "number", INT32)
        assert entry.is_lossy
        assert "amount" in entry.render()

    def test_safe_conversion_is_not_flagged(self):
        log = ConversionLog()
        entry = log.record("amount", "excel", "number", FLOAT64)
        assert not entry.is_lossy

    def test_log_collects_lossy_entries(self):
        log = ConversionLog()
        log.record("a", "excel", "number", FLOAT64)
        log.record("b", "excel", "number", INT32)
        assert len(log) == 2
        assert len(log.lossy_entries) == 1

