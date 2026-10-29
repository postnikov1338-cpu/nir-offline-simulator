"""Shared metadata for reproducible component profiles.

The classes in this module describe where a value came from.  They do not
provide generator, rectifier or DC-bus physics; those belong to later phases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_SOURCE_TYPES = frozenset({
    "manufacturer",
    "literature",
    "digitized_manufacturer_graph",
    "derived_system_requirement",
    "engineering_analogue",
    "engineering_assumption",
    "user_defined",
    "synthetic_test_fixture",
    "scientific_analogue",
    "derived_engineering_profile",
})


@dataclass(frozen=True)
class SourceRef:
    """A stable source reference stored with a component profile."""

    source_id: str
    title: str
    url: str
    source_type: str
    scope: str
    accessed_on: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("source_id", "title", "url", "scope"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"Источник: поле {field_name} не должно быть пустым")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Неизвестный тип источника: {self.source_type}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterProvenance:
    """Origin and confidence of one profile field or a group of fields."""

    source_id: str
    source_type: str
    confidence: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Неизвестный тип происхождения: {self.source_type}")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence должен быть high, medium или low")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
