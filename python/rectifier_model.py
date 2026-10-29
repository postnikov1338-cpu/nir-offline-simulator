"""Autonomous Phase 3 model of a three-phase six-pulse diode bridge.

The module consumes DC current, reverse voltage and measured/assumed case
temperature supplied by its caller.  It is deliberately not connected to the
generator, battery, DC bus or GUI.  Forward-voltage interpolation is confined
to the explicit merged surface of tabulated and digitized manufacturer data
and never extrapolates silently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from component_profiles import ALLOWED_SOURCE_TYPES, SourceRef


ROOT = Path(__file__).resolve().parent
BUILTIN_RECTIFIER_DIR = ROOT / "data" / "rectifiers"
SUPPORTED_TOPOLOGY = "six_pulse_diode"
CURRENT_RATING_BASES = frozenset({
    "per_diode_continuous",
    "per_diode_average",
    "per_module_continuous",
    "per_module_average",
})


class RectifierTableOutOfRangeError(ValueError):
    """Requested forward-current/temperature point is outside the table."""


@dataclass(frozen=True)
class RectifierParameterProvenance:
    source_id: str
    source_type: str
    confidence: str
    units: str
    conditions: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Неизвестный source_type: {self.source_type}")
        if self.confidence not in {"high", "medium", "low", "approximate"}:
            raise ValueError("confidence должен быть high, medium, low или approximate")
        if not all(value.strip() for value in
                   (self.source_id, self.units, self.conditions)):
            raise ValueError("provenance требует source_id, units и conditions")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForwardVoltagePoint:
    forward_current_a: float
    junction_temperature_c: float
    forward_voltage_v: float
    value_kind: str
    source_id: str
    source_type: str
    units: str
    conditions: str
    confidence: str = "high"

    def __post_init__(self) -> None:
        if (not math.isfinite(self.forward_current_a) or self.forward_current_a < 0
                or not math.isfinite(self.forward_voltage_v) or self.forward_voltage_v <= 0):
            raise ValueError("Точка Vf должна содержать неотрицательный ток и положительное напряжение")
        if not math.isfinite(self.junction_temperature_c):
            raise ValueError("Температура точки Vf должна быть конечной")
        if self.value_kind not in {"typical", "maximum"}:
            raise ValueError("value_kind должен быть typical или maximum")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError("Неизвестный source_type точки Vf")
        if self.confidence not in {"high", "approximate"}:
            raise ValueError("confidence точки Vf должен быть high или approximate")
        if self.units != "V" or not self.conditions.strip() or not self.source_id.strip():
            raise ValueError("Точка Vf требует units='V', source_id и conditions")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurrentRating:
    rating_id: str
    basis: str
    current_a: float
    case_temperature_c: float
    conduction_condition: str
    source_id: str
    source_type: str
    units: str
    conditions: str
    bridge_interpretation: str = "direct_conducting_diode_limit"

    def __post_init__(self) -> None:
        if self.basis not in CURRENT_RATING_BASES:
            raise ValueError(f"Неизвестный basis токового рейтинга: {self.basis}")
        if not math.isfinite(self.current_a) or self.current_a <= 0:
            raise ValueError("Токовый рейтинг должен быть положительным")
        if not math.isfinite(self.case_temperature_c):
            raise ValueError("Температура токового рейтинга должна быть конечной")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError("Неизвестный source_type токового рейтинга")
        if self.bridge_interpretation not in {
            "direct_conducting_diode_limit", "derived_conditional_limit",
            "derived_duty_cycle_limit",
        }:
            raise ValueError("Неизвестная интерпретация рейтинга для моста")
        if self.units != "A" or not all(value.strip() for value in (
            self.rating_id, self.conduction_condition, self.source_id, self.conditions,
        )):
            raise ValueError("Токовый рейтинг требует идентификаторы, условия и units='A'")

    def equivalent_bridge_dc_current_limit_a(self) -> float:
        if self.basis == "per_diode_continuous":
            return self.current_a
        if self.basis == "per_diode_average":
            return 3.0 * self.current_a
        if self.basis == "per_module_continuous":
            return self.current_a
        if self.basis == "per_module_average":
            return 1.5 * self.current_a
        raise AssertionError("unreachable")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bridge_interpretation != "direct_conducting_diode_limit":
            formula = {
                "per_module_average": "2*Idc/3 <= IF(AV)_module",
                "per_diode_average": "Idc/3 <= IF(AV)_diode",
                "per_module_continuous": "Idc <= IF_module",
            }.get(self.basis, "Idc <= IF_diode")
            data["derived_bridge_limit"] = {
                "classification": self.bridge_interpretation,
                "source_type": "derived_system_requirement",
                "formula": formula,
                "conditional_limit_a": self.equivalent_bridge_dc_current_limit_a(),
                "valid_only_under_rating_conditions": self.conditions,
            }
        return data


@dataclass(frozen=True)
class RectifierReadiness:
    identity_ready: bool
    topology_ready: bool
    vf_table_ready: bool
    electrical_limits_ready: bool
    current_ratings_ready: bool
    static_thermal_ready: bool
    phase3_ready: bool
    dynamic_thermal_ready: bool
    missing_for_phase3: tuple[str, ...]
    missing_for_dynamic_thermal: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["missing_for_phase3"] = list(self.missing_for_phase3)
        data["missing_for_dynamic_thermal"] = list(self.missing_for_dynamic_thermal)
        return data


@dataclass(frozen=True)
class RectifierProfile:
    profile_id: str
    revision: int
    manufacturer: str
    model: str
    source_type: str
    validation_status: str
    bridge_topology: str | None
    module_count: int | None
    diode_count: int | None
    diodes_per_module: int | None
    reverse_voltage_limit_v: float | None
    junction_temperature_min_c: float | None
    junction_temperature_max_c: float | None
    rth_junction_to_case_single_leg_k_per_w: float | None
    rth_junction_to_case_both_legs_k_per_w: float | None
    rth_case_to_heatsink_k_per_w: float | None
    vf_model_value_kind: str | None
    tabulated_vf_points: tuple[ForwardVoltagePoint, ...]
    digitized_vf_points: tuple[ForwardVoltagePoint, ...]
    current_ratings: tuple[CurrentRating, ...]
    sources: tuple[SourceRef, ...]
    provenance: dict[str, RectifierParameterProvenance]
    warnings: tuple[str, ...] = ()
    thermal_capacitance_j_per_k: float | None = None
    heatsink_thermal_resistance_k_per_w: float | None = None
    airflow_model: Any | None = None
    transient_thermal_impedance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or self.revision < 1:
            raise ValueError("Профиль выпрямителя: неверные profile_id или revision")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError("Неизвестный source_type профиля выпрямителя")
        for name in ("module_count", "diode_count", "diodes_per_module"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} должен быть положительным или null")
        for name in (
            "reverse_voltage_limit_v", "rth_junction_to_case_single_leg_k_per_w",
            "rth_junction_to_case_both_legs_k_per_w", "rth_case_to_heatsink_k_per_w",
            "thermal_capacitance_j_per_k", "heatsink_thermal_resistance_k_per_w",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} должен быть положительным конечным числом или null")
        if self.vf_model_value_kind not in {"typical", "maximum", None}:
            raise ValueError("vf_model_value_kind должен быть typical, maximum или null")
        for point in self.digitized_vf_points:
            if (point.source_type != "digitized_manufacturer_graph"
                    or point.confidence != "approximate"):
                raise ValueError("Оцифрованная Vf-точка требует digitized_manufacturer_graph/approximate")
        if (self.junction_temperature_min_c is not None
                and self.junction_temperature_max_c is not None
                and self.junction_temperature_max_c <= self.junction_temperature_min_c):
            raise ValueError("Неверный диапазон температуры перехода")

    @staticmethod
    def _vf_grid_ready(points: tuple[ForwardVoltagePoint, ...], kind: str | None) -> bool:
        selected = tuple(point for point in points if point.value_kind == kind)
        if not selected:
            return False
        currents = {point.forward_current_a for point in selected}
        temperatures = {point.junction_temperature_c for point in selected}
        pairs = {(point.forward_current_a, point.junction_temperature_c) for point in selected}
        return len(pairs) == len(selected) and all(
            (current, temperature) in pairs
            for current in currents for temperature in temperatures
        )

    def effective_vf_points(self) -> tuple[ForwardVoltagePoint, ...]:
        """Merge sources with exact tabulated values taking coordinate priority."""
        kind = self.vf_model_value_kind
        merged = {
            (point.forward_current_a, point.junction_temperature_c): point
            for point in self.digitized_vf_points if point.value_kind == kind
        }
        for point in self.tabulated_vf_points:
            if point.value_kind == kind:
                merged[(point.forward_current_a, point.junction_temperature_c)] = point
        return tuple(merged[key] for key in sorted(merged))

    @property
    def readiness(self) -> RectifierReadiness:
        identity = bool(self.profile_id and self.manufacturer and self.model)
        topology = (self.bridge_topology == SUPPORTED_TOPOLOGY
                    and self.module_count == 3 and self.diode_count == 6
                    and self.diodes_per_module == 2)
        vf_table = self._vf_grid_ready(self.effective_vf_points(), self.vf_model_value_kind)
        electrical_limits = all(value is not None for value in (
            self.reverse_voltage_limit_v, self.junction_temperature_min_c,
            self.junction_temperature_max_c,
        ))
        current_ratings = bool(self.current_ratings)
        static_thermal = all(value is not None for value in (
            self.rth_junction_to_case_single_leg_k_per_w,
            self.rth_junction_to_case_both_legs_k_per_w,
            self.rth_case_to_heatsink_k_per_w,
        ))
        requirements = {
            "identity": identity, "six_pulse_topology_3_modules_6_diodes": topology,
            "vf_table": vf_table, "reverse_and_junction_limits": electrical_limits,
            "conditioned_current_ratings": current_ratings,
            "static_thermal_parameters": static_thermal,
        }
        missing = tuple(key for key, available in requirements.items() if not available)
        dynamic_requirements = {
            "thermal_capacitance_j_per_k": self.thermal_capacitance_j_per_k is not None,
            "heatsink_thermal_resistance_k_per_w": self.heatsink_thermal_resistance_k_per_w is not None,
            "airflow_model": self.airflow_model is not None,
        }
        missing_dynamic = tuple(key for key, available in dynamic_requirements.items()
                                if not available)
        return RectifierReadiness(
            identity_ready=identity, topology_ready=topology, vf_table_ready=vf_table,
            electrical_limits_ready=electrical_limits,
            current_ratings_ready=current_ratings, static_thermal_ready=static_thermal,
            phase3_ready=not missing, dynamic_thermal_ready=not missing_dynamic,
            missing_for_phase3=missing,
            missing_for_dynamic_thermal=missing_dynamic,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RectifierProfile":
        topology = data.get("topology", {})
        limits = data.get("limits", {})
        thermal = data.get("thermal", {})
        vf = data.get("forward_voltage_model", {})
        tabulated_points = vf.get("tabulated_points", vf.get("points", ()))
        return cls(
            profile_id=data["profile_id"], revision=int(data["revision"]),
            manufacturer=data.get("manufacturer", ""), model=data.get("model", ""),
            source_type=data["source_type"], validation_status=data["validation_status"],
            bridge_topology=topology.get("bridge_topology"),
            module_count=topology.get("module_count"), diode_count=topology.get("diode_count"),
            diodes_per_module=topology.get("diodes_per_module"),
            reverse_voltage_limit_v=_optional_float(limits.get("reverse_voltage_limit_v")),
            junction_temperature_min_c=_optional_float(limits.get("junction_temperature_min_c")),
            junction_temperature_max_c=_optional_float(limits.get("junction_temperature_max_c")),
            rth_junction_to_case_single_leg_k_per_w=_optional_float(
                thermal.get("rth_junction_to_case_single_leg_k_per_w")),
            rth_junction_to_case_both_legs_k_per_w=_optional_float(
                thermal.get("rth_junction_to_case_both_legs_k_per_w")),
            rth_case_to_heatsink_k_per_w=_optional_float(
                thermal.get("rth_case_to_heatsink_k_per_w")),
            vf_model_value_kind=vf.get("value_kind"),
            tabulated_vf_points=tuple(ForwardVoltagePoint(**point)
                                      for point in tabulated_points),
            digitized_vf_points=tuple(ForwardVoltagePoint(**point)
                                      for point in vf.get("digitized_points", ())),
            current_ratings=tuple(CurrentRating(**rating)
                                  for rating in data.get("current_ratings", ())),
            sources=tuple(SourceRef(**source) for source in data.get("sources", ())),
            provenance={key: RectifierParameterProvenance(**value)
                        for key, value in data.get("provenance", {}).items()},
            warnings=tuple(data.get("warnings", ())),
            thermal_capacitance_j_per_k=_optional_float(thermal.get("thermal_capacitance_j_per_k")),
            heatsink_thermal_resistance_k_per_w=_optional_float(
                thermal.get("heatsink_thermal_resistance_k_per_w")),
            airflow_model=thermal.get("airflow_model"),
            transient_thermal_impedance=thermal.get("transient_thermal_impedance"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RectifierProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_builtin(cls, profile_id: str) -> "RectifierProfile":
        path = BUILTIN_RECTIFIER_DIR / f"{profile_id}.json"
        if not path.is_file():
            raise ValueError(f"Неизвестный встроенный профиль выпрямителя: {profile_id}")
        return cls.load(path)

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "profile_id": self.profile_id, "revision": self.revision,
            "manufacturer": self.manufacturer, "model": self.model,
            "source_type": self.source_type, "validation_status": self.validation_status,
            "topology": {
                "bridge_topology": self.bridge_topology, "module_count": self.module_count,
                "diode_count": self.diode_count, "diodes_per_module": self.diodes_per_module,
            },
            "limits": {
                "reverse_voltage_limit_v": self.reverse_voltage_limit_v,
                "junction_temperature_min_c": self.junction_temperature_min_c,
                "junction_temperature_max_c": self.junction_temperature_max_c,
            },
            "thermal": {
                "rth_junction_to_case_single_leg_k_per_w": self.rth_junction_to_case_single_leg_k_per_w,
                "rth_junction_to_case_both_legs_k_per_w": self.rth_junction_to_case_both_legs_k_per_w,
                "rth_case_to_heatsink_k_per_w": self.rth_case_to_heatsink_k_per_w,
                "thermal_capacitance_j_per_k": self.thermal_capacitance_j_per_k,
                "heatsink_thermal_resistance_k_per_w": self.heatsink_thermal_resistance_k_per_w,
                "airflow_model": self.airflow_model,
                "transient_thermal_impedance": self.transient_thermal_impedance,
            },
            "forward_voltage_model": {
                "value_kind": self.vf_model_value_kind,
                "interpolation": "piecewise_bilinear_inside_confirmed_table_only",
                "tabulated_points": [point.to_dict() for point in self.tabulated_vf_points],
                "digitized_points": [point.to_dict() for point in self.digitized_vf_points],
                "effective_points": [point.to_dict() for point in self.effective_vf_points()],
            },
            "current_ratings": [rating.to_dict() for rating in self.current_ratings],
            "sources": [source.to_dict() for source in self.sources],
            "provenance": {key: value.to_dict() for key, value in self.provenance.items()},
            "warnings": list(self.warnings), "readiness": self.readiness.to_dict(),
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class RectifierOperatingPoint:
    status: str
    diagnostics: tuple[str, ...]
    active_limits: tuple[str, ...]
    dc_current_a: float
    reverse_voltage_v: float
    case_temperature_c: float
    diode_conducting_current_a: float
    diode_average_current_a: float
    diode_rms_current_a: float
    forward_voltage_per_conducting_diode_v: float | None
    bridge_forward_voltage_drop_v: float | None
    rectifier_loss_w: float | None
    average_loss_per_diode_w: float | None
    average_loss_per_module_w: float | None
    junction_temperature_c: float | None
    current_limit_at_case_temperature_a: float | None
    case_to_heatsink_temperature_drop_k: float | None
    implied_heatsink_interface_temperature_c: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["diagnostics"] = list(self.diagnostics)
        data["active_limits"] = list(self.active_limits)
        return data


class RectifierModel:
    """Static six-pulse bridge loss and junction-temperature model."""

    def __init__(self, profile: RectifierProfile):
        self.profile = profile
        self._vf_points = profile.effective_vf_points()

    @staticmethod
    def diode_currents(dc_current_a: float) -> tuple[float, float, float]:
        current = float(dc_current_a)
        if not math.isfinite(current) or current < 0:
            raise ValueError("I_dc должен быть неотрицательным конечным числом")
        return current, current / 3.0, current / math.sqrt(3.0)

    @staticmethod
    def _bracket(values: tuple[float, ...], query: float, label: str) -> tuple[float, float]:
        if not values or query < values[0] or query > values[-1]:
            lo = values[0] if values else math.nan
            hi = values[-1] if values else math.nan
            raise RectifierTableOutOfRangeError(
                f"vf_table_range: {label}={query:g} вне [{lo:g}, {hi:g}]")
        for value in values:
            if math.isclose(query, value, rel_tol=0.0, abs_tol=1e-12):
                return value, value
        for left, right in zip(values, values[1:]):
            if left < query < right:
                return left, right
        raise RectifierTableOutOfRangeError(f"vf_table_range: нет интервала для {label}")

    def forward_voltage(self, forward_current_a: float, junction_temperature_c: float) -> float:
        current = float(forward_current_a)
        temperature = float(junction_temperature_c)
        if not math.isfinite(current) or current < 0 or not math.isfinite(temperature):
            raise ValueError("I_forward должен быть неотрицательным, T_junction — конечной")
        if not self.profile.readiness.vf_table_ready:
            raise RectifierTableOutOfRangeError("vf_table_range: таблица Vf не готова")
        currents = tuple(sorted({point.forward_current_a for point in self._vf_points}))
        temperatures = tuple(sorted({point.junction_temperature_c for point in self._vf_points}))
        i0, i1 = self._bracket(currents, current, "I_forward")
        t0, t1 = self._bracket(temperatures, temperature, "T_junction")
        table = {(point.forward_current_a, point.junction_temperature_c): point.forward_voltage_v
                 for point in self._vf_points}

        def at_temperature(temp: float) -> float:
            v0 = table[(i0, temp)]
            if i0 == i1:
                return v0
            v1 = table[(i1, temp)]
            return v0 + (current - i0) * (v1 - v0) / (i1 - i0)

        low = at_temperature(t0)
        if t0 == t1:
            return low
        high = at_temperature(t1)
        return low + (temperature - t0) * (high - low) / (t1 - t0)

    def current_limit_at_case_temperature(self, case_temperature_c: float) -> tuple[float | None, tuple[str, ...]]:
        case_temperature = float(case_temperature_c)
        if not math.isfinite(case_temperature):
            raise ValueError("Температура корпуса должна быть конечной")
        applicable: list[float] = []
        unavailable: list[str] = []
        for rating in self.profile.current_ratings:
            if case_temperature <= rating.case_temperature_c:
                applicable.append(rating.equivalent_bridge_dc_current_limit_a())
            else:
                unavailable.append(f"{rating.rating_id}:case_condition_exceeded")
        if not applicable:
            return None, ("current_limit_not_available_for_case_temperature", *unavailable)
        return min(applicable), tuple(unavailable)

    def _empty_point(self, status: str, diagnostics: tuple[str, ...], active_limits: tuple[str, ...],
                     dc_current_a: float, reverse_voltage_v: float, case_temperature_c: float,
                     current_limit_a: float | None = None) -> RectifierOperatingPoint:
        conducting, average, rms = self.diode_currents(dc_current_a)
        return RectifierOperatingPoint(
            status=status, diagnostics=diagnostics, active_limits=active_limits,
            dc_current_a=dc_current_a, reverse_voltage_v=reverse_voltage_v,
            case_temperature_c=case_temperature_c,
            diode_conducting_current_a=conducting, diode_average_current_a=average,
            diode_rms_current_a=rms, forward_voltage_per_conducting_diode_v=None,
            bridge_forward_voltage_drop_v=None, rectifier_loss_w=None,
            average_loss_per_diode_w=None, average_loss_per_module_w=None,
            junction_temperature_c=None,
            current_limit_at_case_temperature_a=current_limit_a,
            case_to_heatsink_temperature_drop_k=None,
            implied_heatsink_interface_temperature_c=None,
        )

    def operating_point(self, dc_current_a: float, case_temperature_c: float,
                        reverse_voltage_v: float) -> RectifierOperatingPoint:
        current = float(dc_current_a)
        case_temperature = float(case_temperature_c)
        reverse_voltage = float(reverse_voltage_v)
        if not all(math.isfinite(value) for value in (current, case_temperature, reverse_voltage)):
            raise ValueError("Ток, обратное напряжение и температура корпуса должны быть конечными")
        if current < 0 or reverse_voltage < 0:
            raise ValueError("Ток и обратное напряжение не могут быть отрицательными")
        readiness = self.profile.readiness
        if not readiness.phase3_ready:
            return self._empty_point(
                "not_ready", readiness.missing_for_phase3, (), current, reverse_voltage,
                case_temperature,
            )
        current_limit, current_diagnostics = self.current_limit_at_case_temperature(case_temperature)
        active_limits: list[str] = []
        diagnostics = list(current_diagnostics)
        if reverse_voltage > self.profile.reverse_voltage_limit_v:  # type: ignore[operator]
            active_limits.append("reverse_voltage_limit")
        if current_limit is None:
            active_limits.append("current_limit_not_evaluable")
        elif current > current_limit:
            active_limits.append("current_limit")

        if current == 0:
            junction_temperature = case_temperature
            if not (self.profile.junction_temperature_min_c <= junction_temperature  # type: ignore[operator]
                    <= self.profile.junction_temperature_max_c):  # type: ignore[operator]
                active_limits.append("junction_temperature_limit")
            return RectifierOperatingPoint(
                status="limit_exceeded" if active_limits else "ok",
                diagnostics=tuple(diagnostics), active_limits=tuple(active_limits),
                dc_current_a=current, reverse_voltage_v=reverse_voltage,
                case_temperature_c=case_temperature,
                diode_conducting_current_a=0.0, diode_average_current_a=0.0,
                diode_rms_current_a=0.0, forward_voltage_per_conducting_diode_v=None,
                bridge_forward_voltage_drop_v=0.0, rectifier_loss_w=0.0,
                average_loss_per_diode_w=0.0, average_loss_per_module_w=0.0,
                junction_temperature_c=junction_temperature,
                current_limit_at_case_temperature_a=current_limit,
                case_to_heatsink_temperature_drop_k=0.0,
                implied_heatsink_interface_temperature_c=case_temperature,
            )

        junction_temperature = case_temperature
        try:
            for _ in range(100):
                vf = self.forward_voltage(current, junction_temperature)
                average_loss_per_diode = vf * current / 3.0
                next_temperature = case_temperature + average_loss_per_diode * (
                    self.profile.rth_junction_to_case_single_leg_k_per_w)  # type: ignore[operator]
                if abs(next_temperature - junction_temperature) <= 1e-10:
                    junction_temperature = next_temperature
                    break
                junction_temperature = next_temperature
            else:
                diagnostics.append("static_thermal_iteration_not_converged")
            vf = self.forward_voltage(current, junction_temperature)
        except RectifierTableOutOfRangeError as exc:
            diagnostics.append(str(exc))
            active_limits.append("vf_table_range")
            return self._empty_point(
                "vf_table_out_of_range", tuple(diagnostics), tuple(active_limits),
                current, reverse_voltage, case_temperature, current_limit,
            )

        rectifier_loss = 2.0 * vf * current
        average_loss_per_diode = rectifier_loss / self.profile.diode_count  # type: ignore[operator]
        average_loss_per_module = rectifier_loss / self.profile.module_count  # type: ignore[operator]
        junction_temperature = case_temperature + average_loss_per_diode * (
            self.profile.rth_junction_to_case_single_leg_k_per_w)  # type: ignore[operator]
        if not (self.profile.junction_temperature_min_c <= junction_temperature  # type: ignore[operator]
                <= self.profile.junction_temperature_max_c):  # type: ignore[operator]
            active_limits.append("junction_temperature_limit")
        case_to_sink_drop = average_loss_per_module * (
            self.profile.rth_case_to_heatsink_k_per_w)  # type: ignore[operator]
        return RectifierOperatingPoint(
            status="limit_exceeded" if active_limits else "ok",
            diagnostics=tuple(diagnostics), active_limits=tuple(active_limits),
            dc_current_a=current, reverse_voltage_v=reverse_voltage,
            case_temperature_c=case_temperature,
            diode_conducting_current_a=current, diode_average_current_a=current / 3.0,
            diode_rms_current_a=current / math.sqrt(3.0),
            forward_voltage_per_conducting_diode_v=vf,
            bridge_forward_voltage_drop_v=2.0 * vf,
            rectifier_loss_w=rectifier_loss,
            average_loss_per_diode_w=average_loss_per_diode,
            average_loss_per_module_w=average_loss_per_module,
            junction_temperature_c=junction_temperature,
            current_limit_at_case_temperature_a=current_limit,
            case_to_heatsink_temperature_drop_k=case_to_sink_drop,
            implied_heatsink_interface_temperature_c=case_temperature - case_to_sink_drop,
        )


def load_builtin_rectifier_profile(profile_id: str) -> RectifierProfile:
    return RectifierProfile.load_builtin(profile_id)
