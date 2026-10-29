"""Autonomous Phase 5 thermal sensitivity models for generator and rectifier.

Electrical losses are inputs.  This module never recalculates I²R, diode loss,
iron loss, or mechanical loss and does not update mission, battery, or engine state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable

from rectifier_model import RectifierModel


ROOT = Path(__file__).resolve().parent
BUILTIN_THERMAL_DIR = ROOT / "data" / "thermal"


class ThermalStatus(str, Enum):
    NORMAL = "normal"
    APPROACHING_LIMIT = "approaching_limit"
    THERMAL_LIMIT = "thermal_limit"
    NOT_EVALUABLE = "not_evaluable"


@dataclass(frozen=True)
class GeneratorThermalReadiness:
    single_node_ready: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"single_node_ready": self.single_node_ready,
                "missing": list(self.missing)}


@dataclass(frozen=True)
class GeneratorThermalProfile:
    profile_id: str
    source_type: str
    validation_status: str
    model_kind: str
    rth_k_per_w: float | None
    cth_j_per_k: float | None
    ambient_coupling: str
    cooling_mode: str
    valid_loss_range_w: tuple[float, float] | None
    valid_speed_range_rpm: tuple[float, float] | None
    valid_ambient_range_c: tuple[float, float] | None
    winding_temperature_limit_c: float | None
    approaching_limit_margin_k: float
    approaching_limit_source_type: str
    sources: tuple[dict[str, Any], ...]
    provenance: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.model_kind != "single_node":
            raise ValueError("Phase 5 поддерживает только generator model_kind=single_node")
        for name in ("rth_k_per_w", "cth_j_per_k"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} должно быть положительным")
        if self.approaching_limit_margin_k < 0:
            raise ValueError("approaching_limit_margin_k не может быть отрицательным")
        if self.approaching_limit_source_type != "engineering_assumption":
            raise ValueError("Порог approaching_limit должен быть engineering_assumption")

    @property
    def readiness(self) -> GeneratorThermalReadiness:
        required = {
            "rth_k_per_w": self.rth_k_per_w,
            "cth_j_per_k": self.cth_j_per_k,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        return GeneratorThermalReadiness(not missing, missing)

    @property
    def time_constant_s(self) -> float | None:
        if not self.readiness.single_node_ready:
            return None
        return self.rth_k_per_w * self.cth_j_per_k  # type: ignore[operator]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorThermalProfile":
        def pair(value: Any) -> tuple[float, float] | None:
            return None if value is None else (float(value[0]), float(value[1]))
        return cls(
            profile_id=data["profile_id"], source_type=data["source_type"],
            validation_status=data["validation_status"], model_kind=data["model_kind"],
            rth_k_per_w=_optional_float(data.get("rth_k_per_w")),
            cth_j_per_k=_optional_float(data.get("cth_j_per_k")),
            ambient_coupling=data["ambient_coupling"], cooling_mode=data["cooling_mode"],
            valid_loss_range_w=pair(data.get("valid_loss_range_w")),
            valid_speed_range_rpm=pair(data.get("valid_speed_range_rpm")),
            valid_ambient_range_c=pair(data.get("valid_ambient_range_c")),
            winding_temperature_limit_c=_optional_float(
                data.get("winding_temperature_limit_c")),
            approaching_limit_margin_k=float(data["approaching_limit_margin_k"]),
            approaching_limit_source_type=data["approaching_limit_source_type"],
            sources=tuple(data.get("sources", ())),
            provenance=dict(data.get("provenance", {})),
            warnings=tuple(data.get("warnings", ())),
        )

    @classmethod
    def load_builtin(cls, profile_id: str) -> "GeneratorThermalProfile":
        path = BUILTIN_THERMAL_DIR / f"{profile_id}.json"
        if not path.is_file():
            raise ValueError(f"Неизвестный generator thermal profile: {profile_id}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class HeatSinkProfile:
    profile_id: str
    rth_sink_to_ambient_k_per_w: float | None
    thermal_capacitance_j_per_k: float | None
    mass_kg: float | None
    cooling_mode: str
    air_velocity_m_s: float | None
    source_type: str
    validation_status: str
    approaching_limit_margin_k: float
    provenance: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("rth_sink_to_ambient_k_per_w", "thermal_capacitance_j_per_k",
                     "mass_kg", "air_velocity_m_s"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} должно быть положительным или null")
        if self.approaching_limit_margin_k < 0:
            raise ValueError("approaching_limit_margin_k не может быть отрицательным")

    @property
    def ready(self) -> bool:
        return (self.rth_sink_to_ambient_k_per_w is not None
                and self.thermal_capacitance_j_per_k is not None)

    @property
    def time_constant_s(self) -> float | None:
        if not self.ready:
            return None
        return (self.rth_sink_to_ambient_k_per_w
                * self.thermal_capacitance_j_per_k)  # type: ignore[operator]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeatSinkProfile":
        return cls(
            profile_id=data["profile_id"],
            rth_sink_to_ambient_k_per_w=_optional_float(
                data.get("rth_sink_to_ambient_k_per_w")),
            thermal_capacitance_j_per_k=_optional_float(
                data.get("thermal_capacitance_j_per_k")),
            mass_kg=_optional_float(data.get("mass_kg")),
            cooling_mode=data["cooling_mode"],
            air_velocity_m_s=_optional_float(data.get("air_velocity_m_s")),
            source_type=data["source_type"], validation_status=data["validation_status"],
            approaching_limit_margin_k=float(data["approaching_limit_margin_k"]),
            provenance=dict(data.get("provenance", {})),
            warnings=tuple(data.get("warnings", ())),
        )

    @classmethod
    def load_builtin(cls, profile_id: str) -> "HeatSinkProfile":
        path = BUILTIN_THERMAL_DIR / f"{profile_id}.json"
        if not path.is_file():
            raise ValueError(f"Неизвестный heatsink profile: {profile_id}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ThermalSimulationInput:
    time_s: float
    loss_power_w: float
    rpm: float
    ambient_temperature_c: float
    copper_loss_w: float | None = None
    iron_loss_w: float | None = None
    magnet_loss_w: float | None = None
    mechanical_loss_w: float | None = None
    loss_source_type: str = "unknown"


@dataclass(frozen=True)
class ThermalState:
    time_s: float
    node_temperatures_c: dict[str, float]


@dataclass(frozen=True)
class ThermalStepResult:
    status: ThermalStatus
    component: str
    previous_state: ThermalState
    state: ThermalState
    heat_input_w: float
    time_constant_s: float | None
    steady_state_temperature_c: float | None
    thermal_margin_c: float | None
    provenance_status: dict[str, str]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class ThermalSensitivityResult:
    profile_id: str
    temperature_after_time_c: float
    steady_state_temperature_c: float
    time_constant_s: float
    time_to_thermal_limit_s: float | None
    thermal_margin_c: float | None
    provenance_status: str


class GeneratorThermalModel:
    """One-node exact first-order winding model using externally supplied loss."""

    NODE = "generator_winding"

    def __init__(self, profile: GeneratorThermalProfile):
        self.profile = profile

    @staticmethod
    def initial_state(temperature_c: float, time_s: float = 0.0) -> ThermalState:
        return ThermalState(float(time_s), {GeneratorThermalModel.NODE: float(temperature_c)})

    def step(self, state: ThermalState,
             inputs: ThermalSimulationInput) -> ThermalStepResult:
        _validate_step(state, inputs.time_s, inputs.loss_power_w,
                       inputs.ambient_temperature_c)
        if not self.profile.readiness.single_node_ready:
            return _not_evaluable("generator", state, inputs.loss_power_w,
                                  self.profile.warnings + self.profile.readiness.missing)
        if self.NODE not in state.node_temperatures_c:
            raise ValueError("ThermalState не содержит generator_winding")
        dt = inputs.time_s - state.time_s
        rth = self.profile.rth_k_per_w  # type: ignore[assignment]
        cth = self.profile.cth_j_per_k  # type: ignore[assignment]
        tau = rth * cth
        steady = inputs.ambient_temperature_c + inputs.loss_power_w * rth
        initial = state.node_temperatures_c[self.NODE]
        final = steady + (initial - steady) * math.exp(-dt / tau)
        margin = _margin(self.profile.winding_temperature_limit_c, final)
        status = _thermal_status(margin, self.profile.approaching_limit_margin_k)
        warnings = list(self.profile.warnings)
        if inputs.copper_loss_w is not None and not math.isclose(
                inputs.loss_power_w, inputs.copper_loss_w, rel_tol=1e-9, abs_tol=1e-9):
            warnings.append("single_node_heat_input_differs_from_copper_loss")
        for name in ("iron_loss_w", "magnet_loss_w", "mechanical_loss_w"):
            if getattr(inputs, name) is not None:
                warnings.append(f"{name}_not_automatically_applied_to_winding_node")
        warnings.extend(_range_warnings(self.profile, inputs))
        new_state = ThermalState(inputs.time_s, {self.NODE: final})
        return ThermalStepResult(
            status=status, component="generator", previous_state=state, state=new_state,
            heat_input_w=inputs.loss_power_w, time_constant_s=tau,
            steady_state_temperature_c=steady, thermal_margin_c=margin,
            provenance_status={
                "thermal_parameters": self.profile.source_type,
                "loss_input": inputs.loss_source_type,
                "integration": "validated_equation",
                "approaching_limit_threshold": self.profile.approaching_limit_source_type,
            }, warnings=tuple(dict.fromkeys(warnings)))

    def run(self, initial_state: ThermalState,
            sequence: Iterable[ThermalSimulationInput]) -> tuple[ThermalStepResult, ...]:
        results: list[ThermalStepResult] = []
        state = initial_state
        for inputs in sequence:
            result = self.step(state, inputs)
            results.append(result)
            state = result.state
        return tuple(results)


class RectifierThermalModel:
    """Dynamic sink/case sensitivity model fed by Phase 3 rectifier losses."""

    MODES = frozenset({"steady_state", "transient_zth"})

    def __init__(self, rectifier: RectifierModel, heatsink: HeatSinkProfile,
                 thermal_mode: str = "steady_state"):
        if thermal_mode not in self.MODES:
            raise ValueError("Неизвестный thermal_mode выпрямителя")
        self.rectifier = rectifier
        self.heatsink = heatsink
        self.thermal_mode = thermal_mode

    @staticmethod
    def initial_state(temperature_c: float, time_s: float = 0.0) -> ThermalState:
        value = float(temperature_c)
        return ThermalState(float(time_s), {
            "rectifier_heatsink": value,
            "rectifier_case": value,
            **{f"rectifier_junction_{index}": value for index in range(6)},
        })

    def fixed_case_junction_temperature(self, rectifier_loss_w: float,
                                        case_temperature_c: float) -> float:
        profile = self.rectifier.profile
        if profile.diode_count is None or profile.rth_junction_to_case_single_leg_k_per_w is None:
            raise ValueError("Статические thermal parameters выпрямителя не готовы")
        per_diode = float(rectifier_loss_w) / profile.diode_count
        return case_temperature_c + per_diode * profile.rth_junction_to_case_single_leg_k_per_w

    def step(self, state: ThermalState, inputs: ThermalSimulationInput) -> ThermalStepResult:
        _validate_step(state, inputs.time_s, inputs.loss_power_w,
                       inputs.ambient_temperature_c)
        profile = self.rectifier.profile
        if self.thermal_mode == "transient_zth":
            transient = profile.transient_thermal_impedance or {}
            return _not_evaluable(
                "rectifier", state, inputs.loss_power_w,
                (str(transient.get("status", "transient_not_ready")),
                 "manufacturer_zth_pulse_family_not_digitized"))
        if not self.heatsink.ready or not profile.readiness.static_thermal_ready:
            return _not_evaluable(
                "rectifier", state, inputs.loss_power_w,
                (*self.heatsink.warnings, *profile.readiness.missing_for_phase3))
        if "rectifier_heatsink" not in state.node_temperatures_c:
            raise ValueError("ThermalState не содержит rectifier_heatsink")
        dt = inputs.time_s - state.time_s
        r_sa = self.heatsink.rth_sink_to_ambient_k_per_w  # type: ignore[assignment]
        c_sink = self.heatsink.thermal_capacitance_j_per_k  # type: ignore[assignment]
        tau = r_sa * c_sink
        sink_steady = inputs.ambient_temperature_c + inputs.loss_power_w * r_sa
        sink_initial = state.node_temperatures_c["rectifier_heatsink"]
        sink = sink_steady + (sink_initial - sink_steady) * math.exp(-dt / tau)
        module_count = profile.module_count  # type: ignore[assignment]
        diode_count = profile.diode_count  # type: ignore[assignment]
        per_module = inputs.loss_power_w / module_count
        per_diode = inputs.loss_power_w / diode_count
        case = sink + per_module * profile.rth_case_to_heatsink_k_per_w  # type: ignore[operator]
        junction = case + per_diode * profile.rth_junction_to_case_single_leg_k_per_w  # type: ignore[operator]
        steady_case = sink_steady + per_module * profile.rth_case_to_heatsink_k_per_w  # type: ignore[operator]
        steady_junction = steady_case + per_diode * profile.rth_junction_to_case_single_leg_k_per_w  # type: ignore[operator]
        margin = _margin(profile.junction_temperature_max_c, junction)
        status = _thermal_status(margin, self.heatsink.approaching_limit_margin_k)
        nodes = {"rectifier_heatsink": sink, "rectifier_case": case}
        nodes.update({f"rectifier_junction_{index}": junction
                      for index in range(diode_count)})
        new_state = ThermalState(inputs.time_s, nodes)
        return ThermalStepResult(
            status=status, component="rectifier", previous_state=state, state=new_state,
            heat_input_w=inputs.loss_power_w, time_constant_s=tau,
            steady_state_temperature_c=steady_junction, thermal_margin_c=margin,
            provenance_status={
                "junction_to_case": "manufacturer_parameter",
                "case_to_heatsink": "manufacturer_parameter",
                "sink_to_ambient": self.heatsink.source_type,
                "loss_input": inputs.loss_source_type,
                "integration": "validated_equation",
                "approaching_limit_threshold": "engineering_assumption",
            }, warnings=tuple(dict.fromkeys((
                *self.heatsink.warnings,
                "junction_uses_steady_rthjc_not_transient_zth",
                "case_temperature_is_not_assumed_equal_to_ambient",
            ))))


def generator_sensitivity(
        profiles: Iterable[GeneratorThermalProfile], initial_temperature_c: float,
        inputs: ThermalSimulationInput) -> tuple[ThermalSensitivityResult, ...]:
    results: list[ThermalSensitivityResult] = []
    for profile in profiles:
        model = GeneratorThermalModel(profile)
        step = model.step(model.initial_state(initial_temperature_c), inputs)
        if step.status == ThermalStatus.NOT_EVALUABLE:
            continue
        final = step.state.node_temperatures_c[model.NODE]
        time_to_limit = _time_to_limit(
            initial_temperature_c, step.steady_state_temperature_c,
            profile.winding_temperature_limit_c, step.time_constant_s)
        results.append(ThermalSensitivityResult(
            profile_id=profile.profile_id, temperature_after_time_c=final,
            steady_state_temperature_c=step.steady_state_temperature_c,  # type: ignore[arg-type]
            time_constant_s=step.time_constant_s,  # type: ignore[arg-type]
            time_to_thermal_limit_s=time_to_limit,
            thermal_margin_c=step.thermal_margin_c,
            provenance_status=profile.source_type,
        ))
    return tuple(results)


def _validate_step(state: ThermalState, time_s: float, loss_w: float,
                   ambient_c: float) -> None:
    values = (state.time_s, time_s, loss_w, ambient_c,
              *state.node_temperatures_c.values())
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Thermal inputs должны быть конечными")
    if time_s <= state.time_s:
        raise ValueError("time_s должен возрастать")
    if loss_w < 0:
        raise ValueError("Тепловая мощность не может быть отрицательной")


def _not_evaluable(component: str, state: ThermalState, heat_w: float,
                    warnings: tuple[str, ...]) -> ThermalStepResult:
    return ThermalStepResult(
        status=ThermalStatus.NOT_EVALUABLE, component=component,
        previous_state=state, state=state, heat_input_w=heat_w,
        time_constant_s=None, steady_state_temperature_c=None,
        thermal_margin_c=None, provenance_status={}, warnings=warnings)


def _margin(limit_c: float | None, temperature_c: float) -> float | None:
    return None if limit_c is None else limit_c - temperature_c


def _thermal_status(margin_c: float | None,
                    approaching_margin_k: float) -> ThermalStatus:
    if margin_c is None:
        return ThermalStatus.NORMAL
    if margin_c <= 0:
        return ThermalStatus.THERMAL_LIMIT
    if margin_c <= approaching_margin_k:
        return ThermalStatus.APPROACHING_LIMIT
    return ThermalStatus.NORMAL


def _time_to_limit(initial_c: float, steady_c: float | None,
                   limit_c: float | None, tau_s: float | None) -> float | None:
    if steady_c is None or limit_c is None or tau_s is None:
        return None
    if initial_c >= limit_c:
        return 0.0
    if steady_c <= limit_c:
        return None
    ratio = (limit_c - steady_c) / (initial_c - steady_c)
    return -tau_s * math.log(ratio)


def _range_warnings(profile: GeneratorThermalProfile,
                    inputs: ThermalSimulationInput) -> list[str]:
    warnings: list[str] = []
    for value, bounds, label in (
        (inputs.loss_power_w, profile.valid_loss_range_w, "loss"),
        (inputs.rpm, profile.valid_speed_range_rpm, "speed"),
        (inputs.ambient_temperature_c, profile.valid_ambient_range_c, "ambient"),
    ):
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            warnings.append(f"{label}_outside_sensitivity_profile_range")
    return warnings


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)

