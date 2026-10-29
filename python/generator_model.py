"""Autonomous Phase 2 permanent-magnet generator model.

The model is intentionally isolated from the legacy mission solver.  It uses
one unambiguous back-EMF convention: line-to-line RMS volts per mechanical
rpm.  Rectification is represented only by the ideal six-pulse DC equivalent;
diode losses and control belong to a later phase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from component_profiles import ALLOWED_SOURCE_TYPES, ParameterProvenance, SourceRef


ROOT = Path(__file__).resolve().parent
BUILTIN_GENERATOR_DIR = ROOT / "data" / "generators"
IDEAL_SIX_PULSE_VOLTAGE_FACTOR = 3.0 * math.sqrt(2.0) / math.pi
IDEAL_SIX_PULSE_LINE_RMS_CURRENT_FACTOR = math.sqrt(2.0 / 3.0)
SUPPORTED_CONNECTIONS = frozenset({"star", "delta"})
SUPPORTED_LOSS_MODELS = frozenset({"constant"})


class GeneratorNotReadyError(ValueError):
    """The requested calculation needs data absent from the profile."""


class GeneratorOperatingRangeError(ValueError):
    """The requested operating point violates a hard profile limit."""


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class GeneratorReadiness:
    identity_ready: bool
    emf_ready: bool
    connection_ready: bool
    resistance_value_ready: bool
    temperature_resistance_ready: bool
    current_limits_ready: bool
    dc_equivalent_ready: bool
    loss_model_ready: bool
    inductance_ready: bool
    thermal_model_ready: bool
    phase2_ready: bool
    full_model_ready: bool
    missing_for_phase2: tuple[str, ...]
    missing_for_full_model: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["missing_for_phase2"] = list(self.missing_for_phase2)
        data["missing_for_full_model"] = list(self.missing_for_full_model)
        return data


@dataclass(frozen=True)
class GeneratorProfile:
    """Source-backed generator data; unknown quantities remain ``None``."""

    profile_id: str
    revision: int
    manufacturer: str | None
    model: str
    status: str
    source_type: str
    validation_status: str
    phase_count: int
    three_phase_system_count: int
    phase_system_shift_electrical_deg: float | None
    winding_connection: str | None
    pole_pairs: int | None
    ke_ll_rms_v_per_rpm: float | None
    phase_resistance_ohm: float | None
    line_to_line_resistance_ohm: float | None
    resistance_reference_temperature_c: float | None
    winding_material: str | None
    conductor_form: str | None
    resistance_temperature_coefficient_per_k: float | None
    phase_inductance_h: float | None
    maximum_speed_rpm: float | None
    continuous_line_current_rms_a: float | None
    peak_line_current_rms_a: float | None
    maximum_winding_temperature_c: float | None
    loss_model_kind: str | None
    constant_iron_loss_w: float | None
    constant_mechanical_loss_w: float | None
    iron_loss_map: Any | None
    mechanical_loss_map: Any | None
    thermal_parameters: dict[str, float] | None
    series_turns_per_phase: int | None
    sources: tuple[SourceRef, ...]
    provenance: dict[str, ParameterProvenance]
    warnings: tuple[str, ...] = ()
    design_requirements: dict[str, Any] | None = None
    published_observations: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or self.revision < 1 or not self.model.strip():
            raise ValueError("Профиль генератора: неверные profile_id, revision или model")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Неизвестный source_type генератора: {self.source_type}")
        if self.phase_count < 3 or self.three_phase_system_count < 1:
            raise ValueError("Генератор должен содержать хотя бы одну трёхфазную систему")
        if self.phase_count != 3 * self.three_phase_system_count:
            raise ValueError("phase_count должен равняться 3 * three_phase_system_count")
        if self.winding_connection not in SUPPORTED_CONNECTIONS | {None}:
            raise ValueError("winding_connection должен быть star, delta или null")
        if self.winding_material is not None and not self.winding_material.strip():
            raise ValueError("winding_material не может быть пустой строкой")
        if self.loss_model_kind not in SUPPORTED_LOSS_MODELS | {None}:
            raise ValueError("Неподдерживаемый loss_model_kind")
        positive_optional = (
            "pole_pairs", "ke_ll_rms_v_per_rpm", "phase_resistance_ohm",
            "line_to_line_resistance_ohm", "phase_inductance_h", "maximum_speed_rpm",
            "continuous_line_current_rms_a", "peak_line_current_rms_a",
            "maximum_winding_temperature_c", "series_turns_per_phase",
        )
        for name in positive_optional:
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} должен быть положительным конечным числом или null")
        for name in ("constant_iron_loss_w", "constant_mechanical_loss_w"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} должен быть неотрицательным конечным числом или null")
        for name in ("phase_system_shift_electrical_deg", "resistance_reference_temperature_c",
                     "resistance_temperature_coefficient_per_k"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} должен быть конечным числом или null")
        if (self.continuous_line_current_rms_a is not None
                and self.peak_line_current_rms_a is not None
                and self.peak_line_current_rms_a < self.continuous_line_current_rms_a):
            raise ValueError("Пиковый ток не может быть меньше длительного")
        if self.phase_resistance_ohm is not None and self.line_to_line_resistance_ohm is not None:
            if self.winding_connection is None:
                raise ValueError("Два вида сопротивления требуют известной схемы соединения")
            expected = GeneratorModel.line_to_line_resistance_from_phase(
                self.phase_resistance_ohm, self.winding_connection)
            if not math.isclose(expected, self.line_to_line_resistance_ohm, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Фазное и линейное сопротивления противоречат схеме соединения")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorProfile":
        topology = data.get("topology", {})
        electrical = data.get("electrical", {})
        limits = data.get("limits", {})
        losses = data.get("losses", {})
        construction = data.get("construction", {})
        sources = tuple(SourceRef(**source) for source in data.get("sources", ()))
        provenance = {
            key: ParameterProvenance(**value)
            for key, value in data.get("provenance", {}).items()
        }
        return cls(
            profile_id=data["profile_id"], revision=int(data["revision"]),
            manufacturer=data.get("manufacturer"), model=data["model"],
            status=data["status"], source_type=data["source_type"],
            validation_status=data["validation_status"],
            phase_count=int(topology["phase_count"]),
            three_phase_system_count=int(topology["three_phase_system_count"]),
            phase_system_shift_electrical_deg=_optional_float(topology.get("phase_system_shift_electrical_deg")),
            winding_connection=topology.get("winding_connection"),
            pole_pairs=None if topology.get("pole_pairs") is None else int(topology["pole_pairs"]),
            ke_ll_rms_v_per_rpm=_optional_float(electrical.get("ke_ll_rms_v_per_rpm")),
            phase_resistance_ohm=_optional_float(electrical.get("phase_resistance_ohm")),
            line_to_line_resistance_ohm=_optional_float(electrical.get("line_to_line_resistance_ohm")),
            resistance_reference_temperature_c=_optional_float(electrical.get("resistance_reference_temperature_c")),
            winding_material=electrical.get("winding_material"),
            conductor_form=electrical.get("conductor_form"),
            resistance_temperature_coefficient_per_k=_optional_float(
                electrical.get("resistance_temperature_coefficient_per_k")),
            phase_inductance_h=_optional_float(electrical.get("phase_inductance_h")),
            maximum_speed_rpm=_optional_float(limits.get("maximum_speed_rpm")),
            continuous_line_current_rms_a=_optional_float(limits.get("continuous_line_current_rms_a")),
            peak_line_current_rms_a=_optional_float(limits.get("peak_line_current_rms_a")),
            maximum_winding_temperature_c=_optional_float(limits.get("maximum_winding_temperature_c")),
            loss_model_kind=losses.get("loss_model_kind"),
            constant_iron_loss_w=_optional_float(losses.get("constant_iron_loss_w")),
            constant_mechanical_loss_w=_optional_float(losses.get("constant_mechanical_loss_w")),
            iron_loss_map=losses.get("iron_loss_map"), mechanical_loss_map=losses.get("mechanical_loss_map"),
            thermal_parameters=data.get("thermal_parameters"),
            series_turns_per_phase=(None if construction.get("series_turns_per_phase") is None
                                    else int(construction["series_turns_per_phase"])),
            sources=sources, provenance=provenance, warnings=tuple(data.get("warnings", ())),
            design_requirements=data.get("design_requirements"),
            published_observations=data.get("published_observations"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeneratorProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_builtin(cls, profile_id: str) -> "GeneratorProfile":
        path = BUILTIN_GENERATOR_DIR / f"{profile_id}.json"
        if not path.is_file():
            raise ValueError(f"Неизвестный встроенный профиль генератора: {profile_id}")
        return cls.load(path)

    @property
    def readiness(self) -> GeneratorReadiness:
        identity = bool(self.profile_id and self.model and self.phase_count)
        emf = self.ke_ll_rms_v_per_rpm is not None
        connection = self.winding_connection in SUPPORTED_CONNECTIONS
        resistance = self.phase_resistance_ohm is not None or self.line_to_line_resistance_ohm is not None
        temperature_resistance = bool(self.winding_material) and all(
            value is not None for value in (
                self.resistance_reference_temperature_c,
                self.resistance_temperature_coefficient_per_k,
            )
        )
        current_limits = all(value is not None for value in (
            self.continuous_line_current_rms_a, self.peak_line_current_rms_a,
        ))
        dc_equivalent = connection and resistance
        loss_model = (self.loss_model_kind == "constant"
                      and self.constant_iron_loss_w is not None
                      and self.constant_mechanical_loss_w is not None)
        phase2_requirements = {
            "ke_ll_rms_v_per_rpm": emf,
            "winding_connection": connection,
            "winding_resistance": resistance,
            "winding_material": self.winding_material is not None,
            "resistance_reference_temperature_c": self.resistance_reference_temperature_c is not None,
            "resistance_temperature_coefficient_per_k": self.resistance_temperature_coefficient_per_k is not None,
            "continuous_line_current_rms_a": self.continuous_line_current_rms_a is not None,
            "peak_line_current_rms_a": self.peak_line_current_rms_a is not None,
            "maximum_speed_rpm": self.maximum_speed_rpm is not None,
            "supported_loss_model": loss_model,
        }
        missing_phase2 = tuple(key for key, available in phase2_requirements.items() if not available)
        full_extra = {
            "pole_pairs": self.pole_pairs is not None,
            "phase_inductance_h": self.phase_inductance_h is not None,
            "maximum_winding_temperature_c": self.maximum_winding_temperature_c is not None,
            "thermal_parameters": self.thermal_parameters is not None,
        }
        missing_full = (*missing_phase2,
                        *(key for key, available in full_extra.items() if not available))
        return GeneratorReadiness(
            identity_ready=identity, emf_ready=emf, connection_ready=connection,
            resistance_value_ready=resistance,
            temperature_resistance_ready=temperature_resistance,
            current_limits_ready=current_limits, dc_equivalent_ready=dc_equivalent,
            loss_model_ready=loss_model, inductance_ready=self.phase_inductance_h is not None,
            thermal_model_ready=self.thermal_parameters is not None,
            phase2_ready=identity and not missing_phase2,
            full_model_ready=identity and not missing_full,
            missing_for_phase2=missing_phase2, missing_for_full_model=tuple(missing_full),
        )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "profile_id": self.profile_id, "revision": self.revision,
            "manufacturer": self.manufacturer, "model": self.model, "status": self.status,
            "source_type": self.source_type, "validation_status": self.validation_status,
            "topology": {
                "phase_count": self.phase_count,
                "three_phase_system_count": self.three_phase_system_count,
                "phase_system_shift_electrical_deg": self.phase_system_shift_electrical_deg,
                "winding_connection": self.winding_connection, "pole_pairs": self.pole_pairs,
            },
            "electrical": {
                "ke_convention": "ke_ll_rms_v_per_rpm",
                "ke_ll_rms_v_per_rpm": self.ke_ll_rms_v_per_rpm,
                "phase_resistance_ohm": self.phase_resistance_ohm,
                "line_to_line_resistance_ohm": self.line_to_line_resistance_ohm,
                "resistance_reference_temperature_c": self.resistance_reference_temperature_c,
                "winding_material": self.winding_material, "conductor_form": self.conductor_form,
                "resistance_temperature_coefficient_per_k": self.resistance_temperature_coefficient_per_k,
                "phase_inductance_h": self.phase_inductance_h,
            },
            "limits": {
                "maximum_speed_rpm": self.maximum_speed_rpm,
                "continuous_line_current_rms_a": self.continuous_line_current_rms_a,
                "peak_line_current_rms_a": self.peak_line_current_rms_a,
                "maximum_winding_temperature_c": self.maximum_winding_temperature_c,
            },
            "losses": {
                "loss_model_kind": self.loss_model_kind,
                "constant_iron_loss_w": self.constant_iron_loss_w,
                "constant_mechanical_loss_w": self.constant_mechanical_loss_w,
                "iron_loss_map": self.iron_loss_map, "mechanical_loss_map": self.mechanical_loss_map,
            },
            "thermal_parameters": self.thermal_parameters,
            "construction": {"series_turns_per_phase": self.series_turns_per_phase},
            "sources": [source.to_dict() for source in self.sources],
            "provenance": {key: value.to_dict() for key, value in self.provenance.items()},
            "warnings": list(self.warnings), "design_requirements": self.design_requirements,
            "published_observations": self.published_observations,
            "readiness": self.readiness.to_dict(),
        }


@dataclass(frozen=True)
class GeneratorOperatingPoint:
    rpm: float
    winding_temperature_c: float
    emf_line_to_line_rms_v: float
    ideal_no_load_dc_voltage_v: float
    ideal_generator_branch_dc_voltage_v: float
    total_dc_current_a: float
    dc_current_per_three_phase_system_a: float
    line_current_rms_per_three_phase_system_a: float
    phase_current_rms_a: float
    phase_resistance_at_temperature_ohm: float
    line_to_line_resistance_at_temperature_ohm: float
    dc_equivalent_resistance_per_system_ohm: float
    electromagnetic_power_w: float
    ideal_generator_branch_electrical_power_w: float
    copper_loss_w: float
    iron_loss_w: float
    mechanical_loss_w: float
    shaft_power_w: float
    mechanical_torque_nm: float
    electrical_balance_residual_w: float
    mechanical_balance_residual_w: float
    continuous_current_limit_exceeded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneratorModel:
    """Steady-state Phase 2 model with an ideal six-pulse DC equivalent."""

    def __init__(self, profile: GeneratorProfile):
        self.profile = profile

    @staticmethod
    def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
        value = float(value)
        valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
        if not valid:
            relation = "неотрицательным" if allow_zero else "положительным"
            raise ValueError(f"{name} должно быть {relation} конечным числом")
        return value

    @staticmethod
    def line_to_line_resistance_from_phase(phase_resistance_ohm: float, connection: str) -> float:
        resistance = GeneratorModel._positive(phase_resistance_ohm, "R_phase")
        if connection == "star":
            return 2.0 * resistance
        if connection == "delta":
            return 2.0 * resistance / 3.0
        raise ValueError("connection должен быть star или delta")

    @staticmethod
    def phase_resistance_from_line_to_line(line_to_line_resistance_ohm: float, connection: str) -> float:
        resistance = GeneratorModel._positive(line_to_line_resistance_ohm, "R_LL")
        if connection == "star":
            return resistance / 2.0
        if connection == "delta":
            return 1.5 * resistance
        raise ValueError("connection должен быть star или delta")

    @staticmethod
    def dc_equivalent_resistance_from_line_to_line(line_to_line_resistance_ohm: float) -> float:
        """Ideal continuous-current six-pulse bridge: R_eq,DC = R_LL."""
        return GeneratorModel._positive(line_to_line_resistance_ohm, "R_LL")

    def emf_line_to_line_rms(self, rpm: float) -> float:
        rpm = self._positive(rpm, "rpm", allow_zero=True)
        if self.profile.ke_ll_rms_v_per_rpm is None:
            raise GeneratorNotReadyError("ke_ll_rms_v_per_rpm отсутствует")
        return self.profile.ke_ll_rms_v_per_rpm * rpm

    def phase_resistance_at_temperature(self, winding_temperature_c: float) -> float:
        readiness = self.profile.readiness
        if not readiness.connection_ready or not readiness.resistance_value_ready:
            raise GeneratorNotReadyError("Неизвестны схема соединения или опорное сопротивление")
        if not readiness.temperature_resistance_ready:
            raise GeneratorNotReadyError(
                "R(T) требует явных winding_material, reference temperature и temperature coefficient")
        phase_ref = self.profile.phase_resistance_ohm
        if phase_ref is None:
            phase_ref = self.phase_resistance_from_line_to_line(
                self.profile.line_to_line_resistance_ohm, self.profile.winding_connection)  # type: ignore[arg-type]
        temperature = float(winding_temperature_c)
        if not math.isfinite(temperature):
            raise ValueError("Температура обмотки должна быть конечным числом")
        factor = 1.0 + self.profile.resistance_temperature_coefficient_per_k * (  # type: ignore[operator]
            temperature - self.profile.resistance_reference_temperature_c)  # type: ignore[operator]
        if factor <= 0:
            raise GeneratorOperatingRangeError("Линейная модель R(T) дала неположительное сопротивление")
        return phase_ref * factor

    @staticmethod
    def phase_current_from_line_rms(line_current_rms_a: float, connection: str) -> float:
        current = GeneratorModel._positive(line_current_rms_a, "I_line", allow_zero=True)
        if connection == "star":
            return current
        if connection == "delta":
            return current / math.sqrt(3.0)
        raise ValueError("connection должен быть star или delta")

    def operating_point(self, rpm: float, total_dc_current_a: float,
                        winding_temperature_c: float) -> GeneratorOperatingPoint:
        readiness = self.profile.readiness
        if not readiness.phase2_ready:
            missing = ", ".join(readiness.missing_for_phase2)
            raise GeneratorNotReadyError(f"Профиль генератора not_ready; отсутствуют: {missing}")
        rpm = self._positive(rpm, "rpm")
        total_current = self._positive(total_dc_current_a, "I_DC", allow_zero=True)
        if rpm > self.profile.maximum_speed_rpm:  # type: ignore[operator]
            raise GeneratorOperatingRangeError("Обороты превышают maximum_speed_rpm")
        if (self.profile.maximum_winding_temperature_c is not None
                and float(winding_temperature_c) > self.profile.maximum_winding_temperature_c):
            raise GeneratorOperatingRangeError("Температура превышает maximum_winding_temperature_c")
        systems = self.profile.three_phase_system_count
        current_per_system = total_current / systems
        line_current = IDEAL_SIX_PULSE_LINE_RMS_CURRENT_FACTOR * current_per_system
        if line_current > self.profile.peak_line_current_rms_a:  # type: ignore[operator]
            raise GeneratorOperatingRangeError("Ток превышает peak_line_current_rms_a")
        phase_current = self.phase_current_from_line_rms(line_current, self.profile.winding_connection)  # type: ignore[arg-type]
        r_phase = self.phase_resistance_at_temperature(winding_temperature_c)
        r_ll = self.line_to_line_resistance_from_phase(r_phase, self.profile.winding_connection)  # type: ignore[arg-type]
        r_dc = self.dc_equivalent_resistance_from_line_to_line(r_ll)
        emf_ll = self.emf_line_to_line_rms(rpm)
        emf_dc = IDEAL_SIX_PULSE_VOLTAGE_FACTOR * emf_ll
        generator_branch_voltage = emf_dc - current_per_system * r_dc
        if generator_branch_voltage < 0:
            raise GeneratorOperatingRangeError("Заданный ток превышает ток короткого замыкания модели")
        copper_loss = systems * current_per_system ** 2 * r_dc
        electromagnetic_power = emf_dc * total_current
        generator_branch_power = generator_branch_voltage * total_current
        # The constant loss fields refer to the complete machine, whereas
        # winding copper loss is assembled from its three-phase systems.
        iron_loss = float(self.profile.constant_iron_loss_w)
        mechanical_loss = float(self.profile.constant_mechanical_loss_w)
        shaft_power = electromagnetic_power + iron_loss + mechanical_loss
        omega = 2.0 * math.pi * rpm / 60.0
        electrical_residual = electromagnetic_power - generator_branch_power - copper_loss
        mechanical_residual = (shaft_power - generator_branch_power - copper_loss
                               - iron_loss - mechanical_loss)
        return GeneratorOperatingPoint(
            rpm=rpm, winding_temperature_c=float(winding_temperature_c),
            emf_line_to_line_rms_v=emf_ll, ideal_no_load_dc_voltage_v=emf_dc,
            ideal_generator_branch_dc_voltage_v=generator_branch_voltage,
            total_dc_current_a=total_current,
            dc_current_per_three_phase_system_a=current_per_system,
            line_current_rms_per_three_phase_system_a=line_current,
            phase_current_rms_a=phase_current,
            phase_resistance_at_temperature_ohm=r_phase,
            line_to_line_resistance_at_temperature_ohm=r_ll,
            dc_equivalent_resistance_per_system_ohm=r_dc,
            electromagnetic_power_w=electromagnetic_power,
            ideal_generator_branch_electrical_power_w=generator_branch_power,
            copper_loss_w=copper_loss,
            iron_loss_w=iron_loss, mechanical_loss_w=mechanical_loss,
            shaft_power_w=shaft_power, mechanical_torque_nm=shaft_power / omega,
            electrical_balance_residual_w=electrical_residual,
            mechanical_balance_residual_w=mechanical_residual,
            continuous_current_limit_exceeded=(
                line_current > self.profile.continuous_line_current_rms_a),  # type: ignore[operator]
        )


def load_builtin_generator_profile(profile_id: str) -> GeneratorProfile:
    return GeneratorProfile.load_builtin(profile_id)
