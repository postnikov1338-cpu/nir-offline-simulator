"""Quasi-static Phase 4B powertrain operating-point solver.

The solver searches externally prescribed engine/generator speeds and delegates
each electrical candidate to the independent Phase 4A DCBusSolver.  It does
not integrate time, update battery or thermal state, or optimise fuel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

from ecms_control import FixedFactorECMS, ProportionalSOCAdaptiveECMS
from dc_bus_solver import (
    DCBusInputs,
    DCBusOperatingPoint,
    DCBusSolver,
    DCBusSolverStatus,
    load_synthetic_dc_bus_fixture,
)
from engine_model import EngineEnvelopePoint, EngineMapOutOfRangeError, EngineModel, load_builtin_engine_profile
from engine_part_load import EnginePartLoadModel, load_builtin_part_load_profile


ROOT = Path(__file__).resolve().parent
BUILTIN_POWERTRAIN_DIR = ROOT / "data" / "powertrain"
SYNTHETIC_POWERTRAIN_FIXTURE_ID = "synthetic_powertrain_fixture_v1"
OPERATING_POLICIES = frozenset({
    "lowest_feasible_rpm",
    "closest_to_target_rpm",
    "minimum_engine_fuel_rate",
    "minimum_equivalent_fuel_rate",
    "soc_adaptive_minimum_equivalent_fuel_rate",
})
MINIMUM_FUEL_TIE_EPSILON_KG_H = 1e-9


class PowertrainSolverStatus(str, Enum):
    OK = "ok"
    NO_FEASIBLE_OPERATING_POINT = "no_feasible_powertrain_operating_point"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class PowertrainSolverConfig:
    source_type: str
    validation_status: str
    gear_ratio: float
    operating_policy: str
    target_rpm: float | None
    operating_min_rpm: float
    operating_max_rpm: float
    rpm_grid_step: float
    include_confirmed_rated_point: bool
    refine_infeasible_to_feasible_boundary: bool
    rpm_refinement_tolerance: float
    full_load_exact_relative_torque_tolerance: float
    near_full_load_relative_torque_tolerance: float
    fuel_tolerance_source_type: str
    part_load_policy: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_type not in {"synthetic_test_fixture", "engineering_assumption"}:
            raise ValueError("Неизвестный source_type Phase 4B")
        if self.validation_status not in {
            "synthetic_physics_validation", "engineering_simulation",
        }:
            raise ValueError("Неизвестный validation_status Phase 4B")
        if not math.isclose(self.gear_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Phase 4B поддерживает только жёсткую связь gear_ratio=1")
        if self.operating_policy not in OPERATING_POLICIES:
            raise ValueError("Неизвестная operating policy")
        if self.part_load_policy not in {
                "report_missing_without_estimate", "use_attached_engine_part_load_profile"}:
            raise ValueError("Неизвестная part-load fuel policy")
        values = (
            self.operating_min_rpm, self.operating_max_rpm, self.rpm_grid_step,
            self.rpm_refinement_tolerance,
            self.full_load_exact_relative_torque_tolerance,
            self.near_full_load_relative_torque_tolerance,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Параметры поиска rpm должны быть положительными")
        if self.operating_max_rpm <= self.operating_min_rpm:
            raise ValueError("operating_max_rpm должен превышать operating_min_rpm")
        if not 0 <= self.full_load_exact_relative_torque_tolerance < 1:
            raise ValueError("Численный допуск full-load должен находиться в [0, 1)")
        if not (self.full_load_exact_relative_torque_tolerance
                < self.near_full_load_relative_torque_tolerance < 1):
            raise ValueError("Допуск near-full-load должен быть шире численного full-load")
        if self.fuel_tolerance_source_type != "engineering_assumption":
            raise ValueError("Допуски топливной классификации являются engineering_assumption")
        if self.target_rpm is not None and (
                not math.isfinite(self.target_rpm) or self.target_rpm <= 0):
            raise ValueError("target_rpm должно быть положительным")


@dataclass(frozen=True)
class PowertrainOperatingPointInputs:
    load_power_w: float
    battery_soc: float
    battery_temperature_c: float
    generator_winding_temperature_c: float
    rectifier_case_temperature_c: float
    previous_bus_voltage_v: float | None = None
    target_rpm: float | None = None


@dataclass(frozen=True)
class PowertrainCandidate:
    engine_rpm: float
    generator_rpm: float
    dc_bus_status: str
    electrical_feasible: bool
    mechanical_feasible: bool
    physically_feasible: bool
    engine_max_torque_nm: float
    engine_required_torque_nm: float | None
    engine_torque_margin_nm: float | None
    engine_max_power_kw: float
    engine_required_power_kw: float | None
    fuel_flow_kg_h: float | None
    fuel_model_status: str
    engine_load_fraction_lambda: float | None
    engine_normalized_speed_nu: float | None
    engine_full_load_bsfc_g_kwh: float | None
    engine_part_load_factor_k: float | None
    engine_effective_bsfc_g_kwh: float | None
    fuel_model_source_type: str
    fuel_model_confidence: str
    generator_emf_ll_rms_v: float | None
    generator_dc_current_a: float | None
    generator_copper_loss_w: float | None
    generator_other_losses_w: float | None
    generator_required_shaft_power_w: float | None
    generator_required_torque_nm: float | None
    rectifier_drop_v: float | None
    rectifier_loss_w: float | None
    rectifier_junction_temperature_c: float | None
    bus_voltage_v: float | None
    battery_current_a: float | None
    battery_power_w: float | None
    load_current_a: float | None
    load_power_w: float
    electrical_balance_error_w: float | None
    mechanical_balance_error_w: float | None
    active_limits: tuple[str, ...]
    margins: dict[str, float | None]
    provenance_status: dict[str, str]
    warnings: tuple[str, ...]
    dc_bus_point: DCBusOperatingPoint

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["active_limits"] = list(self.active_limits)
        data["warnings"] = list(self.warnings)
        data["dc_bus_point"] = self.dc_bus_point.to_dict()
        return data

    @property
    def thermally_feasible(self) -> bool | None:
        """Static thermal-limit result; dynamic thermal feasibility is unknown."""
        thermal_limits = {
            "rectifier_junction_temperature_limit",
            "generator_temperature_limit",
            "rectifier_temperature_limit",
            "battery_temperature_limit",
        }
        if thermal_limits.intersection(self.active_limits):
            return False
        return None

    @property
    def overall_feasible(self) -> bool:
        return self.physically_feasible and self.thermally_feasible is not False

    @property
    def status(self) -> str:
        if self.overall_feasible:
            return "feasible"
        if self.active_limits:
            return ";".join(self.active_limits)
        return self.dc_bus_status

    @property
    def battery_mode(self) -> str:
        if self.battery_current_a is None:
            return "unknown"
        if self.battery_current_a > 1e-9:
            return "discharge"
        if self.battery_current_a < -1e-9:
            return "charge"
        return "neutral"


@dataclass(frozen=True)
class PowertrainOperatingPoint:
    operating_point_status: PowertrainSolverStatus
    validation_status: str
    operating_policy: str
    gear_ratio: float
    engine_rpm: float | None
    generator_rpm: float | None
    engine_max_torque_nm: float | None
    engine_required_torque_nm: float | None
    engine_torque_margin_nm: float | None
    engine_max_power_kw: float | None
    engine_required_power_kw: float | None
    fuel_flow_kg_h: float | None
    fuel_model_status: str
    engine_load_fraction_lambda: float | None
    engine_normalized_speed_nu: float | None
    engine_full_load_bsfc_g_kwh: float | None
    engine_part_load_factor_k: float | None
    engine_effective_bsfc_g_kwh: float | None
    fuel_model_source_type: str
    fuel_model_confidence: str
    generator_emf_ll_rms_v: float | None
    generator_dc_current_a: float | None
    generator_copper_loss_w: float | None
    generator_other_losses_w: float | None
    generator_required_shaft_power_w: float | None
    generator_required_torque_nm: float | None
    rectifier_drop_v: float | None
    rectifier_loss_w: float | None
    rectifier_junction_temperature_c: float | None
    bus_voltage_v: float | None
    battery_current_a: float | None
    battery_power_w: float | None
    load_current_a: float | None
    load_power_w: float
    electrical_balance_error_w: float | None
    mechanical_balance_error_w: float | None
    active_limits: tuple[str, ...]
    margins: dict[str, float | None]
    provenance_status: dict[str, str]
    warnings: tuple[str, ...]
    candidates: tuple[PowertrainCandidate, ...]
    selected_candidate: int | None
    ecms_selection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operating_point_status"] = self.operating_point_status.value
        data["active_limits"] = list(self.active_limits)
        data["warnings"] = list(self.warnings)
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        if self.ecms_selection is None:
            data.pop("ecms_selection")
        return data


@dataclass(frozen=True)
class SyntheticPowertrainFixture:
    fixture_id: str
    engine_model: EngineModel
    dc_bus_solver: DCBusSolver
    solver_config: PowertrainSolverConfig


class PowertrainOperatingPointSolver:
    """Enumerate immutable candidates, then apply one selection policy."""

    def __init__(self, engine_model: EngineModel, dc_bus_solver: DCBusSolver,
                 config: PowertrainSolverConfig,
                 ecms_controller: (FixedFactorECMS |
                                   ProportionalSOCAdaptiveECMS | None) = None):
        self.engine = engine_model
        self.dc_bus = dc_bus_solver
        self.config = config
        self.ecms = ecms_controller

    def readiness_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.engine.profile.curve_kind != "full_load":
            errors.append("engine_curve_is_not_full_load")
        errors.extend(self.dc_bus.readiness_errors())
        if self.config.validation_status == "synthetic_physics_validation":
            if self.dc_bus.generator.profile.source_type != "synthetic_test_fixture":
                errors.append("generator_not_synthetic_test_fixture")
            if self.dc_bus.rectifier.profile.source_type != "synthetic_test_fixture":
                errors.append("rectifier_not_synthetic_test_fixture")
            if self.dc_bus.battery.config.get("source_type") != "synthetic_test_fixture":
                errors.append("battery_not_synthetic_test_fixture")
        return tuple(dict.fromkeys(errors))

    def powertrain_profile_readiness(self) -> str:
        """Expose analogue/synthetic status without upgrading component evidence."""
        return ";".join((
            f"validation={self.config.validation_status}",
            f"engine={self.engine.profile.source_type}",
            f"generator={self.dc_bus.generator.profile.source_type}",
            f"rectifier={self.dc_bus.rectifier.profile.source_type}",
            f"battery={self.dc_bus.battery.config.get('source_type', 'unknown')}",
        ))

    @staticmethod
    def _validate_inputs(inputs: PowertrainOperatingPointInputs) -> None:
        values = (
            inputs.load_power_w, inputs.battery_soc,
            inputs.battery_temperature_c, inputs.generator_winding_temperature_c,
            inputs.rectifier_case_temperature_c,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Входы Phase 4B должны быть конечными")
        if inputs.load_power_w < 0 or not 0 <= inputs.battery_soc <= 1:
            raise ValueError("Неверные мощность нагрузки или SOC")
        for value, name in (
            (inputs.previous_bus_voltage_v, "previous_bus_voltage_v"),
            (inputs.target_rpm, "target_rpm"),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} должно быть положительным")

    def candidate_rpm_grid(self) -> tuple[float, ...]:
        generator_max_engine_rpm = (
            self.dc_bus.generator.profile.maximum_speed_rpm / self.config.gear_ratio)
        continuous_min = max(
            self.config.operating_min_rpm,
            self.engine.profile.digitized_map_min_rpm,
        )
        continuous_max = min(
            self.config.operating_max_rpm,
            self.engine.profile.digitized_map_max_rpm,
            generator_max_engine_rpm,
        )
        values: list[float] = []
        if continuous_min <= continuous_max:
            rpm = continuous_min
            while rpm <= continuous_max + 1e-9:
                values.append(min(rpm, continuous_max))
                rpm += self.config.rpm_grid_step
            if not values or not math.isclose(values[-1], continuous_max,
                                               rel_tol=0.0, abs_tol=1e-9):
                values.append(continuous_max)
        rated = self.engine.profile.rated_power_rpm
        if (self.config.include_confirmed_rated_point
                and self.config.operating_min_rpm <= rated <= self.config.operating_max_rpm
                and rated <= generator_max_engine_rpm):
            values.append(rated)
        return tuple(sorted(set(values)))

    def _fuel_result(self, envelope: EngineEnvelopePoint,
                     required_torque_nm: float | None,
                     required_power_kw: float | None,
                     mechanically_feasible: bool) -> tuple[float | None, str]:
        if required_torque_nm is None or required_power_kw is None:
            return None, "electrical_operating_point_missing"
        if not mechanically_feasible:
            return None, "engine_torque_limit"
        relative_gap = ((envelope.maximum_torque_nm - required_torque_nm)
                        / envelope.maximum_torque_nm)
        if relative_gap <= self.config.full_load_exact_relative_torque_tolerance:
            if envelope.full_load_point is None:
                return None, "full_load_sfc_missing"
            fuel_flow = (envelope.full_load_point.bsfc_g_kwh
                         * required_power_kw / 1000.0)
            return fuel_flow, "full_load_supported"
        if relative_gap <= self.config.near_full_load_relative_torque_tolerance:
            if envelope.full_load_point is None:
                return None, "full_load_sfc_missing"
            fuel_flow = (envelope.full_load_point.bsfc_g_kwh
                         * required_power_kw / 1000.0)
            return fuel_flow, "near_full_load_estimate"
        return None, "part_load_map_missing"

    def evaluate_candidate(self, engine_rpm: float,
                           inputs: PowertrainOperatingPointInputs) -> PowertrainCandidate:
        self._validate_inputs(inputs)
        envelope = self.engine.maximum_available_at_rpm(engine_rpm)
        generator_rpm = engine_rpm * self.config.gear_ratio
        dc_point = self.dc_bus.solve(DCBusInputs(
            engine_or_generator_speed_rpm=generator_rpm,
            load_power_w=inputs.load_power_w,
            battery_soc=inputs.battery_soc,
            battery_temperature_c=inputs.battery_temperature_c,
            generator_winding_temperature_c=inputs.generator_winding_temperature_c,
            rectifier_case_temperature_c=inputs.rectifier_case_temperature_c,
            previous_bus_voltage_v=inputs.previous_bus_voltage_v,
        ))
        electrical_feasible = dc_point.solver_status == DCBusSolverStatus.OK
        required_power_w = dc_point.generator_shaft_power_w
        required_power_kw = (None if required_power_w is None else required_power_w / 1000.0)
        required_torque = dc_point.generator_mechanical_torque_nm
        torque_margin = (None if required_torque is None
                         else envelope.maximum_torque_nm - required_torque)
        mechanical_feasible = bool(required_torque is not None and torque_margin >= -1e-9)
        active = list(dc_point.active_limits)
        if dc_point.solver_status == DCBusSolverStatus.NO_ROOT:
            active.append("no_dc_bus_root")
        elif dc_point.solver_status == DCBusSolverStatus.NOT_READY:
            active.append("dc_bus_not_ready")
        elif dc_point.solver_status == DCBusSolverStatus.UNSTABLE_ONLY:
            active.append("dc_bus_unstable_only")
        if not mechanical_feasible and required_torque is not None:
            active.append("engine_torque_limit")
        if (dc_point.solver_status in {DCBusSolverStatus.NO_ROOT,
                                       DCBusSolverStatus.NOT_READY}
                and any("rectifier_characteristic_out_of_range" in warning
                        for warning in dc_point.warnings)):
            active.append("rectifier_characteristic_out_of_range")
        physically_feasible = electrical_feasible and mechanical_feasible
        part_load_point = None
        if (required_torque is not None and mechanical_feasible
                and self.engine.supports_part_load):
            part_load_point = self.engine.part_load_at(engine_rpm, required_torque)
        if part_load_point is None:
            fuel_flow, fuel_status = self._fuel_result(
                envelope, required_torque, required_power_kw, mechanical_feasible)
            load_fraction = (None if required_torque is None
                             else required_torque / envelope.maximum_torque_nm)
            normalized_speed = None
            full_load_bsfc = (None if envelope.full_load_point is None
                              else envelope.full_load_point.bsfc_g_kwh)
            part_load_factor = None
            effective_bsfc = (full_load_bsfc if fuel_flow is not None else None)
            fuel_source_type = (
                "digitized_manufacturer_graph" if fuel_status == "full_load_supported"
                else "engineering_assumption" if fuel_status == "near_full_load_estimate"
                else "unknown")
            fuel_confidence = (envelope.confidence if fuel_flow is not None else "low")
            fuel_warnings: tuple[str, ...] = ()
        else:
            fuel_flow = part_load_point.fuel_flow_kg_h
            fuel_status = part_load_point.fuel_model_status
            load_fraction = part_load_point.load_fraction_lambda
            normalized_speed = part_load_point.normalized_speed_nu
            full_load_bsfc = part_load_point.full_load_bsfc_g_kwh
            part_load_factor = part_load_point.part_load_factor_k
            effective_bsfc = part_load_point.effective_bsfc_g_kwh
            fuel_source_type = part_load_point.fuel_model_source_type
            fuel_confidence = part_load_point.fuel_model_confidence
            fuel_warnings = part_load_point.warnings
        generator_other_losses = None
        if (dc_point.generator_shaft_power_w is not None
                and dc_point.generator_electromagnetic_power_w is not None):
            generator_other_losses = (dc_point.generator_shaft_power_w
                                      - dc_point.generator_electromagnetic_power_w)
        mechanical_balance = None
        if required_torque is not None and required_power_w is not None:
            omega = 2.0 * math.pi * engine_rpm / 60.0
            mechanical_balance = required_torque * omega - required_power_w
        margins = dict(dc_point.margins)
        margins["engine_torque_nm"] = torque_margin
        margins["engine_power_kw"] = (None if required_power_kw is None
                                      else envelope.maximum_power_kw - required_power_kw)
        provenance = {
            "electrical_equations": "validated_equation",
            "engine_envelope": envelope.source_type,
            "engine_fuel": fuel_source_type,
            "fuel_classification_tolerances": self.config.fuel_tolerance_source_type,
            "generator_parameters": self.dc_bus.generator.profile.source_type,
            "rectifier_parameters": self.dc_bus.rectifier.profile.source_type,
            "battery_parameters": self.dc_bus.battery.config.get("source_type", "unknown"),
            "mechanical_coupling": "derived_system_requirement",
        }
        warnings = tuple(dict.fromkeys((
            *self.config.warnings, *dc_point.warnings, *envelope.warnings,
            *fuel_warnings,
            *(('near_full_load_fuel_is_engineering_estimate',)
              if fuel_status == "near_full_load_estimate" else ()),
            *(('part_load_bsfc_map_missing',) if fuel_status == "part_load_map_missing" else ()),
            *(('part_load_map_out_of_range',) if fuel_status == "part_load_map_out_of_range" else ()),
        )))
        return PowertrainCandidate(
            engine_rpm=engine_rpm, generator_rpm=generator_rpm,
            dc_bus_status=dc_point.solver_status.value,
            electrical_feasible=electrical_feasible,
            mechanical_feasible=mechanical_feasible,
            physically_feasible=physically_feasible,
            engine_max_torque_nm=envelope.maximum_torque_nm,
            engine_required_torque_nm=required_torque,
            engine_torque_margin_nm=torque_margin,
            engine_max_power_kw=envelope.maximum_power_kw,
            engine_required_power_kw=required_power_kw,
            fuel_flow_kg_h=fuel_flow, fuel_model_status=fuel_status,
            engine_load_fraction_lambda=load_fraction,
            engine_normalized_speed_nu=normalized_speed,
            engine_full_load_bsfc_g_kwh=full_load_bsfc,
            engine_part_load_factor_k=part_load_factor,
            engine_effective_bsfc_g_kwh=effective_bsfc,
            fuel_model_source_type=fuel_source_type,
            fuel_model_confidence=fuel_confidence,
            generator_emf_ll_rms_v=dc_point.generator_emf_ll_rms_v,
            generator_dc_current_a=dc_point.generator_current_dc_a,
            generator_copper_loss_w=dc_point.generator_copper_loss_w,
            generator_other_losses_w=generator_other_losses,
            generator_required_shaft_power_w=required_power_w,
            generator_required_torque_nm=required_torque,
            rectifier_drop_v=dc_point.rectifier_drop_v,
            rectifier_loss_w=dc_point.rectifier_loss_w,
            rectifier_junction_temperature_c=dc_point.rectifier_junction_temperature_c,
            bus_voltage_v=dc_point.bus_voltage_v,
            battery_current_a=dc_point.battery_current_a,
            battery_power_w=dc_point.battery_power_w,
            load_current_a=dc_point.load_current_a, load_power_w=inputs.load_power_w,
            electrical_balance_error_w=dc_point.electrical_balance_error_w,
            mechanical_balance_error_w=mechanical_balance,
            active_limits=tuple(dict.fromkeys(active)), margins=margins,
            provenance_status=provenance, warnings=warnings,
            dc_bus_point=dc_point,
        )

    def _refine_transition(self, left: PowertrainCandidate,
                           right: PowertrainCandidate,
                           inputs: PowertrainOperatingPointInputs) -> PowertrainCandidate:
        if left.physically_feasible or not right.physically_feasible:
            return right
        low, high = left.engine_rpm, right.engine_rpm
        best = right
        while high - low > self.config.rpm_refinement_tolerance:
            midpoint = (low + high) / 2.0
            candidate = self.evaluate_candidate(midpoint, inputs)
            if candidate.physically_feasible:
                high, best = midpoint, candidate
            else:
                low = midpoint
        return best

    def enumerate_operating_points(
            self, inputs: PowertrainOperatingPointInputs
            ) -> tuple[PowertrainCandidate, ...]:
        """Return every grid and optional boundary-refinement candidate."""
        self._validate_inputs(inputs)
        readiness = self.readiness_errors()
        if readiness:
            raise ValueError("Powertrain solver not ready: " + ", ".join(readiness))
        rpm_grid = self.candidate_rpm_grid()
        if not rpm_grid:
            return ()
        candidates = [self.evaluate_candidate(rpm, inputs) for rpm in rpm_grid]
        if self.config.refine_infeasible_to_feasible_boundary:
            refined: list[PowertrainCandidate] = []
            digitized_max = self.engine.profile.digitized_map_max_rpm
            for left, right in zip(candidates, candidates[1:]):
                if (not left.physically_feasible and right.physically_feasible
                        and right.engine_rpm <= digitized_max):
                    refined.append(self._refine_transition(left, right, inputs))
            candidates.extend(refined)
            candidates.sort(key=lambda candidate: candidate.engine_rpm)
        return tuple(candidates)

    # Readable alias used by Phase 7A reports.
    build_candidate_map = enumerate_operating_points

    def _empty_result(self, status: PowertrainSolverStatus,
                      inputs: PowertrainOperatingPointInputs,
                      warnings: tuple[str, ...], candidates: tuple[PowertrainCandidate, ...] = (),
                      active_limits: tuple[str, ...] = (),
                      operating_policy: str | None = None) -> PowertrainOperatingPoint:
        return PowertrainOperatingPoint(
            operating_point_status=status,
            validation_status=self.config.validation_status,
            operating_policy=(self.config.operating_policy if operating_policy is None
                              else operating_policy),
            gear_ratio=self.config.gear_ratio,
            engine_rpm=None, generator_rpm=None,
            engine_max_torque_nm=None, engine_required_torque_nm=None,
            engine_torque_margin_nm=None, engine_max_power_kw=None,
            engine_required_power_kw=None, fuel_flow_kg_h=None,
            fuel_model_status="not_available",
            engine_load_fraction_lambda=None, engine_normalized_speed_nu=None,
            engine_full_load_bsfc_g_kwh=None, engine_part_load_factor_k=None,
            engine_effective_bsfc_g_kwh=None,
            fuel_model_source_type="not_available", fuel_model_confidence="low",
            generator_emf_ll_rms_v=None, generator_dc_current_a=None,
            generator_copper_loss_w=None, generator_other_losses_w=None,
            generator_required_shaft_power_w=None,
            generator_required_torque_nm=None, rectifier_drop_v=None,
            rectifier_loss_w=None, rectifier_junction_temperature_c=None,
            bus_voltage_v=None, battery_current_a=None, battery_power_w=None,
            load_current_a=None, load_power_w=inputs.load_power_w,
            electrical_balance_error_w=None, mechanical_balance_error_w=None,
            active_limits=active_limits, margins={}, provenance_status={},
            warnings=warnings, candidates=candidates, selected_candidate=None,
        )

    def _selection_index(self, candidates: tuple[PowertrainCandidate, ...],
                         inputs: PowertrainOperatingPointInputs,
                         operating_policy: str) -> int | None:
        feasible = [(index, candidate)
                    for index, candidate in enumerate(candidates)
                    if candidate.overall_feasible]
        if not feasible:
            return None
        if operating_policy == "lowest_feasible_rpm":
            return min(feasible, key=lambda item: item[1].engine_rpm)[0]
        if operating_policy == "closest_to_target_rpm":
            target = (inputs.target_rpm if inputs.target_rpm is not None
                      else self.config.target_rpm)
            if target is None:
                raise ValueError("closest_to_target_rpm требует target_rpm")
            return min(feasible,
                       key=lambda item: abs(item[1].engine_rpm - target))[0]
        if operating_policy == "minimum_engine_fuel_rate":
            fuel_candidates = [
                item for item in feasible
                if item[1].fuel_flow_kg_h is not None
                and math.isfinite(item[1].fuel_flow_kg_h)
            ]
            if not fuel_candidates:
                return None
            minimum = min(item[1].fuel_flow_kg_h for item in fuel_candidates)
            tied = [item for item in fuel_candidates
                    if item[1].fuel_flow_kg_h
                    <= minimum + MINIMUM_FUEL_TIE_EPSILON_KG_H]
            return min(
                tied,
                key=lambda item: (
                    math.inf if item[1].battery_power_w is None
                    else abs(item[1].battery_power_w),
                    item[1].engine_rpm,
                ))[0]
        raise ValueError(f"Неизвестная operating policy: {operating_policy}")

    def select_operating_point(
            self, inputs: PowertrainOperatingPointInputs,
            candidates: tuple[PowertrainCandidate, ...],
            operating_policy: str | None = None) -> PowertrainOperatingPoint:
        """Apply a policy without rebuilding or mutating the candidate map."""
        self._validate_inputs(inputs)
        policy = self.config.operating_policy if operating_policy is None else operating_policy
        if policy not in OPERATING_POLICIES:
            raise ValueError(f"Неизвестная operating policy: {policy}")
        feasible = [candidate for candidate in candidates
                    if candidate.overall_feasible]
        ecms_result = None
        if policy in {"minimum_equivalent_fuel_rate",
                      "soc_adaptive_minimum_equivalent_fuel_rate"}:
            expected_type = (FixedFactorECMS if policy == "minimum_equivalent_fuel_rate"
                             else ProportionalSOCAdaptiveECMS)
            if not isinstance(self.ecms, expected_type):
                return self._empty_result(
                    PowertrainSolverStatus.NOT_READY, inputs,
                    tuple(dict.fromkeys((
                        *self.config.warnings,
                        f"{policy}_requires_matching_ecms_profile",
                    ))), candidates,
                    ("ecms_profile_missing",), policy)
            if policy == "minimum_equivalent_fuel_rate":
                ecms_result = self.ecms.select(
                    candidates, self.powertrain_profile_readiness())
            else:
                ecms_result = self.ecms.select(
                    candidates, inputs.battery_soc,
                    self.powertrain_profile_readiness())
            selected_index = ecms_result.selected_candidate_index
        else:
            selected_index = self._selection_index(candidates, inputs, policy)
        if selected_index is None:
            active = tuple(dict.fromkeys(
                limit for candidate in candidates
                for limit in candidate.active_limits))
            if feasible and policy == "minimum_engine_fuel_rate":
                active = (*active, "minimum_engine_fuel_rate_not_evaluable")
                warnings = tuple(dict.fromkeys((
                    *self.config.warnings,
                    "minimum_engine_fuel_rate_requires_fuel_estimate",
                    "objective_does_not_price_battery_energy",
                )))
                return self._empty_result(
                    PowertrainSolverStatus.NOT_READY, inputs, warnings,
                    candidates, tuple(dict.fromkeys(active)), policy)
            if feasible and policy in {"minimum_equivalent_fuel_rate",
                                       "soc_adaptive_minimum_equivalent_fuel_rate"}:
                active = (*active, f"{policy}_not_evaluable")
                warnings = tuple(dict.fromkeys((
                    *self.config.warnings,
                    *(ecms_result.warnings if ecms_result is not None else ()),
                    "minimum_equivalent_fuel_rate_requires_fuel_and_battery_power",
                )))
                return self._empty_result(
                    PowertrainSolverStatus.NOT_READY, inputs, warnings,
                    candidates, tuple(dict.fromkeys(active)), policy)
            warnings = tuple(dict.fromkeys((
                *self.config.warnings,
                "no_feasible_powertrain_operating_point",
            )))
            return self._empty_result(
                PowertrainSolverStatus.NO_FEASIBLE_OPERATING_POINT,
                inputs, warnings, candidates, active, policy)

        selected = candidates[selected_index]
        provenance = dict(selected.provenance_status)
        warnings = selected.warnings
        if policy == "minimum_engine_fuel_rate":
            provenance.update({
                "selection_policy": "engineering_diagnostic",
                "selection_objective": "instantaneous_engine_fuel_rate",
                "objective_does_not_price_battery_energy": True,
                "fuel_tie_tolerance_kg_h": MINIMUM_FUEL_TIE_EPSILON_KG_H,
                "tie_break_order": "fuel_rate_then_abs_battery_power_then_rpm",
            })
            warnings = tuple(dict.fromkeys((
                *warnings,
                "minimum_engine_fuel_rate_is_engineering_diagnostic",
                "objective_does_not_price_battery_energy",
            )))
        elif policy in {"minimum_equivalent_fuel_rate",
                        "soc_adaptive_minimum_equivalent_fuel_rate"} \
                and ecms_result is not None:
            provenance.update(ecms_result.provenance)
            warnings = tuple(dict.fromkeys((*warnings, *ecms_result.warnings)))
        return PowertrainOperatingPoint(
            operating_point_status=PowertrainSolverStatus.OK,
            validation_status=self.config.validation_status,
            operating_policy=policy,
            gear_ratio=self.config.gear_ratio,
            engine_rpm=selected.engine_rpm, generator_rpm=selected.generator_rpm,
            engine_max_torque_nm=selected.engine_max_torque_nm,
            engine_required_torque_nm=selected.engine_required_torque_nm,
            engine_torque_margin_nm=selected.engine_torque_margin_nm,
            engine_max_power_kw=selected.engine_max_power_kw,
            engine_required_power_kw=selected.engine_required_power_kw,
            fuel_flow_kg_h=selected.fuel_flow_kg_h,
            fuel_model_status=selected.fuel_model_status,
            engine_load_fraction_lambda=selected.engine_load_fraction_lambda,
            engine_normalized_speed_nu=selected.engine_normalized_speed_nu,
            engine_full_load_bsfc_g_kwh=selected.engine_full_load_bsfc_g_kwh,
            engine_part_load_factor_k=selected.engine_part_load_factor_k,
            engine_effective_bsfc_g_kwh=selected.engine_effective_bsfc_g_kwh,
            fuel_model_source_type=selected.fuel_model_source_type,
            fuel_model_confidence=selected.fuel_model_confidence,
            generator_emf_ll_rms_v=selected.generator_emf_ll_rms_v,
            generator_dc_current_a=selected.generator_dc_current_a,
            generator_copper_loss_w=selected.generator_copper_loss_w,
            generator_other_losses_w=selected.generator_other_losses_w,
            generator_required_shaft_power_w=selected.generator_required_shaft_power_w,
            generator_required_torque_nm=selected.generator_required_torque_nm,
            rectifier_drop_v=selected.rectifier_drop_v,
            rectifier_loss_w=selected.rectifier_loss_w,
            rectifier_junction_temperature_c=selected.rectifier_junction_temperature_c,
            bus_voltage_v=selected.bus_voltage_v,
            battery_current_a=selected.battery_current_a,
            battery_power_w=selected.battery_power_w,
            load_current_a=selected.load_current_a, load_power_w=selected.load_power_w,
            electrical_balance_error_w=selected.electrical_balance_error_w,
            mechanical_balance_error_w=selected.mechanical_balance_error_w,
            active_limits=selected.active_limits, margins=selected.margins,
            provenance_status=provenance,
            warnings=warnings, candidates=candidates,
            selected_candidate=selected_index,
            ecms_selection=(None if ecms_result is None else ecms_result.to_dict()),
        )

    def solve(self, inputs: PowertrainOperatingPointInputs) -> PowertrainOperatingPoint:
        self._validate_inputs(inputs)
        readiness = self.readiness_errors()
        if readiness:
            return self._empty_result(
                PowertrainSolverStatus.NOT_READY, inputs,
                tuple(dict.fromkeys((*self.config.warnings, *readiness))))
        rpm_grid = self.candidate_rpm_grid()
        if not rpm_grid:
            return self._empty_result(
                PowertrainSolverStatus.NOT_READY, inputs,
                (*self.config.warnings, "empty_confirmed_rpm_intersection"))
        candidates = self.enumerate_operating_points(inputs)
        return self.select_operating_point(inputs, candidates)


def load_synthetic_powertrain_fixture(
        path: str | Path | None = None) -> SyntheticPowertrainFixture:
    fixture_path = (Path(path) if path is not None else BUILTIN_POWERTRAIN_DIR
                    / f"{SYNTHETIC_POWERTRAIN_FIXTURE_ID}.json")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if data.get("source_type") != "synthetic_test_fixture":
        raise ValueError("Synthetic powertrain fixture имеет неверный source_type")
    dc_fixture = load_synthetic_dc_bus_fixture()
    if data["dc_bus_fixture_id"] != dc_fixture.fixture_id:
        raise ValueError("Неизвестная ссылка на DC-bus fixture")
    dc_config = replace(dc_fixture.solver_config,
                        **data.get("dc_bus_solver_overrides", {}))
    dc_solver = DCBusSolver(
        dc_fixture.generator_model, dc_fixture.rectifier_model,
        dc_fixture.battery_model, dc_config)
    part_load = None
    if data.get("engine_part_load_profile_id") is not None:
        part_load = EnginePartLoadModel(
            load_builtin_part_load_profile(data["engine_part_load_profile_id"]),
            data.get("low_load_sensitivity"))
    engine = EngineModel(load_builtin_engine_profile(data["engine_profile_id"]), part_load)
    coupling = data["mechanical_coupling"]
    search = data["search"]
    fuel = data["fuel"]
    config = PowertrainSolverConfig(
        source_type=data["source_type"],
        validation_status=data["validation_status"],
        gear_ratio=float(coupling["gear_ratio"]),
        operating_policy=search["operating_policy"],
        target_rpm=(None if search.get("target_rpm") is None
                    else float(search["target_rpm"])),
        operating_min_rpm=float(search["operating_min_rpm"]),
        operating_max_rpm=float(search["operating_max_rpm"]),
        rpm_grid_step=float(search["rpm_grid_step"]),
        include_confirmed_rated_point=bool(search["include_confirmed_rated_point"]),
        refine_infeasible_to_feasible_boundary=bool(
            search["refine_infeasible_to_feasible_boundary"]),
        rpm_refinement_tolerance=float(search["rpm_refinement_tolerance"]),
        full_load_exact_relative_torque_tolerance=float(
            fuel["full_load_exact_relative_torque_tolerance"]),
        near_full_load_relative_torque_tolerance=float(
            fuel["near_full_load_relative_torque_tolerance"]),
        fuel_tolerance_source_type=fuel["tolerance_source_type"],
        part_load_policy=fuel["part_load_policy"],
        warnings=tuple(data.get("warnings", ())),
    )
    return SyntheticPowertrainFixture(
        fixture_id=data["fixture_id"], engine_model=engine,
        dc_bus_solver=dc_solver, solver_config=config)


def build_synthetic_powertrain_solver() -> PowertrainOperatingPointSolver:
    fixture = load_synthetic_powertrain_fixture()
    return PowertrainOperatingPointSolver(
        fixture.engine_model, fixture.dc_bus_solver, fixture.solver_config)
