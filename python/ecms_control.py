"""Phase 7B1 fixed-factor equivalent-consumption selection layer.

The module consumes immutable Phase 7A candidates.  It never solves the DC
bus, changes SOC, or modifies any physical component model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
BUILTIN_CONTROL_DIR = ROOT / "data" / "control"
DEFAULT_ECMS_PROFILE_ID = "phase7b1_fixed_ecms_avgas100ll_v1"
ECMS_OBJECTIVE_TIE_EPSILON_KG_H = 1e-9
BATTERY_NEUTRAL_TOLERANCE_W = 1e-9


class ECMSReadiness(str, Enum):
    READY_FOR_FIXED_FACTOR_DIAGNOSTIC_ECMS = (
        "ready_for_fixed_factor_diagnostic_ecms")
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class ECMSProfile:
    profile_id: str
    strategy: str
    fuel_lhv_j_per_kg: float
    fuel_lhv_basis: str
    equivalence_factor_discharge: float
    equivalence_factor_charge: float
    soc_feedback_enabled: bool
    aging_penalty_enabled: bool
    thermal_penalty_enabled: bool
    plating_penalty_enabled: bool
    engineering_reference_only: bool
    validated_for_target_powertrain: bool
    exact_two_stroke_oil_blend_lhv_known: bool
    factor_calibration_status: str
    source_type: str
    source: tuple[str, ...]
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("ECMS profile_id не должен быть пустым")
        if self.strategy != "fixed_factor_ecms":
            raise ValueError("Phase 7B1 поддерживает только fixed_factor_ecms")
        if (not math.isfinite(self.fuel_lhv_j_per_kg)
                or self.fuel_lhv_j_per_kg <= 0):
            raise ValueError("fuel_lhv_j_per_kg должен быть положительным")
        for value, name in (
                (self.equivalence_factor_discharge,
                 "equivalence_factor_discharge"),
                (self.equivalence_factor_charge,
                 "equivalence_factor_charge")):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} должен быть конечным и неотрицательным")
        if self.soc_feedback_enabled:
            raise ValueError("SOC feedback относится к Phase 7B2")
        if any((self.aging_penalty_enabled, self.thermal_penalty_enabled,
                self.plating_penalty_enabled)):
            raise ValueError("Phase 7B1 не поддерживает дополнительные penalties")

    @property
    def readiness(self) -> ECMSReadiness:
        return ECMSReadiness.READY_FOR_FIXED_FACTOR_DIAGNOSTIC_ECMS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["readiness"] = self.readiness.value
        data["source"] = list(self.source)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class ECMSCandidateCost:
    candidate_index: int
    engine_rpm: float
    ecms_eligible: bool
    ecms_ineligible_reason: str | None
    physical_fuel_rate_kg_h: float | None
    battery_power_kw: float | None
    battery_mode: str
    battery_equivalent_fuel_rate_kg_h: float | None
    ecms_equivalent_fuel_rate_kg_h: float | None
    equivalence_factor_used: float | None
    equivalence_factor_mode: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ECMSSelectionResult:
    readiness: ECMSReadiness | "AdaptiveECMSReadiness"
    profile_id: str
    costs: tuple[ECMSCandidateCost, ...]
    selected_candidate_index: int | None
    warnings: tuple[str, ...]
    provenance: dict[str, Any]
    adaptive_diagnostics: dict[str, Any] | None = None

    @property
    def selected_cost(self) -> ECMSCandidateCost | None:
        if self.selected_candidate_index is None:
            return None
        return next((cost for cost in self.costs
                     if cost.candidate_index == self.selected_candidate_index), None)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "readiness": self.readiness.value,
            "profile_id": self.profile_id,
            "costs": [cost.to_dict() for cost in self.costs],
            "selected_candidate_index": self.selected_candidate_index,
            "selected_cost": (None if self.selected_cost is None
                              else self.selected_cost.to_dict()),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }
        if self.adaptive_diagnostics is not None:
            data["adaptive_diagnostics"] = dict(self.adaptive_diagnostics)
        return data


class FixedFactorECMS:
    """Evaluate and select candidates using a fixed equivalent-fuel factor."""

    def __init__(self, profile: ECMSProfile):
        self.profile = profile

    def battery_equivalent_fuel_rate(
            self, battery_power_w: float) -> tuple[float, float, str]:
        if not math.isfinite(battery_power_w):
            raise ValueError("battery_power_w должен быть конечным")
        if abs(battery_power_w) < BATTERY_NEUTRAL_TOLERANCE_W:
            return 0.0, 0.0, "neutral"
        if battery_power_w > 0:
            factor = self.profile.equivalence_factor_discharge
            mode = "discharge"
        else:
            factor = self.profile.equivalence_factor_charge
            mode = "charge"
        # W / (J/kg) = kg/s; multiplying by 3600 converts to kg/h.
        equivalent = factor * battery_power_w / self.profile.fuel_lhv_j_per_kg * 3600.0
        return equivalent, factor, mode

    def evaluate_candidate(self, candidate: Any,
                           candidate_index: int) -> ECMSCandidateCost:
        reasons: list[str] = []
        if not bool(candidate.overall_feasible):
            reasons.append("not_overall_feasible")
        fuel = candidate.fuel_flow_kg_h
        if fuel is None or not math.isfinite(fuel):
            reasons.append("physical_fuel_rate_missing")
        battery_w = candidate.battery_power_w
        if battery_w is None or not math.isfinite(battery_w):
            reasons.append("battery_power_missing")
        if reasons:
            return ECMSCandidateCost(
                candidate_index, float(candidate.engine_rpm), False,
                ";".join(reasons), fuel,
                None if battery_w is None else battery_w / 1000.0,
                getattr(candidate, "battery_mode", "unknown"),
                None, None, None, None)
        battery_eq, factor, mode = self.battery_equivalent_fuel_rate(battery_w)
        return ECMSCandidateCost(
            candidate_index=candidate_index,
            engine_rpm=float(candidate.engine_rpm),
            ecms_eligible=True,
            ecms_ineligible_reason=None,
            physical_fuel_rate_kg_h=float(fuel),
            battery_power_kw=float(battery_w) / 1000.0,
            battery_mode=mode,
            battery_equivalent_fuel_rate_kg_h=battery_eq,
            ecms_equivalent_fuel_rate_kg_h=float(fuel) + battery_eq,
            equivalence_factor_used=factor,
            equivalence_factor_mode=mode,
        )

    def select(self, candidates: Sequence[Any],
               powertrain_profile_readiness: str = "not_provided"
               ) -> ECMSSelectionResult:
        costs = tuple(self.evaluate_candidate(candidate, index)
                      for index, candidate in enumerate(candidates))
        eligible = [cost for cost in costs if cost.ecms_eligible]
        selected_index = None
        if eligible:
            minimum = min(cost.ecms_equivalent_fuel_rate_kg_h
                          for cost in eligible)
            tied = [cost for cost in eligible
                    if cost.ecms_equivalent_fuel_rate_kg_h
                    <= minimum + ECMS_OBJECTIVE_TIE_EPSILON_KG_H]
            selected = min(tied, key=lambda cost: (
                abs(cost.battery_power_kw),
                cost.physical_fuel_rate_kg_h,
                cost.engine_rpm,
            ))
            selected_index = selected.candidate_index
        warnings = tuple(dict.fromkeys((
            *self.profile.warnings,
            "EQUIVALENCE_FACTORS_NOT_CALIBRATED",
            "NO_SOC_FEEDBACK",
            "NOT_MISSION_OPTIMAL",
            "NO_AGING_COST",
            "NO_THERMAL_COST",
        )))
        provenance = {
            "ecms_profile_id": self.profile.profile_id,
            "fuel_lhv_j_per_kg": self.profile.fuel_lhv_j_per_kg,
            "fuel_lhv_basis": self.profile.fuel_lhv_basis,
            "equivalence_factor_discharge": (
                self.profile.equivalence_factor_discharge),
            "equivalence_factor_charge": self.profile.equivalence_factor_charge,
            "factor_calibration_status": self.profile.factor_calibration_status,
            "soc_feedback_enabled": self.profile.soc_feedback_enabled,
            "powertrain_profile_readiness": powertrain_profile_readiness,
            "selection_policy": "engineering_diagnostic",
            "selection_objective": "instantaneous_equivalent_fuel_rate",
            "equivalent_fuel_is_control_cost_not_physical_consumption": True,
            "objective_tie_tolerance_kg_h": ECMS_OBJECTIVE_TIE_EPSILON_KG_H,
            "tie_break_order": (
                "equivalent_fuel_then_abs_battery_power_then_physical_fuel_then_rpm"),
        }
        return ECMSSelectionResult(
            readiness=self.profile.readiness,
            profile_id=self.profile.profile_id,
            costs=costs,
            selected_candidate_index=selected_index,
            warnings=warnings,
            provenance=provenance,
        )


def load_builtin_ecms_profile(
        profile_id: str = DEFAULT_ECMS_PROFILE_ID) -> ECMSProfile:
    path = BUILTIN_CONTROL_DIR / f"{profile_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("profile_id") != profile_id:
        raise ValueError("ECMS profile_id не совпадает с именем профиля")
    return ECMSProfile(
        profile_id=data["profile_id"], strategy=data["strategy"],
        fuel_lhv_j_per_kg=float(data["fuel_lhv_j_per_kg"]),
        fuel_lhv_basis=data["fuel_lhv_basis"],
        equivalence_factor_discharge=float(data["equivalence_factor_discharge"]),
        equivalence_factor_charge=float(data["equivalence_factor_charge"]),
        soc_feedback_enabled=bool(data["soc_feedback_enabled"]),
        aging_penalty_enabled=bool(data["aging_penalty_enabled"]),
        thermal_penalty_enabled=bool(data["thermal_penalty_enabled"]),
        plating_penalty_enabled=bool(data["plating_penalty_enabled"]),
        engineering_reference_only=bool(data["engineering_reference_only"]),
        validated_for_target_powertrain=bool(
            data["validated_for_target_powertrain"]),
        exact_two_stroke_oil_blend_lhv_known=bool(
            data["exact_two_stroke_oil_blend_lhv_known"]),
        factor_calibration_status=data["factor_calibration_status"],
        source_type=data["source_type"],
        source=tuple(data.get("source", ())),
        provenance=dict(data.get("provenance", {})),
        warnings=tuple(data.get("warnings", ())),
    )


class SOCManagementState(str, Enum):
    BELOW_MANAGEMENT_BAND = "below_management_band"
    WITHIN_MANAGEMENT_BAND = "within_management_band"
    ABOVE_MANAGEMENT_BAND = "above_management_band"
    AT_REFERENCE = "at_reference"


class AdaptiveECMSReadiness(str, Enum):
    READY_FOR_PROPORTIONAL_SOC_ADAPTIVE_ECMS_DIAGNOSTICS = (
        "ready_for_proportional_soc_adaptive_ecms_diagnostics")
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class AdaptiveECMSProfile:
    profile_id: str
    strategy: str
    fuel_lhv_j_per_kg: float
    fuel_lhv_basis: str
    base_equivalence_factor_discharge: float
    base_equivalence_factor_charge: float
    soc_management_low: float
    soc_management_reference: float
    soc_management_high: float
    soc_proportional_gain: float
    equivalence_factor_discharge_min: float
    equivalence_factor_discharge_max: float
    equivalence_factor_charge_min: float
    equivalence_factor_charge_max: float
    selection_hysteresis_enabled: bool
    selection_hysteresis_relative: float
    hysteresis_objective_scale_floor_kg_h: float
    soc_feedback_enabled: bool
    integral_soc_control_enabled: bool
    minimum_dwell_time_enabled: bool
    minimum_dwell_time_s: float
    aging_penalty_enabled: bool
    thermal_penalty_enabled: bool
    plating_penalty_enabled: bool
    engineering_diagnostic_only: bool
    parameters_calibrated_for_target_powertrain: bool
    exact_two_stroke_oil_blend_lhv_known: bool
    source_type: str
    source: tuple[str, ...]
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy != "proportional_soc_adaptive_ecms":
            raise ValueError("Phase 7B2 поддерживает proportional SOC-adaptive ECMS")
        if not math.isfinite(self.fuel_lhv_j_per_kg) or self.fuel_lhv_j_per_kg <= 0:
            raise ValueError("fuel_lhv_j_per_kg должен быть положительным")
        if not (0 <= self.soc_management_low
                < self.soc_management_reference
                < self.soc_management_high <= 1):
            raise ValueError("SOC management band должен удовлетворять low<ref<high")
        if (not math.isfinite(self.soc_proportional_gain)
                or self.soc_proportional_gain < 0):
            raise ValueError("soc_proportional_gain должен быть неотрицательным")
        factor_triplets = (
            (self.equivalence_factor_discharge_min,
             self.base_equivalence_factor_discharge,
             self.equivalence_factor_discharge_max, "discharge"),
            (self.equivalence_factor_charge_min,
             self.base_equivalence_factor_charge,
             self.equivalence_factor_charge_max, "charge"),
        )
        for minimum, base, maximum, name in factor_triplets:
            if not all(math.isfinite(value) for value in (minimum, base, maximum)):
                raise ValueError(f"{name} equivalence factors должны быть конечными")
            if not 0 <= minimum <= base <= maximum:
                raise ValueError(f"{name} factors должны удовлетворять 0<=min<=base<=max")
        if (not math.isfinite(self.selection_hysteresis_relative)
                or self.selection_hysteresis_relative < 0):
            raise ValueError("selection_hysteresis_relative должен быть >=0")
        if (not math.isfinite(self.hysteresis_objective_scale_floor_kg_h)
                or self.hysteresis_objective_scale_floor_kg_h <= 0):
            raise ValueError("hysteresis objective scale floor должен быть >0")
        if not self.soc_feedback_enabled:
            raise ValueError("Adaptive ECMS требует soc_feedback_enabled=true")
        if self.integral_soc_control_enabled:
            raise ValueError("Integral SOC control отложен после Phase 7B2")
        if self.minimum_dwell_time_enabled or self.minimum_dwell_time_s != 0:
            raise ValueError("Minimum dwell time не реализуется в Phase 7B2")
        if any((self.aging_penalty_enabled, self.thermal_penalty_enabled,
                self.plating_penalty_enabled)):
            raise ValueError("Phase 7B2 не поддерживает дополнительные penalties")

    @property
    def readiness(self) -> AdaptiveECMSReadiness:
        return AdaptiveECMSReadiness.READY_FOR_PROPORTIONAL_SOC_ADAPTIVE_ECMS_DIAGNOSTICS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["readiness"] = self.readiness.value
        data["source"] = list(self.source)
        data["warnings"] = list(self.warnings)
        return data


@dataclass
class AdaptiveECMSState:
    previous_selected_rpm: float | None = None
    switch_count: int = 0
    suppressed_switch_count: int = 0
    last_selected_candidate_id: str | None = None


@dataclass(frozen=True)
class AdaptiveFactorState:
    soc: float
    soc_management_state: str
    soc_error: float
    delta_s: float
    s_dis_base: float
    s_dis_raw: float
    s_dis_effective: float
    s_dis_clipped: bool
    s_chg_base: float
    s_chg_raw: float
    s_chg_effective: float
    s_chg_clipped: bool


@dataclass(frozen=True)
class AdaptiveECMSStepResult:
    selection: ECMSSelectionResult
    factors: AdaptiveFactorState
    diagnostics: dict[str, Any]


class ProportionalSOCAdaptiveECMS:
    """Bounded P-only SOC adaptation followed by supervisory hysteresis."""

    def __init__(self, profile: AdaptiveECMSProfile):
        self.profile = profile
        self.state = AdaptiveECMSState()

    def reset(self) -> None:
        self.state = AdaptiveECMSState()

    def soc_management_state(self, soc: float) -> SOCManagementState:
        if math.isclose(soc, self.profile.soc_management_reference,
                        rel_tol=0.0, abs_tol=1e-12):
            return SOCManagementState.AT_REFERENCE
        if soc < self.profile.soc_management_low:
            return SOCManagementState.BELOW_MANAGEMENT_BAND
        if soc > self.profile.soc_management_high:
            return SOCManagementState.ABOVE_MANAGEMENT_BAND
        return SOCManagementState.WITHIN_MANAGEMENT_BAND

    def factor_state(self, soc: float) -> AdaptiveFactorState:
        if not math.isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("SOC для adaptive ECMS должен находиться в [0,1]")
        error = self.profile.soc_management_reference - soc
        delta = self.profile.soc_proportional_gain * error
        dis_raw = self.profile.base_equivalence_factor_discharge + delta
        chg_raw = self.profile.base_equivalence_factor_charge + delta
        dis = min(max(dis_raw, self.profile.equivalence_factor_discharge_min),
                  self.profile.equivalence_factor_discharge_max)
        chg = min(max(chg_raw, self.profile.equivalence_factor_charge_min),
                  self.profile.equivalence_factor_charge_max)
        return AdaptiveFactorState(
            soc=soc,
            soc_management_state=self.soc_management_state(soc).value,
            soc_error=error,
            delta_s=delta,
            s_dis_base=self.profile.base_equivalence_factor_discharge,
            s_dis_raw=dis_raw,
            s_dis_effective=dis,
            s_dis_clipped=not math.isclose(dis, dis_raw, rel_tol=0.0, abs_tol=1e-15),
            s_chg_base=self.profile.base_equivalence_factor_charge,
            s_chg_raw=chg_raw,
            s_chg_effective=chg,
            s_chg_clipped=not math.isclose(chg, chg_raw, rel_tol=0.0, abs_tol=1e-15),
        )

    def _fixed_profile(self, factors: AdaptiveFactorState) -> ECMSProfile:
        return ECMSProfile(
            profile_id=self.profile.profile_id,
            strategy="fixed_factor_ecms",
            fuel_lhv_j_per_kg=self.profile.fuel_lhv_j_per_kg,
            fuel_lhv_basis=self.profile.fuel_lhv_basis,
            equivalence_factor_discharge=factors.s_dis_effective,
            equivalence_factor_charge=factors.s_chg_effective,
            soc_feedback_enabled=False,
            aging_penalty_enabled=False,
            thermal_penalty_enabled=False,
            plating_penalty_enabled=False,
            engineering_reference_only=True,
            validated_for_target_powertrain=(
                self.profile.parameters_calibrated_for_target_powertrain),
            exact_two_stroke_oil_blend_lhv_known=(
                self.profile.exact_two_stroke_oil_blend_lhv_known),
            factor_calibration_status="soc_adaptive_not_calibrated",
            source_type=self.profile.source_type,
            source=self.profile.source,
            provenance=self.profile.provenance,
            warnings=self.profile.warnings,
        )

    def select(self, candidates: Sequence[Any], battery_soc: float,
               powertrain_profile_readiness: str = "not_provided"
               ) -> ECMSSelectionResult:
        factors = self.factor_state(battery_soc)
        raw = FixedFactorECMS(self._fixed_profile(factors)).select(
            candidates, powertrain_profile_readiness)
        raw_index = raw.selected_candidate_index
        if raw_index is None:
            diagnostics = self._diagnostics(
                factors, None, None, None, None, None, False, False, False,
                "no_raw_eligible_candidate")
            warnings = tuple(dict.fromkeys((
                               *(item for item in raw.warnings
                                 if item != "NO_SOC_FEEDBACK"),
                               "SOC_FEEDBACK_PARAMETERS_NOT_CALIBRATED",
                               "NOT_MISSION_GLOBAL_OPTIMUM")))
            provenance = dict(raw.provenance)
            provenance.update(self._adaptive_provenance())
            return replace(
                raw, readiness=self.profile.readiness,
                adaptive_diagnostics=diagnostics, warnings=warnings,
                provenance=provenance)

        raw_cost = raw.selected_cost
        raw_rpm = raw_cost.engine_rpm
        previous_rpm = self.state.previous_selected_rpm
        selected_index = raw_index
        previous_cost = None
        improvement = None
        relative = None
        switch_requested = previous_rpm is not None and not math.isclose(
            raw_rpm, previous_rpm, rel_tol=0.0, abs_tol=1e-9)
        switch_applied = False
        switch_suppressed = False
        reason = "first_step_raw_minimum" if previous_rpm is None else "raw_best_unchanged"

        previous_entry = None
        if previous_rpm is not None:
            previous_entry = next((cost for cost in raw.costs
                                   if math.isclose(cost.engine_rpm, previous_rpm,
                                                   rel_tol=0.0, abs_tol=1e-9)), None)
        if switch_requested:
            if previous_entry is None:
                reason = "previous_candidate_missing_switch_immediately"
                switch_applied = True
            elif not previous_entry.ecms_eligible:
                reason = "previous_candidate_ineligible_switch_immediately"
                switch_applied = True
            else:
                previous_cost = previous_entry.ecms_equivalent_fuel_rate_kg_h
                improvement = previous_cost - raw_cost.ecms_equivalent_fuel_rate_kg_h
                relative = improvement / max(
                    abs(previous_cost),
                    self.profile.hysteresis_objective_scale_floor_kg_h)
                if (self.profile.selection_hysteresis_enabled
                        and relative <= self.profile.selection_hysteresis_relative):
                    selected_index = previous_entry.candidate_index
                    switch_suppressed = True
                    self.state.suppressed_switch_count += 1
                    reason = "relative_improvement_within_hysteresis_hold_previous"
                else:
                    switch_applied = True
                    reason = ("hysteresis_disabled_accept_raw" if
                              not self.profile.selection_hysteresis_enabled else
                              "relative_improvement_exceeds_hysteresis")

        selected_entry = next(
            cost for cost in raw.costs
            if cost.candidate_index == selected_index)
        selected_rpm = selected_entry.engine_rpm
        if (previous_rpm is not None
                and not math.isclose(selected_rpm, previous_rpm,
                                     rel_tol=0.0, abs_tol=1e-9)):
            self.state.switch_count += 1
        self.state.previous_selected_rpm = selected_rpm
        self.state.last_selected_candidate_id = f"rpm:{selected_rpm:g}"

        diagnostics = self._diagnostics(
            factors, raw_rpm, selected_rpm, previous_rpm,
            raw_cost.ecms_equivalent_fuel_rate_kg_h, previous_cost,
            switch_requested, switch_applied, switch_suppressed, reason,
            improvement, relative)
        provenance = dict(raw.provenance)
        provenance.update(self._adaptive_provenance())
        warnings = tuple(dict.fromkeys((
            *(item for item in raw.warnings if item != "NO_SOC_FEEDBACK"),
            "SOC_FEEDBACK_PARAMETERS_NOT_CALIBRATED",
            "NO_TERMINAL_SOC_OPTIMIZATION",
            "NO_ENGINE_TRANSIENT_MODEL",
            "NO_INTEGRAL_SOC_CONTROL",
            "NOT_MISSION_GLOBAL_OPTIMUM",
        )))
        return replace(raw, readiness=self.profile.readiness,
                       selected_candidate_index=selected_index,
                       warnings=warnings, provenance=provenance,
                       adaptive_diagnostics=diagnostics)

    def _adaptive_provenance(self) -> dict[str, Any]:
        return {
            "strategy": self.profile.strategy,
            "ecms_profile_id": self.profile.profile_id,
            "soc_feedback_scientific_basis": "bounded_proportional_feedback",
            "soc_proportional_gain": self.profile.soc_proportional_gain,
            "soc_management_low": self.profile.soc_management_low,
            "soc_management_reference": self.profile.soc_management_reference,
            "soc_management_high": self.profile.soc_management_high,
            "selection_hysteresis_enabled": self.profile.selection_hysteresis_enabled,
            "selection_hysteresis_relative": self.profile.selection_hysteresis_relative,
            "parameters_calibrated_for_target_powertrain": (
                self.profile.parameters_calibrated_for_target_powertrain),
        }

    def _diagnostics(self, factors: AdaptiveFactorState,
                     raw_best_rpm: float | None, selected_rpm: float | None,
                     previous_rpm: float | None,
                     raw_objective: float | None,
                     previous_objective: float | None,
                     switch_requested: bool, switch_applied: bool,
                     switch_suppressed: bool, switch_reason: str,
                     improvement: float | None = None,
                     relative: float | None = None) -> dict[str, Any]:
        data = asdict(factors)
        data.update({
            "raw_best_rpm": raw_best_rpm,
            "selected_rpm": selected_rpm,
            "previous_selected_rpm": previous_rpm,
            "raw_best_objective": raw_objective,
            "previous_candidate_objective": previous_objective,
            "objective_improvement": improvement,
            "relative_objective_improvement": relative,
            "hysteresis_threshold": self.profile.selection_hysteresis_relative,
            "hysteresis_enabled": self.profile.selection_hysteresis_enabled,
            "switch_requested": switch_requested,
            "switch_applied": switch_applied,
            "switch_suppressed": switch_suppressed,
            "switch_reason": switch_reason,
            "switch_count": self.state.switch_count,
            "suppressed_switch_count": self.state.suppressed_switch_count,
            "last_selected_candidate_id": self.state.last_selected_candidate_id,
        })
        return data


def load_builtin_adaptive_ecms_profile(
        profile_id: str = "phase7b2_soc_adaptive_ecms_diagnostic_v1"
        ) -> AdaptiveECMSProfile:
    path = BUILTIN_CONTROL_DIR / f"{profile_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("profile_id") != profile_id:
        raise ValueError("Adaptive ECMS profile_id не совпадает")
    return AdaptiveECMSProfile(
        profile_id=data["profile_id"], strategy=data["strategy"],
        fuel_lhv_j_per_kg=float(data["fuel_lhv_j_per_kg"]),
        fuel_lhv_basis=data["fuel_lhv_basis"],
        base_equivalence_factor_discharge=float(
            data["base_equivalence_factor_discharge"]),
        base_equivalence_factor_charge=float(
            data["base_equivalence_factor_charge"]),
        soc_management_low=float(data["soc_management_low"]),
        soc_management_reference=float(data["soc_management_reference"]),
        soc_management_high=float(data["soc_management_high"]),
        soc_proportional_gain=float(data["soc_proportional_gain"]),
        equivalence_factor_discharge_min=float(
            data["equivalence_factor_discharge_min"]),
        equivalence_factor_discharge_max=float(
            data["equivalence_factor_discharge_max"]),
        equivalence_factor_charge_min=float(data["equivalence_factor_charge_min"]),
        equivalence_factor_charge_max=float(data["equivalence_factor_charge_max"]),
        selection_hysteresis_enabled=bool(data["selection_hysteresis_enabled"]),
        selection_hysteresis_relative=float(data["selection_hysteresis_relative"]),
        hysteresis_objective_scale_floor_kg_h=float(
            data["hysteresis_objective_scale_floor_kg_h"]),
        soc_feedback_enabled=bool(data["soc_feedback_enabled"]),
        integral_soc_control_enabled=bool(data["integral_soc_control_enabled"]),
        minimum_dwell_time_enabled=bool(data["minimum_dwell_time_enabled"]),
        minimum_dwell_time_s=float(data["minimum_dwell_time_s"]),
        aging_penalty_enabled=bool(data["aging_penalty_enabled"]),
        thermal_penalty_enabled=bool(data["thermal_penalty_enabled"]),
        plating_penalty_enabled=bool(data["plating_penalty_enabled"]),
        engineering_diagnostic_only=bool(data["engineering_diagnostic_only"]),
        parameters_calibrated_for_target_powertrain=bool(
            data["parameters_calibrated_for_target_powertrain"]),
        exact_two_stroke_oil_blend_lhv_known=bool(
            data["exact_two_stroke_oil_blend_lhv_known"]),
        source_type=data["source_type"], source=tuple(data.get("source", ())),
        provenance=dict(data.get("provenance", {})),
        warnings=tuple(data.get("warnings", ())),
    )
