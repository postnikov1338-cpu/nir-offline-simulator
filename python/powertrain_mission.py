"""Phase 5.5 quasi-steady mission integration for the autonomous powertrain.

Each step freezes state while PowertrainOperatingPointSolver searches rpm and
the DC-bus root.  Battery charge and component temperatures are updated only
after that static operating point has been accepted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ecms_control import (
    AdaptiveECMSProfile,
    ECMSProfile,
    FixedFactorECMS,
    ProportionalSOCAdaptiveECMS,
)
from cold_charge_observer import (
    ColdChargeObserver,
    ColdChargeObserverInput,
)
from powertrain_solver import (
    PowertrainOperatingPoint,
    PowertrainOperatingPointInputs,
    PowertrainOperatingPointSolver,
    PowertrainSolverStatus,
    load_synthetic_powertrain_fixture,
)
from powertrain_thermal import (
    GeneratorThermalModel,
    GeneratorThermalProfile,
    HeatSinkProfile,
    RectifierThermalModel,
    ThermalSimulationInput,
    ThermalState,
    ThermalStatus,
)
from thermal_model import ThermalModel


ROOT = Path(__file__).resolve().parent
BUILTIN_POWERTRAIN_DIR = ROOT / "data" / "powertrain"
SYNTHETIC_MISSION_FIXTURE_ID = "synthetic_powertrain_mission_fixture_v1"


@dataclass(frozen=True)
class PowertrainMissionSegment:
    segment_id: str
    duration_s: float
    required_electrical_power_w: float
    ambient_temperature_c: float
    target_rpm: float | None = None
    demand_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        values = (self.duration_s, self.required_electrical_power_w,
                  self.ambient_temperature_c)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Параметры участка миссии должны быть конечными")
        if self.duration_s <= 0 or self.required_electrical_power_w < 0:
            raise ValueError("Участок требует duration>0 и power>=0")
        if self.ambient_temperature_c <= -273.15:
            raise ValueError("Температура воздуха ниже абсолютного нуля")
        if self.target_rpm is not None and (
                not math.isfinite(self.target_rpm) or self.target_rpm <= 0):
            raise ValueError("target_rpm должен быть положительным")


@dataclass(frozen=True)
class PowertrainMissionConfig:
    dt_s: float
    battery_temperature_limit_c: float
    battery_temperature_limit_source_type: str
    shaft_transient_model: str = "not_implemented"
    source_type: str = "synthetic_test_fixture"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s должен быть положительным")
        if (not math.isfinite(self.battery_temperature_limit_c)
                or self.battery_temperature_limit_c <= -273.15):
            raise ValueError("Неверный предел температуры АКБ")
        if self.battery_temperature_limit_source_type not in {
                "manufacturer", "scientific_analogue", "engineering_assumption",
                "synthetic_test_fixture"}:
            raise ValueError("Неизвестный source_type температурного предела АКБ")
        if self.shaft_transient_model != "not_implemented":
            raise ValueError("Phase 5.5 не реализует shaft transient model")


@dataclass(frozen=True)
class PowertrainMissionState:
    time_s: float
    battery_soc: float
    battery_core_temperature_c: float
    battery_surface_temperature_c: float
    generator_temperature_c: float
    rectifier_sink_temperature_c: float
    rectifier_case_temperature_c: float
    rectifier_junction_temperature_c: float
    fuel_remaining_kg: float | None
    previous_bus_voltage_v: float | None
    previous_engine_rpm: float | None

    def __post_init__(self) -> None:
        values = (
            self.time_s, self.battery_soc, self.battery_core_temperature_c,
            self.battery_surface_temperature_c, self.generator_temperature_c,
            self.rectifier_sink_temperature_c, self.rectifier_case_temperature_c,
            self.rectifier_junction_temperature_c,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Mission state должен содержать конечные значения")
        if self.time_s < 0 or not 0 <= self.battery_soc <= 1:
            raise ValueError("Неверные time или SOC")
        if self.fuel_remaining_kg is not None and (
                not math.isfinite(self.fuel_remaining_kg)
                or self.fuel_remaining_kg < 0):
            raise ValueError("fuel_remaining_kg должен быть неотрицательным")


@dataclass(frozen=True)
class PowertrainMissionStep:
    time_start_s: float
    time_s: float
    dt_s: float
    segment_id: str
    requested_power_w: float
    ambient_temperature_c: float
    engine_rpm: float | None
    engine_required_torque_nm: float | None
    engine_max_torque_nm: float | None
    bus_voltage_v: float | None
    generator_current_a: float | None
    battery_current_a: float | None
    battery_soc_start: float
    battery_soc: float
    battery_voltage_v: float | None
    battery_temperature_c: float
    battery_surface_temperature_c: float
    generator_temperature_c: float
    rectifier_sink_temperature_c: float
    rectifier_case_temperature_c: float
    rectifier_junction_temperature_c: float
    generator_loss_w: float | None
    generator_winding_heat_w: float | None
    generator_phase_resistance_ohm: float | None
    rectifier_loss_w: float | None
    battery_loss_w: float | None
    generator_bus_power_w: float | None
    battery_power_w: float | None
    fuel_flow_kg_h: float | None
    fuel_model_status: str
    fuel_accounting_step_status: str
    active_limits: tuple[str, ...]
    operating_point_status: str
    demand_provenance: dict[str, Any]
    warnings: tuple[str, ...]
    requested_battery_current_a: float | None = None
    cold_charge_observer: dict[str, Any] | None = None
    ecms_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["active_limits"] = list(self.active_limits)
        data["warnings"] = list(self.warnings)
        if self.requested_battery_current_a is None:
            data.pop("requested_battery_current_a")
        if self.cold_charge_observer is None:
            data.pop("cold_charge_observer")
        if self.ecms_diagnostics is None:
            data.pop("ecms_diagnostics")
        return data


@dataclass(frozen=True)
class PowertrainMissionResult:
    completed: bool
    mission_terminated: bool
    termination_time_s: float
    termination_reason: str | None
    initial_soc: float
    final_soc: float
    minimum_soc: float
    peak_battery_temperature_c: float
    peak_generator_temperature_c: float
    peak_rectifier_junction_temperature_c: float
    peak_battery_current_a: float
    peak_generator_current_a: float
    total_load_energy_wh: float
    total_generator_bus_energy_wh: float
    net_battery_energy_wh: float
    total_generator_loss_wh: float
    total_rectifier_loss_wh: float
    total_battery_loss_wh: float
    mission_energy_balance_error_wh: float
    fuel_consumed_kg: float
    fuel_accounting_status: str
    active_limits_encountered: tuple[str, ...]
    warnings: tuple[str, ...]
    engine_provenance: str
    generator_provenance: str
    rectifier_provenance: str
    battery_provenance: str
    thermal_provenance: dict[str, str]
    fuel_provenance: str
    initial_state: PowertrainMissionState
    final_state: PowertrainMissionState
    trace: tuple[PowertrainMissionStep, ...]
    cold_charge_observer_summary: dict[str, Any] | None = None
    cold_charge_events: tuple[dict[str, Any], ...] = ()
    ecms_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["active_limits_encountered"] = list(self.active_limits_encountered)
        data["warnings"] = list(self.warnings)
        data["trace"] = [step.to_dict() for step in self.trace]
        if self.cold_charge_observer_summary is None:
            data.pop("cold_charge_observer_summary")
            data.pop("cold_charge_events")
        else:
            data["cold_charge_events"] = [dict(event)
                                          for event in self.cold_charge_events]
        if self.ecms_diagnostics is None:
            data.pop("ecms_diagnostics")
        return data


class PowertrainMissionSimulator:
    """Sequential wrapper around Phase 4B, BatteryModel and Phase 5 thermal models."""

    def __init__(self, powertrain: PowertrainOperatingPointSolver,
                 battery_thermal: ThermalModel,
                 generator_thermal: GeneratorThermalModel,
                 rectifier_thermal: RectifierThermalModel,
                 config: PowertrainMissionConfig,
                 cold_charge_observer: ColdChargeObserver | None = None):
        self.powertrain = powertrain
        self.battery = powertrain.dc_bus.battery
        self.battery_thermal = battery_thermal
        self.generator_thermal = generator_thermal
        self.rectifier_thermal = rectifier_thermal
        self.config = config
        self.cold_charge_observer = cold_charge_observer

    def simulate(self, segments: Iterable[PowertrainMissionSegment],
                 initial_state: PowertrainMissionState) -> PowertrainMissionResult:
        segment_list = tuple(segments)
        if not segment_list:
            raise ValueError("Миссия должна содержать хотя бы один участок")
        state = initial_state
        trace: list[PowertrainMissionStep] = []
        totals = {key: 0.0 for key in (
            "load", "generator_bus", "battery", "generator_loss",
            "rectifier_loss", "battery_loss")}
        fuel_consumed = 0.0
        fuel_status = "complete"
        adaptive_policy = (self.powertrain.config.operating_policy
                           == "soc_adaptive_minimum_equivalent_fuel_rate")
        if adaptive_policy and isinstance(
                self.powertrain.ecms, ProportionalSOCAdaptiveECMS):
            self.powertrain.ecms.reset()
        ecms_totals = (None if self.powertrain.config.operating_policy not in {
                           "minimum_equivalent_fuel_rate",
                           "soc_adaptive_minimum_equivalent_fuel_rate"} else {
                           "battery_discharge_energy_kwh": 0.0,
                           "battery_charge_energy_kwh": 0.0,
                           "integrated_ecms_cost_kg_equivalent": 0.0,
                           "selection_count_by_rpm": {},
                           "selected_rpm_values": [],
                           "adaptive": adaptive_policy,
                           "time_below_management_band_s": 0.0,
                           "time_inside_management_band_s": 0.0,
                           "time_above_management_band_s": 0.0,
                           "soc_weighted_sum": 0.0,
                           "soc_weight_s": 0.0,
                           "maximum_absolute_soc_error": 0.0,
                           "s_dis_values": [],
                           "s_chg_values": [],
                           "discharge_factor_clip_count": 0,
                           "charge_factor_clip_count": 0,
                           "rpm_switch_count": 0,
                           "suppressed_switch_count": 0,
                       })
        termination_reason: str | None = None
        active_limits: list[str] = []
        warnings = list(self.config.warnings)
        if self.cold_charge_observer is not None:
            self.cold_charge_observer.reset()

        for segment in segment_list:
            elapsed = 0.0
            while elapsed < segment.duration_s - 1e-12:
                requested_dt = min(self.config.dt_s, segment.duration_s - elapsed)
                operating = self.powertrain.solve(PowertrainOperatingPointInputs(
                    load_power_w=segment.required_electrical_power_w,
                    battery_soc=state.battery_soc,
                    battery_temperature_c=state.battery_core_temperature_c,
                    generator_winding_temperature_c=state.generator_temperature_c,
                    rectifier_case_temperature_c=state.rectifier_case_temperature_c,
                    previous_bus_voltage_v=state.previous_bus_voltage_v,
                    target_rpm=segment.target_rpm,
                ))
                if operating.operating_point_status != PowertrainSolverStatus.OK:
                    active_limits.extend(operating.active_limits)
                    termination_reason = self._infeasible_reason(operating)
                    warnings.extend(operating.warnings)
                    trace.append(self._failed_step(state, segment, operating))
                    break

                selected = operating.candidates[operating.selected_candidate]  # type: ignore[index]
                dc_point = selected.dc_bus_point
                if operating.generator_copper_loss_w is None:
                    termination_reason = "generator_temperature_not_evaluable"
                    warnings.append("generator_copper_loss_missing")
                    break
                if operating.rectifier_loss_w is None:
                    termination_reason = "rectifier_temperature_not_evaluable"
                    warnings.append("rectifier_loss_missing")
                    break
                dt, fuel_shortage = self._available_fuel_dt(
                    state, operating, requested_dt)
                if dt <= 0:
                    termination_reason = "fuel_depleted"
                    break

                battery_loss_w = (operating.battery_current_a ** 2
                                  * self.battery.resistance_at_temperature(
                                      state.battery_core_temperature_c))
                battery_temperatures = np.array([
                    state.battery_core_temperature_c,
                    state.battery_surface_temperature_c,
                ])
                bounds = self.battery_thermal.temperature_bounds(
                    battery_temperatures, battery_loss_w, dt,
                    segment.ambient_temperature_c)
                try:
                    new_soc, battery_voltage, battery_current, battery_loss_kw = (
                        self.battery.step(
                            state.battery_soc, operating.battery_power_w / 1000.0,
                            dt, state.battery_core_temperature_c,
                            state.battery_surface_temperature_c, bounds))
                except ValueError as exc:
                    termination_reason = self._battery_step_reason(
                        str(exc), state, operating)
                    warnings.append(f"battery_step_rejected:{exc}")
                    break
                if not math.isclose(battery_current, operating.battery_current_a,
                                    rel_tol=1e-7, abs_tol=1e-6):
                    raise RuntimeError("BatteryModel и DCBusSolver дали разные токи")
                if not math.isclose(battery_loss_kw * 1000.0, battery_loss_w,
                                    rel_tol=1e-7, abs_tol=1e-6):
                    raise RuntimeError("Battery loss не согласован с BatteryModel")
                cold_charge_step = None
                if self.cold_charge_observer is not None:
                    # The present quasi-static solver exposes the accepted
                    # battery-branch request.  A rejected unconstrained request
                    # is not reconstructed from a neighbouring infeasible root.
                    cold_charge_step = self.cold_charge_observer.step(
                        ColdChargeObserverInput(
                            dt_s=dt,
                            battery_temperature_c=(
                                state.battery_core_temperature_c),
                            battery_soc=state.battery_soc,
                            requested_battery_current_a=battery_current,
                            actual_battery_current_a=battery_current,
                            battery_capacity_ah=self.battery.q_ah,
                            battery_voltage_v=battery_voltage,
                            cell_voltage_v=(battery_voltage
                                            / self.battery.config["series_cells"]),
                            soc_charge_limit=self.battery.config["soc_target"],
                            cell_voltage_charge_limit_v=(
                                self.battery.config["max_cell_v"]),
                        ))
                new_battery_temperatures, _ = self.battery_thermal.step(
                    battery_temperatures, battery_loss_w, dt,
                    segment.ambient_temperature_c)

                end_time = state.time_s + dt
                generator_step = self.generator_thermal.step(
                    ThermalState(state.time_s, {
                        "generator_winding": state.generator_temperature_c}),
                    ThermalSimulationInput(
                        time_s=end_time,
                        loss_power_w=operating.generator_copper_loss_w,
                        rpm=operating.generator_rpm or 0.0,
                        ambient_temperature_c=segment.ambient_temperature_c,
                        copper_loss_w=operating.generator_copper_loss_w,
                        iron_loss_w=self.powertrain.dc_bus.generator.profile.constant_iron_loss_w,
                        mechanical_loss_w=(
                            self.powertrain.dc_bus.generator.profile.constant_mechanical_loss_w),
                        loss_source_type=(
                            self.powertrain.dc_bus.generator.profile.source_type),
                    ))
                rectifier_state = ThermalState(state.time_s, {
                    "rectifier_heatsink": state.rectifier_sink_temperature_c,
                    "rectifier_case": state.rectifier_case_temperature_c,
                    **{f"rectifier_junction_{index}":
                       state.rectifier_junction_temperature_c for index in range(6)},
                })
                rectifier_step = self.rectifier_thermal.step(
                    rectifier_state,
                    ThermalSimulationInput(
                        time_s=end_time,
                        loss_power_w=operating.rectifier_loss_w,
                        rpm=operating.generator_rpm or 0.0,
                        ambient_temperature_c=segment.ambient_temperature_c,
                        loss_source_type=(
                            self.powertrain.dc_bus.rectifier.profile.source_type),
                    ))
                if generator_step.status == ThermalStatus.NOT_EVALUABLE:
                    termination_reason = "generator_temperature_not_evaluable"
                    warnings.extend(generator_step.warnings)
                    break
                if rectifier_step.status == ThermalStatus.NOT_EVALUABLE:
                    termination_reason = "rectifier_temperature_not_evaluable"
                    warnings.extend(rectifier_step.warnings)
                    break

                generator_temperature = generator_step.state.node_temperatures_c[
                    "generator_winding"]
                rectifier_sink = rectifier_step.state.node_temperatures_c[
                    "rectifier_heatsink"]
                rectifier_case = rectifier_step.state.node_temperatures_c[
                    "rectifier_case"]
                rectifier_junction = rectifier_step.state.node_temperatures_c[
                    "rectifier_junction_0"]
                new_fuel, consumed, step_fuel_status = self._update_fuel(
                    state.fuel_remaining_kg, operating, dt)
                fuel_consumed += consumed
                fuel_status = self._merge_fuel_status(fuel_status, step_fuel_status)

                generator_loss = self._generator_loss_w(operating)
                factor = dt / 3600.0
                totals["load"] += operating.load_power_w * factor
                totals["generator_bus"] += dc_point.generator_power_to_bus_w * factor
                totals["battery"] += operating.battery_power_w * factor
                totals["generator_loss"] += generator_loss * factor
                totals["rectifier_loss"] += operating.rectifier_loss_w * factor
                totals["battery_loss"] += battery_loss_w * factor
                selected_ecms = (None if operating.ecms_selection is None else
                                 operating.ecms_selection.get("selected_cost"))
                adaptive_step = (None if operating.ecms_selection is None else
                                 operating.ecms_selection.get(
                                     "adaptive_diagnostics"))
                if ecms_totals is not None and selected_ecms is not None:
                    power_kw = float(selected_ecms["battery_power_kw"])
                    energy_kwh = power_kw * dt / 3600.0
                    if power_kw > 0:
                        ecms_totals["battery_discharge_energy_kwh"] += energy_kwh
                    elif power_kw < 0:
                        ecms_totals["battery_charge_energy_kwh"] += -energy_kwh
                    ecms_totals["integrated_ecms_cost_kg_equivalent"] += (
                        float(selected_ecms["ecms_equivalent_fuel_rate_kg_h"])
                        * dt / 3600.0)
                    rpm_key = f"{operating.engine_rpm:g}"
                    counts = ecms_totals["selection_count_by_rpm"]
                    counts[rpm_key] = counts.get(rpm_key, 0) + 1
                    ecms_totals["selected_rpm_values"].append(operating.engine_rpm)
                    if adaptive_step is not None:
                        band = adaptive_step["soc_management_state"]
                        if band == "below_management_band":
                            ecms_totals["time_below_management_band_s"] += dt
                        elif band == "above_management_band":
                            ecms_totals["time_above_management_band_s"] += dt
                        else:
                            ecms_totals["time_inside_management_band_s"] += dt
                        ecms_totals["soc_weighted_sum"] += adaptive_step["soc"] * dt
                        ecms_totals["soc_weight_s"] += dt
                        ecms_totals["maximum_absolute_soc_error"] = max(
                            ecms_totals["maximum_absolute_soc_error"],
                            abs(adaptive_step["soc_error"]))
                        ecms_totals["s_dis_values"].append(
                            adaptive_step["s_dis_effective"])
                        ecms_totals["s_chg_values"].append(
                            adaptive_step["s_chg_effective"])
                        ecms_totals["discharge_factor_clip_count"] += int(
                            adaptive_step["s_dis_clipped"])
                        ecms_totals["charge_factor_clip_count"] += int(
                            adaptive_step["s_chg_clipped"])
                        ecms_totals["rpm_switch_count"] = adaptive_step[
                            "switch_count"]
                        ecms_totals["suppressed_switch_count"] = adaptive_step[
                            "suppressed_switch_count"]

                new_state = PowertrainMissionState(
                    time_s=end_time, battery_soc=new_soc,
                    battery_core_temperature_c=float(new_battery_temperatures[0]),
                    battery_surface_temperature_c=float(new_battery_temperatures[1]),
                    generator_temperature_c=generator_temperature,
                    rectifier_sink_temperature_c=rectifier_sink,
                    rectifier_case_temperature_c=rectifier_case,
                    rectifier_junction_temperature_c=rectifier_junction,
                    fuel_remaining_kg=new_fuel,
                    previous_bus_voltage_v=operating.bus_voltage_v,
                    previous_engine_rpm=operating.engine_rpm,
                )
                step_warnings = tuple(dict.fromkeys((
                    *operating.warnings, *generator_step.warnings,
                    *rectifier_step.warnings,
                    "shaft_transient_model_not_implemented",
                )))
                trace.append(PowertrainMissionStep(
                    time_start_s=state.time_s, time_s=end_time, dt_s=dt,
                    segment_id=segment.segment_id,
                    requested_power_w=segment.required_electrical_power_w,
                    ambient_temperature_c=segment.ambient_temperature_c,
                    engine_rpm=operating.engine_rpm,
                    engine_required_torque_nm=operating.engine_required_torque_nm,
                    engine_max_torque_nm=operating.engine_max_torque_nm,
                    bus_voltage_v=operating.bus_voltage_v,
                    generator_current_a=operating.generator_dc_current_a,
                    battery_current_a=battery_current,
                    battery_soc_start=state.battery_soc, battery_soc=new_soc,
                    battery_voltage_v=battery_voltage,
                    battery_temperature_c=new_state.battery_core_temperature_c,
                    battery_surface_temperature_c=new_state.battery_surface_temperature_c,
                    generator_temperature_c=generator_temperature,
                    rectifier_sink_temperature_c=rectifier_sink,
                    rectifier_case_temperature_c=rectifier_case,
                    rectifier_junction_temperature_c=rectifier_junction,
                    generator_loss_w=generator_loss,
                    generator_winding_heat_w=operating.generator_copper_loss_w,
                    generator_phase_resistance_ohm=(
                        self.powertrain.dc_bus.generator.phase_resistance_at_temperature(
                            state.generator_temperature_c)),
                    rectifier_loss_w=operating.rectifier_loss_w,
                    battery_loss_w=battery_loss_w,
                    generator_bus_power_w=dc_point.generator_power_to_bus_w,
                    battery_power_w=operating.battery_power_w,
                    fuel_flow_kg_h=operating.fuel_flow_kg_h,
                    fuel_model_status=operating.fuel_model_status,
                    fuel_accounting_step_status=step_fuel_status,
                    active_limits=operating.active_limits,
                    operating_point_status=operating.operating_point_status.value,
                    demand_provenance=dict(segment.demand_provenance or {}),
                    warnings=step_warnings,
                    requested_battery_current_a=(
                        battery_current if cold_charge_step is not None else None),
                    cold_charge_observer=(
                        cold_charge_step.to_dict()
                        if cold_charge_step is not None else None),
                    ecms_diagnostics=(
                        selected_ecms if adaptive_step is None else
                        {**selected_ecms, **adaptive_step}),
                ))
                active_limits.extend(operating.active_limits)
                warnings.extend(step_warnings)
                state = new_state
                elapsed += dt

                termination_reason = self._post_step_termination(
                    state, operating, generator_step.status,
                    rectifier_step.status, bounds)
                if fuel_shortage:
                    termination_reason = "fuel_depleted"
                if termination_reason is not None:
                    break
            if termination_reason is not None:
                break

        return self._result(
            initial_state, state, trace, totals, fuel_consumed, fuel_status,
            termination_reason, active_limits, warnings, ecms_totals)

    def _failed_step(self, state: PowertrainMissionState,
                     segment: PowertrainMissionSegment,
                     operating: PowertrainOperatingPoint) -> PowertrainMissionStep:
        return PowertrainMissionStep(
            time_start_s=state.time_s, time_s=state.time_s, dt_s=0.0,
            segment_id=segment.segment_id,
            requested_power_w=segment.required_electrical_power_w,
            ambient_temperature_c=segment.ambient_temperature_c,
            engine_rpm=None, engine_required_torque_nm=None, engine_max_torque_nm=None,
            bus_voltage_v=None, generator_current_a=None, battery_current_a=None,
            battery_soc_start=state.battery_soc, battery_soc=state.battery_soc,
            battery_voltage_v=None,
            battery_temperature_c=state.battery_core_temperature_c,
            battery_surface_temperature_c=state.battery_surface_temperature_c,
            generator_temperature_c=state.generator_temperature_c,
            rectifier_sink_temperature_c=state.rectifier_sink_temperature_c,
            rectifier_case_temperature_c=state.rectifier_case_temperature_c,
            rectifier_junction_temperature_c=state.rectifier_junction_temperature_c,
            generator_loss_w=None, generator_winding_heat_w=None,
            generator_phase_resistance_ohm=None,
            rectifier_loss_w=None, battery_loss_w=None,
            generator_bus_power_w=None, battery_power_w=None,
            fuel_flow_kg_h=None, fuel_model_status="not_available",
            fuel_accounting_step_status="not_available",
            active_limits=operating.active_limits,
            operating_point_status=operating.operating_point_status.value,
            demand_provenance=dict(segment.demand_provenance or {}),
            warnings=operating.warnings,
        )

    def _available_fuel_dt(self, state: PowertrainMissionState,
                           operating: PowertrainOperatingPoint,
                           requested_dt: float) -> tuple[float, bool]:
        if state.fuel_remaining_kg is None or operating.fuel_flow_kg_h is None:
            return requested_dt, False
        if state.fuel_remaining_kg <= 0:
            return 0.0, True
        required = operating.fuel_flow_kg_h * requested_dt / 3600.0
        if required <= state.fuel_remaining_kg + 1e-15:
            return requested_dt, False
        return state.fuel_remaining_kg * 3600.0 / operating.fuel_flow_kg_h, True

    @staticmethod
    def _update_fuel(fuel_remaining_kg: float | None,
                     operating: PowertrainOperatingPoint,
                     dt: float) -> tuple[float | None, float, str]:
        status = operating.fuel_model_status
        if status in {"part_load_map_missing", "part_load_map_out_of_range"}:
            return fuel_remaining_kg, 0.0, "incomplete_due_to_part_load_map"
        if operating.fuel_flow_kg_h is None:
            return fuel_remaining_kg, 0.0, "not_available"
        consumed = operating.fuel_flow_kg_h * dt / 3600.0
        remaining = (None if fuel_remaining_kg is None
                     else max(0.0, fuel_remaining_kg - consumed))
        if status == "full_load_supported":
            return remaining, consumed, "complete"
        if status in {
                "near_full_load_estimate", "part_load_scientific_analogue",
                "near_full_load_blend", "high_speed_part_load_transition",
                "very_low_load_scientific_analogue"}:
            return remaining, consumed, "estimated"
        return remaining, 0.0, "not_available"

    @staticmethod
    def _merge_fuel_status(current: str, update: str) -> str:
        order = {
            "complete": 0, "estimated": 1, "not_available": 2,
            "incomplete_due_to_part_load_map": 3,
        }
        return update if order[update] > order[current] else current

    @staticmethod
    def _generator_loss_w(operating: PowertrainOperatingPoint) -> float:
        copper = operating.generator_copper_loss_w or 0.0
        other = operating.generator_other_losses_w
        return copper if other is None else copper + other

    def _post_step_termination(self, state: PowertrainMissionState,
                               operating: PowertrainOperatingPoint,
                               generator_status: ThermalStatus,
                               rectifier_status: ThermalStatus,
                               battery_bounds: tuple[float, float]) -> str | None:
        if max(battery_bounds) >= self.config.battery_temperature_limit_c:
            return "battery_temperature_limit"
        if generator_status == ThermalStatus.THERMAL_LIMIT:
            return "generator_temperature_limit"
        if rectifier_status == ThermalStatus.THERMAL_LIMIT:
            return "rectifier_temperature_limit"
        current = operating.battery_current_a or 0.0
        if (current > 0 and state.battery_soc <= self.battery.soc_floor_at_current(
                current, state.battery_core_temperature_c) + 1e-10):
            return "battery_depleted"
        minimum = self.battery.config["series_cells"] * self.battery.config["min_cell_v"]
        maximum = self.battery.config["series_cells"] * self.battery.config["max_cell_v"]
        if (operating.bus_voltage_v is not None
                and not minimum <= operating.bus_voltage_v <= maximum):
            return "battery_voltage_limit"
        return None

    @staticmethod
    def _infeasible_reason(operating: PowertrainOperatingPoint) -> str:
        limits = set(operating.active_limits)
        if limits and limits <= {"battery_soc_discharge_limit"}:
            return "battery_depleted"
        if limits and limits <= {
                "battery_min_voltage_limit", "battery_max_voltage_limit",
                "bus_undervoltage", "bus_overvoltage"}:
            return "battery_voltage_limit"
        return "no_feasible_powertrain_operating_point"

    def _battery_step_reason(self, message: str, state: PowertrainMissionState,
                             operating: PowertrainOperatingPoint) -> str:
        lowered = message.lower()
        current = operating.battery_current_a or 0.0
        floor = self.battery.soc_floor_at_current(
            max(0.0, current), state.battery_core_temperature_c)
        if current > 0 and state.battery_soc - floor < 1e-3:
            return "battery_depleted"
        if "заряд" in lowered or "soc" in lowered:
            return "battery_depleted"
        if "мощност" in lowered or "напряж" in lowered:
            return "battery_voltage_limit"
        return "battery_depleted"

    def _result(self, initial: PowertrainMissionState,
                final: PowertrainMissionState,
                trace: list[PowertrainMissionStep], totals: dict[str, float],
                fuel_consumed: float, fuel_status: str,
                termination_reason: str | None, active_limits: list[str],
                 warnings: list[str],
                 ecms_totals: dict[str, Any] | None = None) -> PowertrainMissionResult:
        completed = termination_reason is None
        battery_temperatures = [initial.battery_core_temperature_c,
                                initial.battery_surface_temperature_c]
        generator_temperatures = [initial.generator_temperature_c]
        rectifier_temperatures = [initial.rectifier_junction_temperature_c]
        battery_currents: list[float] = []
        generator_currents: list[float] = []
        soc_values = [initial.battery_soc]
        for step in trace:
            battery_temperatures.extend((step.battery_temperature_c,
                                         step.battery_surface_temperature_c))
            generator_temperatures.append(step.generator_temperature_c)
            rectifier_temperatures.append(step.rectifier_junction_temperature_c)
            soc_values.append(step.battery_soc)
            if step.battery_current_a is not None:
                battery_currents.append(abs(step.battery_current_a))
            if step.generator_current_a is not None:
                generator_currents.append(abs(step.generator_current_a))
        provenance = self._provenance(fuel_status)
        ecms_diagnostics = None
        if ecms_totals is not None:
            rpm_values = ecms_totals.pop("selected_rpm_values")
            discharge = ecms_totals["battery_discharge_energy_kwh"]
            charge = ecms_totals["battery_charge_energy_kwh"]
            profile = self.powertrain.ecms.profile if self.powertrain.ecms else None
            ecms_diagnostics = {
                "ecms_profile_id": None if profile is None else profile.profile_id,
                "total_physical_fuel_kg": fuel_consumed,
                "initial_soc": initial.battery_soc,
                "final_soc": final.battery_soc,
                "delta_soc": final.battery_soc - initial.battery_soc,
                "battery_discharge_energy_kwh": discharge,
                "battery_charge_energy_kwh": charge,
                "net_battery_energy_kwh": discharge - charge,
                "integrated_equivalent_fuel_kg": ecms_totals[
                    "integrated_ecms_cost_kg_equivalent"],
                "integrated_ecms_cost_kg_equivalent": ecms_totals[
                    "integrated_ecms_cost_kg_equivalent"],
                "selected_rpm_statistics": ({
                    "minimum_rpm": min(rpm_values),
                    "maximum_rpm": max(rpm_values),
                    "mean_rpm": sum(rpm_values) / len(rpm_values),
                } if rpm_values else {}),
                "selection_count_by_rpm": ecms_totals[
                    "selection_count_by_rpm"],
                "powertrain_profile_readiness": (
                    self.powertrain.powertrain_profile_readiness()),
                "equivalent_fuel_is_control_cost_not_physical_consumption": True,
            }
            if ecms_totals["adaptive"]:
                dis_values = ecms_totals["s_dis_values"]
                chg_values = ecms_totals["s_chg_values"]
                weight = ecms_totals["soc_weight_s"]
                ecms_diagnostics.update({
                    "minimum_soc": min(soc_values),
                    "maximum_soc": max(soc_values),
                    "time_below_management_band_s": ecms_totals[
                        "time_below_management_band_s"],
                    "time_inside_management_band_s": ecms_totals[
                        "time_inside_management_band_s"],
                    "time_above_management_band_s": ecms_totals[
                        "time_above_management_band_s"],
                    "mean_soc": (ecms_totals["soc_weighted_sum"] / weight
                                 if weight else initial.battery_soc),
                    "maximum_absolute_soc_error": ecms_totals[
                        "maximum_absolute_soc_error"],
                    "s_dis_effective_min": min(dis_values, default=None),
                    "s_dis_effective_max": max(dis_values, default=None),
                    "s_chg_effective_min": min(chg_values, default=None),
                    "s_chg_effective_max": max(chg_values, default=None),
                    "discharge_factor_clip_count": ecms_totals[
                        "discharge_factor_clip_count"],
                    "charge_factor_clip_count": ecms_totals[
                        "charge_factor_clip_count"],
                    "rpm_switch_count": ecms_totals["rpm_switch_count"],
                    "suppressed_switch_count": ecms_totals[
                        "suppressed_switch_count"],
                })
        return PowertrainMissionResult(
            completed=completed, mission_terminated=not completed,
            termination_time_s=final.time_s,
            termination_reason=termination_reason,
            initial_soc=initial.battery_soc, final_soc=final.battery_soc,
            minimum_soc=min(soc_values),
            peak_battery_temperature_c=max(battery_temperatures),
            peak_generator_temperature_c=max(generator_temperatures),
            peak_rectifier_junction_temperature_c=max(rectifier_temperatures),
            peak_battery_current_a=max(battery_currents, default=0.0),
            peak_generator_current_a=max(generator_currents, default=0.0),
            total_load_energy_wh=totals["load"],
            total_generator_bus_energy_wh=totals["generator_bus"],
            net_battery_energy_wh=totals["battery"],
            total_generator_loss_wh=totals["generator_loss"],
            total_rectifier_loss_wh=totals["rectifier_loss"],
            total_battery_loss_wh=totals["battery_loss"],
            mission_energy_balance_error_wh=(
                totals["generator_bus"] + totals["battery"] - totals["load"]),
            fuel_consumed_kg=fuel_consumed,
            fuel_accounting_status=fuel_status,
            active_limits_encountered=tuple(dict.fromkeys(active_limits)),
            warnings=tuple(dict.fromkeys(warnings)),
            engine_provenance=provenance["engine"],
            generator_provenance=provenance["generator"],
            rectifier_provenance=provenance["rectifier"],
            battery_provenance=provenance["battery"],
            thermal_provenance=provenance["thermal"],
            fuel_provenance=provenance["fuel"],
            initial_state=initial, final_state=final, trace=tuple(trace),
            cold_charge_observer_summary=(
                None if self.cold_charge_observer is None else
                self.cold_charge_observer.summary().to_dict()),
            cold_charge_events=(
                () if self.cold_charge_observer is None else
                tuple(event.to_dict()
                      for event in self.cold_charge_observer.events)),
            ecms_diagnostics=ecms_diagnostics,
        )

    def _provenance(self, fuel_status: str) -> dict[str, Any]:
        return {
            "engine": self.powertrain.engine.profile.source_type,
            "generator": self.powertrain.dc_bus.generator.profile.source_type,
            "rectifier": self.powertrain.dc_bus.rectifier.profile.source_type,
            "battery": self.battery.config.get("source_type", "unknown"),
            "thermal": {
                "battery": "engineering_model",
                "generator": self.generator_thermal.profile.source_type,
                "rectifier_junction_case": "manufacturer_parameter",
                "rectifier_heatsink": self.rectifier_thermal.heatsink.source_type,
            },
            "fuel": ("incomplete_due_to_part_load_map"
                     if fuel_status == "incomplete_due_to_part_load_map"
                     else "engineering_assumption" if fuel_status == "estimated"
                     else "digitized_manufacturer_graph"),
        }


def load_synthetic_mission_fixture(path: str | Path | None = None,
                                   *, operating_policy: str | None = None,
                                   ecms_profile: (ECMSProfile |
                                                  AdaptiveECMSProfile | None) = None,
                                   solver_overrides: dict[str, Any] | None = None
                                   ) -> tuple[PowertrainMissionSimulator,
                                              PowertrainMissionState]:
    fixture_path = (Path(path) if path is not None else BUILTIN_POWERTRAIN_DIR
                    / f"{SYNTHETIC_MISSION_FIXTURE_ID}.json")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if data.get("source_type") != "synthetic_test_fixture":
        raise ValueError("Mission fixture должен быть synthetic_test_fixture")
    powertrain_fixture = load_synthetic_powertrain_fixture()
    solver_config = replace(
        powertrain_fixture.solver_config,
        **data.get("powertrain_solver_overrides", {}))
    if solver_overrides:
        solver_config = replace(solver_config, **solver_overrides)
    if operating_policy is not None:
        solver_config = replace(solver_config, operating_policy=operating_policy)
    controller = None
    if isinstance(ecms_profile, ECMSProfile):
        controller = FixedFactorECMS(ecms_profile)
    elif isinstance(ecms_profile, AdaptiveECMSProfile):
        controller = ProportionalSOCAdaptiveECMS(ecms_profile)
    solver = PowertrainOperatingPointSolver(
        powertrain_fixture.engine_model, powertrain_fixture.dc_bus_solver,
        solver_config, controller)
    battery_thermal = ThermalModel(
        solver.dc_bus.battery.config["mass_kg"], data["battery_thermal"])
    generator_thermal = GeneratorThermalModel(
        GeneratorThermalProfile.load_builtin(data["generator_thermal_profile_id"]))
    rectifier_thermal = RectifierThermalModel(
        solver.dc_bus.rectifier,
        HeatSinkProfile.load_builtin(data["heatsink_profile_id"]),
        data["rectifier_thermal_mode"])
    mission = data["mission_config"]
    simulator = PowertrainMissionSimulator(
        solver, battery_thermal, generator_thermal, rectifier_thermal,
        PowertrainMissionConfig(
            dt_s=float(mission["dt_s"]),
            battery_temperature_limit_c=float(
                mission["battery_temperature_limit_c"]),
            battery_temperature_limit_source_type=(
                mission["battery_temperature_limit_source_type"]),
            shaft_transient_model=mission["shaft_transient_model"],
            source_type=data["source_type"],
            warnings=tuple(data.get("warnings", ())),
        ))
    initial = data["initial_state"]
    state = PowertrainMissionState(**initial)
    return simulator, state

