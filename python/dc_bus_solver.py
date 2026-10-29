"""Autonomous static Phase 4A DC-bus solver.

The solver coordinates the existing generator, rectifier and battery models.
It does not select generator speed, advance battery state, update temperatures
or participate in the legacy mission simulation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Callable

from battery_model import BatteryModel
from generator_model import (
    GeneratorModel,
    GeneratorOperatingPoint,
    IDEAL_SIX_PULSE_LINE_RMS_CURRENT_FACTOR,
    IDEAL_SIX_PULSE_VOLTAGE_FACTOR,
)
from rectifier_model import RectifierModel, RectifierOperatingPoint


ROOT = Path(__file__).resolve().parent
BUILTIN_DC_BUS_DIR = ROOT / "data" / "dc_bus"
SYNTHETIC_FIXTURE_ID = "synthetic_dc_bus_fixture_v1"


class DCBusSolverStatus(str, Enum):
    OK = "ok"
    INFEASIBLE = "infeasible"
    NO_ROOT = "no_root"
    UNSTABLE_ONLY = "unstable_only"
    NOT_READY = "not_ready"


class DCBusBranchRangeError(ValueError):
    """The generator branch needs current outside its defined model domain."""


@dataclass(frozen=True)
class DCBusSolverConfig:
    source_type: str
    minimum_search_voltage_v: float
    component_bus_voltage_max_v: float
    bus_undervoltage_v: float
    bus_overvoltage_v: float
    voltage_scan_points: int = 801
    root_voltage_tolerance_v: float = 1e-9
    branch_current_tolerance_a: float = 1e-9
    current_balance_tolerance_a: float = 1e-9
    balance_tolerance_w: float = 1e-5
    maximum_iterations: int = 100
    stability_delta_v: float = 1e-3
    commutation_overlap_model: str = "disabled_missing_inductance"
    reverse_voltage_model: str = "static_ideal_estimate"
    validation_mode: str = "synthetic_physics_validation"
    allow_rectifier_characteristic_without_zero: bool = False

    def __post_init__(self) -> None:
        if self.validation_mode not in {
            "synthetic_physics_validation", "engineering_simulation",
        }:
            raise ValueError("Неизвестный validation_mode DC-шины")
        if (self.validation_mode == "synthetic_physics_validation"
                and self.source_type != "synthetic_test_fixture"):
            raise ValueError("Synthetic validation требует synthetic_test_fixture")
        if not isinstance(self.allow_rectifier_characteristic_without_zero, bool):
            raise ValueError("Флаг диапазона Vf должен быть boolean")
        positive = (
            self.minimum_search_voltage_v, self.component_bus_voltage_max_v,
            self.bus_undervoltage_v, self.bus_overvoltage_v,
            self.root_voltage_tolerance_v, self.branch_current_tolerance_a,
            self.current_balance_tolerance_a, self.balance_tolerance_w,
            self.stability_delta_v,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Параметры DCBusSolver должны быть положительными")
        if not (self.minimum_search_voltage_v < self.component_bus_voltage_max_v
                and self.bus_undervoltage_v < self.bus_overvoltage_v):
            raise ValueError("Неверные границы напряжения DC-шины")
        if self.voltage_scan_points < 3 or self.maximum_iterations < 1:
            raise ValueError("Слишком мало точек или итераций solver")
        if self.commutation_overlap_model != "disabled_missing_inductance":
            raise ValueError("Phase 4A требует disabled_missing_inductance")
        if self.reverse_voltage_model != "static_ideal_estimate":
            raise ValueError("Phase 4A поддерживает только static_ideal_estimate")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DCBusSolverConfig":
        return cls(**data)


@dataclass(frozen=True)
class DCBusInputs:
    engine_or_generator_speed_rpm: float
    load_power_w: float
    battery_soc: float
    battery_temperature_c: float
    generator_winding_temperature_c: float
    rectifier_case_temperature_c: float
    previous_bus_voltage_v: float | None = None


@dataclass(frozen=True)
class GeneratorBranchSolution:
    current_dc_a: float
    conducting: bool
    voltage_residual_v: float
    iterations: int
    generator_point: GeneratorOperatingPoint
    rectifier_point: RectifierOperatingPoint
    reverse_voltage_static_estimate_v: float


@dataclass(frozen=True)
class DCBusRoot:
    bus_voltage_v: float
    generator_current_dc_a: float
    battery_current_a: float
    load_current_a: float
    current_balance_error_a: float
    derivative_a_per_v: float | None
    stability: str
    constraint_feasible: bool
    active_limits: tuple[str, ...]
    margins: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["active_limits"] = list(self.active_limits)
        return data


@dataclass(frozen=True)
class DCBusOperatingPoint:
    solver_status: DCBusSolverStatus
    bus_voltage_v: float | None
    generator_current_dc_a: float | None
    generator_power_to_bus_w: float | None
    battery_current_a: float | None
    battery_power_w: float | None
    load_current_a: float | None
    generator_emf_ll_rms_v: float | None
    generator_ideal_rectified_voltage_v: float | None
    generator_copper_loss_w: float | None
    rectifier_drop_v: float | None
    rectifier_loss_w: float | None
    rectifier_junction_temperature_c: float | None
    generator_electromagnetic_power_w: float | None
    generator_shaft_power_w: float | None
    generator_mechanical_torque_nm: float | None
    generator_branch_balance_error_w: float | None
    electrical_balance_error_w: float | None
    reverse_voltage_static_estimate_v: float | None
    reverse_voltage_model: str
    commutation_overlap_model: str
    generator_conducting: bool
    battery_charging: bool
    roots_found: tuple[DCBusRoot, ...]
    selected_root: int | None
    root_stability: str | None
    active_limits: tuple[str, ...]
    margins: dict[str, float | None]
    search_voltage_range_v: tuple[float, float] | None
    iterations: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["solver_status"] = self.solver_status.value
        data["roots_found"] = [root.to_dict() for root in self.roots_found]
        data["active_limits"] = list(self.active_limits)
        data["warnings"] = list(self.warnings)
        if self.search_voltage_range_v is not None:
            data["search_voltage_range_v"] = list(self.search_voltage_range_v)
        return data


@dataclass(frozen=True)
class SyntheticDCBusFixture:
    fixture_id: str
    source_type: str
    generator_model: GeneratorModel
    rectifier_model: RectifierModel
    battery_model: BatteryModel
    solver_config: DCBusSolverConfig
    warning: str


@dataclass(frozen=True)
class _BusState:
    bus_voltage_v: float
    generator_branch: GeneratorBranchSolution
    battery_current_a: float
    load_current_a: float
    current_residual_a: float


class DCBusSolver:
    """Find every static current-balance root at externally prescribed rpm."""

    def __init__(self, generator_model: GeneratorModel, rectifier_model: RectifierModel,
                 battery_model: BatteryModel, config: DCBusSolverConfig):
        self.generator = generator_model
        self.rectifier = rectifier_model
        self.battery = battery_model
        self.config = config

    def readiness_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        synthetic_mode = self.config.validation_mode == "synthetic_physics_validation"
        if synthetic_mode and self.generator.profile.source_type != "synthetic_test_fixture":
            errors.append("generator_not_synthetic_test_fixture")
        if not self.generator.profile.readiness.phase2_ready:
            errors.append("generator_not_ready")
        if self.generator.profile.three_phase_system_count != 1:
            errors.append("phase4a_requires_one_three_phase_system")
        if synthetic_mode and self.rectifier.profile.source_type != "synthetic_test_fixture":
            errors.append("rectifier_not_synthetic_test_fixture")
        if not self.rectifier.profile.readiness.phase3_ready:
            errors.append("rectifier_not_ready")
        currents = {point.forward_current_a
                    for point in self.rectifier.profile.effective_vf_points()}
        if (0.0 not in currents
                and not self.config.allow_rectifier_characteristic_without_zero):
            errors.append("rectifier_vf_domain_must_include_zero_current")
        if synthetic_mode and self.battery.config.get("source_type") != "synthetic_test_fixture":
            errors.append("battery_not_synthetic_test_fixture")
        return tuple(errors)

    @staticmethod
    def _validate_inputs(inputs: DCBusInputs) -> None:
        values = (
            inputs.engine_or_generator_speed_rpm, inputs.load_power_w,
            inputs.battery_soc, inputs.battery_temperature_c,
            inputs.generator_winding_temperature_c,
            inputs.rectifier_case_temperature_c,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Входы DCBusSolver должны быть конечными")
        if inputs.engine_or_generator_speed_rpm <= 0 or inputs.load_power_w < 0:
            raise ValueError("Phase 4A требует положительных оборотов и неотрицательной мощности")
        if not 0 <= inputs.battery_soc <= 1:
            raise ValueError("SOC должен находиться в диапазоне 0...1")
        if (inputs.previous_bus_voltage_v is not None
                and (not math.isfinite(inputs.previous_bus_voltage_v)
                     or inputs.previous_bus_voltage_v <= 0)):
            raise ValueError("previous_bus_voltage_v должно быть положительным")

    def evaluate_battery_current_for_bus_voltage(
            self, bus_voltage_v: float, soc: float, temperature_c: float) -> float:
        """Evaluate immutable OCV-R branch current; positive means discharge."""
        voltage = float(bus_voltage_v)
        if not math.isfinite(voltage) or voltage <= 0:
            raise ValueError("V_bus должно быть положительным конечным числом")
        if not math.isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("SOC должен находиться в диапазоне 0...1")
        ocv = self.battery.ocv(soc)
        resistance = self.battery.resistance_at_temperature(temperature_c)
        return (ocv - voltage) / resistance

    def _reverse_voltage(self, rpm: float) -> tuple[float, float, float]:
        emf_ll = self.generator.emf_line_to_line_rms(rpm)
        ideal_dc = IDEAL_SIX_PULSE_VOLTAGE_FACTOR * emf_ll
        reverse = math.sqrt(2.0) * emf_ll
        return emf_ll, ideal_dc, reverse

    def _maximum_branch_current_a(self) -> float:
        peak_line = self.generator.profile.peak_line_current_rms_a
        if peak_line is None:
            raise DCBusBranchRangeError("generator_peak_current_missing")
        systems = self.generator.profile.three_phase_system_count
        generator_domain = systems * peak_line / IDEAL_SIX_PULSE_LINE_RMS_CURRENT_FACTOR
        points = self.rectifier.profile.effective_vf_points()
        if not points:
            raise DCBusBranchRangeError("rectifier_vf_table_missing")
        rectifier_domain = max(point.forward_current_a for point in points)
        return min(generator_domain, rectifier_domain)

    def solve_generator_branch_current(
            self, bus_voltage_v: float, rpm: float, winding_temperature_c: float,
            rectifier_case_temperature_c: float) -> GeneratorBranchSolution:
        """Solve the nonlinear one-way generator/rectifier branch at fixed Vbus."""
        voltage = float(bus_voltage_v)
        if not math.isfinite(voltage) or voltage <= 0:
            raise ValueError("V_bus должно быть положительным")
        emf_ll, ideal_dc, reverse = self._reverse_voltage(rpm)
        if rpm == 0:
            raise DCBusBranchRangeError("generator_operating_point_requires_positive_rpm")
        generator_zero = self.generator.operating_point(rpm, 0.0, winding_temperature_c)
        rectifier_zero = self.rectifier.operating_point(
            0.0, rectifier_case_temperature_c, reverse)
        points = self.rectifier.profile.effective_vf_points()
        minimum_confirmed_current = min(point.forward_current_a for point in points)
        if minimum_confirmed_current == 0:
            vf_zero = self.rectifier.forward_voltage(0.0, rectifier_case_temperature_c)
            residual_lower = ideal_dc - 2.0 * vf_zero - voltage
            lower = 0.0
        else:
            if ideal_dc <= voltage:
                return GeneratorBranchSolution(
                    current_dc_a=0.0, conducting=False,
                    voltage_residual_v=ideal_dc - voltage, iterations=0,
                    generator_point=generator_zero, rectifier_point=rectifier_zero,
                    reverse_voltage_static_estimate_v=reverse,
                )
            if not self.config.allow_rectifier_characteristic_without_zero:
                raise DCBusBranchRangeError(
                    "rectifier_characteristic_out_of_range: zero-current Vf unavailable")
            lower = minimum_confirmed_current
            lower_generator = self.generator.operating_point(
                rpm, lower, winding_temperature_c)
            lower_rectifier = self.rectifier.operating_point(
                lower, rectifier_case_temperature_c, reverse)
            if lower_rectifier.bridge_forward_voltage_drop_v is None:
                raise DCBusBranchRangeError(
                    "rectifier_characteristic_out_of_range: minimum-current point unavailable")
            residual_lower = (lower_generator.ideal_generator_branch_dc_voltage_v
                              - lower_rectifier.bridge_forward_voltage_drop_v - voltage)
            if residual_lower < -self.config.root_voltage_tolerance_v:
                raise DCBusBranchRangeError(
                    "rectifier_characteristic_out_of_range: required current is below "
                    f"{minimum_confirmed_current:g} A")
        if residual_lower <= self.config.root_voltage_tolerance_v and lower == 0.0:
            return GeneratorBranchSolution(
                current_dc_a=0.0, conducting=False,
                voltage_residual_v=min(0.0, residual_lower), iterations=0,
                generator_point=generator_zero, rectifier_point=rectifier_zero,
                reverse_voltage_static_estimate_v=reverse,
            )
        def branch_at(current: float) -> tuple[float, GeneratorOperatingPoint,
                                                RectifierOperatingPoint]:
            generator_point = self.generator.operating_point(
                rpm, current, winding_temperature_c)
            rectifier_point = self.rectifier.operating_point(
                current, rectifier_case_temperature_c, reverse)
            drop = rectifier_point.bridge_forward_voltage_drop_v
            if drop is None:
                raise DCBusBranchRangeError(
                    "rectifier_branch_not_evaluable:" + rectifier_point.status)
            residual = (generator_point.ideal_generator_branch_dc_voltage_v
                        - drop - voltage)
            return residual, generator_point, rectifier_point

        if lower > 0.0 and abs(residual_lower) <= self.config.root_voltage_tolerance_v:
            residual, generator_point, rectifier_point = branch_at(lower)
            return GeneratorBranchSolution(
                current_dc_a=lower, conducting=True,
                voltage_residual_v=residual, iterations=0,
                generator_point=generator_point, rectifier_point=rectifier_point,
                reverse_voltage_static_estimate_v=reverse,
            )

        upper = self._maximum_branch_current_a()
        if lower > upper:
            raise DCBusBranchRangeError("rectifier_characteristic_out_of_range")
        residual_upper, _, _ = branch_at(upper)
        if residual_upper > 0:
            raise DCBusBranchRangeError(
                "generator_branch_root_above_confirmed_current_domain")
        iterations = 0
        for iterations in range(1, self.config.maximum_iterations + 1):
            midpoint = (lower + upper) / 2.0
            residual_mid, _, _ = branch_at(midpoint)
            if (abs(residual_mid) <= self.config.root_voltage_tolerance_v
                    or upper - lower <= self.config.branch_current_tolerance_a):
                lower = upper = midpoint
                break
            if residual_mid > 0:
                lower = midpoint
            else:
                upper = midpoint
        current = (lower + upper) / 2.0
        residual, generator_point, rectifier_point = branch_at(current)
        return GeneratorBranchSolution(
            current_dc_a=current, conducting=True,
            voltage_residual_v=residual, iterations=iterations,
            generator_point=generator_point, rectifier_point=rectifier_point,
            reverse_voltage_static_estimate_v=reverse,
        )

    def _bus_state(self, voltage: float, inputs: DCBusInputs) -> _BusState:
        branch = self.solve_generator_branch_current(
            voltage, inputs.engine_or_generator_speed_rpm,
            inputs.generator_winding_temperature_c,
            inputs.rectifier_case_temperature_c,
        )
        battery_current = self.evaluate_battery_current_for_bus_voltage(
            voltage, inputs.battery_soc, inputs.battery_temperature_c)
        load_current = inputs.load_power_w / voltage
        return _BusState(
            bus_voltage_v=voltage, generator_branch=branch,
            battery_current_a=battery_current, load_current_a=load_current,
            current_residual_a=(branch.current_dc_a + battery_current - load_current),
        )

    def _search_voltage_range(self, inputs: DCBusInputs) -> tuple[float, float]:
        battery_max = (self.battery.config["series_cells"]
                       * self.battery.config["max_cell_v"])
        _, ideal_dc, _ = self._reverse_voltage(inputs.engine_or_generator_speed_rpm)
        upper = min(self.config.component_bus_voltage_max_v,
                    max(battery_max, ideal_dc))
        lower = self.config.minimum_search_voltage_v
        if upper <= lower:
            raise ValueError("Профили не задают положительный диапазон поиска V_bus")
        return lower, upper

    def _bisect_bus_root(self, left: float, right: float,
                         evaluate: Callable[[float], _BusState]) -> _BusState:
        left_state, right_state = evaluate(left), evaluate(right)
        if left_state.current_residual_a == 0:
            return left_state
        if right_state.current_residual_a == 0:
            return right_state
        for _ in range(self.config.maximum_iterations):
            midpoint = (left + right) / 2.0
            middle_state = evaluate(midpoint)
            if (abs(middle_state.current_residual_a)
                    <= self.config.current_balance_tolerance_a
                    or right - left <= self.config.root_voltage_tolerance_v):
                return middle_state
            if left_state.current_residual_a * middle_state.current_residual_a < 0:
                right, right_state = midpoint, middle_state
            else:
                left, left_state = midpoint, middle_state
        return evaluate((left + right) / 2.0)

    def _root_stability(self, state: _BusState, lower: float, upper: float,
                        evaluate: Callable[[float], _BusState]) -> tuple[str, float | None]:
        delta = min(self.config.stability_delta_v,
                    (state.bus_voltage_v - lower) / 2.0,
                    (upper - state.bus_voltage_v) / 2.0)
        if delta <= self.config.root_voltage_tolerance_v:
            return "indeterminate", None
        try:
            left = evaluate(state.bus_voltage_v - delta).current_residual_a
            right = evaluate(state.bus_voltage_v + delta).current_residual_a
        except (ValueError, DCBusBranchRangeError):
            return "indeterminate", None
        derivative = (right - left) / (2.0 * delta)
        tolerance = self.config.current_balance_tolerance_a / delta
        if derivative < -tolerance:
            return "stable", derivative
        if derivative > tolerance:
            return "unstable", derivative
        return "indeterminate", derivative

    def _limits_and_margins(self, state: _BusState, inputs: DCBusInputs
                            ) -> tuple[tuple[str, ...], dict[str, float | None]]:
        active: list[str] = []
        voltage = state.bus_voltage_v
        branch = state.generator_branch
        generator_point = branch.generator_point
        rectifier_point = branch.rectifier_point
        discharge_limit, charge_limit, _ = self.battery.temperature_current_limits(
            inputs.battery_temperature_c)
        battery_current = state.battery_current_a
        if battery_current > discharge_limit + self.config.current_balance_tolerance_a:
            active.append("battery_discharge_current_limit")
        if -battery_current > charge_limit + self.config.current_balance_tolerance_a:
            active.append("battery_charge_current_limit")
        if (battery_current > 0
                and inputs.battery_soc < self.battery.soc_floor_at_current(
                    battery_current, inputs.battery_temperature_c)):
            active.append("battery_soc_discharge_limit")
        if battery_current < 0 and inputs.battery_soc >= self.battery.config["soc_target"]:
            active.append("battery_soc_charge_limit")
        if voltage < self.config.bus_undervoltage_v:
            active.append("bus_undervoltage")
        if voltage > self.config.bus_overvoltage_v:
            active.append("bus_overvoltage")
        battery_min_voltage = (self.battery.config["series_cells"]
                               * self.battery.config["min_cell_v"])
        battery_max_voltage = (self.battery.config["series_cells"]
                               * self.battery.config["max_cell_v"])
        if voltage < battery_min_voltage:
            active.append("battery_min_voltage_limit")
        if voltage > battery_max_voltage:
            active.append("battery_max_voltage_limit")
        if generator_point.continuous_current_limit_exceeded:
            active.append("generator_current_limit")
        rectifier_limit_names = {
            "current_limit": "rectifier_current_limit",
            "reverse_voltage_limit": "rectifier_reverse_voltage_limit",
            "junction_temperature_limit": "rectifier_junction_temperature_limit",
            "vf_table_range": "rectifier_vf_table_range",
            "current_limit_not_evaluable": "rectifier_current_limit_not_evaluable",
        }
        for item in rectifier_point.active_limits:
            active.append(rectifier_limit_names.get(item, f"rectifier_{item}"))
        rectifier_current_limit = rectifier_point.current_limit_at_case_temperature_a
        reverse_limit = self.rectifier.profile.reverse_voltage_limit_v
        continuous_line_limit = self.generator.profile.continuous_line_current_rms_a
        margins = {
            "battery_discharge_current_a": discharge_limit - max(0.0, battery_current),
            "battery_charge_current_a": charge_limit - max(0.0, -battery_current),
            "generator_continuous_line_current_rms_a": (
                None if continuous_line_limit is None else continuous_line_limit
                - generator_point.line_current_rms_per_three_phase_system_a),
            "rectifier_current_a": (
                None if rectifier_current_limit is None else rectifier_current_limit
                - branch.current_dc_a),
            "rectifier_reverse_voltage_v": (
                None if reverse_limit is None else reverse_limit
                - branch.reverse_voltage_static_estimate_v),
            "bus_undervoltage_v": voltage - self.config.bus_undervoltage_v,
            "bus_overvoltage_v": self.config.bus_overvoltage_v - voltage,
            "battery_min_voltage_v": voltage - battery_min_voltage,
            "battery_max_voltage_v": battery_max_voltage - voltage,
        }
        return tuple(dict.fromkeys(active)), margins

    def _make_root(self, state: _BusState, stability: str, derivative: float | None,
                   inputs: DCBusInputs) -> DCBusRoot:
        active, margins = self._limits_and_margins(state, inputs)
        return DCBusRoot(
            bus_voltage_v=state.bus_voltage_v,
            generator_current_dc_a=state.generator_branch.current_dc_a,
            battery_current_a=state.battery_current_a,
            load_current_a=state.load_current_a,
            current_balance_error_a=state.current_residual_a,
            derivative_a_per_v=derivative, stability=stability,
            constraint_feasible=not active, active_limits=active, margins=margins,
        )

    def _empty_result(self, status: DCBusSolverStatus, warnings: tuple[str, ...],
                      roots: tuple[DCBusRoot, ...] = (), iterations: int = 0,
                      search_range: tuple[float, float] | None = None
                      ) -> DCBusOperatingPoint:
        return DCBusOperatingPoint(
            solver_status=status, bus_voltage_v=None,
            generator_current_dc_a=None, generator_power_to_bus_w=None,
            battery_current_a=None, battery_power_w=None, load_current_a=None,
            generator_emf_ll_rms_v=None, generator_ideal_rectified_voltage_v=None,
            generator_copper_loss_w=None, rectifier_drop_v=None,
            rectifier_loss_w=None, rectifier_junction_temperature_c=None,
            generator_electromagnetic_power_w=None, generator_shaft_power_w=None,
            generator_mechanical_torque_nm=None,
            generator_branch_balance_error_w=None, electrical_balance_error_w=None,
            reverse_voltage_static_estimate_v=None,
            reverse_voltage_model=self.config.reverse_voltage_model,
            commutation_overlap_model=self.config.commutation_overlap_model,
            generator_conducting=False, battery_charging=False,
            roots_found=roots, selected_root=None, root_stability=None,
            active_limits=(), margins={}, search_voltage_range_v=search_range,
            iterations=iterations, warnings=warnings,
        )

    def _select_root(self, roots: tuple[DCBusRoot, ...], inputs: DCBusInputs
                     ) -> tuple[int | None, DCBusSolverStatus]:
        stable_feasible = [index for index, root in enumerate(roots)
                           if root.stability == "stable" and root.constraint_feasible]
        stable_any = [index for index, root in enumerate(roots)
                      if root.stability == "stable"]
        if stable_feasible:
            candidates = stable_feasible
            status = DCBusSolverStatus.OK
        elif stable_any:
            candidates = stable_any
            status = DCBusSolverStatus.INFEASIBLE
        elif roots:
            candidates = list(range(len(roots)))
            status = DCBusSolverStatus.UNSTABLE_ONLY
        else:
            return None, DCBusSolverStatus.NO_ROOT
        reference = (inputs.previous_bus_voltage_v if inputs.previous_bus_voltage_v is not None
                     else self.battery.ocv(inputs.battery_soc))
        selected = min(candidates,
                       key=lambda index: abs(roots[index].bus_voltage_v - reference))
        return selected, status

    def solve(self, inputs: DCBusInputs) -> DCBusOperatingPoint:
        self._validate_inputs(inputs)
        readiness = self.readiness_errors()
        validation_warning = (
            "synthetic_test_fixture_not_physical_hardware"
            if self.config.validation_mode == "synthetic_physics_validation"
            else "engineering_simulation_not_validated_real_powertrain")
        base_warnings = (
            validation_warning,
            "commutation_overlap_model=disabled_missing_inductance",
            "reverse_voltage_model=static_ideal_estimate_excludes_switching_spikes",
        )
        if readiness:
            return self._empty_result(
                DCBusSolverStatus.NOT_READY, (*base_warnings, *readiness))
        try:
            lower, upper = self._search_voltage_range(inputs)
        except ValueError as exc:
            return self._empty_result(
                DCBusSolverStatus.NOT_READY, (*base_warnings, str(exc)))
        evaluations = 0
        cache: dict[float, _BusState] = {}

        def evaluate(voltage: float) -> _BusState:
            nonlocal evaluations
            key = float(voltage)
            cached = cache.get(key)
            if cached is not None:
                return cached
            evaluations += 1
            state = self._bus_state(key, inputs)
            cache[key] = state
            return state

        samples: list[tuple[float, _BusState | None]] = []
        domain_warnings: list[str] = []
        for index in range(self.config.voltage_scan_points):
            voltage = lower + (upper - lower) * index / (self.config.voltage_scan_points - 1)
            try:
                samples.append((voltage, evaluate(voltage)))
            except (ValueError, DCBusBranchRangeError) as exc:
                samples.append((voltage, None))
                message = str(exc)
                if message not in domain_warnings:
                    domain_warnings.append(message)

        root_states: list[_BusState] = []
        for (_, left_state), (_, right_state) in zip(samples, samples[1:]):
            if left_state is not None and abs(left_state.current_residual_a) <= (
                    self.config.current_balance_tolerance_a):
                root_states.append(left_state)
            if left_state is None or right_state is None:
                continue
            if left_state.current_residual_a * right_state.current_residual_a < 0:
                try:
                    root_states.append(self._bisect_bus_root(
                        left_state.bus_voltage_v, right_state.bus_voltage_v, evaluate))
                except (ValueError, DCBusBranchRangeError) as exc:
                    message = str(exc)
                    if message not in domain_warnings:
                        domain_warnings.append(message)
        if samples[-1][1] is not None and abs(samples[-1][1].current_residual_a) <= (
                self.config.current_balance_tolerance_a):
            root_states.append(samples[-1][1])
        root_states.sort(key=lambda state: state.bus_voltage_v)
        deduplicated: list[_BusState] = []
        for state in root_states:
            if (not deduplicated or abs(state.bus_voltage_v - deduplicated[-1].bus_voltage_v)
                    > 10.0 * self.config.root_voltage_tolerance_v):
                deduplicated.append(state)
        roots: list[DCBusRoot] = []
        for state in deduplicated:
            stability, derivative = self._root_stability(
                state, lower, upper, evaluate)
            roots.append(self._make_root(state, stability, derivative, inputs))
        roots_tuple = tuple(roots)
        selected_index, status = self._select_root(roots_tuple, inputs)
        warnings = (*base_warnings, *domain_warnings)
        if selected_index is None:
            return self._empty_result(status, warnings, roots_tuple,
                                      evaluations, (lower, upper))
        selected_root = roots_tuple[selected_index]
        selected_state = evaluate(selected_root.bus_voltage_v)
        branch = selected_state.generator_branch
        generator_point = branch.generator_point
        rectifier_point = branch.rectifier_point
        voltage = selected_state.bus_voltage_v
        generator_bus_power = voltage * branch.current_dc_a
        battery_power = voltage * selected_state.battery_current_a
        rectifier_loss = float(rectifier_point.rectifier_loss_w or 0.0)
        generator_balance = (generator_point.electromagnetic_power_w
                             - generator_point.copper_loss_w - rectifier_loss
                             - generator_bus_power)
        bus_balance = generator_bus_power + battery_power - inputs.load_power_w
        active_limits = list(selected_root.active_limits)
        if abs(generator_balance) > self.config.balance_tolerance_w:
            active_limits.append("generator_branch_balance_error")
        if abs(bus_balance) > self.config.balance_tolerance_w:
            active_limits.append("dc_bus_balance_error")
        if active_limits and status == DCBusSolverStatus.OK:
            status = DCBusSolverStatus.INFEASIBLE
        return DCBusOperatingPoint(
            solver_status=status, bus_voltage_v=voltage,
            generator_current_dc_a=branch.current_dc_a,
            generator_power_to_bus_w=generator_bus_power,
            battery_current_a=selected_state.battery_current_a,
            battery_power_w=battery_power,
            load_current_a=selected_state.load_current_a,
            generator_emf_ll_rms_v=generator_point.emf_line_to_line_rms_v,
            generator_ideal_rectified_voltage_v=generator_point.ideal_no_load_dc_voltage_v,
            generator_copper_loss_w=generator_point.copper_loss_w,
            rectifier_drop_v=rectifier_point.bridge_forward_voltage_drop_v,
            rectifier_loss_w=rectifier_loss,
            rectifier_junction_temperature_c=rectifier_point.junction_temperature_c,
            generator_electromagnetic_power_w=generator_point.electromagnetic_power_w,
            generator_shaft_power_w=generator_point.shaft_power_w,
            generator_mechanical_torque_nm=generator_point.mechanical_torque_nm,
            generator_branch_balance_error_w=generator_balance,
            electrical_balance_error_w=bus_balance,
            reverse_voltage_static_estimate_v=branch.reverse_voltage_static_estimate_v,
            reverse_voltage_model=self.config.reverse_voltage_model,
            commutation_overlap_model=self.config.commutation_overlap_model,
            generator_conducting=branch.conducting,
            battery_charging=selected_state.battery_current_a < 0,
            roots_found=roots_tuple, selected_root=selected_index,
            root_stability=selected_root.stability,
            active_limits=tuple(dict.fromkeys(active_limits)),
            margins=selected_root.margins,
            search_voltage_range_v=(lower, upper), iterations=evaluations,
            warnings=warnings,
        )


def load_synthetic_dc_bus_fixture(
        path: str | Path | None = None) -> SyntheticDCBusFixture:
    """Load the one fully specified, explicitly nonphysical Phase 4A fixture."""
    from generator_model import GeneratorProfile
    from rectifier_model import RectifierProfile

    fixture_path = (Path(path) if path is not None
                    else BUILTIN_DC_BUS_DIR / f"{SYNTHETIC_FIXTURE_ID}.json")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if data.get("source_type") != "synthetic_test_fixture":
        raise ValueError("Phase 4A fixture должен быть synthetic_test_fixture")
    return SyntheticDCBusFixture(
        fixture_id=data["fixture_id"], source_type=data["source_type"],
        generator_model=GeneratorModel(GeneratorProfile.from_dict(data["generator_profile"])),
        rectifier_model=RectifierModel(RectifierProfile.from_dict(data["rectifier_profile"])),
        battery_model=BatteryModel(data["battery_config"]),
        solver_config=DCBusSolverConfig.from_dict(data["solver_config"]),
        warning=data["warning"],
    )


def build_synthetic_dc_bus_solver() -> DCBusSolver:
    fixture = load_synthetic_dc_bus_fixture()
    return DCBusSolver(
        fixture.generator_model, fixture.rectifier_model,
        fixture.battery_model, fixture.solver_config,
    )
