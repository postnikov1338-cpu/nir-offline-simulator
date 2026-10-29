"""L275E Part-Load Fuel Model v1.

The manufacturer/digitized Limbach full-load curve remains owned by
``engine_model.EngineModel``.  This module adds a normalized scientific-
analogue correction and never changes the full-load torque envelope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

from monotone_interpolation import pchip_derivative, pchip_interpolate


ROOT = Path(__file__).resolve().parent
BUILTIN_PART_LOAD_DIR = ROOT / "data" / "engine_part_load"
DEFAULT_PROFILE_ID = "two_stroke_gasoline_carbureted_v1"
LOW_LOAD_SENSITIVITIES = frozenset({"optimistic", "nominal", "pessimistic"})


class EnginePartLoadStatus(str, Enum):
    FULL_LOAD_SUPPORTED = "full_load_supported"
    PART_LOAD_SCIENTIFIC_ANALOGUE = "part_load_scientific_analogue"
    NEAR_FULL_LOAD_BLEND = "near_full_load_blend"
    HIGH_SPEED_PART_LOAD_TRANSITION = "high_speed_part_load_transition"
    VERY_LOW_LOAD_SCIENTIFIC_ANALOGUE = "very_low_load_scientific_analogue"
    ENGINE_TORQUE_LIMIT = "engine_torque_limit"
    RPM_OUT_OF_RANGE = "rpm_out_of_range"
    PART_LOAD_MAP_OUT_OF_RANGE = "part_load_map_out_of_range"
    FULL_LOAD_SFC_MISSING = "full_load_sfc_missing"


@dataclass(frozen=True)
class EnginePartLoadProvenance:
    source_type: str
    confidence: str
    primary_analogue: str | None
    validation_source: str | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnginePartLoadReadiness:
    ready: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EnginePartLoadProfile:
    profile_id: str
    revision: int
    engine_part_load_family: str
    part_load_method: str
    source_type: str
    validation_status: str
    rpm_grid: tuple[float, ...]
    lambda_grid: tuple[float, ...]
    k_pl_grid: tuple[tuple[float, ...], ...]
    high_speed_transition_start_rpm: float
    high_speed_transition_end_rpm: float
    low_load_lambda_boundary: float
    near_full_load_lambda_boundary: float
    peak_torque_reference_rpm: float
    low_load_fuel_floors_kg_h: dict[str, float]
    default_low_load_sensitivity: str
    cell_provenance: dict[str, dict[str, str]]
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def readiness(self) -> EnginePartLoadReadiness:
        errors: list[str] = []
        if self.part_load_method != "normalized_scientific_analogue":
            errors.append("unsupported_part_load_method")
        if self.source_type != "normalized_scientific_analogue":
            errors.append("invalid_source_type")
        if len(self.rpm_grid) < 2 or any(b <= a for a, b in zip(self.rpm_grid, self.rpm_grid[1:])):
            errors.append("invalid_rpm_grid")
        if len(self.lambda_grid) < 2 or any(b <= a for a, b in zip(self.lambda_grid, self.lambda_grid[1:])):
            errors.append("invalid_lambda_grid")
        if len(self.k_pl_grid) != len(self.rpm_grid) or any(
                len(row) != len(self.lambda_grid) for row in self.k_pl_grid):
            errors.append("invalid_k_pl_shape")
        if any(not math.isfinite(value) or value <= 0
               for row in self.k_pl_grid for value in row):
            errors.append("invalid_k_pl_value")
        if not math.isclose(self.rpm_grid[-1], self.high_speed_transition_start_rpm):
            errors.append("high_speed_transition_must_start_at_last_map_row")
        if self.high_speed_transition_end_rpm <= self.high_speed_transition_start_rpm:
            errors.append("invalid_high_speed_transition")
        if not math.isclose(self.lambda_grid[0], self.low_load_lambda_boundary):
            errors.append("low_load_boundary_must_match_first_lambda")
        if not math.isclose(self.lambda_grid[-1], self.near_full_load_lambda_boundary):
            errors.append("near_full_boundary_must_match_last_lambda")
        if self.default_low_load_sensitivity not in LOW_LOAD_SENSITIVITIES:
            errors.append("invalid_default_low_load_sensitivity")
        if set(self.low_load_fuel_floors_kg_h) != LOW_LOAD_SENSITIVITIES:
            errors.append("incomplete_low_load_sensitivity_set")
        if any(not math.isfinite(value) or value < 0
               for value in self.low_load_fuel_floors_kg_h.values()):
            errors.append("invalid_low_load_fuel_floor")
        if self.peak_torque_reference_rpm <= 0:
            errors.append("invalid_peak_torque_reference")
        return EnginePartLoadReadiness(not errors, tuple(errors))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnginePartLoadProfile":
        profile = cls(
            profile_id=data["profile_id"], revision=int(data["revision"]),
            engine_part_load_family=data["engine_part_load_family"],
            part_load_method=data["part_load_method"], source_type=data["source_type"],
            validation_status=data["validation_status"],
            rpm_grid=tuple(float(x) for x in data["rpm_grid"]),
            lambda_grid=tuple(float(x) for x in data["lambda_grid"]),
            k_pl_grid=tuple(tuple(float(x) for x in row) for row in data["k_pl_grid"]),
            high_speed_transition_start_rpm=float(data["high_speed_transition"]["start_rpm"]),
            high_speed_transition_end_rpm=float(data["high_speed_transition"]["end_rpm"]),
            low_load_lambda_boundary=float(data["load_regions"]["low_load_boundary"]),
            near_full_load_lambda_boundary=float(data["load_regions"]["near_full_boundary"]),
            peak_torque_reference_rpm=float(data["peak_torque_reference_rpm"]),
            low_load_fuel_floors_kg_h={k: float(v) for k, v in data["low_load_fuel_floors_kg_h"].items()},
            default_low_load_sensitivity=data["default_low_load_sensitivity"],
            cell_provenance=dict(data.get("cell_provenance", {})),
            provenance=dict(data["provenance"]), warnings=tuple(data.get("warnings", ())),
        )
        readiness = profile.readiness()
        if not readiness.ready:
            raise ValueError(f"Part-load profile not ready: {readiness.errors}")
        return profile

    @classmethod
    def load_builtin(cls, profile_id: str = DEFAULT_PROFILE_ID) -> "EnginePartLoadProfile":
        path = BUILTIN_PART_LOAD_DIR / f"engine_part_load_{profile_id}.json"
        if not path.is_file():
            raise ValueError(f"Unknown part-load profile: {profile_id}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class EnginePartLoadOperatingPoint:
    rpm: float
    torque_nm: float
    max_torque_nm: float | None
    load_fraction_lambda: float | None
    normalized_speed_nu: float | None
    shaft_power_kw: float | None
    full_load_bsfc_g_kwh: float | None
    part_load_factor_k: float | None
    effective_bsfc_g_kwh: float | None
    fuel_flow_kg_h: float | None
    fuel_model_status: str
    fuel_model_source_type: str
    fuel_model_confidence: str
    provenance: EnginePartLoadProvenance
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


def _smoothstep(value: float) -> float:
    return 3.0 * value * value - 2.0 * value * value * value


def _cubic_hermite(start: float, end: float,
                   start_derivative: float, end_derivative: float,
                   fraction: float, interval_width: float) -> float:
    """C1 transition; h01 is the ordinary cubic smoothstep basis."""
    h01 = _smoothstep(fraction)
    h00 = 1.0 - h01
    h10 = fraction**3 - 2.0 * fraction**2 + fraction
    h11 = fraction**3 - fraction**2
    return (h00 * start + h10 * interval_width * start_derivative
            + h01 * end + h11 * interval_width * end_derivative)


class EnginePartLoadModel:
    def __init__(self, profile: EnginePartLoadProfile,
                 low_load_sensitivity: str | None = None):
        readiness = profile.readiness()
        if not readiness.ready:
            raise ValueError(f"Part-load profile not ready: {readiness.errors}")
        sensitivity = low_load_sensitivity or profile.default_low_load_sensitivity
        if sensitivity not in LOW_LOAD_SENSITIVITIES:
            raise ValueError("Unknown low-load sensitivity")
        self.profile = profile
        self.low_load_sensitivity = sensitivity

    @property
    def low_load_fuel_floor_kg_h(self) -> float:
        return self.profile.low_load_fuel_floors_kg_h[self.low_load_sensitivity]

    def _base_map_factor(self, rpm: float, load_fraction: float) -> float:
        p = self.profile
        if not p.rpm_grid[0] <= rpm <= p.rpm_grid[-1]:
            raise ValueError("Base part-load map rpm out of range")
        rows = [pchip_interpolate(p.lambda_grid, row, load_fraction)
                for row in p.k_pl_grid]
        value = pchip_interpolate(p.rpm_grid, rows, rpm)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid interpolated K_PL")
        return value

    def _map_and_high_speed_factor(self, rpm: float,
                                   load_fraction: float) -> float:
        p = self.profile
        if rpm <= p.high_speed_transition_start_rpm:
            return self._base_map_factor(rpm, load_fraction)
        at_start = self._base_map_factor(
            p.high_speed_transition_start_rpm, load_fraction)
        rows = [pchip_interpolate(p.lambda_grid, row, load_fraction)
                for row in p.k_pl_grid]
        start_derivative = pchip_derivative(
            p.rpm_grid, rows, p.high_speed_transition_start_rpm)
        width = (p.high_speed_transition_end_rpm
                 - p.high_speed_transition_start_rpm)
        fraction = (rpm - p.high_speed_transition_start_rpm) / width
        return _cubic_hermite(
            at_start, 1.0, start_derivative, 0.0, fraction, width)

    def part_load_factor(self, rpm: float, load_fraction: float) -> float:
        p = self.profile
        rpm = float(rpm)
        load_fraction = float(load_fraction)
        if not 0.2 <= load_fraction <= 1.0:
            raise ValueError("K_PL is defined for lambda in [0.2, 1]")
        if not p.rpm_grid[0] <= rpm <= p.high_speed_transition_end_rpm:
            raise ValueError("Part-load rpm out of range")
        map_lambda = min(load_fraction, p.near_full_load_lambda_boundary)
        base = self._map_and_high_speed_factor(rpm, map_lambda)
        if load_fraction <= p.near_full_load_lambda_boundary:
            return base
        boundary = p.near_full_load_lambda_boundary
        derivative_step = 1e-5
        boundary_derivative = (
            self._map_and_high_speed_factor(rpm, boundary)
            - self._map_and_high_speed_factor(rpm, boundary - derivative_step)
        ) / derivative_step
        width = 1.0 - boundary
        fraction = (load_fraction - boundary) / width
        return _cubic_hermite(
            base, 1.0, boundary_derivative, 0.0, fraction, width)

    def _provenance(self, rpm: float, load_fraction: float) -> EnginePartLoadProvenance:
        p = self.profile
        if math.isclose(load_fraction, 1.0, abs_tol=1e-12):
            return EnginePartLoadProvenance(
                "manufacturer_or_digitized_engine_data", "high", None, None)
        if load_fraction < p.low_load_lambda_boundary:
            return EnginePartLoadProvenance(
                "displacement_scaled_low_load_scientific_analogue", "low",
                p.provenance["low_load"]["analogue"], None,
                (f"sensitivity={self.low_load_sensitivity}",))
        if rpm > p.high_speed_transition_start_rpm:
            return EnginePartLoadProvenance(
                "engineering_high_speed_transition", "low_medium",
                p.provenance["main_map"]["primary_analogue"],
                p.provenance["high_speed_transition"]["validation_source"])
        key = f"{int(rpm)}:{load_fraction:.1f}"
        if key in p.cell_provenance:
            cell = p.cell_provenance[key]
            return EnginePartLoadProvenance(
                cell["source_type"], cell["confidence"],
                p.provenance["main_map"]["primary_analogue"], None,
                (cell["note"],))
        return EnginePartLoadProvenance(
            "normalized_scientific_analogue", "medium",
            p.provenance["main_map"]["primary_analogue"], None)

    def evaluate(self, engine_model: Any, rpm: float,
                 torque_nm: float) -> EnginePartLoadOperatingPoint:
        rpm = float(rpm)
        torque_nm = float(torque_nm)
        if not math.isfinite(rpm) or not math.isfinite(torque_nm) or torque_nm < 0:
            raise ValueError("rpm and torque must be finite; torque must be non-negative")
        try:
            envelope = engine_model.maximum_available_at_rpm(rpm)
        except Exception as exc:
            if exc.__class__.__name__ != "EngineMapOutOfRangeError":
                raise
            return self._empty(rpm, torque_nm, EnginePartLoadStatus.RPM_OUT_OF_RANGE)
        maximum = envelope.maximum_torque_nm
        load_fraction = torque_nm / maximum
        normalized_speed = rpm / self.profile.peak_torque_reference_rpm
        shaft_power = 2 * math.pi * rpm * torque_nm / (60 * 1000)
        full_bsfc = (None if envelope.full_load_point is None
                     else envelope.full_load_point.bsfc_g_kwh)
        if load_fraction > 1.0 + 1e-12:
            return self._point(
                rpm, torque_nm, maximum, load_fraction, normalized_speed,
                shaft_power, full_bsfc, None, None, None,
                EnginePartLoadStatus.ENGINE_TORQUE_LIMIT,
                EnginePartLoadProvenance("manufacturer_or_digitized_engine_data",
                                         "high", None, None),
                ("requested_torque_exceeds_full_load_envelope",))
        load_fraction = min(load_fraction, 1.0)
        if math.isclose(load_fraction, 1.0, abs_tol=1e-12):
            if full_bsfc is None:
                return self._point(
                    rpm, torque_nm, maximum, load_fraction, normalized_speed,
                    shaft_power, None, 1.0, None, None,
                    EnginePartLoadStatus.FULL_LOAD_SFC_MISSING,
                    self._provenance(rpm, 1.0), ("rated_point_has_no_confirmed_sfc",))
            return self._point(
                rpm, torque_nm, maximum, load_fraction, normalized_speed,
                shaft_power, full_bsfc, 1.0, full_bsfc,
                envelope.full_load_point.fuel_flow_kg_h,
                EnginePartLoadStatus.FULL_LOAD_SUPPORTED,
                self._provenance(rpm, 1.0))
        if not self.profile.rpm_grid[0] <= rpm <= self.profile.high_speed_transition_end_rpm:
            return self._point(
                rpm, torque_nm, maximum, load_fraction, normalized_speed,
                shaft_power, full_bsfc, None, None, None,
                EnginePartLoadStatus.PART_LOAD_MAP_OUT_OF_RANGE,
                EnginePartLoadProvenance("not_available", "low", None, None))
        if load_fraction == 0:
            provenance = self._provenance(rpm, load_fraction)
            return self._point(
                rpm, torque_nm, maximum, 0.0, normalized_speed, 0.0,
                full_bsfc, None, None, self.low_load_fuel_floor_kg_h,
                EnginePartLoadStatus.VERY_LOW_LOAD_SCIENTIFIC_ANALOGUE,
                provenance)
        if full_bsfc is None:
            factor = (self.part_load_factor(rpm, max(load_fraction, 0.2))
                      if load_fraction >= 0.2 else None)
            return self._point(
                rpm, torque_nm, maximum, load_fraction, normalized_speed,
                shaft_power, None, factor, None, None,
                EnginePartLoadStatus.FULL_LOAD_SFC_MISSING,
                self._provenance(rpm, load_fraction),
                ("part_load_factor_available_but_full_load_sfc_missing",))
        if load_fraction < self.profile.low_load_lambda_boundary:
            anchor_lambda = self.profile.low_load_lambda_boundary
            anchor_factor = self.part_load_factor(rpm, anchor_lambda)
            maximum_shaft_power = (
                2 * math.pi * rpm * envelope.maximum_torque_nm / (60 * 1000))
            anchor_power = maximum_shaft_power * anchor_lambda
            anchor_flow = full_bsfc * anchor_factor * anchor_power / 1000
            derivative_step = 1e-5
            factor_derivative = (
                self.part_load_factor(rpm, anchor_lambda + derivative_step)
                - anchor_factor
            ) / derivative_step
            anchor_flow_derivative = (
                full_bsfc * maximum_shaft_power / 1000
                * (anchor_factor + anchor_lambda * factor_derivative)
            )
            x = load_fraction / anchor_lambda
            fuel_flow = _cubic_hermite(
                self.low_load_fuel_floor_kg_h, anchor_flow,
                0.0, anchor_flow_derivative, x, anchor_lambda)
            effective_bsfc = fuel_flow * 1000 / shaft_power
            return self._point(
                rpm, torque_nm, maximum, load_fraction, normalized_speed,
                shaft_power, full_bsfc, None, effective_bsfc, fuel_flow,
                EnginePartLoadStatus.VERY_LOW_LOAD_SCIENTIFIC_ANALOGUE,
                self._provenance(rpm, load_fraction))
        factor = self.part_load_factor(rpm, load_fraction)
        effective_bsfc = full_bsfc * factor
        fuel_flow = effective_bsfc * shaft_power / 1000
        if load_fraction >= self.profile.near_full_load_lambda_boundary:
            status = EnginePartLoadStatus.NEAR_FULL_LOAD_BLEND
        elif rpm > self.profile.high_speed_transition_start_rpm:
            status = EnginePartLoadStatus.HIGH_SPEED_PART_LOAD_TRANSITION
        else:
            status = EnginePartLoadStatus.PART_LOAD_SCIENTIFIC_ANALOGUE
        return self._point(
            rpm, torque_nm, maximum, load_fraction, normalized_speed,
            shaft_power, full_bsfc, factor, effective_bsfc, fuel_flow,
            status, self._provenance(rpm, load_fraction))

    @staticmethod
    def _point(rpm: float, torque: float, maximum: float, load: float,
               nu: float, power: float, full_bsfc: float | None,
               factor: float | None, bsfc: float | None,
               fuel: float | None, status: EnginePartLoadStatus,
               provenance: EnginePartLoadProvenance,
               warnings: tuple[str, ...] = ()) -> EnginePartLoadOperatingPoint:
        return EnginePartLoadOperatingPoint(
            rpm, torque, maximum, load, nu, power, full_bsfc, factor, bsfc,
            fuel, status.value, provenance.source_type, provenance.confidence,
            provenance, warnings)

    @staticmethod
    def _empty(rpm: float, torque: float,
               status: EnginePartLoadStatus) -> EnginePartLoadOperatingPoint:
        provenance = EnginePartLoadProvenance("not_available", "low", None, None)
        return EnginePartLoadOperatingPoint(
            rpm, torque, None, None, None, None,
            None, None, None, None, status.value, provenance.source_type,
            provenance.confidence, provenance, ())


def generate_part_load_diagnostic_plots(engine_model: Any,
                                        output_path: str | Path) -> Path:
    """Create the four requested diagnostics in one non-GUI PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model = engine_model.part_load_model
    if model is None:
        raise ValueError("Engine has no part-load model")
    speeds = (4300, 5000, 5700, 6400, 6800, 7000, 7200)
    lambdas = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for rpm in speeds:
        envelope = engine_model.maximum_available_at_rpm(rpm)
        factors, bsfc, fuel = [], [], []
        for load in lambdas:
            point = model.evaluate(engine_model, rpm, envelope.maximum_torque_nm * load)
            factors.append(np.nan if point.part_load_factor_k is None else point.part_load_factor_k)
            bsfc.append(np.nan if point.effective_bsfc_g_kwh is None else point.effective_bsfc_g_kwh)
            fuel.append(np.nan if point.fuel_flow_kg_h is None else point.fuel_flow_kg_h)
        axes[0, 0].plot(lambdas, factors, label=f"{rpm} rpm")
        if np.count_nonzero(np.isfinite(bsfc)) > 1:
            axes[0, 1].plot(lambdas, bsfc, label=f"{rpm} rpm")
        if np.count_nonzero(np.isfinite(fuel)) > 1:
            axes[1, 0].plot(lambdas, fuel, label=f"{rpm} rpm")
    axes[0, 0].set(title="K_PL", xlabel="lambda", ylabel="K_PL")
    axes[0, 1].set(title="BSFC", xlabel="lambda", ylabel="g/kWh")
    axes[1, 0].set(title="Fuel flow", xlabel="lambda", ylabel="kg/h", xlim=(0, 1))
    rpm_grid = np.linspace(4300, 7200, 80)
    load_grid = np.linspace(.2, 1, 80)
    surface = np.array([[model.part_load_factor(rpm, load) for load in load_grid]
                        for rpm in rpm_grid])
    contour = axes[1, 1].contourf(load_grid, rpm_grid, surface, levels=20)
    fig.colorbar(contour, ax=axes[1, 1], label="K_PL")
    axes[1, 1].set(title="K_PL surface", xlabel="lambda", ylabel="rpm")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.axvline(.2, color="grey", ls="--", lw=.8, alpha=.7)
        axis.axvline(.9, color="grey", ls="--", lw=.8, alpha=.7)
        axis.legend(fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.suptitle("L275E Part-Load Model v1 — diagnostic scientific analogue")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_builtin_part_load_profile(
        profile_id: str = DEFAULT_PROFILE_ID) -> EnginePartLoadProfile:
    return EnginePartLoadProfile.load_builtin(profile_id)
