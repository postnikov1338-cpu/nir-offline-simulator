"""Phase 6C cold-charge and lithium-plating evidence observer.

The observer deliberately separates an exact manufacturer charge policy from
published evidence for other cells.  It reports evidence states and charge
exposures; it does not estimate plating damage or alter electrical physics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
BUILTIN_COLD_CHARGE_DIR = ROOT / "data" / "cold_charge"
DEFAULT_POLICY_PROFILE_ID = "tattu_taa22k6s30asx_charge_policy_v1"
DEFAULT_EVIDENCE_PROFILE_IDS = (
    "huang_nmc532_graphite_200mah_2022_v1",
    "koleti_nca_graphite_31ah_2019_v1",
    "gao_p2d_low_temperature_plating_2022_v1",
)


class ChargePolicyStatus(str, Enum):
    NO_CHARGE_REQUEST = "no_charge_request"
    ALLOWED = "allowed"
    BLOCKED_BELOW_MIN_TEMPERATURE = "blocked_below_min_temperature"
    BLOCKED_ABOVE_MAX_TEMPERATURE = "blocked_above_max_temperature"
    LIMITED_BY_MAX_C_RATE = "limited_by_max_c_rate"
    LIMITED_BY_SOC = "limited_by_soc"
    LIMITED_BY_VOLTAGE = "limited_by_voltage"
    BLOCKED_BY_EXISTING_BATTERY_LIMIT = "blocked_by_existing_battery_limit"


class PlatingEvidenceState(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"
    OUTSIDE_EVIDENCE = "outside_evidence"
    PLATING_NOT_OBSERVED_EXPERIMENT = "plating_not_observed_experiment"
    PLATING_OBSERVED_EXPERIMENT = "plating_observed_experiment"
    BELOW_PUBLISHED_MODEL_ONSET = "below_published_model_onset"
    MODEL_PREDICTED_PLATING = "model_predicted_plating"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class ColdChargeReadiness(str, Enum):
    READY_FOR_EVIDENCE_DIAGNOSTICS = "ready_for_evidence_diagnostics"
    NOT_VALIDATED_FOR_TATTU_PLATING_PREDICTION = (
        "not_validated_for_tattu_plating_prediction")
    NO_QUANTITATIVE_DAMAGE_MODEL = "no_quantitative_damage_model"
    READY_FOR_TARGET_BATTERY_POLICY = "ready_for_target_battery_policy"


@dataclass(frozen=True)
class BatteryChargePolicy:
    profile_id: str
    source_id: str
    source_type: str
    manufacturer: str
    sku: str
    capacity_ah: float
    series_cells: int
    nominal_voltage_v: float
    charge_temperature_min_c: float
    charge_temperature_max_c: float
    normal_charge_c_rate: float
    max_charge_c_rate: float
    validated_for_target_battery: bool
    provenance: dict[str, Any]
    sources: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.capacity_ah, self.nominal_voltage_v,
            self.charge_temperature_min_c, self.charge_temperature_max_c,
            self.normal_charge_c_rate, self.max_charge_c_rate,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Charge policy должна содержать конечные числа")
        if (self.capacity_ah <= 0 or self.series_cells <= 0
                or self.nominal_voltage_v <= 0
                or self.charge_temperature_max_c
                <= self.charge_temperature_min_c
                or self.normal_charge_c_rate <= 0
                or self.max_charge_c_rate <= 0
                or self.normal_charge_c_rate > self.max_charge_c_rate):
            raise ValueError("Неверные пределы manufacturer charge policy")
        if self.source_type != "manufacturer":
            raise ValueError("Target charge policy должна иметь manufacturer provenance")

    @property
    def maximum_charge_current_a(self) -> float:
        return self.capacity_ah * self.max_charge_c_rate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatteryChargePolicy":
        return cls(
            profile_id=data["profile_id"], source_id=data["source_id"],
            source_type=data["source_type"], manufacturer=data["manufacturer"],
            sku=data["sku"], capacity_ah=float(data["capacity_ah"]),
            series_cells=int(data["series_cells"]),
            nominal_voltage_v=float(data["nominal_voltage_v"]),
            charge_temperature_min_c=float(data["charge_temperature_min_c"]),
            charge_temperature_max_c=float(data["charge_temperature_max_c"]),
            normal_charge_c_rate=float(data["normal_charge_c_rate"]),
            max_charge_c_rate=float(data["max_charge_c_rate"]),
            validated_for_target_battery=bool(
                data["validated_for_target_battery"]),
            provenance=dict(data.get("provenance", {})),
            sources=tuple(dict(item) for item in data.get("sources", ())),
            warnings=tuple(data.get("warnings", ())),
        )

    @classmethod
    def load_builtin(cls, profile_id: str = DEFAULT_POLICY_PROFILE_ID
                     ) -> "BatteryChargePolicy":
        path = BUILTIN_COLD_CHARGE_DIR / f"{profile_id}.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [dict(item) for item in self.sources]
        data["warnings"] = list(self.warnings)
        data["readiness"] = ColdChargeReadiness.READY_FOR_TARGET_BATTERY_POLICY.value
        return data

    def to_battery_model_config(self) -> dict[str, Any]:
        """Return only the limit fields consumed by BatteryModel."""
        return {
            "profile_id": self.profile_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "charge_temperature_min_c": self.charge_temperature_min_c,
            "charge_temperature_max_c": self.charge_temperature_max_c,
            "normal_charge_c_rate": self.normal_charge_c_rate,
            "max_charge_c_rate": self.max_charge_c_rate,
            "validated_for_target_battery": self.validated_for_target_battery,
        }


@dataclass(frozen=True)
class ColdChargeEvidencePoint:
    point_id: str
    temperature_c: float
    charge_c_rate: float
    soc_onset: float | None
    evidence_type: str
    evidence_result: PlatingEvidenceState
    confidence: str
    charge_protocol: dict[str, Any]
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColdChargeEvidencePoint":
        onset = data.get("soc_onset")
        point = cls(
            point_id=data["point_id"],
            temperature_c=float(data["temperature_c"]),
            charge_c_rate=float(data["charge_c_rate"]),
            soc_onset=None if onset is None else float(onset),
            evidence_type=data["evidence_type"],
            evidence_result=PlatingEvidenceState(data["evidence_result"]),
            confidence=data["confidence"],
            charge_protocol=dict(data.get("charge_protocol", {})),
            notes=data.get("notes"),
        )
        if point.charge_c_rate <= 0:
            raise ValueError("Evidence C-rate должен быть положительным")
        if point.soc_onset is not None and not 0 <= point.soc_onset <= 1:
            raise ValueError("Evidence SOC onset должен находиться в 0...1")
        return point

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_result"] = self.evidence_result.value
        return data


@dataclass(frozen=True)
class ColdChargeEvidenceProfile:
    profile_id: str
    source_id: str
    source_type: str
    authors: tuple[str, ...]
    year: int
    doi: str
    chemistry: str | None
    cathode: str | None
    anode_type: str | None
    cell_format: str | None
    nominal_capacity_ah: float | None
    points: tuple[ColdChargeEvidencePoint, ...]
    confidence: str
    validated_for_tattu: bool
    source_internal_conflict: bool
    source_internal_conflict_note: str | None
    diagnostic_context: dict[str, Any]
    provenance: dict[str, Any]
    sources: tuple[dict[str, Any], ...]
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColdChargeEvidenceProfile":
        capacity = data.get("nominal_capacity_ah")
        profile = cls(
            profile_id=data["profile_id"], source_id=data["source_id"],
            source_type=data["source_type"],
            authors=tuple(data.get("authors", ())), year=int(data["year"]),
            doi=data["doi"], chemistry=data.get("chemistry"),
            cathode=data.get("cathode"), anode_type=data.get("anode_type"),
            cell_format=data.get("cell_format"),
            nominal_capacity_ah=(None if capacity is None else float(capacity)),
            points=tuple(ColdChargeEvidencePoint.from_dict(item)
                         for item in data["points"]),
            confidence=data["confidence"],
            validated_for_tattu=bool(data["validated_for_tattu"]),
            source_internal_conflict=bool(
                data.get("source_internal_conflict", False)),
            source_internal_conflict_note=data.get(
                "source_internal_conflict_note"),
            diagnostic_context=dict(data.get("diagnostic_context", {})),
            provenance=dict(data.get("provenance", {})),
            sources=tuple(dict(item) for item in data.get("sources", ())),
            notes=tuple(data.get("notes", ())),
        )
        if not profile.points:
            raise ValueError("Evidence profile не содержит опубликованных точек")
        if profile.validated_for_tattu:
            raise ValueError("Published analogue evidence не валидировано для Tattu")
        return profile

    @classmethod
    def load_builtin(cls, profile_id: str) -> "ColdChargeEvidenceProfile":
        path = BUILTIN_COLD_CHARGE_DIR / f"{profile_id}.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["authors"] = list(self.authors)
        data["points"] = [point.to_dict() for point in self.points]
        data["sources"] = [dict(item) for item in self.sources]
        data["notes"] = list(self.notes)
        data["readiness"] = [
            ColdChargeReadiness.READY_FOR_EVIDENCE_DIAGNOSTICS.value,
            ColdChargeReadiness.NOT_VALIDATED_FOR_TATTU_PLATING_PREDICTION.value,
            ColdChargeReadiness.NO_QUANTITATIVE_DAMAGE_MODEL.value,
        ]
        return data


@dataclass(frozen=True)
class ColdChargeObserverInput:
    dt_s: float
    battery_temperature_c: float
    battery_soc: float
    requested_battery_current_a: float
    actual_battery_current_a: float
    battery_capacity_ah: float
    battery_voltage_v: float | None = None
    cell_voltage_v: float | None = None
    soc_charge_limit: float | None = None
    cell_voltage_charge_limit_v: float | None = None
    existing_battery_limit_active: bool = False

    def __post_init__(self) -> None:
        values = (
            self.dt_s, self.battery_temperature_c, self.battery_soc,
            self.requested_battery_current_a, self.actual_battery_current_a,
            self.battery_capacity_ah,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Observer input должен содержать конечные числа")
        if (self.dt_s <= 0 or self.battery_capacity_ah <= 0
                or not 0 <= self.battery_soc <= 1
                or self.battery_temperature_c <= -273.15):
            raise ValueError("Неверный observer input")


@dataclass
class ColdChargeObserverState:
    elapsed_s: float = 0.0
    actual_cold_charge_exposure_ah: float = 0.0
    blocked_charge_request_ah: float = 0.0
    counterfactual_plating_exposure_ah: float = 0.0
    min_requested_charge_temperature_c: float | None = None
    max_requested_charge_c_rate: float = 0.0
    max_soc_during_cold_charge_request: float | None = None
    experimental_plating_match_count: int = 0
    experimental_no_plating_match_count: int = 0
    model_predicted_plating_match_count: int = 0
    unknown_match_count: int = 0


@dataclass(frozen=True)
class ColdChargeEvidenceMatch:
    profile_id: str
    source_id: str
    point_id: str
    state: PlatingEvidenceState
    evidence_type: str
    source_cell_chemistry: str | None
    source_cell_format: str | None
    validated_for_tattu: bool
    confidence: str
    source_internal_conflict: bool
    source_internal_conflict_note: str | None
    point: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


@dataclass(frozen=True)
class ColdChargeObserverStepResult:
    time_s: float
    temperature_c: float
    soc: float
    requested_charge_current_a: float
    actual_charge_current_a: float
    requested_c_rate: float
    actual_c_rate: float
    charge_policy_status: ChargePolicyStatus
    plating_evidence_state: PlatingEvidenceState
    matched_evidence: tuple[ColdChargeEvidenceMatch, ...]
    actual_exposure_increment_ah: float
    blocked_request_increment_ah: float
    counterfactual_exposure_increment_ah: float
    research_only: bool = True
    counterfactual: bool = False
    not_actual_operation: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["charge_policy_status"] = self.charge_policy_status.value
        data["plating_evidence_state"] = self.plating_evidence_state.value
        data["matched_evidence"] = [match.to_dict()
                                    for match in self.matched_evidence]
        return data


@dataclass(frozen=True)
class ColdChargeObserverSummary:
    actual_cold_charge_exposure_ah: float
    blocked_charge_request_ah: float
    counterfactual_plating_exposure_ah: float
    minimum_requested_charge_temperature_c: float | None
    maximum_requested_charge_c_rate: float
    maximum_soc_during_cold_charge_request: float | None
    experimental_plating_events: int
    experimental_no_plating_events: int
    model_predicted_plating_events: int
    unknown_events: int
    validated_for_tattu: bool
    manufacturer_policy_validated_for_tattu: bool
    readiness: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["readiness"] = list(self.readiness)
        return data


class ColdChargeObserver:
    """Strict-point evidence query and stateful exposure accumulator."""

    def __init__(self, charge_policy: BatteryChargePolicy,
                 evidence_profiles: Iterable[ColdChargeEvidenceProfile],
                 numerical_epsilon: float = 1e-9):
        self.charge_policy = charge_policy
        self.evidence_profiles = tuple(evidence_profiles)
        if not self.evidence_profiles:
            raise ValueError("Observer требует хотя бы один evidence profile")
        if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0:
            raise ValueError("numerical_epsilon должен быть положительным")
        self.numerical_epsilon = numerical_epsilon
        self.state = ColdChargeObserverState()
        self.events: list[ColdChargeObserverStepResult] = []

    @classmethod
    def load_builtin(cls) -> "ColdChargeObserver":
        return cls(
            BatteryChargePolicy.load_builtin(),
            tuple(ColdChargeEvidenceProfile.load_builtin(profile_id)
                  for profile_id in DEFAULT_EVIDENCE_PROFILE_IDS),
        )

    def reset(self) -> None:
        self.state = ColdChargeObserverState()
        self.events = []

    def _policy_status(self, inputs: ColdChargeObserverInput,
                       requested_charge_a: float,
                       requested_c_rate: float) -> ChargePolicyStatus:
        if requested_charge_a <= self.numerical_epsilon:
            return ChargePolicyStatus.NO_CHARGE_REQUEST
        if (inputs.battery_temperature_c
                < self.charge_policy.charge_temperature_min_c):
            return ChargePolicyStatus.BLOCKED_BELOW_MIN_TEMPERATURE
        if (inputs.battery_temperature_c
                > self.charge_policy.charge_temperature_max_c):
            return ChargePolicyStatus.BLOCKED_ABOVE_MAX_TEMPERATURE
        if requested_c_rate > self.charge_policy.max_charge_c_rate + self.numerical_epsilon:
            return ChargePolicyStatus.LIMITED_BY_MAX_C_RATE
        if (inputs.soc_charge_limit is not None
                and inputs.battery_soc >= inputs.soc_charge_limit):
            return ChargePolicyStatus.LIMITED_BY_SOC
        if (inputs.cell_voltage_v is not None
                and inputs.cell_voltage_charge_limit_v is not None
                and inputs.cell_voltage_v >= inputs.cell_voltage_charge_limit_v):
            return ChargePolicyStatus.LIMITED_BY_VOLTAGE
        if inputs.existing_battery_limit_active:
            return ChargePolicyStatus.BLOCKED_BY_EXISTING_BATTERY_LIMIT
        return ChargePolicyStatus.ALLOWED

    def query_evidence(self, temperature_c: float, charge_c_rate: float,
                       soc: float) -> tuple[ColdChargeEvidenceMatch, ...]:
        matches: list[ColdChargeEvidenceMatch] = []
        for profile in self.evidence_profiles:
            for point in profile.points:
                if (abs(temperature_c - point.temperature_c) > self.numerical_epsilon
                        or abs(charge_c_rate - point.charge_c_rate)
                        > self.numerical_epsilon):
                    continue
                state = point.evidence_result
                if point.soc_onset is not None:
                    state = (PlatingEvidenceState.MODEL_PREDICTED_PLATING
                             if soc + self.numerical_epsilon >= point.soc_onset
                             else PlatingEvidenceState.BELOW_PUBLISHED_MODEL_ONSET)
                matches.append(ColdChargeEvidenceMatch(
                    profile_id=profile.profile_id, source_id=profile.source_id,
                    point_id=point.point_id, state=state,
                    evidence_type=point.evidence_type,
                    source_cell_chemistry=profile.chemistry,
                    source_cell_format=profile.cell_format,
                    validated_for_tattu=profile.validated_for_tattu,
                    confidence=point.confidence,
                    source_internal_conflict=profile.source_internal_conflict,
                    source_internal_conflict_note=(
                        profile.source_internal_conflict_note),
                    point=point.to_dict(),
                ))
        return tuple(matches)

    @staticmethod
    def _aggregate_state(matches: tuple[ColdChargeEvidenceMatch, ...]
                         ) -> PlatingEvidenceState:
        if not matches:
            return PlatingEvidenceState.UNKNOWN
        states = {match.state for match in matches}
        positive = states & {
            PlatingEvidenceState.PLATING_OBSERVED_EXPERIMENT,
            PlatingEvidenceState.MODEL_PREDICTED_PLATING,
        }
        negative = states & {
            PlatingEvidenceState.PLATING_NOT_OBSERVED_EXPERIMENT,
            PlatingEvidenceState.BELOW_PUBLISHED_MODEL_ONSET,
        }
        if positive and negative:
            return PlatingEvidenceState.CONFLICTING_EVIDENCE
        for state in (
                PlatingEvidenceState.PLATING_OBSERVED_EXPERIMENT,
                PlatingEvidenceState.MODEL_PREDICTED_PLATING,
                PlatingEvidenceState.PLATING_NOT_OBSERVED_EXPERIMENT,
                PlatingEvidenceState.BELOW_PUBLISHED_MODEL_ONSET):
            if state in states:
                return state
        return PlatingEvidenceState.UNKNOWN

    def step(self, inputs: ColdChargeObserverInput) -> ColdChargeObserverStepResult:
        requested_charge = max(0.0, -inputs.requested_battery_current_a)
        actual_charge = max(0.0, -inputs.actual_battery_current_a)
        requested_c_rate = requested_charge / inputs.battery_capacity_ah
        actual_c_rate = actual_charge / inputs.battery_capacity_ah
        status = self._policy_status(inputs, requested_charge, requested_c_rate)
        matches = (() if requested_charge <= self.numerical_epsilon else
                   self.query_evidence(inputs.battery_temperature_c,
                                       requested_c_rate, inputs.battery_soc))
        evidence_state = (PlatingEvidenceState.NOT_APPLICABLE
                          if requested_charge <= self.numerical_epsilon
                          else self._aggregate_state(matches))

        # "Cold charge" is tied to the selected target policy boundary rather
        # than to a plating verdict from an analogue.  At the exact lower
        # boundary charging is permitted but still retained diagnostically.
        actual_cold = (actual_charge > self.numerical_epsilon
                       and inputs.battery_temperature_c
                       <= self.charge_policy.charge_temperature_min_c)
        blocked_by_temperature = status in {
            ChargePolicyStatus.BLOCKED_BELOW_MIN_TEMPERATURE,
            ChargePolicyStatus.BLOCKED_ABOVE_MAX_TEMPERATURE,
        }
        request_positive = evidence_state in {
            PlatingEvidenceState.PLATING_OBSERVED_EXPERIMENT,
            PlatingEvidenceState.MODEL_PREDICTED_PLATING,
            PlatingEvidenceState.CONFLICTING_EVIDENCE,
        }
        actual_increment = (actual_charge * inputs.dt_s / 3600.0
                            if actual_cold else 0.0)
        blocked_increment = (requested_charge * inputs.dt_s / 3600.0
                             if blocked_by_temperature
                             and actual_charge <= self.numerical_epsilon else 0.0)
        counterfactual_increment = (requested_charge * inputs.dt_s / 3600.0
                                    if blocked_increment > 0 and request_positive
                                    else 0.0)
        counterfactual_condition = (
            bool(matches)
            and abs(requested_charge - actual_charge) > self.numerical_epsilon)

        result = ColdChargeObserverStepResult(
            time_s=self.state.elapsed_s + inputs.dt_s,
            temperature_c=inputs.battery_temperature_c,
            soc=inputs.battery_soc,
            requested_charge_current_a=requested_charge,
            actual_charge_current_a=actual_charge,
            requested_c_rate=requested_c_rate,
            actual_c_rate=actual_c_rate,
            charge_policy_status=status,
            plating_evidence_state=evidence_state,
            matched_evidence=matches,
            actual_exposure_increment_ah=actual_increment,
            blocked_request_increment_ah=blocked_increment,
            counterfactual_exposure_increment_ah=counterfactual_increment,
            counterfactual=counterfactual_condition,
            not_actual_operation=counterfactual_condition,
        )
        state = self.state
        state.elapsed_s += inputs.dt_s
        state.actual_cold_charge_exposure_ah += actual_increment
        state.blocked_charge_request_ah += blocked_increment
        state.counterfactual_plating_exposure_ah += counterfactual_increment
        if requested_charge > self.numerical_epsilon:
            state.min_requested_charge_temperature_c = (
                inputs.battery_temperature_c
                if state.min_requested_charge_temperature_c is None else
                min(state.min_requested_charge_temperature_c,
                    inputs.battery_temperature_c))
            state.max_requested_charge_c_rate = max(
                state.max_requested_charge_c_rate, requested_c_rate)
            if (inputs.battery_temperature_c
                    <= self.charge_policy.charge_temperature_min_c):
                state.max_soc_during_cold_charge_request = (
                    inputs.battery_soc
                    if state.max_soc_during_cold_charge_request is None else
                    max(state.max_soc_during_cold_charge_request,
                        inputs.battery_soc))
        if not matches and requested_charge > self.numerical_epsilon:
            state.unknown_match_count += 1
        for match in matches:
            if match.state == PlatingEvidenceState.PLATING_OBSERVED_EXPERIMENT:
                state.experimental_plating_match_count += 1
            elif match.state == PlatingEvidenceState.PLATING_NOT_OBSERVED_EXPERIMENT:
                state.experimental_no_plating_match_count += 1
            elif match.state == PlatingEvidenceState.MODEL_PREDICTED_PLATING:
                state.model_predicted_plating_match_count += 1
        self.events.append(result)
        return result

    def summary(self) -> ColdChargeObserverSummary:
        state = self.state
        return ColdChargeObserverSummary(
            actual_cold_charge_exposure_ah=state.actual_cold_charge_exposure_ah,
            blocked_charge_request_ah=state.blocked_charge_request_ah,
            counterfactual_plating_exposure_ah=(
                state.counterfactual_plating_exposure_ah),
            minimum_requested_charge_temperature_c=(
                state.min_requested_charge_temperature_c),
            maximum_requested_charge_c_rate=state.max_requested_charge_c_rate,
            maximum_soc_during_cold_charge_request=(
                state.max_soc_during_cold_charge_request),
            experimental_plating_events=(
                state.experimental_plating_match_count),
            experimental_no_plating_events=(
                state.experimental_no_plating_match_count),
            model_predicted_plating_events=(
                state.model_predicted_plating_match_count),
            unknown_events=state.unknown_match_count,
            validated_for_tattu=False,
            manufacturer_policy_validated_for_tattu=(
                self.charge_policy.validated_for_target_battery),
            readiness=(
                ColdChargeReadiness.READY_FOR_EVIDENCE_DIAGNOSTICS.value,
                ColdChargeReadiness.NOT_VALIDATED_FOR_TATTU_PLATING_PREDICTION.value,
                ColdChargeReadiness.NO_QUANTITATIVE_DAMAGE_MODEL.value,
            ),
        )

    def provenance_snapshot(self) -> dict[str, Any]:
        return {
            "manufacturer_policy": self.charge_policy.to_snapshot(),
            "evidence_profiles": [profile.to_snapshot()
                                  for profile in self.evidence_profiles],
            "cross_source_interpolation": False,
            "nearest_neighbour_matching": False,
            "quantitative_damage_model": False,
        }


def export_cold_charge_diagnostics(observer: ColdChargeObserver,
                                   output_dir: str | Path) -> tuple[Path, Path]:
    """Write one CSV row per source match and a provenance-rich JSON summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "cold_charge_events.csv"
    fields = (
        "time_s", "temperature_c", "soc", "requested_charge_current_a",
        "actual_charge_current_a", "requested_c_rate", "actual_c_rate",
        "charge_policy_status", "plating_evidence_state", "source_id",
        "evidence_type", "source_cell_chemistry", "source_cell_format",
        "validated_for_tattu", "confidence", "counterfactual",
        "not_actual_operation",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in observer.events:
            matches: tuple[ColdChargeEvidenceMatch | None, ...] = (
                event.matched_evidence or (None,))
            for match in matches:
                writer.writerow({
                    "time_s": event.time_s,
                    "temperature_c": event.temperature_c,
                    "soc": event.soc,
                    "requested_charge_current_a": event.requested_charge_current_a,
                    "actual_charge_current_a": event.actual_charge_current_a,
                    "requested_c_rate": event.requested_c_rate,
                    "actual_c_rate": event.actual_c_rate,
                    "charge_policy_status": event.charge_policy_status.value,
                    "plating_evidence_state": event.plating_evidence_state.value,
                    "source_id": "" if match is None else match.source_id,
                    "evidence_type": "" if match is None else match.evidence_type,
                    "source_cell_chemistry": ("" if match is None else
                                              match.source_cell_chemistry),
                    "source_cell_format": ("" if match is None else
                                           match.source_cell_format),
                    "validated_for_tattu": (False if match is None else
                                             match.validated_for_tattu),
                    "confidence": "" if match is None else match.confidence,
                    "counterfactual": event.counterfactual,
                    "not_actual_operation": event.not_actual_operation,
                })
    json_path = output / "cold_charge_summary.json"
    payload = {
        "summary": observer.summary().to_dict(),
        "provenance": observer.provenance_snapshot(),
        "events": [event.to_dict() for event in observer.events],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return csv_path, json_path
