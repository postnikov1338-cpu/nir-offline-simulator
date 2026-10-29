"""Engine profiles and the Phase 1 Limbach full-load curve model.

The digitized Limbach data are a one-dimensional full-load characteristic.
They are deliberately not exposed as a two-dimensional BSFC map and are not
used by the legacy mission power dispatch.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from component_profiles import ParameterProvenance, SourceRef
from monotone_interpolation import pchip_interpolate


ROOT = Path(__file__).resolve().parent
BUILTIN_ENGINE_DIR = ROOT / "data" / "engines"
MECHANICAL_HP_TO_KW = 0.7457
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


class EngineMapOutOfRangeError(ValueError):
    """Requested speed is outside the digitized engine characteristic."""


@dataclass(frozen=True)
class EngineCurvePoint:
    """Only source axes plus digitization quality metadata are stored."""

    rpm: float
    power_hp_raw: float
    sfc_g_hph_raw: float
    confidence: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value > 0 for value in
                   (self.rpm, self.power_hp_raw, self.sfc_g_hph_raw)):
            raise ValueError("Точка характеристики ДВС должна содержать положительные конечные значения")
        if self.confidence not in _CONFIDENCE_RANK:
            raise ValueError("confidence точки должен быть high, medium или low")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class EngineProfile:
    profile_id: str
    revision: int
    manufacturer: str
    model: str
    engine_type: str
    displacement_cm3: float
    dry_mass_kg: float
    rated_power_kw: float
    rated_power_rpm: float
    engine_max_rpm: float
    curve_kind: str
    source_type: str
    precision: str
    points: tuple[EngineCurvePoint, ...]
    source: SourceRef
    provenance: dict[str, ParameterProvenance]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or self.revision < 1:
            raise ValueError("Профиль ДВС: неверные profile_id или revision")
        if self.curve_kind != "full_load":
            raise ValueError("Phase 1 поддерживает только curve_kind='full_load'")
        if self.source_type != "digitized_from_manufacturer_graph":
            raise ValueError("Оцифрованный профиль должен иметь явный source_type")
        if self.precision != "approximate":
            raise ValueError("Ручная оцифровка должна быть помечена как approximate")
        if len(self.points) < 2:
            raise ValueError("Характеристика ДВС должна содержать минимум две точки")
        speeds = [point.rpm for point in self.points]
        if any(right <= left for left, right in zip(speeds, speeds[1:])):
            raise ValueError("Обороты в характеристике ДВС должны строго возрастать")
        if not all(math.isfinite(value) and value > 0 for value in (
            self.displacement_cm3, self.dry_mass_kg, self.rated_power_kw,
            self.rated_power_rpm, self.engine_max_rpm,
        )):
            raise ValueError("Паспортные параметры ДВС должны быть положительными")
        if self.engine_max_rpm < self.rated_power_rpm:
            raise ValueError("Максимальные обороты не могут быть ниже паспортных")

    @property
    def digitized_map_min_rpm(self) -> float:
        return self.points[0].rpm

    @property
    def digitized_map_max_rpm(self) -> float:
        return self.points[-1].rpm

    @property
    def supports_part_load(self) -> bool:
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineProfile":
        rated = data["rated"]
        source_data = data["source"]
        source = SourceRef(
            source_id=source_data["source_id"],
            title=source_data["title"],
            url=source_data["url"],
            source_type=source_data["source_type"],
            scope=source_data["scope"],
            accessed_on=source_data.get("accessed_on"),
            notes=source_data.get("notes", ""),
        )
        provenance = {
            key: ParameterProvenance(
                source_id=value["source_id"],
                source_type=value["source_type"],
                confidence=value["confidence"],
                note=value.get("note", ""),
            )
            for key, value in data.get("provenance", {}).items()
        }
        points = tuple(EngineCurvePoint(
            rpm=float(point["rpm"]),
            power_hp_raw=float(point["power_hp_raw"]),
            sfc_g_hph_raw=float(point["sfc_g_hph_raw"]),
            confidence=point["confidence"],
            warnings=tuple(point.get("warnings", ())),
        ) for point in data["points"])
        profile = cls(
            profile_id=data["profile_id"], revision=int(data["revision"]),
            manufacturer=data["manufacturer"], model=data["model"],
            engine_type=data["engine_type"], displacement_cm3=float(data["displacement_cm3"]),
            dry_mass_kg=float(data["dry_mass_kg"]),
            rated_power_kw=float(rated["power_kw"]), rated_power_rpm=float(rated["rpm"]),
            engine_max_rpm=float(data["engine_max_rpm"]), curve_kind=data["curve_kind"],
            source_type=data["source_type"], precision=data["precision"], points=points,
            source=source, provenance=provenance, warnings=tuple(data.get("warnings", ())),
        )
        validity = data.get("validity", {})
        if validity and (float(validity["digitized_map_min_rpm"]) != profile.digitized_map_min_rpm
                         or float(validity["digitized_map_max_rpm"]) != profile.digitized_map_max_rpm):
            raise ValueError("Границы validity не совпадают с оцифрованными точками")
        return profile

    @classmethod
    def load(cls, path: str | Path) -> "EngineProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def load_builtin(cls, profile_id: str = "limbach_l275e_digitized_v1") -> "EngineProfile":
        path = BUILTIN_ENGINE_DIR / f"{profile_id}.json"
        if not path.is_file():
            raise ValueError(f"Неизвестный встроенный профиль ДВС: {profile_id}")
        return cls.load(path)

    def to_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe source data; derived curve values are not stored."""
        return {
            "profile_id": self.profile_id,
            "revision": self.revision,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "engine_type": self.engine_type,
            "displacement_cm3": self.displacement_cm3,
            "dry_mass_kg": self.dry_mass_kg,
            "rated": {"power_kw": self.rated_power_kw, "rpm": self.rated_power_rpm},
            "engine_max_rpm": self.engine_max_rpm,
            "curve_kind": self.curve_kind,
            "source_type": self.source_type,
            "precision": self.precision,
            "validity": {
                "digitized_map_min_rpm": self.digitized_map_min_rpm,
                "digitized_map_max_rpm": self.digitized_map_max_rpm,
                "out_of_range": "stop",
            },
            "source": self.source.to_dict(),
            "provenance": {key: value.to_dict() for key, value in self.provenance.items()},
            "warnings": list(self.warnings),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class EngineOperatingPoint:
    """Derived point on the one-dimensional full-load characteristic."""

    rpm: float
    power_kw: float
    torque_nm: float
    bsfc_g_kwh: float
    fuel_flow_kg_h: float
    confidence: str
    warnings: tuple[str, ...]
    curve_kind: str = "full_load"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class EngineEnvelopePoint:
    """Maximum available shaft envelope without inventing part-load fuel data."""

    rpm: float
    maximum_power_kw: float
    maximum_torque_nm: float
    support_kind: str
    source_type: str
    confidence: str
    full_load_point: EngineOperatingPoint | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["full_load_point"] = (None if self.full_load_point is None
                                   else self.full_load_point.to_dict())
        data["warnings"] = list(self.warnings)
        return data


class EngineModel:
    """Shape-preserving PCHIP evaluation of a full-load source curve."""

    def __init__(self, profile: EngineProfile, part_load_model=None):
        if profile.curve_kind != "full_load":
            raise ValueError("EngineModel Phase 1 принимает только full_load")
        self.profile = profile
        self._rpm = tuple(point.rpm for point in profile.points)
        self._power_hp = tuple(point.power_hp_raw for point in profile.points)
        self._sfc_g_hph = tuple(point.sfc_g_hph_raw for point in profile.points)
        self.part_load_model = part_load_model

    @property
    def supports_part_load(self) -> bool:
        return self.part_load_model is not None

    def part_load_at(self, rpm: float, torque_nm: float):
        """Delegate optional reconstructed fuel physics to its own model."""
        if self.part_load_model is None:
            return None
        return self.part_load_model.evaluate(self, rpm, torque_nm)

    @staticmethod
    def _derived(rpm: float, power_hp: float, sfc_g_hph: float,
                 confidence: str, warnings: tuple[str, ...]) -> EngineOperatingPoint:
        power_kw = power_hp * MECHANICAL_HP_TO_KW
        torque_nm = 9550.0 * power_kw / rpm
        bsfc_g_kwh = sfc_g_hph / MECHANICAL_HP_TO_KW
        fuel_flow_kg_h = bsfc_g_kwh * power_kw / 1000
        return EngineOperatingPoint(
            rpm=rpm, power_kw=power_kw, torque_nm=torque_nm,
            bsfc_g_kwh=bsfc_g_kwh, fuel_flow_kg_h=fuel_flow_kg_h,
            confidence=confidence, warnings=warnings,
        )

    def full_load_at_rpm(self, rpm: float) -> EngineOperatingPoint:
        rpm = float(rpm)
        if not math.isfinite(rpm):
            raise ValueError("Обороты ДВС должны быть конечным числом")
        lo, hi = self._rpm[0], self._rpm[-1]
        if not lo <= rpm <= hi:
            raise EngineMapOutOfRangeError(
                f"engine_map_out_of_range: {rpm:g} rpm вне оцифрованной full-load характеристики {lo:g}…{hi:g} rpm"
            )
        if rpm in self._rpm:
            index = self._rpm.index(rpm)
            point = self.profile.points[index]
            return self._derived(rpm, point.power_hp_raw, point.sfc_g_hph_raw,
                                 point.confidence, point.warnings)
        right = next(index for index, source_rpm in enumerate(self._rpm)
                     if source_rpm > rpm)
        left_point, right_point = self.profile.points[right - 1], self.profile.points[right]
        power_hp = pchip_interpolate(self._rpm, self._power_hp, rpm)
        sfc = pchip_interpolate(self._rpm, self._sfc_g_hph, rpm)
        confidence = min((left_point.confidence, right_point.confidence), key=_CONFIDENCE_RANK.get)
        warnings = tuple(dict.fromkeys((*left_point.warnings, *right_point.warnings,
                                        "interpolated_between_digitized_points",
                                        "pchip_shape_preserving_interpolation")))
        return self._derived(rpm, power_hp, sfc, confidence, warnings)

    def digitized_operating_points(self) -> tuple[EngineOperatingPoint, ...]:
        return tuple(self.full_load_at_rpm(rpm) for rpm in self._rpm)

    def maximum_available_at_rpm(self, rpm: float) -> EngineEnvelopePoint:
        """Evaluate the confirmed full-load envelope without extrapolation."""
        rpm = float(rpm)
        if not math.isfinite(rpm):
            raise ValueError("Обороты ДВС должны быть конечным числом")
        if self.profile.digitized_map_min_rpm <= rpm <= self.profile.digitized_map_max_rpm:
            point = self.full_load_at_rpm(rpm)
            return EngineEnvelopePoint(
                rpm=rpm, maximum_power_kw=point.power_kw,
                maximum_torque_nm=point.torque_nm,
                support_kind="digitized_full_load_curve",
                source_type="digitized_manufacturer_graph",
                confidence=point.confidence, full_load_point=point,
                warnings=point.warnings,
            )
        if math.isclose(rpm, self.profile.rated_power_rpm,
                        rel_tol=0.0, abs_tol=1e-9):
            power_kw = self.profile.rated_power_kw
            torque_nm = 9550.0 * power_kw / rpm
            return EngineEnvelopePoint(
                rpm=rpm, maximum_power_kw=power_kw,
                maximum_torque_nm=torque_nm,
                support_kind="confirmed_rated_point",
                source_type="manufacturer", confidence="high",
                full_load_point=None,
                warnings=("rated_point_has_no_confirmed_sfc",),
            )
        raise EngineMapOutOfRangeError(
            f"engine_envelope_out_of_range: {rpm:g} rpm не принадлежит "
            f"оцифрованному диапазону {self.profile.digitized_map_min_rpm:g}…"
            f"{self.profile.digitized_map_max_rpm:g} rpm и не равно подтверждённой "
            f"паспортной точке {self.profile.rated_power_rpm:g} rpm")

    def maximum_available_power(self, rpm: float) -> float:
        return self.maximum_available_at_rpm(rpm).maximum_power_kw

    def maximum_available_torque(self, rpm: float) -> float:
        return self.maximum_available_at_rpm(rpm).maximum_torque_nm


def draw_engine_full_load_curve(figure, model: EngineModel, operating_rpm: float | None = None) -> None:
    """Draw the reference curve without implying that legacy dispatch uses it."""
    figure.clear()
    axes = figure.subplots(1, 3)
    points = model.digitized_operating_points()
    source_rpm = [point.rpm for point in points]
    dense_rpm = [float(rpm) for rpm in range(
        int(model.profile.digitized_map_min_rpm),
        int(model.profile.digitized_map_max_rpm) + 1, 25)]
    dense_points = [model.full_load_at_rpm(rpm) for rpm in dense_rpm]
    series = (
        ([point.power_kw for point in points],
         [point.power_kw for point in dense_points], "Мощность, кВт", "Power(n)"),
        ([point.torque_nm for point in points],
         [point.torque_nm for point in dense_points], "Момент, Н·м", "Torque(n), рассчитан из мощности"),
        ([point.bsfc_g_kwh for point in points],
         [point.bsfc_g_kwh for point in dense_points], "BSFC, г/(кВт·ч)", "BSFC_FL(n)"),
    )
    current = model.full_load_at_rpm(operating_rpm) if operating_rpm is not None else None
    current_values = (current.power_kw, current.torque_nm, current.bsfc_g_kwh) if current else ()
    for index, (source_values, dense_values, ylabel, title) in enumerate(series):
        axes[index].plot(dense_rpm, dense_values, "-", lw=2,
                         label="PCHIP-модель")
        axes[index].plot(source_rpm, source_values, "o", ms=5,
                         label="Оцифрованные исходные точки")
        if current:
            axes[index].plot([current.rpm], [current_values[index]], "ro", label="Рабочая точка")
        axes[index].set(xlabel="Частота вращения, об/мин", ylabel=ylabel, title=title)
        axes[index].grid(alpha=.25)
        axes[index].legend(fontsize=8)
    figure.suptitle("Limbach L275E — приблизительная full-load характеристика; legacy-расчётом не используется")
    figure.text(.5, .015, "Ручная оцифровка графика производителя; экстраполяция и трактовка как 2D BSFC-карты запрещены.",
                ha="center", fontsize=9)
    figure.subplots_adjust(bottom=.22, top=.80, wspace=.34)


def load_builtin_engine_profile(profile_id: str = "limbach_l275e_digitized_v1") -> EngineProfile:
    return EngineProfile.load_builtin(profile_id)
