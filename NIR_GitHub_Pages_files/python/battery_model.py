"""Отдельная модель АКБ: OCV + R и квазистационарная поправка Пёйкерта.

SOC — физический заряд / номинальная ёмкость. Поправка Пёйкерта меняет
доступность заряда, а не создаёт дополнительный расход энергии или нагрев.
Для переменного тока это приближение, требующее совместной калибровки с OCV/R.
"""
from __future__ import annotations
from bisect import bisect_right
import math
from battery_profiles import temperature_defaults


class TemperatureRangeError(ValueError):
    """No extrapolation outside the measured electrical profile."""


CHARGE_REASONS = {
    'allowed': 'Заряд разрешён',
    'below_zero': 'Заряд запрещён: температура АКБ ниже 0 °C',
    'below_manufacturer_temperature': 'Заряд вне паспортного диапазона: температура ниже минимума',
    'above_manufacturer_temperature': 'Заряд вне паспортного диапазона: температура выше максимума',
    'agri_temperature': 'Заряд запрещён исследовательскими ограничениями AGRI',
    'soc_target': 'Достигнут целевой SOC',
    'voltage_limit': 'Достигнут предел напряжения заряда',
}


class BatteryModel:
    def __init__(self, config):
        self.config = config
        self.q_ah = config.get("capacity_ah")
        if self.q_ah is None:  # Совместимость с прежними конфигурациями DEMO.
            self.q_ah = config["mass_kg"] * config["specific_energy_kwh_kg"] * 1000 / config["nominal_v"]
        p = config.get("peukert", {})
        self.enabled = p.get("enabled", False)
        self.exponent = p.get("exponent", 1.0)
        self.reference_current_a = p.get("reference_current_a", self.q_ah)
        self.temperature = config.get('temperature', {'enabled': False})
        self.temperature_enabled = self.temperature.get('enabled', False)
        self.profile = self.temperature.get('profile', temperature_defaults()['profile'])
        self.limit_policy = self.temperature.get('current_limit_policy', 'basic')
        self.agri_limits = self.temperature.get('agri_limits', temperature_defaults()['agri_limits'])
        # Missing charge_policy is the frozen pre-Phase-6C rule.  New physical
        # target-battery configurations must provide an explicit profile.
        self.charge_policy = config.get('charge_policy')
        self.validate()

    def validate(self):
        b = self.config
        if not isinstance(self.temperature_enabled, bool):
            raise ValueError('temperature.enabled должно быть true или false')
        if self.limit_policy not in ('basic', 'agri_analogue'):
            raise ValueError('Неизвестный профиль температурных ограничений')
        for kind in ('capacity', 'resistance'):
            x, y = self.profile[kind+'_temperature_c'], self.profile[kind+'_factor']
            if len(x) < 2 or len(x) != len(y) or not all(math.isfinite(v) for v in [*x, *y]):
                raise ValueError('Неверная температурная таблица')
            if any(v <= u for u, v in zip(x, x[1:])) or min(x) <= -273.15 or min(y) <= 0:
                raise ValueError('Неверные узлы температурной таблицы')
            if kind == 'capacity' and (max(y) > 1 or any(v < u for u, v in zip(y, y[1:]))):
                raise ValueError('Коэффициент ёмкости должен возрастать и не превышать 1')
            if kind == 'resistance' and any(v > u for u, v in zip(y, y[1:])):
                raise ValueError('Коэффициент сопротивления должен убывать')
            ref = self.profile['reference_temperature_c']
            if not math.isfinite(ref) or not x[0] <= ref <= x[-1] or abs(self._interpolate(x, y, ref)-1) > 1e-9:
                raise ValueError('Температурная таблица должна быть нормирована при Tref')
        if self.profile.get('out_of_range') != 'stop':
            raise ValueError('Экстраполяция температурных характеристик не разрешена')
        for key in ('charge_bands', 'discharge_bands'):
            bands = self.agri_limits[key]
            if not bands:
                raise ValueError('Пустая таблица токовых ограничений')
            for index, band in enumerate(bands):
                if len(band) != 3 or not all(math.isfinite(v) for v in band) or band[0] >= band[1] or band[2] <= 0:
                    raise ValueError('Неверная таблица токовых ограничений')
                if index and bands[index-1][1] != band[0]:
                    raise ValueError('Температурные диапазоны должны стыковаться')
        for name, value in {"capacity_ah": self.q_ah, "reference_current_a": self.reference_current_a,
                            **{k: b[k] for k in ("mass_kg", "nominal_v", "series_cells",
                              "charge_c", "discharge_c", "resistance_ohm", "min_cell_v", "max_cell_v")}}.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"АКБ: {name} должно быть конечным положительным числом")
        if b["series_cells"] != int(b["series_cells"]):
            raise ValueError("АКБ: число последовательных ячеек должно быть целым")
        if not isinstance(self.enabled, bool):
            raise ValueError("АКБ: peukert.enabled должно быть true или false")
        if not math.isfinite(self.exponent) or not 1 <= self.exponent <= 1.5:
            raise ValueError("Показатель Пёйкерта k должен быть в пределах 1–1,5")
        if not 0 <= b["soc_floor"] < b["soc0"] <= b["soc_target"] <= 1:
            raise ValueError("АКБ: неверные границы SOC")
        if b["min_cell_v"] >= b["max_cell_v"]:
            raise ValueError("АКБ: нижнее напряжение должно быть меньше верхнего")
        for key in ("system_discharge_a", "system_charge_a"):
            value = b.get(key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"АКБ: неверный предел {key}")
        if self.charge_policy is not None:
            policy = self.charge_policy
            required = (
                'profile_id', 'source_type', 'charge_temperature_min_c',
                'charge_temperature_max_c', 'max_charge_c_rate')
            if any(key not in policy for key in required):
                raise ValueError('АКБ: неполный charge_policy')
            low = policy['charge_temperature_min_c']
            high = policy['charge_temperature_max_c']
            rate = policy['max_charge_c_rate']
            if (not all(math.isfinite(value) for value in (low, high, rate))
                    or low <= -273.15 or high <= low or rate <= 0):
                raise ValueError('АКБ: неверные пределы charge_policy')
            if policy['source_type'] != 'manufacturer':
                raise ValueError('АКБ: physical charge_policy должна быть manufacturer')
        x, y = b["ocv_soc"], b["ocv_cell_v"]
        if len(x) < 2 or len(x) != len(y) or not all(math.isfinite(v) for v in [*x, *y]):
            raise ValueError("АКБ: неверная OCV-карта")
        if x[0] != 0 or x[-1] != 1 or any(v <= u for u, v in zip(x, x[1:])):
            raise ValueError("АКБ: SOC карты должен возрастать от 0 до 1")
        if any(v < u for u, v in zip(y, y[1:])) or min(y) <= 0:
            raise ValueError("АКБ: напряжение OCV должно быть положительным и не убывать")

    @property
    def nominal_energy_wh(self):
        return self.q_ah * self.config["nominal_v"]

    @property
    def discharge_limit_a(self):
        return min(self.q_ah * self.config["discharge_c"],
                   self.config.get("system_discharge_a") or math.inf)

    @staticmethod
    def _interpolate(x, y, temperature):
        if not math.isfinite(temperature) or temperature <= -273.15:
            raise ValueError('Неверная температура АКБ')
        if not x[0] <= temperature <= x[-1]:
            raise TemperatureRangeError(f'Температура {temperature:.3f} °C вне таблицы {x[0]}…{x[-1]} °C; экстраполяция запрещена')
        j = min(len(x)-2, max(0, bisect_right(x, temperature)-1))
        return y[j] + (y[j+1]-y[j])*(temperature-x[j])/(x[j+1]-x[j])

    def temperature_factors(self, temperature_c=25.0):
        temperature_c = float(temperature_c)
        if not math.isfinite(temperature_c) or temperature_c <= -273.15:
            raise ValueError('Неверная температура АКБ')
        if not self.temperature_enabled:
            return 1.0, 1.0
        return tuple(self._interpolate(self.profile[k+'_temperature_c'], self.profile[k+'_factor'], temperature_c)
                     for k in ('capacity', 'resistance'))

    def resistance_at_temperature(self, temperature_c=25.0):
        return self.config['resistance_ohm']*self.temperature_factors(temperature_c)[1]

    @staticmethod
    def _band_rate(bands, low, high):
        if low < bands[0][0] or high > bands[-1][1]:
            return 0.0
        # Both spatial nodes and ALL intermediate temperatures obey the limit.
        return min(rate for left, right, rate in bands if left <= high and right >= low)

    def temperature_current_limits(self, temperature_c=25.0, surface_c=None, temperature_bounds=None):
        temperature_c = float(temperature_c)
        surface_c = temperature_c if surface_c is None else surface_c
        surface_c = float(surface_c)
        if not all(math.isfinite(t) and t > -273.15 for t in (temperature_c, surface_c)):
            raise ValueError('Неверная температура АКБ')
        low, high = min(temperature_c, surface_c), max(temperature_c, surface_c)
        if temperature_bounds is not None:
            if len(temperature_bounds) != 2 or not all(math.isfinite(t) and t > -273.15 for t in temperature_bounds) or temperature_bounds[0] > low or temperature_bounds[1] < high:
                raise ValueError('Неверные температурные границы шага')
            low, high = temperature_bounds
        discharge = self.discharge_limit_a
        charge = min(self.q_ah*self.config['charge_c'], self.config.get('system_charge_a') or math.inf)
        reason = 'allowed'
        if self.limit_policy == 'agri_analogue':
            discharge = min(discharge, self.q_ah*self._band_rate(self.agri_limits['discharge_bands'], low, high))
            charge = min(charge, self.q_ah*self._band_rate(self.agri_limits['charge_bands'], low, high))
            if charge == 0:
                reason = 'agri_temperature'
        if self.charge_policy is None:
            if low < 0:
                charge, reason = 0.0, 'below_zero'
        else:
            policy = self.charge_policy
            charge = min(charge, self.q_ah*policy['max_charge_c_rate'])
            if low < policy['charge_temperature_min_c']:
                charge, reason = 0.0, 'below_manufacturer_temperature'
            elif high > policy['charge_temperature_max_c']:
                charge, reason = 0.0, 'above_manufacturer_temperature'
        return discharge, charge, reason

    def capacity_factor(self, current_a):
        """Доля Q_nom, доступная от полного заряда при постоянном токе.

        Ниже опорного тока ёмкость не увеличивается сверх номинальной.
        При заряде/покое поправка отключена; это не оценка старения АКБ.
        """
        if not math.isfinite(current_a):
            raise ValueError("Ток должен быть конечным")
        if not self.enabled or current_a <= self.reference_current_a:
            return 1.0
        return (self.reference_current_a / current_a) ** (self.exponent - 1)

    def effective_capacity_ah(self, current_a, temperature_c=25.0):
        return self.q_ah * self.capacity_factor(current_a)*self.temperature_factors(temperature_c)[0]

    def soc_floor_at_current(self, current_a, temperature_c=25.0):
        # Резерв сохраняется в долях НОМИНАЛЬНОГО заряда. Недоступная часть
        # добавляется к резерву: SOC_min = SOC_reserve + 1 - Q_eff/Q_nom.
        return min(1.0, self.config["soc_floor"] + 1 - self.capacity_factor(current_a)*self.temperature_factors(temperature_c)[0])

    def available_charge_ah(self, soc, current_a, temperature_c=25.0):
        return self.q_ah * max(0.0, soc - self.soc_floor_at_current(current_a, temperature_c))

    def ocv(self, soc):
        x, y = self.config["ocv_soc"], self.config["ocv_cell_v"]
        soc = min(1.0, max(0.0, soc))
        j = min(len(x) - 2, max(0, bisect_right(x, soc) - 1))
        return (y[j] + (y[j+1] - y[j]) * (soc - x[j]) / (x[j+1] - x[j])) * self.config["series_cells"]

    def energy_kwh(self, soc):
        """Физическая энергия по OCV, без умножения на поправку Пёйкерта."""
        knots = [0.0] + [x for x in self.config["ocv_soc"] if 0 < x < soc] + [soc]
        return self.q_ah * sum((self.ocv(x)+self.ocv(y))*(y-x)/2
                              for x, y in zip(knots, knots[1:])) / 1000

    def current_for_power(self, soc, power_kw, temperature_c=25.0):
        if not math.isfinite(power_kw) or not math.isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("АКБ: неверные SOC или мощность")
        u, r = self.ocv(soc), self.resistance_at_temperature(temperature_c)
        disc = u*u - 4*r*power_kw*1000
        if disc < -1e-7:
            raise ValueError("АКБ не может выдать эту мощность при заданном OCV и R")
        return 2*power_kw*1000 / (u + math.sqrt(max(disc, 0)))

    def power_limits(self, soc, dt, temperature_c=25.0, surface_c=None, temperature_bounds=None):
        """Пределы на ВСЕСЬ шаг; ток, напряжение и доступный заряд."""
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("АКБ: неверный шаг времени или SOC")
        b = self.config
        u, r = self.ocv(soc), self.resistance_at_temperature(temperature_c)
        discharge_limit, charge_limit, _ = self.temperature_current_limits(temperature_c, surface_c, temperature_bounds)
        umin, umax = b["series_cells"]*b["min_cell_v"], b["series_cells"]*b["max_cell_v"]
        upper = max(0.0, min(discharge_limit, (u-umin)/r, u/(2*r),
                            (soc-b["soc_floor"])*self.q_ah*3600/dt))
        # Условие монотонно по I. Находим предел, при котором SOC в конце
        # шага ещё не ниже границы доступного при этом токе заряда.
        if soc - upper*dt/(3600*self.q_ah) < self.soc_floor_at_current(upper, temperature_c):
            lo, hi = 0.0, upper
            for _ in range(48):
                mid = (lo+hi)/2
                if soc - mid*dt/(3600*self.q_ah) >= self.soc_floor_at_current(mid, temperature_c):
                    lo = mid
                else:
                    hi = mid
            upper = lo
        charge_a = max(0.0, min(charge_limit,
                               (umax-u)/r, (b["soc_target"]-soc)*self.q_ah*3600/dt))
        return (u-r*upper)*upper/1000, (u+r*charge_a)*charge_a/1000

    def step(self, soc, power_kw, dt, temperature_c=25.0, surface_c=None, temperature_bounds=None):
        pd, pc = self.power_limits(soc, dt, temperature_c, surface_c, temperature_bounds)
        # Zero limits are strict, not softened by the general power tolerance.
        if (power_kw < 0 and pc == 0) or (power_kw > 0 and pd == 0):
            raise ValueError('АКБ не принимает/не отдаёт мощность при текущих ограничениях')
        if power_kw > pd + 1e-8 or power_kw < -pc - 1e-8:
            raise ValueError("Превышены допустимые мощность или доступный заряд АКБ")
        current = self.current_for_power(soc, power_kw, temperature_c)
        r = self.resistance_at_temperature(temperature_c)
        # Даже при смене нагрузки заряд меняется ТОЛЬКО через физический ток.
        new_soc = soc - current*dt/(3600*self.q_ah)
        return new_soc, self.ocv(soc)-current*r, current, current*current*r/1000

    def diagnostics(self, soc, dt, temperature_c, surface_c, temperature_bounds=None):
        kc, kr = self.temperature_factors(temperature_c)
        pd, pc = self.power_limits(soc, dt, temperature_c, surface_c, temperature_bounds)
        dcap, ccap, reason = self.temperature_current_limits(temperature_c, surface_c, temperature_bounds)
        if reason == 'allowed' and pc == 0:
            reason = 'soc_target' if soc >= self.config['soc_target'] else 'voltage_limit'
        return dict(capacity_temperature_factor=kc, resistance_temperature_factor=kr,
                    battery_resistance_ohm=self.config['resistance_ohm']*kr,
                    max_charge_current_A=-self.current_for_power(soc, -pc, temperature_c),
                    max_discharge_current_A=self.current_for_power(soc, pd, temperature_c),
                    temperature_charge_current_cap_A=ccap, temperature_discharge_current_cap_A=dcap,
                    charge_allowed=bool(pc > 0), charge_limit_reason=reason,
                    charge_blocked_by_temperature=bool(ccap == 0))


def draw_capacity_curves(figure, battery):
    """Характеристики только АКБ; от модели ВМГ/ДВС не зависят."""
    import numpy as np
    figure.clear()
    axes = figure.subplots(1, 2)
    currents = np.linspace(0, battery.discharge_limit_a, 240)
    capacities = [battery.effective_capacity_ah(i) for i in currents]
    axes[0].plot(currents, capacities, label="При текущей настройке k")
    axes[0].axhline(battery.q_ah, color="gray", ls="--", label="Номинальная ёмкость")
    axes[0].set(xlabel="Ток разряда, А", ylabel="Ёмкость от полного заряда, А·ч")
    axes[0].legend(fontsize=8)
    power = currents*battery.config["nominal_v"]/1000
    axes[1].plot(power, capacities)
    axes[1].set(xlabel=f"Мощность при {battery.config['nominal_v']:g} В, кВт",
                ylabel="Ёмкость от полного заряда, А·ч")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.set_ylim(0, battery.q_ah*1.08)
    k = battery.exponent if battery.enabled else 1.0
    figure.suptitle(f"Пёйкерт: k={k:g}, опорный ток {battery.reference_current_a:g} А — приближение", fontsize=11)
    figure.text(.5, .015, "Справа U = Uном; в полёте ток рассчитывается с учётом просадки напряжения.",
                ha="center", fontsize=9)
    figure.subplots_adjust(bottom=.21, top=.84, wspace=.28)
