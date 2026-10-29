"""Two-node engineering thermal approximation. See THERMAL_MODEL_RU.md."""
import math
import numpy as np


def thermal_defaults():
    return dict(ambient_c=25.0, initial_c=25.0, specific_heat_j_kg_k=1100.0,
                conductivity_w_m_k=0.5, convection_w_m2_k=5.0,
                exposed_area_m2=0.10, conduction_length_m=0.025,
                core_mass_fraction=0.5, warning_c=60.0)


class ThermalModel:
    def __init__(self, mass_kg, config=None):
        self.p = thermal_defaults()
        self.p.update(config or {})
        for key, value in self.p.items():
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'Тепловая модель: неверное значение {key}')
            if key.endswith('_c'):
                if value <= -273.15:
                    raise ValueError('Температура должна быть выше абсолютного нуля')
            elif value <= 0 and not (key == 'convection_w_m2_k' and value == 0):
                raise ValueError(f'Тепловая модель: {key} должно быть положительным')
        if not math.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('Тепловая модель: неверная масса')
        f = self.p['core_mass_fraction']
        if not 0 < f < 1:
            raise ValueError('Доля массы ядра должна быть между 0 и 1')
        capacity = mass_kg * self.p['specific_heat_j_kg_k']
        self.capacities = np.array([capacity*f, capacity*(1-f)])
        g = self.p['conductivity_w_m_k']*self.p['exposed_area_m2']/self.p['conduction_length_m']
        self.ha = self.p['convection_w_m2_k']*self.p['exposed_area_m2']
        # Symmetric similarity transform gives a stable exact exponential,
        # including the insulated (zero eigenvalue) case.
        self.sqrt_c = np.sqrt(self.capacities)
        matrix = np.array([[-g, g], [g, -g-self.ha]])
        self.values, self.vectors = np.linalg.eigh(matrix / np.outer(self.sqrt_c, self.sqrt_c))
        self.cache = {}

    def step(self, temperatures, heat_w, dt, ambient_c=None):
        ambient = self.p['ambient_c'] if ambient_c is None else float(ambient_c)
        if (not math.isfinite(dt) or dt <= 0 or not math.isfinite(heat_w)
                or heat_w < 0 or not math.isfinite(ambient) or ambient <= -273.15):
            raise ValueError('Неверные шаг времени или тепловыделение')
        state = np.asarray(temperatures, dtype=float)
        if state.shape != (2,) or not np.isfinite(state).all():
            raise ValueError('Неверное тепловое состояние')
        if dt not in self.cache:
            z = self.values*dt
            phi = np.full(2, dt, dtype=float)
            np.divide(np.expm1(z), self.values, out=phi, where=np.abs(self.values)>1e-15)
            self.cache[dt] = (np.exp(z), phi)
        decay, phi = self.cache[dt]
        y = self.sqrt_c*(state-ambient)
        source = np.array([heat_w/self.sqrt_c[0], 0.0])
        new = ambient + (self.vectors @ (decay*(self.vectors.T@y) + phi*(self.vectors.T@source)))/self.sqrt_c
        stored = float(self.capacities @ (new-state))
        return new, heat_w-stored/dt

    def temperature_bounds(self, temperatures, heat_w, dt, ambient_c=None):
        """Exact extrema of both nodes on a constant-source interval."""
        ambient = self.p['ambient_c'] if ambient_c is None else float(ambient_c)
        end, _ = self.step(temperatures, heat_w, dt, ambient)
        start = np.asarray(temperatures, dtype=float)
        y = self.sqrt_c*(start-ambient)
        source = np.array([heat_w/self.sqrt_c[0], 0.0])
        derivative_modes = self.values*(self.vectors.T@y) + self.vectors.T@source
        values = [*start, *end]
        for node in range(2):
            a, b = self.vectors[node]*derivative_modes/self.sqrt_c[node]
            if a*b < 0 and self.values[0] != self.values[1]:
                extremum = math.log(-b/a)/(self.values[0]-self.values[1])
                if 0 < extremum < dt:
                    values.append(self.step(start, heat_w, extremum, ambient)[0][node])
        return float(min(values)), float(max(values))
