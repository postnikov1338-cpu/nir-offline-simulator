"""Adapters only: browser values delegate to the existing Python models."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Protocol

from ecms_control import (ProportionalSOCAdaptiveECMS, load_builtin_ecms_profile,
                          load_builtin_adaptive_ecms_profile)
from generator_model import GeneratorModel, GeneratorProfile
from rectifier_model import RectifierModel, RectifierProfile
from powertrain_mission import load_synthetic_mission_fixture, PowertrainMissionSegment
from powertrain_solver import PowertrainOperatingPointInputs
from powertrain_thermal import GeneratorThermalModel, GeneratorThermalProfile, HeatSinkProfile, ThermalSimulationInput

ROOT = Path(__file__).resolve().parent
POLICIES = {'lowest': 'lowest_feasible_rpm', 'minimum_fuel': 'minimum_engine_fuel_rate',
            'fixed_ecms': 'minimum_equivalent_fuel_rate',
            'adaptive_ecms': 'soc_adaptive_minimum_equivalent_fuel_rate',
            'manual': 'closest_to_target_rpm'}
LIMITS = dict(max_duration_s=600, max_steps=600, max_sweep_points=31,
              max_request_bytes=2_000_000, max_load_kw=30, max_jobs=100)


def safe(value, path='$', missing=None):
    """Strict JSON projection. Nonfinite values retain an explicit reason."""
    missing = {} if missing is None else missing
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): safe(v, f'{path}.{k}', missing) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v, f'{path}[{i}]', missing) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        missing[path] = 'nonfinite:' + str(value)
        return None
    return value


def number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f'{name}: требуется число')
    try:
        n = float(value.replace(',', '.') if isinstance(value, str) else value)
    except (ValueError, TypeError):
        raise ValueError(f'{name}: требуется число') from None
    if not math.isfinite(n) or not low <= n <= high:
        raise ValueError(f'{name}: допустимо от {low} до {high}')
    return n


def defaults():
    _, initial = load_synthetic_mission_fixture()
    return dict(config_version=1, mode='predefined', strategy='fixed_ecms', dt_s=1.0,
                manual_rpm=6500.0, initial_state=asdict(initial),
                generator_thermal_profile='generator_thermal_nominal_v1',
                heatsink_profile='heatsink_nominal_v1',
                segments=[dict(duration_s=3.0, load_kw=6.0, ambient_c=25.0)],
                controllers=dict(discharge_factor=1.0, charge_factor=1.0, kp=0.0))


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError('Конфигурация должна быть объектом')
    d = defaults()
    if set(raw) - set(d):
        raise ValueError('Неизвестные поля конфигурации: ' + ', '.join(sorted(set(raw)-set(d))))
    d.update(deepcopy(raw))
    if d['config_version'] != 1 or d['mode'] not in ('predefined', 'interactive') or d['strategy'] not in POLICIES:
        raise ValueError('Неизвестная версия, режим или стратегия')
    d['dt_s'] = number(d['dt_s'], 'Шаг, с', 0.1, 5)
    d['manual_rpm'] = number(d['manual_rpm'], 'Ручные rpm', 5000, 6500)
    if d['strategy'] == 'manual' and d['manual_rpm'] not in (5000, 5750, 6500):
        raise ValueError('Ручной режим: выберите 5000, 5750 или 6500 rpm')
    if not isinstance(d['segments'], list) or not 1 <= len(d['segments']) <= 20:
        raise ValueError('Требуется от 1 до 20 участков')
    segments = []
    steps = 0
    for row in d['segments']:
        if set(row) != {'duration_s', 'load_kw', 'ambient_c'}:
            raise ValueError('Участок: duration_s, load_kw, ambient_c')
        seg = dict(duration_s=number(row['duration_s'], 'Длительность', .1, 600),
                   load_kw=number(row['load_kw'], 'Нагрузка, кВт', 0, LIMITS['max_load_kw']),
                   ambient_c=number(row['ambient_c'], 'Воздух, °C', -60, 80))
        segments.append(seg)
        steps += math.ceil(seg['duration_s']/d['dt_s']-1e-12)
    if sum(s['duration_s'] for s in segments) > LIMITS['max_duration_s'] or steps > LIMITS['max_steps']:
        raise ValueError('Лимит сервиса: не более 600 секунд и 600 физических шагов')
    d['segments'] = segments
    original = defaults()['initial_state']
    if not isinstance(d['initial_state'], dict) or set(d['initial_state']) != set(original):
        raise ValueError('Неверная структура initial_state')
    for key, v in d['initial_state'].items():
        if key == 'battery_soc':
            original[key] = number(v, 'Начальный SOC', .1, .95)
        elif 'temperature' in key:
            original[key] = number(v, key, -60, 180)
        elif key == 'fuel_remaining_kg':
            original[key] = None if v is None else number(v, key, 0, 100)
        elif key == 'previous_engine_rpm':
            original[key] = None if v is None else number(v, key, 4300, 7200)
        elif v is not None and key != 'time_s':
            raise ValueError(f'{key}: должен быть null при новом запуске')
    if d['initial_state']['time_s'] != 0:
        raise ValueError('Новая сессия начинается с time_s=0')
    d['initial_state'] = original
    if set(d['controllers']) != {'discharge_factor', 'charge_factor', 'kp'}:
        raise ValueError('Неверные параметры контроллера')
    d['controllers'] = {k: number(v, k, 0, 10 if k == 'kp' else 5) for k,v in d['controllers'].items()}
    if d['strategy']=='adaptive_ecms' and max(d['controllers']['discharge_factor'],d['controllers']['charge_factor'])>2:
        raise ValueError('Adaptive ECMS: опорные коэффициенты должны находиться в [0,2] согласно профилю')
    if d['generator_thermal_profile'] not in ('generator_thermal_nominal_v1','generator_thermal_optimistic_v1','generator_thermal_pessimistic_v1'):
        raise ValueError('Неизвестный тепловой профиль генератора')
    if d['heatsink_profile'] not in ('heatsink_nominal_v1','heatsink_optimistic_v1','heatsink_pessimistic_v1'):
        raise ValueError('Неизвестный профиль радиатора')
    return d


class SessionAdaptiveECMS(ProportionalSOCAdaptiveECMS):
    """Runner calls reset at each simulate invocation; session owns its lifetime.

    New session always constructs a new instance. Selection equations are inherited.
    """
    def reset(self):
        pass


def build_context(config):
    profile = None
    c = config['controllers']
    if config['strategy'] == 'fixed_ecms':
        profile = replace(load_builtin_ecms_profile(), equivalence_factor_discharge=c['discharge_factor'],
                          equivalence_factor_charge=c['charge_factor'])
    elif config['strategy'] == 'adaptive_ecms':
        profile = replace(load_builtin_adaptive_ecms_profile(),
                          base_equivalence_factor_discharge=c['discharge_factor'],
                          base_equivalence_factor_charge=c['charge_factor'], soc_proportional_gain=c['kp'])
    sim, initial = load_synthetic_mission_fixture(operating_policy=POLICIES[config['strategy']], ecms_profile=profile,
        solver_overrides=dict(operating_min_rpm=5000., operating_max_rpm=6500., rpm_grid_step=750.,
                              include_confirmed_rated_point=False, refine_infeasible_to_feasible_boundary=False))
    if config['strategy'] == 'adaptive_ecms':
        sim.powertrain.ecms = SessionAdaptiveECMS(profile)
    sim.config = replace(sim.config, dt_s=config['dt_s'])
    sim.generator_thermal = GeneratorThermalModel(GeneratorThermalProfile.load_builtin(config['generator_thermal_profile']))
    sim.rectifier_thermal.heatsink = HeatSinkProfile.load_builtin(config['heatsink_profile'])
    return sim, replace(initial, **config['initial_state'])


def metadata():
    sim, _ = build_context(defaults())
    return dict(api_version=1, defaults=defaults(), service_limits=LIMITS,
                units=dict(time_s='s', load_kw='kW', power_w='W', current_a='A', temperature_c='°C', fuel='kg / kg/h', soc='fraction'),
                profiles=dict(engine=sim.powertrain.engine.profile.to_snapshot(),
                              generator=sim.powertrain.dc_bus.generator.profile.to_snapshot(),
                              rectifier=sim.powertrain.dc_bus.rectifier.profile.to_snapshot(),
                              battery=deepcopy(sim.battery.config),
                              generator_thermal=asdict(sim.generator_thermal.profile),
                              heatsink=asdict(sim.rectifier_thermal.heatsink)),
                observers=dict(aging='not_connected', plating='not_connected'),
                available_profiles=dict(generator=['synthetic','keri_afpm_uav_coil3_2021','limbach_l275e_direct_12s_generator_concept_v1'],
                                        rectifier=['synthetic','vishay_vs_qa300fa17','vishay_vs_qa400fa10']),
                warning='Synthetic установка; не подтверждённая реальная установка. Электрические температурные коэффициенты АКБ выключены; charge_policy отсутствует.')


def component_call(kind, inputs, config=None):
    """Fresh context for each bench; never touches active session/controller."""
    config = validate_config(config or {})
    allowed={
        'engine':{'rpm','torque_nm'}, 'generator':{'rpm','temperature_c','current_a','profile_id'},
        'rectifier':{'current_a','temperature_c','reverse_voltage_v','profile_id'},
        'battery':{'soc','dt_s','temperature_c','surface_c','power_kw'},
        'coupled':{'rpm','load_kw','soc','battery_c','temperature_c','case_c'},
        'thermal_generator':{'rpm','temperature_c','duration_s','dt_s','heat_w','ambient_c'},
        'thermal_rectifier':{'temperature_c','duration_s','dt_s','heat_w','ambient_c'},
        'thermal_battery':{'temperature_c','surface_c','duration_s','dt_s','heat_w','ambient_c'}}
    if kind not in allowed or set(inputs)-allowed[kind]:
        raise ValueError('Неизвестные или неподдерживаемые входы испытания')
    sim, initial = build_context(config)
    p = sim.powertrain
    def n(key, default, low=-1e6, high=1e6):
        return number(inputs.get(key, default), key, low, high)
    rpm = n('rpm',6500,0,8000)
    temp = n('temperature_c',60,-60,200)
    if kind == 'engine':
        envelope = p.engine.maximum_available_at_rpm(rpm)
        torque = n('torque_nm',10,0,100)
        return dict(envelope=envelope.to_dict(), part_load=p.engine.part_load_at(rpm,torque).to_dict())
    if kind == 'generator':
        model = p.dc_bus.generator
        profile = inputs.get('profile_id','synthetic')
        if profile != 'synthetic':
            if profile not in metadata_ids('generator'):
                raise ValueError('Неизвестный профиль генератора')
            model = GeneratorModel(GeneratorProfile.load_builtin(profile))
        if not model.profile.readiness.phase2_ready:
            return dict(status='not_ready', readiness=model.profile.readiness.to_dict(), profile=model.profile.to_snapshot())
        return dict(status='calculated', point=model.operating_point(rpm,n('current_a',100,0,1000),temp).to_dict(), profile=model.profile.to_snapshot())
    if kind == 'rectifier':
        model = p.dc_bus.rectifier
        profile = inputs.get('profile_id','synthetic')
        if profile != 'synthetic':
            if profile not in metadata_ids('rectifier'):
                raise ValueError('Неизвестный профиль выпрямителя')
            model = RectifierModel(RectifierProfile.load_builtin(profile))
        return dict(point=model.operating_point(n('current_a',100,0,1000),temp,n('reverse_voltage_v',60,0,500)).to_dict(), profile=model.profile.to_snapshot())
    if kind == 'battery':
        soc = n('soc',.8,0,1)
        dt = n('dt_s',1,.1,60)
        surface = n('surface_c',temp,-60,180)
        diagnostics = sim.battery.diagnostics(soc,dt,temp,surface)
        result = sim.battery.step(soc,n('power_kw',1,-30,30),dt,temp,surface)
        return dict(next_soc=result[0], voltage_v=result[1], current_a=result[2], loss_kw=result[3],
                    diagnostics=diagnostics, effective_capacity_ah=sim.battery.effective_capacity_ah(result[2],temp),
                    observers={'aging':'not_connected','plating':'not_connected'}, profile=sim.battery.config)
    if kind == 'coupled':
        target = n('rpm',6500,5000,6500) if config['strategy']=='manual' else None
        point = p.solve(PowertrainOperatingPointInputs(n('load_kw',6,0,30)*1000,
            n('soc',.8,0,1),n('battery_c',25,-60,180),temp,n('case_c',60,-60,200),target_rpm=target))
        output=point.to_dict()
        if point.selected_candidate is not None:
            output['browser_snapshot']=component_snapshot(sim,rpm=point.engine_rpm,
                current=point.generator_dc_current_a,torque=point.engine_required_torque_nm,
                soc=n('soc',.8,0,1),core=n('battery_c',25,-60,180),surface=n('battery_c',25,-60,180),
                winding=temp,dt=config['dt_s'])
        return output
    if kind in ('thermal_generator','thermal_rectifier','thermal_battery'):
        duration = n('duration_s',60,.1,600)
        dt = n('dt_s',5,.1,60)
        if math.ceil(duration/dt) > 600:
            raise ValueError('Не более 600 тепловых шагов')
        heat = n('heat_w',100,0,10000)
        ambient = n('ambient_c',25,-60,80)
        trace=[]
        time_s=0.
        if kind=='thermal_battery':
            state = [temp,n('surface_c',temp,-60,200)]
        else:
            model = sim.generator_thermal if kind=='thermal_generator' else sim.rectifier_thermal
            state = model.initial_state(temp)
        while time_s < duration-1e-12:
            step=min(dt,duration-time_s)
            time_s+=step
            if kind=='thermal_battery':
                state, rejected = sim.battery_thermal.step(state,heat,step,ambient)
                trace.append(dict(time_s=time_s,core_c=float(state[0]),surface_c=float(state[1]),heat_rejected_w=float(rejected)))
            else:
                result=model.step(state,ThermalSimulationInput(time_s,heat,rpm,ambient,loss_source_type='user_input'))
                trace.append(result.to_dict())
                if result.status.value=='not_evaluable':
                    break
                state=result.state
        return dict(trace=trace, heat_source_type='user_input', thermal_parameters='engineering_sensitivity',
                    initial_temperature_c=temp, ambient_c=ambient)
    raise ValueError('Неизвестное испытание')


def metadata_ids(kind):
    return metadata()['available_profiles'][kind][1:]


def component_snapshot(sim,*,rpm,current,torque,soc,core,surface,winding,dt):
    """Diagnostic projections at the accepted electrical input state; no extra solve."""
    return dict(generator=sim.powertrain.dc_bus.generator.operating_point(rpm,current,winding).to_dict(),
                engine=sim.powertrain.engine.part_load_at(rpm,torque).to_dict(),
                battery=sim.battery.diagnostics(soc,dt,core,surface),
                temperature_basis='start_of_accepted_electrical_step')


class DemandProvider(Protocol):
    """Future extension: state+command -> W, provenance, status; no flight model here."""
    def demand(self, time_s: float, vehicle_state: dict, mission_command: dict) -> dict: ...


def advance(sim, state, duration, load_kw, ambient, manual_rpm=None):
    segment=PowertrainMissionSegment('browser_step',duration,load_kw*1000,ambient,
                                    target_rpm=manual_rpm,
                                    demand_provenance={'provider':'direct_electrical_load','source_type':'user_input'})
    return sim.simulate((segment,),state)
