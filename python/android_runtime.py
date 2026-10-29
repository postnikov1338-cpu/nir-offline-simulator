"""Opt-in synchronous transport adapter. All physical updates delegate to frozen models."""
from copy import deepcopy
from dataclasses import asdict
import json
import platform
import sys
import time
from pathlib import Path

from browser_models import (advance,build_context,component_call,component_snapshot,
                            defaults,metadata,number,safe,validate_config)

IDENTITY_PATH=Path(__file__).parent/'model-identity.json'

def identity():
    return json.loads(IDENTITY_PATH.read_text(encoding='utf-8')) if IDENTITY_PATH.exists() else {'kind':'native_validation'}

def new_config(raw=None):
    config=defaults();config.update(mode='interactive',segments=[dict(duration_s=60,load_kw=6,ambient_c=25)])
    if raw:config.update(deepcopy(raw))
    config=validate_config(config)
    if config['strategy'] not in ('fixed_ecms','manual'):raise ValueError('MVP: Fixed ECMS или ручные обороты')
    return config

def finite_tree(value):
    if isinstance(value,float) and not __import__('math').isfinite(value):raise ValueError('NaN/Infinity в файле')
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,list):
        for v in value:finite_tree(v)

class OfflineExperiment:
    def __init__(self,config=None,session_id='native',scheduled=(),boundary=None):
        self.config=new_config(config)
        self.sim,self.state=build_context(self.config)
        self.id=session_id;self.revision=0;self.status='ready';self.reason=None
        self.trace=[];self.events=[];self.pending=[];self.commands={}
        self.scheduled=deepcopy(list(scheduled));self.boundary=boundary
        self.live=dict(load_kw=self.config['segments'][0]['load_kw'],ambient_c=self.config['segments'][0]['ambient_c'],rpm=self.config['manual_rpm'])
        self.fuel_kg=0.;self.fuel_known=True;self.fuel_status='complete';self.last_attempt=None
        self.timings=[];self.stop_command=None
        self.model_identity=identity()
        self.profiles=metadata()['profiles']
        import numpy
        self.runtime=dict(python=platform.python_version(),numpy=numpy.__version__,platform=sys.platform,
                          physical_executor='unchanged_python_models')

    def header(self):
        return safe(dict(schema='nir.android.experiment',schema_version=1,config=self.config,
            session_id=self.id,revision=self.revision,status=self.status,reason=self.reason,
            state=asdict(self.state),fuel_kg=self.fuel_kg if self.fuel_known else None,
            fuel_accounting_status=self.fuel_status,events=self.events,last_attempt=self.last_attempt,
            live_inputs=self.live,sequence=len(self.trace),model_identity=self.model_identity,
            profiles=self.profiles,runtime=self.runtime,stop_command=self.stop_command,
            performance=dict(step_count=len(self.timings),solver_s=sum(self.timings),last_step_s=self.timings[-1] if self.timings else None),
            observers=dict(aging='not_connected',plating='not_connected')))

    def export(self):
        value=self.header();value['trace']=deepcopy(self.trace);return value

    def command(self,message):
        if message.get('session_id')!=self.id:raise ValueError('Устаревшая сессия')
        cid=message.get('command_id')
        if not isinstance(cid,str) or not 1<=len(cid)<=100:raise ValueError('Нужен command_id')
        if cid in self.commands:
            old,result=self.commands[cid]
            if old!=message:raise ValueError('ID команды повторён с другим содержимым')
            return deepcopy(result)
        if len(self.commands)>=5000:raise ValueError('Лимит команд; начните новый опыт')
        action=message['action']
        previous_count=len(self.trace)
        if action=='event':
            if self.config['mode']!='interactive' or self.status in ('completed','terminated','error'):raise ValueError('Воздействие недоступно')
            field=message.get('field');bounds={'load_kw':(0,30),'ambient_c':(-60,80),'rpm':(5000,6500)}
            if field not in bounds or (field=='rpm' and self.config['strategy']!='manual'):raise ValueError('Воздействие не поддерживается')
            value=number(message.get('value'),field,*bounds[field])
            if field=='rpm' and value not in (5000,5750,6500):raise ValueError('Ручные rpm: 5000 / 5750 / 6500')
            event=dict(sequence=len(self.events)+1,field=field,new_value=value,old_value=None,
                requested_sim_time=self.state.time_s,applied_sim_time=None,status='pending',reason=None)
            self.events.append(event);self.pending.append(event)
        elif action=='step':
            if self.status not in ('ready','paused'):raise ValueError('Шаг недоступен; создайте новый опыт')
            self._step()
        elif action in ('pause','stop'):
            if self.status not in ('completed','terminated','error'):
                self.status='paused' if action=='pause' else 'terminated'
                if action=='stop':self.stop_command=dict(applied_sim_time=self.state.time_s);self.reason='Остановлено пользователем на границе шага'
        else:raise ValueError('Неизвестная команда ядра')
        self.revision+=1
        result=dict(header=self.header(),rows=self.trace[previous_count:] if action=='step' else [])
        self.commands[cid]=(deepcopy(message),deepcopy(result))
        return result

    def _step(self):
        elapsed=self.state.time_s;start=0.;duration=0.
        if self.boundary is not None and elapsed>=self.boundary-1e-10:
            self.status='paused';self.reason='Граница сохранённого префикса replay';return
        for seg in self.config['segments']:
            end=start+seg['duration_s']
            if elapsed<end-1e-12:
                duration=min(self.config['dt_s'],end-elapsed)
                if self.config['mode']=='predefined':self.live.update(load_kw=seg['load_kw'],ambient_c=seg['ambient_c'])
                break
            start=end
        if duration<=0:self.status='completed';return
        if self.boundary is not None:duration=min(duration,self.boundary-elapsed)
        for event in list(self.scheduled):
            if event['applied_sim_time']<=elapsed+1e-10:
                copied=deepcopy(event);copied['sequence']=len(self.events)+1
                self.events.append(copied);self.pending.append(copied);self.scheduled.remove(event)
        for event in self.pending:
            key=event['field'];event.update(old_value=self.live[key],applied_sim_time=elapsed,status='applied')
            self.live[key]=event['new_value']
        self.pending.clear()
        old_state=self.state;started=time.perf_counter()
        result=advance(self.sim,old_state,duration,self.live['load_kw'],self.live['ambient_c'],
                       self.live['rpm'] if self.config['strategy']=='manual' else None)
        self.timings.append(time.perf_counter()-started)
        self.last_attempt=dict(completed=result.completed,reason=result.termination_reason,
            time_start_s=elapsed,requested_load_kw=self.live['load_kw'],ambient_c=self.live['ambient_c'])
        self.state=result.final_state;self.fuel_kg+=result.fuel_consumed_kg
        self.fuel_known &= result.fuel_accounting_status in ('complete','estimated')
        self.fuel_status=self.sim._merge_fuel_status(self.fuel_status,result.fuel_accounting_status)
        for row in result.trace:
            if row.dt_s<=0:continue
            data=row.to_dict()
            data['browser_snapshot']=component_snapshot(self.sim,rpm=row.engine_rpm,current=row.generator_current_a,
                torque=row.engine_required_torque_nm,soc=old_state.battery_soc,core=old_state.battery_core_temperature_c,
                surface=old_state.battery_surface_temperature_c,winding=old_state.generator_temperature_c,dt=row.dt_s)
            data.update(sequence=len(self.trace)+1,cumulative_fuel_kg=self.fuel_kg if self.fuel_known else None)
            self.trace.append(safe(data))
        total=sum(s['duration_s'] for s in self.config['segments'])
        self.status='terminated' if not result.completed else 'completed' if self.state.time_s>=total-1e-10 else 'paused'
        self.reason=result.termination_reason
        if self.boundary is not None and self.state.time_s>=self.boundary-1e-10 and self.status=='paused':
            self.reason='Граница сохранённого префикса replay'
        if self.status in ('terminated','completed'):
            for event in self.events:
                if event['status']=='pending':event.update(status='rejected',reason='Опыт завершён')

def import_plan(document):
    finite_tree(document)
    if not isinstance(document,dict) or document.get('schema')!='nir.android.experiment' or document.get('schema_version')!=1:
        raise ValueError('Нужен JSON автономного приложения версии 1; Windows-отчёт не подменяется этим форматом')
    if document.get('model_identity')!=identity():raise ValueError('Версия моделей/профилей файла не совпадает с установленной')
    config=new_config(document.get('config'))
    if document.get('profiles')!=safe(metadata()['profiles']):raise ValueError('Snapshot физических профилей не совпадает')
    events=document.get('events',[])
    if not isinstance(events,list) or len(events)>5000:raise ValueError('Неверный список событий')
    total=sum(s['duration_s'] for s in config['segments']);scheduled=[]
    for event in events:
        if event.get('status')!='applied':continue
        field=event.get('field');bounds={'load_kw':(0,30),'ambient_c':(-60,80),'rpm':(5000,6500)}
        if field not in bounds or (field=='rpm' and config['strategy']!='manual'):raise ValueError('Неверное воздействие в файле')
        value=number(event['new_value'],field,*bounds[field]);t=number(event['applied_sim_time'],'Время события',0,total)
        if field=='rpm' and value not in (5000,5750,6500):raise ValueError('Неверные ручные rpm')
        # Imported commands must occur on the original physical grid, not split solver steps.
        offset=0.;valid=False
        for seg in config['segments']:
            if offset-1e-10<=t<=offset+seg['duration_s']+1e-10:
                k=(t-offset)/config['dt_s'];valid=abs(k-round(k))<1e-8 or abs(t-offset-seg['duration_s'])<1e-8
                if valid:break
            offset+=seg['duration_s']
        if not valid:raise ValueError('Событие не на физической границе')
        scheduled.append(dict(field=field,new_value=value,applied_sim_time=t,requested_sim_time=event.get('requested_sim_time',t),status='applied'))
    boundary=None
    if document.get('status')!='completed':
        boundary=number(document.get('state',{}).get('time_s'),'Граница replay',0,total)
    return config,scheduled,boundary

SESSION=None
def dispatch_json(payload):
    global SESSION
    message=json.loads(payload);action=message['action']
    if action=='create':SESSION=OfflineExperiment(message.get('config'),message['session_id']);value=dict(header=SESSION.header(),rows=[])
    elif action=='import':
        config,events,boundary=import_plan(message['document'])
        SESSION=OfflineExperiment(config,message['session_id'],events,boundary);value=dict(header=SESSION.header(),rows=[])
    elif action=='bench':value=safe(component_call(message['kind'],message['inputs'],message.get('config')))
    elif action=='validate_import':
        config,events,boundary=import_plan(message['document'])
        value=dict(valid=True,config=config,event_count=len(events),boundary=boundary)
    elif action=='export':value=SESSION.export()
    elif action=='status':value=dict(header=SESSION.header(),rows=[])
    else:
        if SESSION is None:raise ValueError('Нет опыта')
        value=SESSION.command(message)
    missing={};value=safe(value,missing=missing)
    if missing and isinstance(value,dict):value['nonfinite_reasons']=missing
    return json.dumps(value,ensure_ascii=False,allow_nan=False)
