'use strict';
const $=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,d=3)=>typeof v==='number'&&Number.isFinite(v)?v.toLocaleString('ru-RU',{maximumFractionDigits:d}):'—';
let worker=null,kernelReady=false,cacheReady=false,manifest=null,runtime=null,cacheProof=null;
let activeId=null,pendingNewId=null,current=null,rows=[],running=false,busy=false,commandPending=false,waitingBoundary=false,readOnly=false,preview=null;
let commitChain=Promise.resolve(),dbPromise=null;const pending=new Map(),measurements=[],pageStarted=performance.now();
const statusNames={ready:'Готово к расчёту',paused:'Пауза',completed:'Завершено',terminated:'Остановлено',error:'Ошибка'};
function fail(error){$('error').textContent=error.message||String(error);$('error').hidden=false;}
function num(id,min,max){const raw=$(id).value.trim().replace(',','.');if(!raw||!Number.isFinite(Number(raw))||Number(raw)<min||Number(raw)>max)throw Error('Проверьте '+$(id).parentElement.firstChild.textContent+' · допустимо '+min+'…'+max);return Number(raw);}
function db(){if(!dbPromise)dbPromise=new Promise((resolve,reject)=>{const request=indexedDB.open('nir-android-experiments',1);request.onupgradeneeded=()=>{request.result.createObjectStore('runs',{keyPath:'id'});request.result.createObjectStore('meta');};request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});return dbPromise;}
async function store(report){const database=await db();await new Promise((resolve,reject)=>{const tx=database.transaction(['runs','meta'],'readwrite');tx.objectStore('runs').put({id:report.session_id,report,updated:new Date().toISOString()});tx.objectStore('meta').put(report.session_id,'last');tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||Error('Запись отменена'));});}
async function saved(id=null){const database=await db();return new Promise((resolve,reject)=>{const request=database.transaction('runs').objectStore('runs')[id?'get':'getAll'](id||undefined);request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});}
async function listRuns(){const items=await saved();$('savedRuns').replaceChildren(new Option('Выберите опыт',''));for(const item of items.sort((a,b)=>b.updated.localeCompare(a.updated))){$('savedRuns').add(new Option(item.updated.slice(0,19).replace('T',' ')+' · '+fmt(item.report.state.time_s)+' с',item.id));}}
function report(){return current?{...current,trace:rows,checkpoint_execution:running?'running':'stable'}:null;}
function request(action,extra={}){if(!worker)throw Error('Ядро не готово');const command_id=crypto.randomUUID();return new Promise((resolve,reject)=>{pending.set(command_id,{resolve,reject,at:performance.now(),action});worker.postMessage({action,session_id:activeId,command_id,...extra});});}
async function accepted(message){
 const value=message.value;if(!value?.header)return;
 const header=value.header;
 if(header.session_id===pendingNewId){activeId=pendingNewId;pendingNewId=null;rows=[];readOnly=false;preview=null;$('fileStatus').textContent='Новый опыт создан; прежняя история сохранена.';for(const[id,key]of [['load','load_kw'],['air','ambient_c'],['rpm','rpm']])$(id).value=header.live_inputs[key];$('setup').open=false;}
 if(header.session_id!==activeId)return;
 if(current?.session_id===header.session_id&&header.revision<current.revision)return;
 current=header;
 for(const row of value.rows||[]){if(row.sequence===rows.length+1)rows.push(row);else if(row.sequence>rows.length+1)throw Error('Нарушен порядок принятых шагов');}
 running=message.execution==='running';waitingBoundary=message.execution==='waiting_boundary';
 if(value.rows?.length||['completed','terminated','error'].includes(header.status))busy=false;
 const snapshot=report();snapshot.checkpoint_execution=running?'running':'stable';
 await store(snapshot);render();
}
function startWorker(){
 worker=new Worker('worker.js');
 worker.onmessage=({data:message})=>{
  if(message.type==='progress'){$('preparationStatus').textContent=message.message;return;}
  if(message.type==='ready'){if(manifest&&message.build!==manifest.build){fail(Error('Версии страницы и ядра не совпали; закройте приложение и откройте снова'));return;}kernelReady=true;runtime=message.runtime;measurements.push({kind:'kernel_init',ms:message.init_ms},{kind:'page_to_ready',ms:performance.now()-pageStarted});preparationLabel();render();return;}
  if(message.type==='computing'){if(message.session_id===activeId){busy=true;render();}return;}
  if(message.type==='idle'){if(message.session_id===activeId){running=false;busy=false;waitingBoundary=false;render();}return;}
  if(message.type==='fatal'){running=false;busy=false;kernelReady=false;readOnly=true;worker.terminate();worker=null;for(const item of pending.values())item.reject(Error(message.message));pending.clear();fail(Error('Ядро остановлено. Сохранён последний подтверждённый префикс. Перезагрузите страницу для нового опыта. '+message.message));render();return;}
  if(message.type==='error'){const item=pending.get(message.request_id);item?.reject(Error(message.message));pending.delete(message.request_id);return;}
  commitChain=commitChain.then(async()=>{
   try{await accepted(message);if(message.checkpoint)worker?.postMessage({action:'checkpoint_ack',token:message.checkpoint,ok:true});
    const item=pending.get(message.request_id);if(item){measurements.push({kind:'request',action:item.action,ms:performance.now()-item.at});item.resolve(message.value);pending.delete(message.request_id);}
   }catch(error){if(message.checkpoint)worker?.postMessage({action:'checkpoint_ack',token:message.checkpoint,ok:false});running=false;readOnly=true;fail(Error('Не удалось сохранить checkpoint: '+error.message));pending.get(message.request_id)?.reject(error);pending.delete(message.request_id);render();}
  });
 };
 worker.onerror=event=>{kernelReady=false;running=false;busy=false;readOnly=true;worker?.terminate();worker=null;for(const item of pending.values())item.reject(Error('Worker: '+event.message));pending.clear();fail(Error('Worker завершился. Доступен только последний сохранённый префикс; перезагрузите страницу.'));render();};
 request('init').catch(fail);
}
function preparationLabel(){
 if(kernelReady&&cacheReady)$('preparationStatus').textContent='Готово к работе без интернета · ресурсы проверены · физическое ядро проверено';
 else if(cacheReady)$('preparationStatus').textContent='Ресурсы сохранены. Подготовка Python и короткая физическая проверка…';
}
async function prepare(){
 if(!window.isSecureContext||!('serviceWorker'in navigator))throw Error('Нужен HTTPS-адрес. Обычный Wi-Fi HTTP или file:// не устанавливает автономный пакет.');
 manifest=await (await fetch('asset-manifest.json')).json();
 navigator.serviceWorker.addEventListener('message',({data})=>{if(data.type==='cache_progress')$('preparationStatus').textContent='Подготовка автономного набора: '+data.done+' / '+data.total+' · '+fmt(data.bytes/1e6,1)+' МБ';if(data.type==='cache_error')fail(Error('Подготовка не завершена: '+data.message));});
 const registration=await navigator.serviceWorker.register('sw.js',{updateViaCache:'none'});
 const notice=()=>{if(registration.waiting)$('updateStatus').textContent='Новая версия подготовлена. После остановки опыта закройте все вкладки приложения и откройте снова. Текущий опыт не обновляется посреди расчёта.';};notice();registration.addEventListener('updatefound',()=>{registration.installing?.addEventListener('statechange',notice);});
 await navigator.serviceWorker.ready;
 if(!navigator.serviceWorker.controller)await new Promise(resolve=>navigator.serviceWorker.addEventListener('controllerchange',resolve,{once:true}));
 cacheProof=await new Promise((resolve,reject)=>{const channel=new MessageChannel(),timer=setTimeout(()=>reject(Error('Не получено подтверждение целостности кэша')),120000);channel.port1.onmessage=({data})=>{clearTimeout(timer);data.ok?resolve(data):reject(Error(data.message));};navigator.serviceWorker.controller.postMessage({type:'verify'},[channel.port2]);});
 if(cacheProof.build!==manifest.build)throw Error('Кэш и страница разных версий; закройте все вкладки и откройте приложение снова.');
 cacheReady=true;measurements.push({kind:'page_to_cache_verified',ms:performance.now()-pageStarted});preparationLabel();startWorker();
}
function syncSetup(){const manual=$('strategy').value==='manual',scheduled=$('mode').value==='predefined';$('manualConfig').hidden=!manual;$('duration').disabled=scheduled;$('initialLoad').disabled=scheduled;}
async function create(){
 const defaults=(await fetch('initial-config.json').then(r=>r.json())),config=structuredClone(defaults);
 config.mode=$('mode').value;config.strategy=$('strategy').value;config.manual_rpm=Number($('initialRpm').value);
 const ambient=num('initialAir',-60,80),duration=$('mode').value==='predefined'?60:num('duration',1,600);
 if(!Number.isInteger(duration))throw Error('В минимальной версии длительность задаётся целым числом секунд; шаг остаётся 1 с');
 config.segments=config.mode==='predefined'?[6,5,7].map(load_kw=>({duration_s:20,load_kw,ambient_c:ambient})):[{duration_s:duration,load_kw:num('initialLoad',0,30),ambient_c:ambient}];
 Object.assign(config.initial_state,{battery_soc:num('initialSoc',10,95)/100,battery_core_temperature_c:num('initialCore',-60,180),battery_surface_temperature_c:num('initialSurface',-60,180),generator_temperature_c:num('initialWinding',-60,180)});
 pendingNewId=crypto.randomUUID();try{await request('create',{session_id:pendingNewId,config});}catch(error){pendingNewId=null;throw error;}
}
async function control(action){
 if(['pause','stop'].includes(action)){waitingBoundary=busy||running;render();await request(action);return;}
 if(commandPending||busy||running)return;
 commandPending=true;render();try{await request(action);}finally{commandPending=false;render();}
}
async function apply(field,input,min,max){if(commandPending)return;const value=num(input,min,max);commandPending=true;render();try{await request('event',{field,value});}finally{commandPending=false;render();}}
function receipt(){
 if(!current)return 'Создайте новый опыт после подготовки ядра.';
 const names={load_kw:'Нагрузка, кВт',ambient_c:'Воздух, °C',rpm:'Желаемые rpm'},parts=[];
 for(const[key,id]of [['load_kw','load'],['ambient_c','air'],['rpm','rpm']]){
  if(key==='rpm'&&current.config.strategy!=='manual')continue;
  const event=[...current.events].reverse().find(e=>e.field===key);
  parts.push(names[key]+': в поле '+$(id).value+' · ожидает '+(event?.status==='pending'?fmt(event.new_value):'нет')+' · применено '+fmt(current.live_inputs[key])+(event?.status==='applied'?' при t='+fmt(event.applied_sim_time)+' с':''));
 }return parts.join('\n');
}
function metric(label,value,unit=''){return '<div class="metric"><span>'+esc(label)+'</span><strong>'+fmt(value,4)+' <small>'+esc(unit)+'</small></strong></div>';}
function render(){
 const started=performance.now(),stable=Boolean(current)&&!running&&!busy&&!commandPending&&!waitingBoundary;
 const editable=kernelReady&&cacheReady&&!readOnly&&current?.config.mode==='interactive'&&!['completed','terminated','error'].includes(current?.status);
 $('create').disabled=!kernelReady||!cacheReady||running||busy||commandPending||waitingBoundary;
 for(const[id,action]of [['step','step'],['start','start']])$(id).disabled=!kernelReady||readOnly||!current||!['ready','paused'].includes(current.status)||running||busy||commandPending||waitingBoundary;
 $('pause').disabled=!kernelReady||readOnly||(!running&&!busy)||waitingBoundary;$('stop').disabled=!kernelReady||readOnly||!current||(!running&&!busy&&current.status!=='paused')||commandPending;
 for(const id of ['applyLoad','applyAir','applyRpm'])$(id).disabled=!editable||commandPending;
 for(const id of ['load','air'])$(id).disabled=!editable;
 $('manualLive').hidden=current?.config.strategy!=='manual';$('applyRpm').hidden=current?.config.strategy!=='manual';$('rpm').disabled=!editable;
 for(const id of ['exportJson','exportCsv','exportSvg'])$(id).disabled=!stable;
 $('importJson').disabled=!kernelReady||!stable&&Boolean(current)||running||busy||commandPending;$('savedRuns').disabled=running||busy||commandPending;
 $('replay').hidden=!preview;$('replay').disabled=!kernelReady||running||busy||commandPending||waitingBoundary;
 $('status').textContent=readOnly?'Сохранённый префикс · только просмотр':waitingBoundary?'Ожидается завершение текущего шага':busy?'Вычисление шага':commandPending?'Команда отправляется в ядро':running?'Выполняется':current?statusNames[current.status]||current.status:kernelReady?'Создайте опыт':'Ядро загружается';
 if(current){$('clock').textContent=fmt(current.state.time_s)+' / '+fmt(current.config.segments.reduce((n,s)=>n+s.duration_s,0))+' с';$('context').textContent=(current.config.strategy==='manual'?'Желаемые rpm: '+fmt(current.live_inputs.rpm)+'; фактические выбираются из допустимых точек.':'Автоматическое управление: Fixed ECMS.')+' Шаг '+fmt(current.config.dt_s)+' с · вычислено '+fmt(current.performance?.solver_s,2)+' реальных с.'+(current.reason?' Причина: '+current.reason:'');}
 $('receipt').textContent=receipt();
 const row=rows.at(-1);$('sample').textContent=row?'Последний принятый шаг t='+fmt(row.time_s)+' с · электрическая точка при t='+fmt(row.time_start_s)+' с'+(busy?' · новый шаг считается; показан прежний результат':''):'Рассчитанных точек нет. Введённые числа сами не запускают solver.';
 $('metrics').innerHTML=[metric('Нагрузка шага',row?.requested_power_w==null?null:row.requested_power_w/1000,'кВт'),metric('Генератор после моста',row?.generator_bus_power_w==null?null:row.generator_bus_power_w/1000,'кВт'),metric('АКБ: + разряд / − заряд',row?.battery_power_w==null?null:row.battery_power_w/1000,'кВт'),metric('Ток АКБ',row?.battery_current_a,'А'),metric('DC-шина',row?.bus_voltage_v,'В'),metric('SOC',row?.battery_soc==null?null:row.battery_soc*100,'%'),metric('Фактические обороты',row?.engine_rpm,'rpm'),metric('Расход топлива',row?.fuel_flow_kg_h,'кг/ч'),metric('Израсходовано топлива',row?.cumulative_fuel_kg,'кг'),metric('Ядро АКБ',row?.battery_temperature_c,'°C'),metric('Обмотка',row?.generator_temperature_c,'°C'),metric('Переход выпрямителя',row?.rectifier_junction_temperature_c,'°C')].join('');
 draw();measurements.push({kind:'dom_render',ms:performance.now()-started});if(measurements.length>500)measurements.shift();
}
function draw(){
 const defs=[['Мощность, кВт',[['requested_power_w','Потребители',.001,'#647e86'],['generator_bus_power_w','Генератор',.001,'#11796f'],['battery_power_w','АКБ',.001,'#b77732']]],['Заряд АКБ, %',[['battery_soc','SOC',100,'#11796f']]],['Ток АКБ, А',[['battery_current_a','+ разряд, − заряд',1,'#11796f']]],['Температуры, °C',[['ambient_temperature_c','Воздух · вход шага',1,'#647e86'],['battery_temperature_c','Ядро АКБ',1,'#11796f'],['battery_surface_temperature_c','Поверхность',1,'#7696ad'],['generator_temperature_c','Обмотка',1,'#b77732'],['rectifier_junction_temperature_c','Переход',1,'#a56768']]]];
 const holder=$('plots');if(!holder.children.length)for(const _ of defs){const el=document.createElement('section');el.className='plot';holder.append(el);}
 defs.forEach(([title,series],index)=>{
  const card=holder.children[index],width=Math.max(260,Math.floor(card.clientWidth-24)),left=65,right=15,top=38,height=150,bottom=top+height;
  const values=rows.flatMap(r=>series.map(([key,,scale])=>r[key]==null?null:r[key]*scale)).filter(Number.isFinite);let min=values.length?Math.min(...values):0,max=values.length?Math.max(...values):1;
  if(index===0){min=Math.min(min,0);max=Math.max(max,0);}if(min===max){min-=Math.max(1,Math.abs(min)*.01);max+=Math.max(1,Math.abs(max)*.01);}
  const events=(current?.events||[]).filter(e=>e.status==='applied'&&e.applied_sim_time!=null),xmax=Math.max(1,...rows.map(r=>r.time_s),...events.map(e=>e.applied_sim_time)),X=x=>left+x/xmax*(width-left-right),Y=y=>bottom-(y-min)/(max-min)*height;
  let svg='<title>'+esc(title)+'</title><rect width="100%" height="100%" fill="white"/><text x="8" y="20" font-size="15" fill="#193438">'+esc(title)+'</text>';
  for(let n=0;n<3;n++){const v=min+(max-min)*n/2,y=Y(v);svg+='<line x1="'+left+'" x2="'+(width-right)+'" y1="'+y+'" y2="'+y+'" stroke="#dbe5df"/><text x="'+(left-7)+'" y="'+(y+4)+'" text-anchor="end" font-size="12" fill="#526e65">'+esc(fmt(v,3))+'</text>';}
  for(const event of events)svg+='<line data-event-time="'+event.applied_sim_time+'" x1="'+X(event.applied_sim_time)+'" x2="'+X(event.applied_sim_time)+'" y1="'+top+'" y2="'+bottom+'" stroke="#af6542" stroke-dasharray="4 3"><title>'+esc(event.field+' → '+event.new_value+' при t='+event.applied_sim_time+' с')+'</title></line>';
  for(const[key,label,scale,color]of series){let d='',open=false;for(const row of rows){const v=row[key];if(v==null||!Number.isFinite(v)){open=false;continue;}d+=(open?'L':'M')+X(row.time_s)+','+Y(v*scale)+' ';open=true;}svg+='<path d="'+d+'" fill="none" stroke="'+color+'" stroke-width="2"><title>'+esc(label)+'</title></path>';if(rows.length===1&&Number.isFinite(rows[0][key]))svg+='<circle cx="'+X(rows[0].time_s)+'" cy="'+Y(rows[0][key]*scale)+'" r="3" fill="'+color+'"/>';}
  svg+='<text x="'+left+'" y="'+(bottom+21)+'" font-size="12" fill="#526e65">0</text><text x="'+(width-right)+'" y="'+(bottom+21)+'" text-anchor="end" font-size="12" fill="#526e65">'+esc(fmt(xmax))+'</text><text x="'+(width/2)+'" y="'+(bottom+39)+'" text-anchor="middle" font-size="12" fill="#526e65">Время, с</text>';
  let legendY=bottom+60;for(const[,label,,color]of series){svg+='<line x1="8" x2="22" y1="'+(legendY-4)+'" y2="'+(legendY-4)+'" stroke="'+color+'" stroke-width="2"/><text x="30" y="'+legendY+'" font-size="12" fill="#193438">'+esc(label)+'</text>';legendY+=18;}
  card.innerHTML='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+width+' '+(legendY+8)+'" role="img" aria-label="'+esc(title)+'" font-family="Segoe UI,Arial,sans-serif">'+svg+'</svg>'+(rows.length?'':'<p>Нет рассчитанных точек</p>');
 });
}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function csv(){const fields=['sequence','time_start_s','time_s','dt_s','requested_power_w','ambient_temperature_c','engine_rpm','bus_voltage_v','generator_current_a','generator_bus_power_w','battery_power_w','battery_current_a','battery_soc','battery_temperature_c','battery_surface_temperature_c','generator_temperature_c','rectifier_sink_temperature_c','rectifier_case_temperature_c','rectifier_junction_temperature_c','fuel_flow_kg_h','cumulative_fuel_kg','operating_point_status'];const cell=v=>v==null?'':'"'+String(v).replace(/"/g,'""')+'"';return '\ufeff'+[fields.join(','),...rows.map(r=>fields.map(k=>cell(r[k])).join(','))].join('\r\n');}
function guard(fn){return async event=>{const at=performance.now();try{$('error').hidden=true;const task=fn();measurements.push({kind:'event_handler_sync',action:event?.currentTarget?.id,ms:performance.now()-at});requestAnimationFrame(()=>measurements.push({kind:'frame_callback_wait',action:event?.target?.id,ms:performance.now()-at,note:'callback scheduling, not proof of screen presentation'}));await task;}catch(error){fail(error);render();}};}
for(const id of ['mode','strategy'])$(id).onchange=syncSetup;
$('create').onclick=guard(async()=>{if(commandPending||running||busy)return;commandPending=true;render();try{await create();await listRuns();}finally{commandPending=false;render();}});
for(const action of ['step','start','pause','stop'])$(action).onclick=guard(()=>control(action));
$('applyLoad').onclick=guard(()=>apply('load_kw','load',0,30));$('applyAir').onclick=guard(()=>apply('ambient_c','air',-60,80));$('applyRpm').onclick=guard(()=>apply('rpm','rpm',5000,6500));
for(const id of ['load','air','rpm'])$(id).addEventListener('input',()=>{$('receipt').textContent=receipt();});
$('exportJson').onclick=guard(()=>download('nir_android_experiment.json',JSON.stringify(report(),null,2),'application/json'));
$('exportCsv').onclick=guard(()=>download('nir_android_results.csv',csv(),'text/csv;charset=utf-8'));
$('exportSvg').onclick=guard(()=>download('nir_android_power.svg','<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString($('plots').querySelector('svg')),'image/svg+xml'));
$('importJson').onchange=guard(async()=>{const file=$('importJson').files[0];if(!file)return;try{if(file.size>20000000)throw Error('JSON больше 20 МБ');const document=JSON.parse(await file.text());await request('validate_import',{document});preview=document;$('fileStatus').textContent='Проверен файл '+file.name+' · '+fmt(document.state.time_s)+' с. Импорт не запускает расчёт. «Свежий replay» создаст новый опыт с нуля.';render();}finally{$('importJson').value='';}});
$('replay').onclick=guard(async()=>{if(!preview||busy||running||commandPending)return;const document=preview;pendingNewId=crypto.randomUUID();commandPending=true;render();try{await request('import',{session_id:pendingNewId,document});await request('start');}catch(error){pendingNewId=null;throw error;}finally{commandPending=false;render();}});
$('savedRuns').onchange=guard(async()=>{if(!$('savedRuns').value)return;const item=await saved($('savedRuns').value);if(!item)return;current=item.report;rows=current.trace||[];activeId=null;readOnly=true;preview=current;$('fileStatus').textContent='Открыт сохранённый префикс. Продолжение не восстановлено; доступен новый опыт или свежий replay.';render();});
 $('diagnostics').onclick=()=>{const data={runtime,cacheProof,build:manifest?.build,environment:{userAgent:navigator.userAgent,secureContext:window.isSecureContext,online_flag:navigator.onLine,memory:performance.memory?{method:'nonstandard Chromium performance.memory; JS heap only, not complete WASM or process memory',usedJSHeapSize:performance.memory.usedJSHeapSize,totalJSHeapSize:performance.memory.totalJSHeapSize,jsHeapSizeLimit:performance.memory.jsHeapSizeLimit}:null},measurements,current:report()};$('diagnosticText').textContent=JSON.stringify({runtime,cacheProof,measurements:measurements.slice(-20)},null,2);download('nir_android_diagnostics.json',JSON.stringify(data,null,2),'application/json');};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(running||busy)&&worker){waitingBoundary=true;render();request('pause').catch(fail);}});
let lastWidth=0;new ResizeObserver(entries=>{const width=Math.round(entries[0].contentRect.width);if(width!==lastWidth){lastWidth=width;draw();}}).observe($('plots'));
(async()=>{syncSetup();render();try{const items=await saved();if(items.length){const last=items.sort((a,b)=>b.updated.localeCompare(a.updated))[0];current=last.report;rows=current.trace||[];readOnly=true;preview=current;$('fileStatus').textContent='Восстановлен последний подтверждённый префикс. Контроллер не продолжен автоматически.';await listRuns();render();}}catch(error){fail(Error('Хранилище недоступно: '+error.message));}await prepare();})().catch(error=>{fail(error);$('preparationStatus').textContent='Автономный набор не подготовлен. Для первой установки нужен доступный HTTPS-адрес и полная загрузка.';});
