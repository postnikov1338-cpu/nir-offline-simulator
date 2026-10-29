'use strict';
const worker=new Worker('worker.js'),waiting=new Map();let sessionId=null,header=null;
const results={runtime:null,cases:{},timing:{}};
function request(action,extra={}){return new Promise((resolve,reject)=>{const command_id=crypto.randomUUID();waiting.set(command_id,{resolve,reject});worker.postMessage({action,session_id:sessionId,command_id,...extra});});}
worker.onmessage=({data:m})=>{
 if(m.type==='ready'){results.runtime=m.runtime;results.timing.init_ms=m.init_ms;document.getElementById('progress').textContent='Ядро готово; smoke физического шага пройден';document.getElementById('pilot').disabled=false;document.getElementById('mission').disabled=false;}
 if(m.type==='progress')document.getElementById('progress').textContent=m.message;
 if(m.checkpoint)worker.postMessage({action:'checkpoint_ack',token:m.checkpoint,ok:true});
 if(m.type==='response'){if(m.value.header)header=m.value.header;waiting.get(m.request_id)?.resolve(m.value);waiting.delete(m.request_id);}
 if(['error','fatal'].includes(m.type)){document.getElementById('progress').textContent=m.message;waiting.get(m.request_id)?.reject(Error(m.message));waiting.delete(m.request_id);}
};
async function create(config){sessionId=crypto.randomUUID();return request('create',{config});}
async function run(config){await create(config);const rows=[],start=performance.now();while(!['completed','terminated','error'].includes(header.status)){const value=await request('step');rows.push(...value.rows);document.getElementById('progress').textContent='Рассчитано '+header.state.time_s+' с';if(!value.rows.length)break;}return {header,trace:rows,wall_ms:performance.now()-start};}
document.getElementById('pilot').onclick=async()=>{try{document.getElementById('pilot').disabled=true;
 for(const [name,inputs]of [['engine_5000',{rpm:5000,torque_nm:10}],['engine_6500',{rpm:6500,torque_nm:10}],['engine_7200',{rpm:7200,torque_nm:10}]]){results.cases[name]=await request('bench',{kind:'engine',inputs});}
 results.cases.coupled=await request('bench',{kind:'coupled',inputs:{load_kw:6}});
 results.cases.three=await run({mode:'predefined',segments:[6,5,7].map(load_kw=>({duration_s:1,load_kw,ambient_c:25}))});
 results.cases.manual=await run({strategy:'manual',manual_rpm:5750,segments:[{duration_s:2,load_kw:6,ambient_c:25}]});
 results.cases.infeasible=await run({segments:[{duration_s:1,load_kw:30,ambient_c:25}]});
 document.getElementById('output').textContent=JSON.stringify(results);document.getElementById('progress').textContent='Пилот завершён';
 }catch(e){document.getElementById('progress').textContent=String(e);}};
document.getElementById('mission').onclick=async()=>{try{document.getElementById('mission').disabled=true;results.cases.mission60=await run({mode:'predefined',segments:[6,5,7].map(load_kw=>({duration_s:20,load_kw,ambient_c:25}))});document.getElementById('output').textContent=JSON.stringify(results);document.getElementById('progress').textContent='60 с завершены';}catch(e){document.getElementById('progress').textContent=String(e);}};
request('init');
