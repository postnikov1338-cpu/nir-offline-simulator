'use strict';
let py=null,bootPromise=null,current=null,running=false,busy=false,intent=null,checkpoint=null;
const commands=new Map();
const send=value=>self.postMessage(value);
const digest=async data=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),x=>x.toString(16).padStart(2,'0')).join('');
function call(message){py.globals.set('payload_json',JSON.stringify(message));return JSON.parse(py.runPython('android_runtime.dispatch_json(payload_json)'));}
async function boot(){
 if(bootPromise)return bootPromise;
 bootPromise=(async()=>{
  const start=performance.now();const manifest=await (await fetch('asset-manifest.json')).json();
  send({type:'progress',message:'Загрузка Python / WebAssembly…'});
  importScripts('runtime/pyodide.js');
  py=await loadPyodide({indexURL:new URL('runtime/',self.location.href).href});
  await py.loadPackage('numpy');
  const sources=manifest.files.filter(r=>r.path.startsWith('python/'));
  for(const row of sources){const data=await (await fetch(row.path)).arrayBuffer();if(await digest(data)!==row.sha256)throw Error('Контрольная сумма: '+row.path);const dest='/nir/'+row.path.slice(7),dir=dest.slice(0,dest.lastIndexOf('/'));py.FS.mkdirTree(dir);py.FS.writeFile(dest,new Uint8Array(data));}
  py.runPython("import sys\nsys.path.insert(0, '/nir')\nimport android_runtime");
  send({type:'progress',message:'Короткая физическая проверка ядра…'});
  call({action:'create',session_id:'kernel-smoke',config:{segments:[{duration_s:1,load_kw:6,ambient_c:25}]}});
  const smoke=call({action:'step',session_id:'kernel-smoke',command_id:'smoke-1'});
  if(smoke.header.status!=='completed'||smoke.rows.length!==1)throw Error('Физическая проверка ядра не пройдена');
  const runtime=JSON.parse(py.runPython("__import__('json').dumps(android_runtime.SESSION.runtime)"));
  send({type:'ready',runtime,build:manifest.build,init_ms:performance.now()-start,smoke:smoke.rows[0]});
  return true;
 })();return bootPromise;
}
function acknowledge(token,ok){if(checkpoint?.token===token){const waiter=checkpoint;checkpoint=null;ok?waiter.resolve():waiter.reject(Error('Checkpoint не сохранён; запуск остановлен'));}}
function waitCheckpoint(token){return new Promise((resolve,reject)=>{checkpoint={token,resolve,reject};});}
async function physicalStep(request,reply=true){
 if(busy)throw Error('Физический шаг уже выполняется');
 busy=true;
 try{
  send({type:'computing',session_id:current?.session_id});
  const result=call({...request,action:'step'});current=result.header;
  if(!result.rows.length)running=false;
  const token=crypto.randomUUID(),ack=waitCheckpoint(token);
  const response={type:reply?'response':'update',request_id:reply?request.command_id:null,value:result,checkpoint:token,execution:running?'running':'paused'};
  if(reply)commands.set(request.command_id,{request:JSON.stringify(request),response});
  send(response);
  await ack;
 }catch(error){running=false;send({type:'fatal',message:String(error.message||error),session_id:current?.session_id});throw error;}finally{busy=false;}
 if(intent){const action=intent;intent=null;running=false;const result=call({action,session_id:current.session_id,command_id:crypto.randomUUID()});current=result.header;send({type:'update',value:result,execution:action==='stop'?'stopped':'paused'});}
 if(current.status==='completed'||current.status==='terminated'||current.status==='error')running=false;
}
async function loop(){
 if(!running||busy)return;
 try{await physicalStep({session_id:current.session_id,command_id:crypto.randomUUID()},false);}
 catch(error){running=false;send({type:'fatal',message:String(error.message||error),session_id:current?.session_id});return;}
 // A macrotask boundary is required: queued pause/stop/events must run between steps.
 if(running)setTimeout(loop,0);
 else send({type:'idle',session_id:current?.session_id,status:current?.status});
}
self.onmessage=async({data:message})=>{
 if(message.action==='checkpoint_ack'){acknowledge(message.token,message.ok);return;}
 try{
  await boot();
  const key=message.command_id;
  if(typeof key!=='string')throw Error('Нужен уникальный ID команды');
  if(commands.has(key)){
   if(commands.get(key).request!==JSON.stringify(message))throw Error('Повтор ID с другими данными');
   send(commands.get(key).response);return;
  }
  if(message.action==='init'){send({type:'response',request_id:key,value:{ready:true}});return;}
  if(!['create','import','bench','validate_import'].includes(message.action)&&message.session_id!==current?.session_id)throw Error('Ответ/команда другой сессии');
  let value;
  if(['create','import'].includes(message.action)){
   if(busy||running)throw Error('Сначала дождитесь паузы или остановки');
   value=call(message);current=value.header;intent=null;
  }else if(message.action==='start'){
   if(busy||running||!['ready','paused'].includes(current.status))throw Error('Запуск сейчас недоступен');
   running=true;value={header:current,rows:[]};setTimeout(loop,0);
  }else if(['pause','stop'].includes(message.action)){
   running=false;
   if(busy){intent=message.action;value={header:current,rows:[]};}
   else{value=call(message);current=value.header;}
  }else if(message.action==='step'){
   if(running||busy)throw Error('Шаг уже выполняется');
   await physicalStep(message);return;
  }else{
   if(['export','bench','validate_import'].includes(message.action)&&(busy||running))throw Error('Дождитесь подтверждённой паузы');
   value=call(message);if(value.header)current=value.header;
  }
  const response={type:'response',request_id:key,value,execution:intent?'waiting_boundary':running?'running':'paused'};
  if(commands.size>=5000)commands.delete(commands.keys().next().value);
  commands.set(key,{request:JSON.stringify(message),response});send(response);
 }catch(error){send({type:'error',request_id:message.command_id,message:String(error.message||error),session_id:current?.session_id});}
};
