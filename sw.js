/* Immutable resource sets; calculation results are never served from this cache. */
importScripts('release-id.js');
const CACHE='nir-android-assets-'+self.NIR_BUILD;
const base=new URL('./',self.location.href);
const digest=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b)),v=>v.toString(16).padStart(2,'0')).join('');
async function progress(value){for(const client of await self.clients.matchAll({type:'window',includeUncontrolled:true}))client.postMessage(value);}
self.addEventListener('install',event=>event.waitUntil((async()=>{
 const response=await fetch(new URL('asset-manifest.json',base),{cache:'no-store'});
 if(!response.ok)throw Error('Manifest недоступен');
 const manifest=await response.json();if(manifest.build!==self.NIR_BUILD)throw Error('Смешаны версии SW и ресурсов');
 const exists=await caches.has(CACHE),cache=await caches.open(CACHE);
 try{
  let done=0,bytes=0;
  for(const row of manifest.files){
   const url=new URL(row.path,base);
   if(url.origin!==base.origin||!url.pathname.startsWith(base.pathname))throw Error('Путь ресурса вне приложения');
   const resource=await fetch(url,{cache:'no-store'});if(!resource.ok)throw Error('Ресурс: '+row.path);
   const data=await resource.clone().arrayBuffer();
   if(data.byteLength!==row.size_bytes||await digest(data)!==row.sha256)throw Error('SHA-256: '+row.path);
   await cache.put(url,resource);done++;bytes+=data.byteLength;
   await progress({type:'cache_progress',done,total:manifest.files.length,bytes});
  }
  await cache.put(new URL('asset-manifest.json',base),new Response(JSON.stringify(manifest),{headers:{'Content-Type':'application/json'}}));
 }catch(error){if(!exists)await caches.delete(CACHE);await progress({type:'cache_error',message:String(error)});throw error;}
 // No skipWaiting: an installed update cannot replace an open experiment.
})()));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('fetch',event=>{
 const url=new URL(event.request.url);
 event.respondWith((async()=>{
  if(event.request.method!=='GET'||url.origin!==base.origin)return new Response('Offline-only application',{status:503});
  const cache=await caches.open(CACHE);
  if(event.request.mode==='navigate'&&(url.pathname===base.pathname||url.pathname===base.pathname+'index.html'))return await cache.match(new URL('index.html',base))||new Response('Пакет не подготовлен',{status:503});
  return await cache.match(event.request,{ignoreSearch:false})||new Response('Ресурс отсутствует в автономном наборе',{status:503});
 })());
});
self.addEventListener('message',event=>{
 if(event.data?.type!=='verify')return;
 event.waitUntil((async()=>{
  const cache=await caches.open(CACHE),response=await cache.match(new URL('asset-manifest.json',base));
  if(!response)throw Error('Нет manifest в кэше');
  const manifest=await response.json();let bytes=0;
  for(const row of manifest.files){const item=await cache.match(new URL(row.path,base));if(!item)throw Error('Нет '+row.path);const data=await item.arrayBuffer();if(await digest(data)!==row.sha256)throw Error('Повреждён '+row.path);bytes+=data.byteLength;}
  event.ports[0].postMessage({ok:true,build:manifest.build,files:manifest.files.length,bytes});
 })().catch(error=>event.ports[0].postMessage({ok:false,message:String(error)})));
});
