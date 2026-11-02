import {$, state, emit, toast, request, jsonRequest} from './api.js';
import {initGallery, clearSelection} from './gallery.js';
import {initChat, resetDraft, loadMessages, stopWatching, watchJob} from './chat.js';
import {initResources, loadResources, renderResources} from './resources.js';
import {initSettings} from './settings.js';
let sessions = [], nextCursor = null, historyQuery = '';

async function loadHistory(append=false) {
  const query = $('#history-search').value.trim();
  const result = await request(`/api/sessions?q=${encodeURIComponent(query)}&limit=50${append && nextCursor ? '&cursor='+nextCursor : ''}`);
  if (query !== $('#history-search').value.trim()) return;
  sessions = append ? [...sessions,...result.sessions] : result.sessions;
  nextCursor = result.next_cursor; $('#more-sessions').classList.toggle('hidden',!nextCursor);
  $('#session-count').textContent = sessions.length + (nextCursor ? '+' : '');
  $('#history-list').replaceChildren();
  let day = '';
  const today = new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Macau'}).format(new Date());
  for (const session of sessions) {
    const group = session.display_name.slice(0,10)===today ? '今天' : session.display_name.slice(0,10);
    if (group!==day) {const title=document.createElement('div'); title.className='history-group'; title.textContent=group; $('#history-list').append(title); day=group;}
    const button=document.createElement('button'); button.className='session-item'+(session.session_id===state.sid?' active':''); button.dataset.sid=session.session_id;
    const title=document.createElement('strong'); title.textContent=session.display_name;
    const info=document.createElement('small'); info.textContent=`${session.last_turn} 轮对话 · ${session.image_count} 张图片${['queued','running'].includes(session.job_status)?' · 处理中':''}`;
    button.append(title,info); button.onclick=()=>selectSession(session.session_id).catch(error=>toast(error.message)); $('#history-list').append(button);
  }
  if (!sessions.length) {const empty=document.createElement('p'); empty.className='subtle-empty'; empty.textContent=query?'未找到对应时间的会话':'暂无历史会话'; $('#history-list').append(empty);}
}
async function refreshSession() {
  if (!state.sid) return;
  const sid = state.sid;
  const session = await request(`/api/sessions/${sid}`);
  if (sid !== state.sid) return;
  state.session=session;
  $('#session-title').textContent=session.display_name;
  const active=session.images.filter(x=>session.active_image_ids.includes(x.image_id));
  $('#active-images').textContent=active.length ? `分析对象 · ${active.map(x=>x.source_name).join('、')}` : '尚未绑定航拍图 · 可先补充现场信息';
  await loadResources(); await loadMessages(); await loadHistory();
  if (session.job_id && ['queued','running','interrupted'].includes(session.job_status) && !state.busy) watchJob({job_id:session.job_id,status:session.job_status});
}
async function selectSession(sid, preserveDraft=false) {
  if (state.sending && !preserveDraft) return toast('正在提交消息，请稍候切换会话。');
  stopWatching();
  if (!preserveDraft) {resetDraft(); $('#question').value=''; clearSelection();}
  state.sid=sid; state.resources=[]; renderResources();
  $('#messages').replaceChildren(); $('#welcome').classList.remove('hidden');
  $('#resource-turn').value=''; $('#resource-image').value='';
  localStorage.setItem('rescuenet-session',sid);
  await refreshSession();
  $('#message-scroll').scrollTop=$('#message-scroll').scrollHeight;
  if (innerWidth<700) $('#shell').classList.add('no-sidebar');
}
async function ensureSession() {
  if (!state.sid) {const session=await jsonRequest('/api/sessions','POST',{}); await selectSession(session.session_id,true);}
}
function initLayout() {
  const shell=$('#shell');
  if (innerWidth<1180) shell.classList.add('no-resources');
  if (innerWidth<700) shell.classList.add('no-sidebar');
  $('#toggle-sidebar').onclick=()=>shell.classList.toggle('no-sidebar');
  $('#close-sidebar').onclick=()=>shell.classList.add('no-sidebar');
  $('#toggle-resources').onclick=()=>shell.classList.toggle('no-resources');
  $('#close-resources').onclick=()=>shell.classList.add('no-resources');
  const resizer=$('#sidebar-resizer');
  resizer.onpointerdown=event=>{
    resizer.setPointerCapture(event.pointerId);
    resizer.onpointermove=move=>{const top=$('.history-section').getBoundingClientRect().top; const height=Math.max(110,Math.min(innerHeight*.53,move.clientY-top)); document.documentElement.style.setProperty('--history',height+'px');};
    resizer.onpointerup=()=>{resizer.onpointermove=null;};
  };
  resizer.onkeydown=event=>{if (['ArrowUp','ArrowDown'].includes(event.key)) {event.preventDefault(); const height=$('.history-section').getBoundingClientRect().height+(event.key==='ArrowUp'?-20:20); document.documentElement.style.setProperty('--history',Math.max(110,Math.min(innerHeight*.53,height))+'px');}};
}
async function boot() {
  initLayout(); initGallery(); initResources(); initSettings(); initChat(ensureSession);
  $('#new-session').onclick=async()=>{if(state.sending)return; try{const session=await jsonRequest('/api/sessions','POST',{}); await selectSession(session.session_id);}catch(error){toast(error.message);}};
  let timer;
  $('#history-search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>loadHistory().catch(error=>toast(error.message)),220);};
  $('#more-sessions').onclick=()=>loadHistory(true).catch(error=>toast(error.message));
  document.addEventListener('gallery:clear',clearSelection);
  document.addEventListener('session:refresh',()=>refreshSession().catch(error=>toast(error.message)));
  document.addEventListener('resources:refresh',()=>loadResources().catch(error=>toast(error.message)));
  try {
    state.meta=await request('/api/meta'); $('#model-chip').textContent=state.meta.model || 'LangGraph';
    await loadHistory();
    const previous=localStorage.getItem('rescuenet-session');
    if(previous) {try{await selectSession(previous);}catch{localStorage.removeItem('rescuenet-session');state.sid=null;}}
  } catch(error) {toast('服务连接失败：'+error.message);}
}
boot();
