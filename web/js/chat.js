import {$, state, emit, toast, request, jsonRequest, sessionUrl} from './api.js';
let source = null, seen = new Set(), pendingRequest = null;
function markdown(element, content) {
  // Markdown解析后必须净化；本地依赖不可用时安全退化为纯文本。
  if (window.marked && window.DOMPurify) element.innerHTML = window.DOMPurify.sanitize(window.marked.parse(content), {FORBID_TAGS:['iframe','style','script'], FORBID_ATTR:['style']});
  else { element.textContent = content; element.style.whiteSpace = 'pre-wrap'; }
  for (const link of element.querySelectorAll('a')) {link.target = '_blank'; link.rel = 'noopener noreferrer';}
}
export function resetDraft() {
  for (const entry of state.draft) if (entry.ownedUrl) URL.revokeObjectURL(entry.url);
  state.draft = []; pendingRequest = null; renderDraft();
}
function renderDraft() {
  $('#attachments').replaceChildren();
  $('#attachments').classList.toggle('hidden', !state.draft.length);
  for (const [index, entry] of state.draft.entries()) {
    const tile = document.createElement('div'); tile.className = 'attachment';
    if (!entry.url) {entry.url = URL.createObjectURL(entry.file); entry.ownedUrl = true;}
    const image = document.createElement('img'); image.src = entry.url; image.alt = entry.name;
    const label = document.createElement('small'); label.textContent = `${index+1}. ${entry.name}`; label.title = entry.relative;
    const remove = document.createElement('button'); remove.textContent = '×'; remove.ariaLabel = `移除 ${entry.name}`;
    remove.onclick = () => { if(state.sending)return; if (entry.ownedUrl) URL.revokeObjectURL(entry.url); state.draft.splice(index,1); pendingRequest = null; renderDraft(); };
    tile.append(image,label,remove); $('#attachments').append(tile);
  }
  $('#composer-scope').textContent = state.draft.length ? `待发送 ${state.draft.length} 张航拍图` : '支持单图与多图共同分析';
}
export async function loadMessages() {
  if (!state.sid) return;
  const sid = state.sid;
  let after = 0, values = [];
  do {
    const result = await request(`/api/sessions/${sid}/messages?after_message_id=${after}&limit=100`);
    values.push(...result.messages); after = result.next_after;
  } while (after);
  if (sid !== state.sid) return;
  const scroll = $('#message-scroll');
  const atBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 100;
  $('#messages').replaceChildren();
  $('#welcome').classList.toggle('hidden', values.length > 0);
  for (const message of values) {
    const article = document.createElement('article'); article.className = 'message ' + message.role;
    article.dataset.turnId = message.turn_id;
    const label = document.createElement('div'); label.className = 'message-label';
    if (message.role === 'assistant') {const mark = document.createElement('span'); mark.className = 'agent-dot'; mark.textContent = 'R'; label.append(mark);}
    label.append(document.createTextNode(message.role === 'assistant' ? 'RescueNet · 灾情分析' : '你'));
    article.append(label);
    if (message.role === 'user' && message.image_ids?.length) {
      const images = document.createElement('div'); images.className = 'message-images';
      for (const imageId of message.image_ids) {
        const resource = state.resources.find(x => x.image_id === imageId && x.role === 'original');
        if (!resource) continue;
        const button = document.createElement('button');
        const image = document.createElement('img'); image.src = resource.preview_url; image.alt = resource.source_names[0]; image.loading = 'lazy';
        const small = document.createElement('small'); small.textContent = resource.source_names[0];
        button.append(image,small); button.onclick = () => emit('resource:preview', resource); images.append(button);
      }
      article.append(images);
    }
    const content = document.createElement('div'); content.className = 'message-content';
    // 文件的预览入口由资源清单提供，避免将内部磁盘路径当作网页链接。
    const text = message.content.replace(/\n\n报告文件：[^\n]+/g, '\n\n报告已归档，可在会话资源中查看。');
    if (message.role === 'user') content.textContent = text; else markdown(content,text);
    article.append(content);
    const footer = document.createElement('div'); footer.className = 'message-footer';
    footer.textContent = `第 ${Number(message.turn_id)} 轮 · ${new Date(message.timestamp).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',timeZone:'Asia/Macau'})}`;
    if (message.status !== 'completed') footer.append(document.createTextNode(' · ' + ({failed:'未全部完成',running:'处理中',interrupted:'已中断',needs_clarification:'待补充',unsupported:'超出范围'}[message.status] || message.status)));
    const reports = state.resources.filter(x => x.turn_id === message.turn_id && x.kind === 'reports');
    if (message.role === 'assistant' && reports.length) {const button = document.createElement('button'); button.textContent = `查看报告（${reports.length}）`; button.onclick = () => emit('resource:reports', message.turn_id); footer.append(button);}
    article.append(footer); $('#messages').append(article);
  }
  if (atBottom || values.length <= 2) scroll.scrollTop = scroll.scrollHeight;
  else $('#new-message').classList.remove('hidden');
}
export function stopWatching() {source?.close(); source = null; seen.clear(); state.busy = false; $('#job-panel').classList.add('hidden'); $('#retry-job')?.remove(); $('#progress-log').replaceChildren(); updateSend();}
function updateSend() {$('#send-message').disabled = state.busy || state.sending; $('#question').disabled=state.sending; $('#send-message').title = state.busy ? '当前会话正在处理，请等待完成' : '发送分析需求';}
export function watchJob(job) {
  source?.close(); seen.clear();
  state.busy = ['queued','running'].includes(job.status); updateSend();
  $('#retry-job')?.remove();
  if (!state.busy) {
    if(job.status==='interrupted') {
      $('#job-panel').classList.remove('hidden'); $('#job-label').textContent='任务因服务重启中断，已保留落盘成果';
      const retry=document.createElement('button'); retry.id='retry-job'; retry.className='text-button'; retry.textContent='重试原任务';
      const sid=state.sid;
      retry.onclick=async()=>{retry.disabled=true;try{const next=await jsonRequest(`/api/sessions/${sid}/jobs/${job.job_id}/retry`,'POST',{});if(state.sid===sid)watchJob(next);}catch(error){toast(error.message);retry.disabled=false;}};
      $('.job-summary').append(retry);
    }
    return;
  }
  const sid = state.sid;
  $('#job-panel').classList.remove('hidden'); $('#progress-log').replaceChildren(); $('#job-label').textContent = job.status === 'queued' ? '分析任务已排队' : '正在处理灾情分析…';
  source = new EventSource(`/api/sessions/${sid}/jobs/${job.job_id}/events`);
  const watching = source;
  watching.onmessage = async event => {
    if (sid !== state.sid || source !== watching) return;
    const value = JSON.parse(event.data);
    if (seen.has(value.event_id)) return;
    seen.add(value.event_id);
    const label = value.summary || value.node;
    $('#job-label').textContent = label;
    if (value.event !== 'state_update') {const li = document.createElement('li'); li.textContent = label; $('#progress-log').append(li); li.scrollIntoView({block:'nearest'});}
    if (value.event === 'artifact_saved') emit('resources:refresh');
    if (value.event === 'job_result') {
      watching.close(); state.busy = false; updateSend();
      $('#job-panel').classList.add('hidden'); emit('session:refresh');
      if (value.status === 'failed') toast('本轮未全部完成，已保留可用成果，请查看回答与执行状态。');
    }
  };
  watching.onerror = async () => {
    if (sid !== state.sid || source !== watching) return;
    try {
      const latest = await request(`/api/sessions/${sid}/jobs/${job.job_id}`);
      if (sid !== state.sid || source !== watching) return;
      if (!['queued','running'].includes(latest.status)) {watching.close(); state.busy = false; updateSend(); $('#job-panel').classList.add('hidden'); emit('session:refresh');}
      else $('#job-label').textContent = '连接恢复中，后台任务继续执行…';
    } catch {if(source===watching)$('#job-label').textContent = '连接暂时中断，正在重连…';}
  };
}
export function initChat(ensureSession) {
  document.addEventListener('draft:add', event => {
    if(state.sending)return toast('附件正在提交，请稍候。');
    for (const entry of event.detail) {
      if (state.draft.some(existing => existing.file === entry.file)) continue;
      if (state.draft.length >= state.meta.max_images) {toast(`每轮最多 ${state.meta.max_images} 张图片`); break;}
      if (entry.file.size > state.meta.max_upload_bytes) {toast(`${entry.name} 超过单文件大小限制`); continue;}
      // 草稿与图集分别持有URL，切换文件夹或回收离屏缩略图不影响待发送预览。
      state.draft.push({...entry,url:URL.createObjectURL(entry.file),ownedUrl:true});
    }
    pendingRequest = null; renderDraft(); $('#question').focus();
  });
  document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => {$('#question').value = button.dataset.prompt; $('#question').focus();});
  $('#new-message').onclick = () => {$('#message-scroll').scrollTop = $('#message-scroll').scrollHeight; $('#new-message').classList.add('hidden');};
  $('#toggle-progress').onclick = () => $('#progress-log').classList.toggle('hidden');
  $('#question').onkeydown = event => {if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); $('#send-message').click();}};
  $('#send-message').onclick = async () => {
    if (state.busy || state.sending) return;
    const text = $('#question').value.trim();
    if (!text && !state.draft.length) return toast('请描述分析需求，或选择航拍图片。');
    state.sending = true; updateSend();
    try {
      await ensureSession();
      const sid = state.sid;
      const signature = JSON.stringify([text,state.draft.map(x => [x.name,x.file.size,x.file.lastModified])]);
      if (!pendingRequest || pendingRequest.signature !== signature || pendingRequest.sid !== sid) pendingRequest = {sid,signature,id:crypto.randomUUID(),images:null};
      if (!pendingRequest.images) {
        pendingRequest.images = null;
        if (state.draft.length) {
          const form = new FormData(); state.draft.forEach(x => form.append('files',x.file,x.name));
          form.append('relative_paths',JSON.stringify(state.draft.map(x => x.relative)));
          const uploaded = await request(sessionUrl('/uploads'),{method:'POST',body:form});
          pendingRequest.images = uploaded.uploads.map(x => ({kind:'upload',id:x.upload_id}));
          if (uploaded.uploads.length < state.draft.length) toast('相同内容的图片已合并，避免重复分析。');
        } else pendingRequest.images=[];
      }
      const job = await jsonRequest(sessionUrl('/messages'),'POST',{question:text || '分析所选灾情图像',request_id:pendingRequest.id,images:pendingRequest.images});
      if (sid !== state.sid) return;
      $('#question').value = ''; resetDraft(); emit('gallery:clear'); watchJob(job);
      emit('session:refresh');
    } catch (error) {toast(error.message);}
    finally {state.sending = false; updateSend();}
  };
}
