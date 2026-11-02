import {$, state, emit, toast, request, sessionUrl} from './api.js';
let kind = 'images', previewToken = 0;
function size(bytes) {return bytes >= 1048576 ? `${(bytes/1048576).toFixed(1)} MB` : `${Math.max(1,Math.round(bytes/1024))} KB`;}
export async function loadResources() {
  if (!state.sid) return;
  const sid = state.sid;
  const result = await request(sessionUrl('/resources'));
  if (sid !== state.sid) return;
  state.resources = result.resources;
  for (const key of ['images','reports','data']) $(`#count-${key}`).textContent = state.resources.filter(x => x.kind === key).length;
  const turnSelect = $('#resource-turn'), previousTurn = turnSelect.value;
  turnSelect.replaceChildren(new Option('全部轮次',''));
  [...new Set(state.resources.map(x => x.turn_id).filter(Boolean))].sort().reverse().forEach(turn => turnSelect.add(new Option(`第 ${Number(turn)} 轮`,turn)));
  turnSelect.value = previousTurn;
  if (turnSelect.selectedIndex < 0) turnSelect.selectedIndex = 0;
  const imageSelect = $('#resource-image'), previousImage = imageSelect.value;
  imageSelect.replaceChildren(new Option('全部图片',''));
  const images = new Map();
  state.resources.filter(x => x.image_id).forEach(x => images.set(x.image_id,x.source_names[0]));
  for (const [id,name] of images) imageSelect.add(new Option(name,id));
  [...new Set(state.resources.map(x => x.group_id).filter(Boolean))].forEach(id => imageSelect.add(new Option(`多图组 · 第 ${Number(id.replace('group_',''))} 轮`,id)));
  imageSelect.value = previousImage;
  if (imageSelect.selectedIndex < 0) imageSelect.selectedIndex = 0;
  renderResources();
}
export function renderResources() {
  const list = $('#resource-list'); list.replaceChildren();
  const values = state.resources.filter(x => x.kind === kind
    && (!$('#resource-turn').value || x.turn_id === $('#resource-turn').value)
    && (!$('#resource-image').value || x.image_id === $('#resource-image').value || x.group_ids?.includes($('#resource-image').value)));
  values.sort((a,b) => (b.turn_id || '').localeCompare(a.turn_id || '') || Number(b.role==='group_report')-Number(a.role==='group_report'));
  let lastTurn = null;
  for (const value of values) {
    const turn = value.turn_id || 'inputs';
    if (turn !== lastTurn) {const title = document.createElement('div'); title.className = 'resource-group-title'; title.textContent = value.turn_id ? `第 ${Number(value.turn_id)} 轮` : '会话原始影像'; list.append(title); lastTurn = turn;}
    const card = document.createElement('article'); card.className = 'resource-item' + (value.role==='group_report' ? ' group-report' : '');
    if (value.kind === 'images') {const image = document.createElement('img'); image.className = 'resource-thumbnail'; image.src = value.preview_url; image.alt = value.display_name; image.loading = 'lazy'; image.onclick = () => previewResource(value); card.append(image);}
    const info = document.createElement('div'); info.className = 'resource-info';
    if (value.kind !== 'images') {const icon = document.createElement('span'); icon.className = 'resource-file-icon'; icon.textContent = value.kind==='reports' ? '▤' : '{}'; info.append(icon);}
    const text = document.createElement('div'), title = document.createElement('strong'), small = document.createElement('small');
    title.textContent = value.display_name; title.title = value.display_name;
    small.textContent = `${value.source_names.length > 1 ? `${value.source_names.length} 张图片 · ` : ''}${size(value.bytes)}${['failed','partial'].includes(value.status) ? ' · 部分成果' : ''}`;
    if(value.kind==='reports' && value.created_at) small.textContent += ' · '+new Date(value.created_at).toLocaleString('zh-CN',{timeZone:'Asia/Macau',hour12:false});
    text.append(title,small); info.append(text); card.append(info);
    const actions = document.createElement('div'); actions.className = 'resource-actions';
    const preview = document.createElement('button'); preview.textContent = value.role==='mask' ? '查看文件信息' : '预览'; preview.onclick = () => previewResource(value);
    const download = document.createElement('a'); download.href = value.download_url; download.textContent = '下载';
    actions.append(preview,download); card.append(actions); list.append(card);
  }
  if (!values.length) {
    const empty = document.createElement('div'); empty.className = 'resource-empty';
    const icon = document.createElement('span'); icon.textContent = kind==='reports' ? '▤' : '▧';
    const title = document.createElement('strong'); title.textContent = {images:'图像成果将在这里呈现',reports:'当前范围尚未生成报告',data:'当前范围尚无分析数据'}[kind];
    const hint = document.createElement('p'); hint.textContent = kind==='reports' ? '在对话中提出报告需求后查看' : '发送航拍图并提出分析需求';
    empty.append(icon,title,hint); list.append(empty);
  }
}
async function showPreview(value, token) {
  $('#preview-title').textContent = value.display_name;
  $('#preview-download').href = value.download_url;
  $('#preview-caption').textContent = value.source_names.join('、') + (value.turn_id ? ` · 第 ${Number(value.turn_id)} 轮` : '');
  const body = $('#preview-body'); body.replaceChildren();
  $('#preview-legend').replaceChildren(); $('#preview-legend').classList.add('hidden');
  if (value.kind==='images') {
    const image = document.createElement('img'); image.src = value.preview_url; image.alt = value.display_name; body.append(image);
    if(value.role!=='original') {
      const metadata=state.resources.find(x=>x.role==='visualization'&&x.image_id===value.image_id&&x.turn_id===value.turn_id);
      if(metadata)try{
        const response=await fetch(metadata.preview_url);const data=await response.json();
        if(token!==previewToken)return;
        for(const entry of data.legend||[]){const label=document.createElement('span');const color=document.createElement('i');if(/^#[0-9a-f]{6}$/i.test(entry.hex))color.style.background=entry.hex;label.append(color,document.createTextNode(entry.name));$('#preview-legend').append(label);}
        $('#preview-legend').classList.remove('hidden');
      }catch{ /* 图例读取失败不伪造颜色，原始数据入口仍可下载。 */ }
    }
  } else if (value.kind==='reports' && value.preview_url && value.display_name.endsWith('.html')) {
    const frame = document.createElement('iframe'); frame.setAttribute('sandbox',''); frame.title = value.display_name; frame.src = value.preview_url; body.append(frame);
  } else {
    const pre = document.createElement('pre'); pre.textContent = '正在读取…'; body.append(pre);
    if (value.role==='mask') pre.textContent = `原始类别掩膜（NumPy .npy）\n\n来源：${value.source_names.join('、')}\n大小：${size(value.bytes)}\nSHA-256：${value.sha256}\n\n该文件保存原始类别编号，请下载后使用 NumPy 读取。`;
    else {
      try {
        const response = await fetch(value.preview_url); if (!response.ok) throw new Error('资源读取失败');
        const text = await response.text();
        if (token !== previewToken) return;
        try {pre.textContent = JSON.stringify(JSON.parse(text),null,2);} catch {pre.textContent = text;}
      } catch(error) {if (token === previewToken) pre.textContent = error.message;}
    }
  }
}
export function previewResource(value) {
  const token = ++previewToken;
  $('#preview-tabs').replaceChildren();
  $('#preview-body').querySelector('.legend')?.remove();
  if (value.kind==='images') {
    const siblings = ['original','segmentation','overlay'].map(role => state.resources.find(x => x.role===role && x.image_id===value.image_id && (role==='original' || x.turn_id===value.turn_id))).filter(Boolean);
    for (const sibling of siblings) {
      const button = document.createElement('button'); button.textContent = {original:'原图',segmentation:'分割图',overlay:'叠加图'}[sibling.role];
      button.classList.toggle('active',sibling.resource_id===value.resource_id);
      button.onclick = () => {$('#preview-tabs').querySelectorAll('button').forEach(x => x.classList.remove('active')); button.classList.add('active'); showPreview(sibling,++previewToken);};
      $('#preview-tabs').append(button);
    }
  }
  showPreview(value,token);
  if (!$('#preview-dialog').open) $('#preview-dialog').showModal();
}
export function initResources() {
  document.querySelectorAll('[data-kind]').forEach(button => button.onclick = () => {
    kind = button.dataset.kind;
    document.querySelectorAll('[data-kind]').forEach(x => {x.classList.toggle('active',x===button); x.setAttribute('aria-selected',String(x===button));}); renderResources();
  });
  $('#resource-turn').onchange = renderResources; $('#resource-image').onchange = renderResources;
  $('#close-preview').onclick = () => {previewToken++; $('#preview-dialog').close(); $('#preview-body').replaceChildren();};
  document.addEventListener('resource:preview', event => previewResource(event.detail));
  document.addEventListener('resource:reports', event => {document.querySelector('[data-kind="reports"]').click(); $('#resource-turn').value=event.detail; renderResources(); $('#shell').classList.remove('no-resources');});
}
