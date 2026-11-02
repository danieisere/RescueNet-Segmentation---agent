import {$, state, emit, toast} from './api.js';
let entries = [], selected = new Set(), observer;
const allowed = /\.(jpg|jpeg|png|bmp)$/i;

export function clearSelection() { selected.clear(); render(); }
function updateSelection() {
  $('#selected-count').textContent = `已选择 ${selected.size} 张`;
  $('#gallery-actions').classList.toggle('hidden', !entries.length);
  $('#add-selection').disabled = !selected.size;
}
function render() {
  observer?.disconnect();
  for (const entry of entries) if(entry.url){URL.revokeObjectURL(entry.url);entry.url=null;}
  $('#gallery-grid').replaceChildren();
  const query = $('#gallery-search').value.trim().toLowerCase();
  observer = new IntersectionObserver(changes => {
    for (const change of changes) {
      const image = change.target;
      const entry = entries[Number(image.dataset.index)];
      if (entry) {
        if (change.isIntersecting) {entry.url ||= URL.createObjectURL(entry.file); image.src = entry.url;}
        else if (entry.url) {
          // 离开预加载范围就释放图集URL；草稿有独立URL，不会被此处误释放。
          image.removeAttribute('src'); URL.revokeObjectURL(entry.url); entry.url = null;
        }
      }
    }
  }, {root:$('#gallery-scroll'), rootMargin:'160px'});
  entries.forEach((entry, index) => {
    if (query && !entry.relative.toLowerCase().includes(query)) return;
    const button = document.createElement('button');
    button.className = 'gallery-tile' + (selected.has(entry.key) ? ' selected' : '');
    button.title = entry.relative;
    button.setAttribute('aria-label', `选择 ${entry.relative}`);
    button.setAttribute('aria-pressed', String(selected.has(entry.key)));
    const thumb = document.createElement('div'); thumb.className = 'thumb';
    const image = document.createElement('img'); image.alt = entry.name; image.dataset.index = index;
    thumb.append(image); button.append(thumb);
    const label = document.createElement('small'); label.textContent = entry.name; button.append(label);
    if (selected.has(entry.key)) {
      const number = document.createElement('span'); number.className = 'selection-number';
      number.textContent = [...selected].indexOf(entry.key) + 1; button.append(number);
    }
    button.onclick = () => {
      if (selected.has(entry.key)) selected.delete(entry.key);
      else {
        if (selected.size >= (state.meta?.max_images || 5)) return toast(`每轮最多选择 ${state.meta?.max_images || 5} 张图片`);
        selected.add(entry.key);
      }
      render();
    };
    $('#gallery-grid').append(button); observer.observe(image);
  });
  $('#gallery-empty').classList.toggle('hidden', entries.length > 0);
  updateSelection();
}
function loadFolder(files) {
  // 只有此处建立本地URL；不调用上传接口，不上传目录清单。
  for (const entry of entries) if (entry.url) URL.revokeObjectURL(entry.url);
  entries = [...files].filter(file => allowed.test(file.name)).sort((a,b) =>
    (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, 'zh-CN', {numeric:true})).map((file,index) =>
      ({key:`gallery-${index}`,file,name:file.name,relative:file.webkitRelativePath || file.name}));
  selected.clear();
  $('#gallery-count').textContent = `${entries.length} 张`;
  $('#folder-name').textContent = entries[0]?.relative.split('/')[0] || '文件夹中没有支持的图片';
  $('#gallery-search-wrap').classList.toggle('hidden', !entries.length);
  $('#gallery-search').value = '';
  render();
}
export function initGallery() {
  const input = $('#folder-input');
  $('#choose-folder').onclick = () => {
    if ('webkitdirectory' in input) input.click();
    else { toast('此浏览器暂不支持目录选择，请选择多张图片'); $('#files-input').click(); }
  };
  input.onchange = () => loadFolder(input.files);
  $('#gallery-search').oninput = render;
  $('#clear-selection').onclick = clearSelection;
  $('#add-selection').onclick = () => {
    emit('draft:add', [...selected].map(key => entries.find(entry => entry.key === key)));
    if (innerWidth < 700) $('#shell').classList.add('no-sidebar');
  };
  $('#attach-images').onclick = () => $('#files-input').click();
  $('#files-input').onchange = () => {
    const files = [...$('#files-input').files].filter(file => allowed.test(file.name));
    emit('draft:add', files.map(file => ({key:crypto.randomUUID(),file,name:file.name,relative:file.name})));
    $('#files-input').value = '';
  };
  const composer = $('#composer');
  composer.ondragover = event => {event.preventDefault(); composer.style.borderColor = '#7299bd';};
  composer.ondragleave = () => {composer.style.borderColor = '';};
  composer.ondrop = event => {
    event.preventDefault(); composer.style.borderColor = '';
    emit('draft:add', [...event.dataTransfer.files].filter(file => allowed.test(file.name)).map(file =>
      ({key:crypto.randomUUID(),file,name:file.name,relative:file.name})));
  };
}
