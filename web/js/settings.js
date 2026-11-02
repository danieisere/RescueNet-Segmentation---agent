import {$, toast, request, jsonRequest} from './api.js';
let current = null;
const fields = [
  ['大模型服务','llm', [['provider','服务商','text'],['model','模型名称','text'],['base_url','接口地址','text'],['wire_api','接口类型',['responses','chat_completions']],['reasoning_effort','推理强度',['none','minimal','low','medium','high','xhigh']],['verbosity','回答详细程度',['low','medium','high']]]],
  ['图像分割','ensemble',[['models','参与集成的模型','models'],['confidence_lambda','置信度惩罚系数','number']]],
  ['规则评分权重','priority_weights',[['destroyed','建筑物完全毁坏','number'],['major','建筑物严重损坏','number'],['minor','建筑物轻微损坏','number'],['road_blocked','道路阻塞','number']]],
  ['报告','report',[['format','默认报告格式',['html','markdown']],['include_visualization','包含可视化图像','checkbox']]],
  ['运行与上传','agent',[['device','计算设备',['auto','cpu','cuda']]]],
  ['图片数量','vision',[['max_input_images','每轮最多图片数','integer']]],
  ['网页附件','web',[['max_upload_bytes','单图片大小限制（字节）','integer']]],
];
function render() {
  const form=$('#settings-form'); form.replaceChildren();
  $('#settings-notice').textContent=current.restart_required ? '配置已保存。当前后端仍使用启动时的配置，重启后生效。' : '配置保存后，重启后端生效；运行中的任务不受影响。';
  for (const [title,section,items] of fields) {
    const group=document.createElement('section'); group.className='settings-section';
    const heading=document.createElement('h3'); heading.textContent=title; group.append(heading);
    for (const [key,label,type] of items) {
      const row=document.createElement('div'); row.className='setting-row';
      const name=document.createElement('label'); name.textContent=label;
      const value=current.values[section]?.[key];
      const runtime=current.runtime[section]?.[key];
      if (current.restart_required && JSON.stringify(value)!==JSON.stringify(runtime)) {const note=document.createElement('small'); note.textContent=`运行中：${Array.isArray(runtime)?runtime.join('、'):(runtime??'默认')}`; name.append(note);}
      let input;
      if (Array.isArray(type)) {input=document.createElement('select'); type.forEach(option=>input.add(new Option(option,option))); input.value=value??type[0];}
      else if(type==='models') {
        input=document.createElement('div'); input.className='model-choices';
        for(const model of ['unet','pspnet','deeplabv3']) {const label=document.createElement('label'); const checkbox=document.createElement('input'); checkbox.type='checkbox'; checkbox.value=model; checkbox.checked=(value||[]).includes(model); label.append(checkbox,document.createTextNode(model)); input.append(label);}
      } else {
        input=document.createElement('input'); input.type=type==='integer'?'number':type;
        if(type==='number'||type==='integer'){input.min='0';input.step=type==='integer'?'1':'any';}
        if(type==='checkbox')input.checked=Boolean(value);else input.value=value??'';
      }
      input.dataset.section=section; input.dataset.key=key; input.dataset.type=Array.isArray(type)?'select':type;
      name.htmlFor=`setting-${section}-${key}`;input.id=name.htmlFor;
      row.append(name,input);group.append(row);
    }
    form.append(group);
  }
  const group=document.createElement('section');group.className='settings-section';
  const heading=document.createElement('h3');heading.textContent='会话与配置来源（只读）';group.append(heading);
  const info=[['API 密钥',current.credentials.configured?'已配置 · 内容不返回浏览器':'未配置'],['密钥环境变量',current.credentials.api_key_env],['历史压缩阈值',current.readonly.compression_threshold],['压缩目标',current.readonly.compression_target_tokens],['优先保留近期轮次',current.readonly.recent_turns],['模型权重目录',current.readonly.weight_dir],['会话产物目录',current.readonly.report_directory],['实际配置文件',current.source]];
  for(const [label,value]of info){const row=document.createElement('div');row.className='setting-row';const name=document.createElement('label');name.textContent=label;const text=document.createElement('span');text.className='readonly-setting';text.textContent=value;row.append(name,text);group.append(row);}
  form.append(group);
}
export function initSettings() {
  $('#open-settings').onclick=async()=>{try{current=await request('/api/settings');render();$('#settings-status').textContent='';$('#settings-dialog').showModal();}catch(error){toast(error.message);}};
  $('#close-settings').onclick=()=>$('#settings-dialog').close();
  $('#settings-form').onsubmit=event=>event.preventDefault();
  $('#save-settings').onclick=async()=>{
    if(!current)return;
    const changes={};
    try {
      for(const input of $('#settings-form').querySelectorAll('[data-section]')){
        const {section,key,type}=input.dataset;
        const value=type==='models'?[...input.querySelectorAll('input:checked')].map(x=>x.value):type==='checkbox'?input.checked:['integer','number'].includes(type)?Number(input.value):input.value;
        if(JSON.stringify(value)!==JSON.stringify(current.values[section]?.[key])) (changes[section]||={})[key]=value;
      }
      if(!Object.keys(changes).length)return toast('设置没有变化。');
      $('#save-settings').disabled=true;
      current=await jsonRequest('/api/settings','PATCH',{version:current.version,changes});render();$('#settings-status').textContent='已保存，重启后端后生效。';
    }catch(error){toast(error.message);}finally{$('#save-settings').disabled=false;}
  };
}
