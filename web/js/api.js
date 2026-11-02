/* 各区域共享会话身份；本地图集文件始终留在浏览器内存。 */
export const state = {sid:null, session:null, resources:[], draft:[], meta:null, busy:false, sending:false};
export const $ = (selector) => document.querySelector(selector);
export const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, {detail}));
export function toast(message) {
  $('#toast').textContent = message;
  $('#toast').classList.remove('hidden');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $('#toast').classList.add('hidden'), 4500);
}
export async function request(url, options = {}) {
  const response = await fetch(url, options);
  const value = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = value.error?.message || (Array.isArray(value.detail) ? value.detail.map(x => x.msg).join('；') : value.detail) || `请求失败（${response.status}）`;
    throw new Error(message);
  }
  return value;
}
export const jsonRequest = (url, method, body) => request(url, {method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
export const sessionUrl = (suffix='') => `/api/sessions/${state.sid}${suffix}`;
