"""综合报告渲染：保留分图原文与图像，下载单文件也可阅读。"""
import html
import json
import base64
import mimetypes
from pathlib import Path
from .artifact_store import atomic_bytes, sha256
from .report_naming import build_report_name


class GroupReportRenderer:
    def render(self, group_result, output_dir, report_format):
        items = group_result['per_image']
        if not any(item['status'] == 'completed' for item in items):
            raise ValueError('全部分图失败，不能生成综合成功报告')
        name = build_report_name([item['source_name'] for item in items], group_result['image_ids'], report_format)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        status = '全部完成' if group_result['aggregate_status'] == 'completed' else '部分完成：未覆盖的图片与阶段见下文'
        sections, overview_rows, thumbnails = [], [], []
        metadata = f"轮次：{group_result.get('turn_id', '未记录')} · 生成时间：{group_result.get('created_at', '未记录')}"
        markdown = [f"# {name['display_name']}", status, metadata, '## 分析范围', group_result['question'],
                                  '图片：' + '、'.join(item['source_name'] for item in items), group_result['final_answer']]
        nav = ['<a href="#overview">综合总览</a>']
        for item in items:
            content = ''
            stats, priority = {}, {}
            markdown.append('## ' + item['source_name'])
            markdown.append(item['final_answer'])
            for record in item.get('artifacts', []):
                if record['role'] not in {'report', 'statistics', 'priority', 'original'}:
                    continue
                root = Path(item['item_dir']).resolve()
                path = (root / record['path']).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha256(path) != record['sha256']:
                    raise ValueError('分图报告或指标来源验证失败')
                if record['role'] == 'statistics':
                    stats = (json.loads(path.read_text(encoding='utf-8')).get('per_image_stats') or [{}])[0]
                    continue
                if record['role'] == 'priority':
                    priority = (json.loads(path.read_text(encoding='utf-8')).get('priority_zones') or [{}])[0]
                    continue
                if record['role'] == 'original':
                    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
                    mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
                    thumbnails.append(f'<figure><img alt="{html.escape(item["source_name"], quote=True)}" src="data:{mime};base64,{encoded}"><figcaption>{html.escape(item["source_name"])}</figcaption></figure>')
                    continue
                original = path.read_text(encoding='utf-8')
                if path.suffix == '.html':
                    # srcdoc 内嵌完整已有报告及其data URL图像，避免下载后丢失外部资源。
                    content = '<iframe sandbox title="分图报告" srcdoc="' + html.escape(original, quote=True) + '"></iframe>'
                else:
                    content = '<pre>' + html.escape(original) + '</pre>'
                    markdown.append(original)
            # 只展示原清单中已验证的单图指标，不跨图累加像素或按分数排名。
            ratios = stats.get('class_ratios', {})
            metrics = [item['source_name'], item['status'], str(stats.get('total_pixels', '未计算')),
                       f"{ratios['5']:.2%}" if '5' in ratios else '未计算',
                       f"{ratios['8']:.2%}" if '8' in ratios else '未计算', priority.get('level', '未计算')]
            overview_rows.append('<tr>' + ''.join('<td>'+html.escape(value)+'</td>' for value in metrics) + '</tr>')
            image_id = html.escape(item['image_id'], quote=True)
            nav.append(f'<a href="#image-{image_id}">{html.escape(item["source_name"])}</a>')
            sections.append(f'<section id="image-{image_id}"><h2>{html.escape(item["source_name"])}</h2>'
                            f'<p>{html.escape(item["status"])}</p><pre>{html.escape(item["final_answer"])}</pre>{content}</section>')
        if report_format == 'html':
            body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(name['display_name'])}</title><style>
body{{margin:0;background:#f5f7fa;color:#253247;font:15px/1.8 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:32px}}
nav{{position:sticky;top:0;background:white;padding:12px;display:flex;gap:16px;flex-wrap:wrap;border-bottom:1px solid #dde3eb}}
a{{color:#3266a3;text-decoration:none}}section{{background:white;margin:20px 0;padding:24px;border:1px solid #e1e6ed;border-radius:8px;scroll-margin-top:80px}}
pre{{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}}iframe{{width:100%;height:1250px;border:1px solid #e1e6ed}}.status{{color:#8a591c}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #e1e6ed;padding:8px;text-align:left}}.gallery{{display:flex;flex-wrap:wrap;gap:16px}}figure{{margin:0;max-width:210px}}figure img{{width:100%;height:140px;object-fit:contain}}.table-scroll{{overflow:auto}}
</style><nav>{''.join(nav)}</nav><main><h1>{html.escape(name['display_name'])}</h1><p class="status">{status}</p>
<p>{html.escape(metadata)}</p>
<section id="overview"><h2>分析范围与综合总览</h2><p>{html.escape(group_result['question'])}</p>
<p>{html.escape('、'.join(item['source_name'] for item in items))}</p><div class="gallery">{''.join(thumbnails)}</div>
<h3>各图指标总览</h3><div class="table-scroll"><table><thead><tr><th>图片</th><th>完成状态</th><th>原始掩膜总像素</th><th>完全毁坏像素占比</th><th>道路阻塞像素占比</th><th>规则等级</th></tr></thead><tbody>{''.join(overview_rows)}</tbody></table></div>
<p>两项占比的分母均为对应图片的原始掩膜总像素；不代表建筑数量或道路实际通行能力。按选图顺序展示。</p><pre>{html.escape(group_result['final_answer'])}</pre>
<p>各图像素指标独立计算，不能累加为实际受灾面积；现场信息与模型判断应分别核实。</p></section>{''.join(sections)}</main></html>'''
        else:
            body = '\n\n'.join(markdown)
        path = directory / name['filename']
        atomic_bytes(path, body.encode('utf-8'))
        record = {'role': 'group_report', 'path': 'report/' + path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size}
        return {**name, 'report_path': str(path), 'format': report_format, 'artifacts': [record]}
