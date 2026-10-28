"""
报告生成工具

职责：基于优先级计算结果生成结构化的应急救援报告（HTML/PDF/Markdown）
"""
import os
import base64
import io
import re
from datetime import datetime
from typing import Dict, Any

from modules.agent.base_tool import BaseTool
from configs.agent_config import REPORT_CONFIG


class ReportGeneratorTool(BaseTool):
    """
    报告生成工具
    
    设计动机：
    - 将结构化数据转换为可读的应急救援报告
    - 支持多种输出格式（HTML/PDF/Markdown）
    - 包含可视化元素和 actionable 建议
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化报告生成器
        
        Args:
            config: 报告配置，默认使用REPORT_CONFIG
        """
        self.config = config or REPORT_CONFIG
        self.output_dir = self.config['output_dir']
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "generate_report"
    
    @property
    def description(self) -> str:
        return "为当前单张图像生成独立应急救援报告，包含单图受灾等级、损坏统计、救援建议和语义分割可视化。"
    
    def execute(
        self,
        priority_result: Dict[str, Any],
        output_filename: str = None,
        segmentation_result: Dict[str, Any] = None,
        statistics_result: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        生成应急救援报告
        
        Args:
            priority_result: PriorityCalculatorTool的输出
            output_filename: 输出文件名（不含扩展名），默认自动生成
        
        Returns:
            {
                'report_path': str,      # 报告文件路径
                'format': str,           # 报告格式
                'summary': str           # 报告摘要文本
            }
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"emergency_report_{timestamp}"
        
        # 根据配置选择格式
        report_format = self.config['format']
        
        original_image_data = self._build_original_image(segmentation_result)
        visualization_data = self._build_segmentation_visualization(segmentation_result)

        if report_format == 'html':
            report_path = self._generate_html(
                priority_result,
                output_filename,
                original_image_data,
                visualization_data,
                statistics_result,
            )
        elif report_format == 'markdown':
            report_path = self._generate_markdown(priority_result, output_filename, visualization_data)
        else:
            raise ValueError(f"Unsupported format: {report_format}")
        
        # 生成摘要
        summary = self._extract_summary(priority_result)
        
        print(f"[INFO] 报告生成完成 | 格式: {report_format} | 路径: {report_path}")
        
        return {
            'report_path': report_path,
            'format': report_format,
            'summary': summary,
            'image_path': segmentation_result.get('image_path') if segmentation_result else None,
        }

    def update_conclusion(self, report_path: str, conclusion: str) -> None:
        """将 Agent 最终应急结论写入已生成的报告。"""
        if not report_path or not os.path.isfile(report_path):
            return
        if os.path.splitext(report_path)[1].lower() in {'.md', '.markdown'}:
            with open(report_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n## 🚨 Agent 应急救援结论\n\n{conclusion}\n")
            return
        with open(report_path, 'r', encoding='utf-8') as f:
            html = f.read()
        marker = '<div id="agent-conclusion"></div>'
        content = (
            '<div id="agent-conclusion" class="agent-conclusion">'
            '<h2>🚨 Agent 应急救援结论</h2>'
            f'{self._markdown_to_html(conclusion)}'
            '</div>'
        )
        if marker in html:
            html = html.replace(marker, content)
        elif '<div id="agent-conclusion" class="agent-conclusion">' in html:
            html = re.sub(
                r'<div id="agent-conclusion" class="agent-conclusion">.*?</div>',
                content,
                html,
                count=1,
                flags=re.DOTALL,
            )
        else:
            html = html.replace('</body>', content + '</body>')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

    @staticmethod
    def _escape_html(value: str) -> str:
        from html import escape
        return escape(value or '')

    @classmethod
    def _markdown_to_html(cls, value: str) -> str:
        """安全渲染 Agent 常用的标题、列表、粗体、代码和表格。"""
        lines = (value or '').splitlines()
        output = []
        index = 0
        list_type = None

        def close_list():
            nonlocal list_type
            if list_type:
                output.append(f'</{list_type}>')
                list_type = None

        def inline(text):
            escaped = cls._escape_html(text)
            escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
            escaped = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', escaped)
            return escaped

        while index < len(lines):
            line = lines[index].strip()
            if not line:
                close_list()
                index += 1
                continue

            if line.startswith('|') and index + 1 < len(lines):
                delimiter = lines[index + 1].strip()
                if delimiter.startswith('|') and re.fullmatch(r'[|:\-\s]+', delimiter):
                    close_list()
                    headers = [inline(cell.strip()) for cell in line.strip('|').split('|')]
                    output.append('<table><thead><tr>' + ''.join(f'<th>{cell}</th>' for cell in headers) + '</tr></thead><tbody>')
                    index += 2
                    while index < len(lines) and lines[index].strip().startswith('|'):
                        cells = [inline(cell.strip()) for cell in lines[index].strip().strip('|').split('|')]
                        output.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
                        index += 1
                    output.append('</tbody></table>')
                    continue

            heading = re.match(r'^(#{1,3})\s+(.+)$', line)
            if heading:
                close_list()
                level = len(heading.group(1))
                output.append(f'<h{level}>{inline(heading.group(2))}</h{level}>')
            elif re.match(r'^\d+\.\s+', line):
                if list_type != 'ol':
                    close_list()
                    output.append('<ol>')
                    list_type = 'ol'
                item_text = re.sub(r'^\d+\.\s+', '', line)
                output.append(f'<li>{inline(item_text)}</li>')
            elif re.match(r'^[-*]\s+', line):
                if list_type != 'ul':
                    close_list()
                    output.append('<ul>')
                    list_type = 'ul'
                item_text = re.sub(r'^[-*]\s+', '', line)
                output.append(f'<li>{inline(item_text)}</li>')
            else:
                close_list()
                output.append(f'<p>{inline(line)}</p>')
            index += 1

        close_list()
        return ''.join(output)

    def _build_segmentation_visualization(self, segmentation_result: Dict[str, Any] = None) -> str:
        """将类别掩膜编码为可嵌入 HTML 的 PNG data URI。"""
        if not segmentation_result or segmentation_result.get('best_mask') is None:
            return ''
        try:
            from PIL import Image
            import numpy as np
            mask = np.asarray(segmentation_result['best_mask'], dtype=np.uint8)
            palette = np.array([
                [0, 0, 0], [40, 120, 220], [150, 150, 150], [255, 220, 0],
                [255, 120, 0], [220, 0, 0], [180, 0, 180], [40, 180, 80],
                [120, 70, 20], [0, 160, 80], [0, 220, 220],
            ], dtype=np.uint8)
            rgb = palette[np.clip(mask, 0, len(palette) - 1)]
            image = Image.fromarray(rgb, mode='RGB')
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f'data:image/png;base64,{encoded}'
        except Exception as exc:
            print(f"⚠️ 语义分割图生成失败: {exc}")
            return ''

    def _build_original_image(self, segmentation_result: Dict[str, Any] = None) -> str:
        """读取原航拍图并编码为可独立携带的 data URI。"""
        if not segmentation_result:
            return ''
        image_path = segmentation_result.get('image_path')
        if not image_path or not os.path.isfile(image_path):
            return ''
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=90, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f'data:image/jpeg;base64,{encoded}'
        except Exception as exc:
            print(f"⚠️ 原航拍图嵌入失败: {exc}")
            return ''

    def _generate_html(
        self,
        priority_result: Dict,
        filename: str,
        original_image_data: str = '',
        visualization_data: str = '',
        statistics_result: Dict = None,
    ) -> str:
        """生成HTML格式报告"""
        from jinja2 import Template
        
        template_str = self._get_html_template()
        template = Template(template_str)
        
        # 渲染模板
        zone = priority_result['priority_zones'][0]
        image_name = os.path.splitext(zone['image_id'])[0]
        image_stats = statistics_result['per_image_stats'][0] if statistics_result else {}
        class_ratios = image_stats.get('class_ratios', {})
        html_content = template.render(
            title=f"{image_name} 灾害应急响应报告",
            generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            zone=zone,
            image_stats=image_stats,
            class_ratios=class_ratios,
            include_visualization=self.config['include_visualization'],
            original_image=original_image_data,
            segmentation_visualization=visualization_data,
        )
        
        # 保存文件
        report_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_markdown(self, priority_result: Dict, filename: str, visualization_data: str = '') -> str:
        """生成Markdown格式报告"""
        md_content = self._build_markdown_content(priority_result)
        if visualization_data:
            md_content = f"![语义分割图]({visualization_data})\n\n" + md_content
        
        report_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return report_path
    
    def _get_html_template(self) -> str:
        """获取HTML模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #d32f2f; border-bottom: 3px solid #d32f2f; padding-bottom: 10px; }
        h2 { color: #1976d2; margin-top: 30px; }
        .summary { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }
        .critical { background: #f8d7da; color: #721c24; padding: 10px; margin: 10px 0; border-radius: 4px; }
        .high { background: #fff3cd; color: #856404; padding: 10px; margin: 10px 0; border-radius: 4px; }
        .medium { background: #d1ecf1; color: #0c5460; padding: 10px; margin: 10px 0; border-radius: 4px; }
        .low { background: #d4edda; color: #155724; padding: 10px; margin: 10px 0; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }
        .report-header { margin-bottom: 24px; }
        .image-comparison { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 24px; align-items: start; margin-top: 22px; }
        .report-image { min-width: 0; margin: 0; }
        .report-image img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; border: 1px solid #bbb; background: #111; }
        .mask-legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3px 10px; margin-top: 8px; font-size: 0.78em; color: #444; }
        .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 4px; vertical-align: middle; }
        .agent-conclusion { background: #e8f4ff; border-left: 4px solid #1976d2; padding: 15px; margin: 20px 0; }
        .agent-conclusion h1, .agent-conclusion h2, .agent-conclusion h3 { color: #1976d2; border: 0; padding: 0; }
        .agent-conclusion p { margin: 10px 0; }
        .agent-conclusion table { background: #fff; }
        @media (max-width: 760px) {
            body { margin: 20px; }
            .image-comparison { grid-template-columns: 1fr; }
            .mask-legend { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="report-header">
        <h1>🏗️ {{ title }}</h1>
        <p><strong>生成时间:</strong> {{ generated_time }}</p>
        {% if original_image or segmentation_visualization %}
        <div class="image-comparison">
            {% if original_image %}
            <figure class="report-image original-image">
                <img src="{{ original_image }}" alt="原航拍图">
            </figure>
            {% endif %}
            {% if segmentation_visualization %}
            <figure class="report-image segmentation-visualization">
                <img src="{{ segmentation_visualization }}" alt="语义分割图">
                <div class="mask-legend">
                    <span><i class="swatch" style="background:#969696"></i>无损建筑</span>
                    <span><i class="swatch" style="background:#ffdc00"></i>轻微损坏</span>
                    <span><i class="swatch" style="background:#ff7800"></i>严重损坏</span>
                    <span><i class="swatch" style="background:#dc0000"></i>完全毁坏</span>
                    <span><i class="swatch" style="background:#28b450"></i>道路畅通</span>
                    <span><i class="swatch" style="background:#784614"></i>道路阻塞</span>
                </div>
            </figure>
            {% endif %}
        </div>
        {% endif %}
    </div>
    <div id="agent-conclusion"></div>
    
    <div class="summary {{ zone.level|lower }}">
        <h2>📊 当前图像灾情摘要</h2>
        <ul>
            <li><strong>图像:</strong> {{ zone.image_id }}</li>
            <li><strong>单图受灾等级:</strong> {{ zone.level }}</li>
            <li><strong>救援优先级评分:</strong> {{ zone.priority_score }}</li>
            <li><strong>道路阻塞占比:</strong> {{ "%.2f"|format(zone.road_blocked_ratio * 100) }}%</li>
        </ul>
    </div>
    
    <h2>🏚️ 当前图像损坏构成</h2>
    <table>
        <tr>
            <th>类别</th>
            <th>像素数量</th>
            <th>占整张图比例</th>
        </tr>
        <tr><td>建筑物完全毁坏</td><td>{{ zone.destroyed_count }}</td><td>{{ "%.2f"|format(class_ratios.get(5, 0) * 100) }}%</td></tr>
        <tr><td>建筑物严重损坏</td><td>{{ zone.major_damage_count }}</td><td>{{ "%.2f"|format(class_ratios.get(4, 0) * 100) }}%</td></tr>
        <tr><td>建筑物轻微损坏</td><td>{{ zone.minor_damage_count }}</td><td>{{ "%.2f"|format(class_ratios.get(3, 0) * 100) }}%</td></tr>
        <tr><td>道路阻塞</td><td>{{ image_stats.class_counts.get(8, 0) }}</td><td>{{ "%.2f"|format(class_ratios.get(8, 0) * 100) }}%</td></tr>
    </table>
    
    <h2>💡 救援建议</h2>
    <ol>
        <li><strong>搜救:</strong> 根据当前图像中的完全毁坏和严重损坏建筑比例配置搜索、破拆与生命探测力量。</li>
        <li><strong>医疗:</strong> 在当前图像覆盖区域外缘设置伤员检伤与转运点。</li>
        <li><strong>道路抢通:</strong> 对图中道路阻塞区域优先投入清障与工程保障力量。</li>
        <li><strong>现场核验:</strong> 结合无人机复拍和地面勘察确认分割结果，避免仅凭模型结果调度。</li>
    </ol>
    
    <div class="footer">
        <p>本报告由 RescueNet Emergency Response Agent (RERA) 自动生成</p>
        <p>⚠️ 注意: 本报告仅供参考，实际救援决策请结合现场情况</p>
    </div>
</body>
</html>
        """
    
    def _build_markdown_content(self, priority_result: Dict) -> str:
        """构建Markdown内容"""
        zone = priority_result['priority_zones'][0]
        image_name = os.path.splitext(zone['image_id'])[0]
        
        md = f"""# 🏗️ {image_name} 灾害应急响应报告

**生成时间:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 当前图像灾情摘要

- 图像: {zone['image_id']}
- 单图受灾等级: {zone['level']}
- 救援优先级评分: {zone['priority_score']}
- 完全毁坏建筑像素: {zone['destroyed_count']}
- 严重损坏建筑像素: {zone['major_damage_count']}
- 轻微损坏建筑像素: {zone['minor_damage_count']}
- 道路阻塞占比: {zone['road_blocked_ratio'] * 100:.2f}%

## 💡 救援建议

1. 根据当前图像中的建筑损坏情况配置搜救、破拆与生命探测力量。
2. 在当前图像覆盖区域外缘设置伤员检伤与转运点。
3. 对图中道路阻塞区域优先投入清障与工程保障力量。
4. 结合无人机复拍和地面勘察核验模型结果。

---

*本报告由 RescueNet Emergency Response Agent (RERA) 自动生成*

⚠️ **注意:** 本报告仅供参考，实际救援决策请结合现场情况
"""
        
        return md
    
    def _extract_summary(self, priority_result: Dict) -> str:
        """提取报告摘要（用于快速预览）"""
        zone = priority_result['priority_zones'][0]
        return (
            f"图像 {zone['image_id']} 分析完成，单图受灾等级为 {zone['level']}，"
            f"优先级评分 {zone['priority_score']}，道路阻塞占比 "
            f"{zone['road_blocked_ratio'] * 100:.2f}%。"
        )
