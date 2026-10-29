"""
报告生成工具

职责：基于优先级计算结果生成结构化的应急救援报告（HTML/PDF/Markdown）
"""
import os
import re
from datetime import datetime
from typing import Dict, Any

from modules.agent.base_tool import BaseTool
from modules.agent.segmentation_visualization import (
    build_segmentation_visualizations,
    colorize_mask,
    image_to_data_url,
    segmentation_legend,
)
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
        
        # 优先复用分割工具生成的三图，保证报告与后续模型输入的颜色、位置完全一致。
        visualizations = self._build_visualizations(segmentation_result)
        original_image_data = visualizations.get('original', '')
        visualization_data = visualizations.get('segmentation', '')
        overlay_data = visualizations.get('overlay', '')
        legend = visualizations.get('legend', segmentation_legend())

        if report_format == 'html':
            report_path = self._generate_html(
                priority_result,
                output_filename,
                original_image_data,
                visualization_data,
                statistics_result,
                overlay_data=overlay_data,
                legend=legend,
            )
        elif report_format == 'markdown':
            report_path = self._generate_markdown(
                priority_result, output_filename, visualization_data,
                original_image_data=original_image_data, overlay_data=overlay_data, legend=legend,
            )
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

    def _build_visualizations(self, segmentation_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """复用三图；旧结果没有该字段时，通过同一实现补齐，兼容既有调用。"""
        if not segmentation_result or segmentation_result.get('best_mask') is None:
            return {}
        if segmentation_result.get('visualizations'):
            return segmentation_result['visualizations']
        try:
            from PIL import Image, ImageOps
            import numpy as np
            mask = segmentation_result['best_mask']
            image_path = segmentation_result.get('image_path')
            if image_path and os.path.isfile(image_path):
                with Image.open(image_path) as image:
                    # 对齐 OpenCV 默认读取的 EXIF 朝向，避免旧结果的叠加图发生旋转错位。
                    original = np.array(ImageOps.exif_transpose(image).convert('RGB'))
                return build_segmentation_visualizations(original, mask)
            # 原图不可用时只展示已有掩膜，不生成没有真实底图的叠加结果。
            return {'segmentation': image_to_data_url(colorize_mask(mask)),
                    'legend': segmentation_legend()}
        except Exception as exc:
            print(f"⚠️ 分割可视化生成失败: {exc}")
            return {}

    def _build_segmentation_visualization(self, segmentation_result: Dict[str, Any] = None) -> str:
        """保留旧接口，统一使用共享配色生成分割图。"""
        return self._build_visualizations(segmentation_result).get('segmentation', '')

    def _build_original_image(self, segmentation_result: Dict[str, Any] = None) -> str:
        """保留旧接口，优先使用本次分割时读取的原航拍图。"""
        return self._build_visualizations(segmentation_result).get('original', '')

    def _generate_html(
        self,
        priority_result: Dict,
        filename: str,
        original_image_data: str = '',
        visualization_data: str = '',
        statistics_result: Dict = None,
        overlay_data: str = '',
        legend: list = None,
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
            overlay_visualization=overlay_data,
            segmentation_legend=legend if legend is not None else segmentation_legend(),
        )
        
        # 保存文件
        report_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_markdown(
        self, priority_result: Dict, filename: str, visualization_data: str = '',
        original_image_data: str = '', overlay_data: str = '', legend: list = None,
    ) -> str:
        """生成Markdown格式报告"""
        md_content = self._build_markdown_content(priority_result)
        # Markdown 与 HTML 展示相同的三图和全类别图例。
        images = [('原航拍图', original_image_data), ('语义分割图', visualization_data),
                  ('原图与分割结果叠加图', overlay_data)]
        image_sections = [f'![{title}]({data})' for title, data in images if data]
        if image_sections:
            legend = legend if legend is not None else segmentation_legend()
            legend_lines = ['| 类别编号 | 类别 | 颜色（RGB） |', '|---|---|---|']
            legend_lines.extend(
                f"| {entry['class_id']} | {entry['name']} | {entry['hex']} |" for entry in legend
            )
            md_content = '\n\n'.join(image_sections + ['\n'.join(legend_lines), md_content])
        
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
        .report-image img { display: block; width: 100%; height: auto; object-fit: contain; border: 1px solid #bbb; background: #111; }
        .overlay-visualization { grid-column: 1 / -1; }
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
        {% if original_image or segmentation_visualization or overlay_visualization %}
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
                    {% for entry in segmentation_legend %}
                    <span><i class="swatch" style="background:{{ entry.hex }}"></i>{{ entry.class_id }} {{ entry.name }}</span>
                    {% endfor %}
                </div>
            </figure>
            {% endif %}
            {% if overlay_visualization %}
            <figure class="report-image overlay-visualization">
                <img src="{{ overlay_visualization }}" alt="原图与分割结果叠加图">
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
