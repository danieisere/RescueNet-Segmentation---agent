"""
报告生成工具

职责：基于优先级计算结果生成结构化的应急救援报告（HTML/PDF/Markdown）
"""
import os
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
        return "基于优先级计算结果生成结构化的应急救援报告。报告包含执行摘要、详细分析、救援建议和风险评估，支持HTML/PDF/Markdown格式。"
    
    def execute(self, priority_result: Dict[str, Any], output_filename: str = None) -> Dict[str, Any]:
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
        
        if report_format == 'html':
            report_path = self._generate_html(priority_result, output_filename)
        elif report_format == 'markdown':
            report_path = self._generate_markdown(priority_result, output_filename)
        else:
            raise ValueError(f"Unsupported format: {report_format}")
        
        # 生成摘要
        summary = self._extract_summary(priority_result)
        
        print(f"[INFO] 报告生成完成 | 格式: {report_format} | 路径: {report_path}")
        
        return {
            'report_path': report_path,
            'format': report_format,
            'summary': summary
        }
    
    def _generate_html(self, priority_result: Dict, filename: str) -> str:
        """生成HTML格式报告"""
        from jinja2 import Template
        
        template_str = self._get_html_template()
        template = Template(template_str)
        
        # 渲染模板
        html_content = template.render(
            title="灾害应急响应报告",
            generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            statistics=priority_result['statistics'],
            priority_zones=priority_result['priority_zones'][:self.config['top_k_priority_zones']],
            include_visualization=self.config['include_visualization']
        )
        
        # 保存文件
        report_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_markdown(self, priority_result: Dict, filename: str) -> str:
        """生成Markdown格式报告"""
        md_content = self._build_markdown_content(priority_result)
        
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
    </style>
</head>
<body>
    <h1>🏗️ {{ title }}</h1>
    <p><strong>生成时间:</strong> {{ generated_time }}</p>
    
    <div class="summary">
        <h2>📊 执行摘要</h2>
        <ul>
            <li>总评估区域: {{ statistics.total_zones }} 个</li>
            <li>🔴 CRITICAL级别: {{ statistics.critical_count }} 个</li>
            <li>🟠 HIGH级别: {{ statistics.high_count }} 个</li>
            <li>🟡 MEDIUM级别: {{ statistics.medium_count }} 个</li>
            <li>🟢 LOW级别: {{ statistics.low_count }} 个</li>
        </ul>
    </div>
    
    <h2>🚨 优先救援区域 (Top {{ priority_zones|length }})</h2>
    <table>
        <tr>
            <th>排名</th>
            <th>区域ID</th>
            <th>优先级等级</th>
            <th>评分</th>
            <th>完全毁坏建筑</th>
            <th>严重损坏建筑</th>
            <th>道路阻塞率</th>
        </tr>
        {% for zone in priority_zones %}
        <tr class="{{ zone.level|lower }}">
            <td>{{ loop.index }}</td>
            <td>{{ zone.image_id }}</td>
            <td>
                {% if zone.level == 'CRITICAL' %}🔴 CRITICAL
                {% elif zone.level == 'HIGH' %}🟠 HIGH
                {% elif zone.level == 'MEDIUM' %}🟡 MEDIUM
                {% else %}🟢 LOW{% endif %}
            </td>
            <td>{{ zone.priority_score }}</td>
            <td>{{ zone.destroyed_count }}</td>
            <td>{{ zone.major_damage_count }}</td>
            <td>{{ "%.2f"|format(zone.road_blocked_ratio * 100) }}%</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>💡 救援建议</h2>
    <ol>
        <li><strong>立即行动:</strong> 优先派遣救援队到CRITICAL级别区域</li>
        <li><strong>资源调配:</strong> 为HIGH级别区域准备医疗设备和临时住所</li>
        <li><strong>道路疏通:</strong> 清理阻塞道路，确保救援通道畅通</li>
        <li><strong>持续监测:</strong> 对MEDIUM和LOW级别区域保持监控</li>
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
        stats = priority_result['statistics']
        zones = priority_result['priority_zones'][:self.config['top_k_priority_zones']]
        
        md = f"""# 🏗️ 灾害应急响应报告

**生成时间:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 执行摘要

- 总评估区域: {stats['total_zones']} 个
- 🔴 CRITICAL级别: {stats['critical_count']} 个
- 🟠 HIGH级别: {stats['high_count']} 个
- 🟡 MEDIUM级别: {stats['medium_count']} 个
- 🟢 LOW级别: {stats['low_count']} 个

## 🚨 优先救援区域 (Top {len(zones)})

| 排名 | 区域ID | 等级 | 评分 | 完全毁坏 | 严重损坏 | 道路阻塞率 |
|------|--------|------|------|----------|----------|------------|
"""
        
        for i, zone in enumerate(zones, 1):
            level_icon = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(zone['level'], '⚪')
            
            md += f"| {i} | {zone['image_id']} | {level_icon} {zone['level']} | {zone['priority_score']} | {zone['destroyed_count']} | {zone['major_damage_count']} | {zone['road_blocked_ratio']*100:.2f}% |\n"
        
        md += """
## 💡 救援建议

1. **立即行动:** 优先派遣救援队到CRITICAL级别区域
2. **资源调配:** 为HIGH级别区域准备医疗设备和临时住所
3. **道路疏通:** 清理阻塞道路，确保救援通道畅通
4. **持续监测:** 对MEDIUM和LOW级别区域保持监控

---

*本报告由 RescueNet Emergency Response Agent (RERA) 自动生成*

⚠️ **注意:** 本报告仅供参考，实际救援决策请结合现场情况
"""
        
        return md
    
    def _extract_summary(self, priority_result: Dict) -> str:
        """提取报告摘要（用于快速预览）"""
        stats = priority_result['statistics']
        
        summary = (
            f"灾情评估完成。共分析{stats['total_zones']}个区域，"
            f"其中{stats['critical_count']}个CRITICAL级别，"
            f"{stats['high_count']}个HIGH级别。"
        )
        
        if stats['top_3_critical']:
            top_zone = stats['top_3_critical'][0]
            summary += f"最紧急区域: {top_zone['image_id']} (评分: {top_zone['score']})"
        
        return summary
