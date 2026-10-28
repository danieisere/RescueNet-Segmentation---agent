import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from agent_cli import build_report_filename
from modules.agent.tools.report_tool import ReportGeneratorTool


class SingleImageReportTests(unittest.TestCase):
    def setUp(self):
        self.priority = {
            'priority_zones': [{
                'image_id': '10782.jpg',
                'image_path': '10782.jpg',
                'priority_score': 12345.0,
                'level': 'CRITICAL',
                'destroyed_count': 20,
                'major_damage_count': 10,
                'minor_damage_count': 5,
                'road_blocked_ratio': 0.25,
                'total_building_pixels': 40,
            }],
            'statistics': {'total_zones': 1},
        }
        self.statistics = {
            'per_image_stats': [{
                'image_id': '10782.jpg',
                'image_path': '10782.jpg',
                'total_pixels': 100,
                'class_counts': {3: 5, 4: 10, 5: 20, 8: 25},
                'class_ratios': {3: 0.05, 4: 0.10, 5: 0.20, 8: 0.25},
            }],
            'global_stats': {'total_images': 1},
        }
        self.segmentation = {
            'image_path': '10782.jpg',
            'best_mask': np.array([[3, 4], [5, 8]], dtype=np.int64),
        }

    def test_html_is_single_image_report_with_top_right_mask_and_conclusion(self):
        with tempfile.TemporaryDirectory() as output_dir:
            original_path = os.path.join(output_dir, '10782.jpg')
            Image.new('RGB', (8, 6), color=(80, 120, 160)).save(original_path)
            self.segmentation['image_path'] = original_path
            tool = ReportGeneratorTool({
                'output_dir': output_dir,
                'format': 'html',
                'include_visualization': True,
                'top_k_priority_zones': 1,
            })
            result = tool.execute(
                self.priority,
                '10782_report',
                self.segmentation,
                self.statistics,
            )
            conclusion = (
                '## 单图结论\n\n'
                '当前图像为 **CRITICAL**。\n\n'
                '| 类别 | 占比 |\n|---|---:|\n|道路阻塞|25%|'
            )
            tool.update_conclusion(result['report_path'], conclusion)

            with open(result['report_path'], 'r', encoding='utf-8') as f:
                html = f.read()

        self.assertIn('10782 灾害应急响应报告', html)
        self.assertNotIn('10782.jpg 灾害应急响应报告', html)
        self.assertIn('grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)', html)
        self.assertIn('data:image/jpeg;base64,', html)
        self.assertIn('data:image/png;base64,', html)
        self.assertIn('class="report-image original-image"', html)
        self.assertIn('class="report-image segmentation-visualization"', html)
        self.assertNotIn('语义分割结果（右上）', html)
        self.assertNotIn('<figcaption>', html)
        self.assertIn('Agent 应急救援结论', html)
        self.assertIn('<h2>单图结论</h2>', html)
        self.assertIn('<strong>CRITICAL</strong>', html)
        self.assertIn('<table>', html)
        self.assertNotIn('## 单图结论', html)
        self.assertNotIn('**CRITICAL**', html)
        self.assertNotIn('总评估区域', html)
        self.assertNotIn('优先救援区域 (Top', html)

    def test_batch_filenames_are_independent(self):
        first = build_report_filename('images/10782.jpg', None, 2)
        second = build_report_filename('images/10783.jpg', None, 2)
        custom = build_report_filename('images/10782.jpg', 'disaster.html', 2)

        self.assertEqual(first, '10782_report.html')
        self.assertEqual(second, '10783_report.html')
        self.assertEqual(custom, 'disaster_10782.html')
        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
