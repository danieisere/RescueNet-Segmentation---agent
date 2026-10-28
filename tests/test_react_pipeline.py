import os
import unittest

import numpy as np

from modules.agent.react_controller import ReActController


class ScriptedController(ReActController):
    def __init__(self, tools, outputs):
        self.tools = tools
        self.outputs = iter(outputs)
        self.max_iterations = 10
        self.temperature = 0.0
        self.max_tokens = 500

    def _call_llm(self, messages):
        return next(self.outputs)


class FakeSegmentationTool:
    name = 'ensemble_segment'
    description = 'segment'

    def __init__(self, result):
        self.result = result

    def execute(self, image_path):
        self.result['image_path'] = image_path
        return self.result


class FakeStatisticsTool:
    name = 'analyze_statistics'
    description = 'statistics'

    def __init__(self):
        self.received = None
        self.result = {
            'per_image_stats': [{'image_id': '10782.jpg'}],
            'global_stats': {'total_images': 1},
        }

    def execute(self, seg_results):
        self.received = seg_results
        if not isinstance(seg_results[0]['best_mask'], np.ndarray):
            raise TypeError('best_mask must remain an ndarray')
        return self.result


class FakePriorityTool:
    name = 'calculate_priority'
    description = 'priority'

    def __init__(self):
        self.received = None
        self.result = {
            'priority_zones': [{'image_id': '10782.jpg', 'priority_score': 10}],
            'statistics': {'total_zones': 1},
        }

    def execute(self, stats_result):
        self.received = stats_result
        return self.result


class FakeReportTool:
    name = 'generate_report'
    description = 'report'

    def __init__(self):
        self.received = None
        self.output_filename = None
        self.result = {
            'report_path': './output/reports/critical_areas_report.html',
            'format': 'html',
            'summary': 'done',
        }

    def execute(
        self,
        priority_result,
        output_filename=None,
        segmentation_result=None,
        statistics_result=None,
    ):
        self.received = priority_result
        self.output_filename = output_filename
        self.segmentation_result = segmentation_result
        self.statistics_result = statistics_result
        return self.result


class ReActPipelineTests(unittest.TestCase):
    def setUp(self):
        self.seg_result = {
            'best_model': 'deeplabv3',
            'best_mask': np.arange(16, dtype=np.int64).reshape(4, 4),
            'scores': {'unet': 0.7, 'pspnet': 0.75, 'deeplabv3': 0.8},
            'all_predictions': {},
            'image_path': '',
        }
        self.statistics = FakeStatisticsTool()
        self.priority = FakePriorityTool()
        self.report = FakeReportTool()
        self.tools = {
            'ensemble_segment': FakeSegmentationTool(self.seg_result),
            'analyze_statistics': self.statistics,
            'calculate_priority': self.priority,
            'generate_report': self.report,
        }

    def test_pipeline_ignores_llm_copies_and_passes_real_objects(self):
        outputs = [
            'Thought: segment\nAction: ensemble_segment\nAction Input: {"image_path": "10782.jpg"}',
            (
                'Thought: stats\nAction: analyze_statistics\n'
                'Action Input: {"seg_results": [{"mask": "[[5 5 ...]]"}]}'
            ),
            (
                'Thought: priority\nAction: calculate_priority\n'
                'Action Input: {"stats_result": "truncated text"}'
            ),
            (
                'Thought: report\nAction: generate_report\n'
                'Action Input: {"priority_result": "truncated text"}'
            ),
            'Thought: done\nFinal Answer: verified answer',
        ]
        controller = ScriptedController(self.tools, outputs)

        result = controller.run(
            'question',
            {
                'image_path': '10782.jpg',
                'output_filename': 'critical_areas_report.html',
            },
        )

        self.assertIs(self.statistics.received[0], self.seg_result)
        self.assertIs(self.priority.received, self.statistics.result)
        self.assertIs(self.report.received, self.priority.result)
        self.assertIs(self.report.segmentation_result, self.seg_result)
        self.assertIs(self.report.statistics_result, self.statistics.result)
        self.assertEqual(self.report.output_filename, 'critical_areas_report')
        self.assertEqual(result['result_data']['report_path'], self.report.result['report_path'])
        self.assertEqual(result['final_answer'], 'verified answer')

    def test_final_answer_is_rejected_until_pipeline_completes(self):
        outputs = [
            'Thought: stop early\nFinal Answer: unsupported answer',
            'Thought: segment\nAction: ensemble_segment\nAction Input: {"image_path": "10782.jpg"}',
            'Thought: stats\nAction: analyze_statistics\nAction Input: {}',
            'Thought: priority\nAction: calculate_priority\nAction Input: {}',
            'Thought: report\nAction: generate_report\nAction Input: {}',
            'Thought: done\nFinal Answer: supported answer',
        ]
        controller = ScriptedController(self.tools, outputs)

        result = controller.run(
            'question',
            {'image_path': '10782.jpg', 'output_filename': 'report.html'},
        )

        self.assertEqual(result['reasoning_trace'][0]['status'], 'failed')
        self.assertIn('generate_report', result['reasoning_trace'][0]['error'])
        self.assertEqual(result['final_answer'], 'supported answer')

    def test_segmentation_observation_never_serializes_mask_contents(self):
        controller = ScriptedController(self.tools, [])

        summary = controller._summarize_observation('ensemble_segment', self.seg_result)

        self.assertIn('"mask_shape": [4, 4]', summary)
        self.assertNotIn('best_mask', summary)
        self.assertNotIn('all_predictions', summary)
        self.assertNotIn('...', summary)

    def test_output_filename_discards_path_and_known_extension(self):
        controller = ScriptedController(self.tools, [])

        normalized = controller._normalize_output_filename(
            os.path.join('unsafe', 'critical_areas_report.html')
        )

        self.assertEqual(normalized, 'critical_areas_report')

    def test_action_input_ignores_text_after_json(self):
        controller = ScriptedController(self.tools, [])
        llm_output = (
            'Thought: segment\n'
            'Action: ensemble_segment\n'
            'Action Input: {"image_path": "10782.jpg"}\n'
            '无法执行后续工具，这段文字不属于 JSON。'
        )

        _, action, action_input = controller._parse_llm_response(llm_output)

        self.assertEqual(action, 'ensemble_segment')
        self.assertEqual(action_input, {'image_path': '10782.jpg'})

    def test_plain_text_response_is_preserved_as_final_answer(self):
        controller = ScriptedController(self.tools, [])
        plain_text = '最紧急区域为 10782.jpg，应优先派遣搜救和道路清障力量。'

        _, action, action_input = controller._parse_llm_response(plain_text)

        self.assertEqual(action, 'Final Answer')
        self.assertEqual(action_input, plain_text)


if __name__ == '__main__':
    unittest.main()
