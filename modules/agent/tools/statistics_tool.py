"""
统计分析工具

职责：统计分割结果中各类别的数量、面积占比等指标
"""
import numpy as np
from typing import Dict, Any, List

from modules.agent.base_tool import BaseTool
from configs.dataset_config import CLASS_NAMES


class StatisticsAnalyzerTool(BaseTool):
    """
    统计分析工具
    
    设计动机：
    - 将像素级分割结果转换为可读的统计数据
    - 为优先级计算和报告生成提供结构化数据
    """
    
    @property
    def name(self) -> str:
        return "analyze_statistics"
    
    @property
    def description(self) -> str:
        return "统计分割结果中各类别（背景、水、建筑物损坏等级、道路、车辆等）的像素数量和面积占比，返回结构化的统计字典。"
    
    def execute(self, seg_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析分割结果的统计信息
        
        Args:
            seg_results: Ensemble分割结果列表，每个元素包含：
                {
                    'best_mask': np.ndarray,  # 分割掩码 (H, W)
                    'image_path': str,        # 图像路径
                    'best_model': str         # 使用的模型
                }
            也支持单条结果字典。
        
        Returns:
            {
                'per_image_stats': List[Dict],  # 每张图像的统计
                'global_stats': Dict            # 全局汇总统计
            }
        """
        seg_results = self._normalize_seg_results(seg_results)
        per_image_stats = []
        global_class_counts = np.zeros(11)  # 11个类别
        
        for result in seg_results:
            mask = result['best_mask']
            image_path = result['image_path']
            
            # 统计各类别像素数量
            class_counts = self._count_classes(mask)
            total_pixels = mask.size
            
            # 计算面积占比
            class_ratios = {
                class_id: count / total_pixels
                for class_id, count in class_counts.items()
            }
            
            # 构建单张图像的统计
            image_stat = {
                'image_path': image_path,
                'image_id': self._extract_image_id(image_path),
                'total_pixels': total_pixels,
                'class_counts': class_counts,
                'class_ratios': class_ratios,
                'class_names': {
                    CLASS_NAMES[k]: v for k, v in class_counts.items() if v > 0
                }
            }
            
            per_image_stats.append(image_stat)
            
            # 累加到全局统计
            for class_id, count in class_counts.items():
                global_class_counts[class_id] += count
        
        # 计算全局统计
        total_global_pixels = np.sum(global_class_counts)
        global_class_ratios = {
            i: count / total_global_pixels if total_global_pixels > 0 else 0
            for i, count in enumerate(global_class_counts)
        }
        
        global_stats = {
            'total_images': len(seg_results),
            'total_pixels': int(total_global_pixels),
            'global_class_counts': {i: int(count) for i, count in enumerate(global_class_counts)},
            'global_class_ratios': {i: round(ratio, 4) for i, ratio in global_class_ratios.items()},
            'building_damage_summary': self._summarize_building_damage(global_class_counts)
        }
        
        print(f"[INFO] 统计分析完成 | 处理{len(seg_results)}张图像")
        
        return {
            'per_image_stats': per_image_stats,
            'global_stats': global_stats
        }
    
    def _count_classes(self, mask: np.ndarray) -> Dict[int, int]:
        """
        统计掩码中各类别的像素数量
        
        Args:
            mask: 分割掩码 (H, W)
        
        Returns:
            Dict[class_id, count]
        """
        unique_classes, counts = np.unique(mask, return_counts=True)
        return {int(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)}

    def _normalize_seg_results(self, seg_results: Any) -> List[Dict[str, Any]]:
        """
        规范化输入，使其支持单条结果字典或结果列表。
        """
        if isinstance(seg_results, dict):
            return [seg_results]
        if isinstance(seg_results, list):
            return seg_results
        raise ValueError("analyze_statistics 期望接收单条结果字典或结果列表")
    
    def _extract_image_id(self, image_path: str) -> str:
        """从图像路径提取ID"""
        import os
        return os.path.basename(image_path)
    
    def _summarize_building_damage(self, global_class_counts: np.ndarray) -> Dict[str, int]:
        """
        总结建筑物损坏情况
        
        Returns:
            {
                'no_damage': int,      # Class 2
                'minor_damage': int,   # Class 3
                'major_damage': int,   # Class 4
                'destroyed': int       # Class 5
            }
        """
        return {
            'no_damage': int(global_class_counts[2]),
            'minor_damage': int(global_class_counts[3]),
            'major_damage': int(global_class_counts[4]),
            'destroyed': int(global_class_counts[5])
        }
