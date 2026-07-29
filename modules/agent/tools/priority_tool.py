"""
优先级计算工具

职责：基于分割统计结果计算各区域的救援优先级评分
"""
import os
from typing import Dict, Any, List

from modules.agent.base_tool import BaseTool
from configs.agent_config import PRIORITY_WEIGHTS


class PriorityCalculatorTool(BaseTool):
    """
    救援优先级计算工具
    
    设计动机：
    - 将统计数据转换为可执行的救援优先级排序
    - 综合考虑建筑物损坏程度和道路通行性
    - 为决策者提供清晰的行动指南
    
    优先级公式：
    Priority_Score = w1 × N_destroyed + w2 × N_major + w3 × N_minor + w4 × Road_Blocked_Ratio
    """
    
    @property
    def name(self) -> str:
        return "calculate_priority"
    
    @property
    def description(self) -> str:
        return "基于分割统计结果计算各区域的救援优先级评分。综合考虑完全毁坏、严重损坏、轻微损坏的建筑数量以及道路阻塞率，返回按优先级排序的区域列表。"
    
    def execute(self, stats_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算救援优先级
        
        Args:
            stats_result: StatisticsAnalyzerTool的输出，包含：
                {
                    'per_image_stats': List[Dict],
                    'global_stats': Dict
                }
        
        Returns:
            {
                'priority_zones': List[Dict],  # 按优先级降序排序的区域
                'statistics': Dict             # 全局统计摘要
            }
        """
        per_image_stats = stats_result['per_image_stats']
        
        priority_zones = []
        
        for image_stat in per_image_stats:
            class_counts = image_stat['class_counts']
            total_pixels = image_stat['total_pixels']
            
            # 提取关键指标
            destroyed_count = class_counts.get(5, 0)  # Class 5: 完全毁坏
            major_damage = class_counts.get(4, 0)     # Class 4: 严重损坏
            minor_damage = class_counts.get(3, 0)     # Class 3: 轻微损坏
            road_blocked = class_counts.get(8, 0)     # Class 8: 道路阻塞
            
            # 计算道路阻塞率
            road_blocked_ratio = road_blocked / total_pixels if total_pixels > 0 else 0
            
            # 计算优先级评分
            score = (
                PRIORITY_WEIGHTS['destroyed'] * destroyed_count +
                PRIORITY_WEIGHTS['major'] * major_damage +
                PRIORITY_WEIGHTS['minor'] * minor_damage +
                PRIORITY_WEIGHTS['road_blocked'] * road_blocked_ratio * total_pixels
            )
            
            # 分级
            level = self._classify_level(score)
            
            priority_zones.append({
                'image_id': image_stat['image_id'],
                'image_path': image_stat['image_path'],
                'priority_score': round(score, 2),
                'level': level,
                'destroyed_count': int(destroyed_count),
                'major_damage_count': int(major_damage),
                'minor_damage_count': int(minor_damage),
                'road_blocked_ratio': round(road_blocked_ratio, 4),
                'total_building_pixels': int(
                    class_counts.get(2, 0) + class_counts.get(3, 0) + 
                    class_counts.get(4, 0) + class_counts.get(5, 0)
                )
            })
        
        # 按优先级降序排序
        priority_zones.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 生成统计摘要
        statistics = self._generate_summary(priority_zones)
        
        print(f"[INFO] 优先级计算完成 | CRITICAL: {statistics['critical_count']} | HIGH: {statistics['high_count']}")
        
        return {
            'priority_zones': priority_zones,
            'statistics': statistics
        }
    
    def _classify_level(self, score: float) -> str:
        """
        根据评分划分优先级等级
        
        Args:
            score: 优先级评分
        
        Returns:
            优先级等级字符串
        """
        if score > 10000:
            return 'CRITICAL'  # 🔴 红色警报：立即救援
        elif score > 5000:
            return 'HIGH'      # 🟠 橙色警报：优先处理
        elif score > 2000:
            return 'MEDIUM'    # 🟡 黄色警报：常规处理
        else:
            return 'LOW'       # 🟢 绿色正常：持续关注
    
    def _generate_summary(self, priority_zones: List[Dict]) -> Dict[str, Any]:
        """
        生成优先级统计摘要
        
        Args:
            priority_zones: 优先级区域列表
        
        Returns:
            统计摘要字典
        """
        level_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }
        
        for zone in priority_zones:
            level_counts[zone['level']] += 1
        
        # Top 3最紧急区域
        top_3_critical = priority_zones[:3] if len(priority_zones) >= 3 else priority_zones
        
        return {
            'total_zones': len(priority_zones),
            'critical_count': level_counts['CRITICAL'],
            'high_count': level_counts['HIGH'],
            'medium_count': level_counts['MEDIUM'],
            'low_count': level_counts['LOW'],
            'top_3_critical': [
                {
                    'image_id': z['image_id'],
                    'score': z['priority_score'],
                    'destroyed': z['destroyed_count']
                }
                for z in top_3_critical
            ]
        }
