"""
Ensemble分割工具

策略：三个模型分别预测，选择置信度评分最高的结果作为最终输出
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any

from modules.agent.base_tool import BaseTool
from configs.agent_config import ENSEMBLE_CONFIG, AGENT_CONFIG


class EnsembleSegmentationTool(BaseTool):
    """
    多模型融合分割工具
    
    设计动机：
    - RescueNet数据集较小，单个模型泛化能力有限
    - 通过Ensemble融合三个模型，利用多样性提升鲁棒性
    - 采用"评分最高者胜出"策略，自动选择最可靠的预测结果
    """
    
    def __init__(self, use_mock: bool = False):
        """
        初始化工具
        
        Args:
            use_mock: 是否使用Mock模式（演示用，无需真实模型）
        """
        self.use_mock = use_mock
        self.models = {}
        
        if not use_mock:
            self._verify_weights()
            self._load_models()
    
    @property
    def name(self) -> str:
        return "ensemble_segment"
    
    @property
    def description(self) -> str:
        return "并行调用UNet、PSPNet、DeepLabV3三个模型，计算各模型的置信度评分，选择评分最高的模型结果作为最终输出。适用于需要高可靠性分割的场景。"
    
    def execute(self, image_path: str) -> Dict[str, Any]:
        """
        执行Ensemble分割
        
        Args:
            image_path: 输入图像路径
        
        Returns:
            {
                'best_model': str,           # 最佳模型名称 ('unet'/'pspnet'/'deeplabv3')
                'best_mask': np.ndarray,     # 最佳分割掩码 (H, W)
                'scores': Dict[str, float],  # 各模型置信度评分
                'all_predictions': Dict,     # 所有模型的原始预测（用于分析）
                'image_path': str            # 输入图像路径
            }
        """
        # Mock模式：返回假数据（用于演示Agent流程）
        if self.use_mock:
            return self._mock_execute(image_path)
        
        # 正常模式：调用真实模型
        return self._real_execute(image_path)
    
    def _mock_execute(self, image_path: str) -> Dict[str, Any]:
        """模拟分割结果（无需真实模型，用于演示）"""
        # 生成随机分割掩码（11个类别）
        fake_mask = np.random.randint(0, 11, size=(768, 768))
        
        # 模拟评分
        scores = {
            'unet': round(np.random.uniform(0.75, 0.90), 3),
            'pspnet': round(np.random.uniform(0.70, 0.88), 3),
            'deeplabv3': round(np.random.uniform(0.72, 0.92), 3)
        }
        
        best_model = max(scores, key=scores.get)
        
        print(f"[MOCK] Ensemble分割完成 | 最佳模型: {best_model} | 评分: {scores}")
        
        return {
            'best_model': best_model,
            'best_mask': fake_mask,
            'scores': scores,
            'all_predictions': {},
            'image_path': image_path
        }
    
    def _real_execute(self, image_path: str) -> Dict[str, Any]:
        """真实分割逻辑"""
        from models import get_model
        from modules.datasets import SegmentationDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2
        
        # Step 1: 加载并预处理图像
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        transform = A.Compose([
            A.Resize(768, 768),
            A.Normalize(),
            ToTensorV2()
        ])
        
        augmented = transform(image=raw_image)
        img_tensor = augmented['image'].unsqueeze(0)  # [1, 3, 768, 768]
        
        # Step 2: 并行推理
        predictions = {}
        scores = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            device = model_info['device']
            
            # 移动到设备
            img_on_device = img_tensor.to(device)
            
            # 推理
            with torch.no_grad():
                output = model(img_on_device)  # [1, 11, 768, 768]
            
            predictions[model_name] = output
            
            # Step 3: 计算置信度评分
            score = self._calculate_confidence(output)
            scores[model_name] = score
        
        # Step 4: 选择评分最高的模型
        best_model = max(scores, key=scores.get)
        best_mask = torch.argmax(predictions[best_model], dim=1).squeeze(0).cpu().numpy()
        
        print(f"[INFO] Ensemble分割完成 | 最佳模型: {best_model} | 评分: {scores}")
        
        return {
            'best_model': best_model,
            'best_mask': best_mask,
            'scores': scores,
            'all_predictions': predictions,
            'image_path': image_path
        }
    
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """
        计算模型预测的置信度评分
        
        方法：
        1. 最大概率均值：预测类别的最大概率的平均值（越高越好）
        2. 熵惩罚：预测分布越集中，熵越低，评分越高
        
        Score = mean(max_prob) - λ * normalized_entropy
        
        Args:
            output: 模型输出 logits [1, 11, H, W]
        
        Returns:
            float: 置信度评分 [0, 1]
        """
        probs = F.softmax(output, dim=1)  # [1, 11, H, W]
        max_probs, _ = torch.max(probs, dim=1)  # [1, H, W]
        
        # 方法1: 平均最大概率
        avg_max_prob = torch.mean(max_probs).item()
        
        # 方法2: 熵惩罚
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        avg_entropy = torch.mean(entropy).item()
        max_entropy = torch.log(torch.tensor(11.0)).item()  # 11个类别的最大熵
        normalized_entropy = avg_entropy / max_entropy
        
        # 综合评分
        lambda_weight = ENSEMBLE_CONFIG['confidence_lambda']
        score = avg_max_prob - lambda_weight * normalized_entropy
        
        return round(score, 4)
    
    def _verify_weights(self):
        """验证所有必需的权重文件是否存在"""
        required_files = [
            'unet_final.pth',
            'pspnet_final.pth',
            'deeplabv3_final.pth'
        ]
        
        weight_dir = ENSEMBLE_CONFIG['weight_dir']
        missing_files = []
        
        for filename in required_files:
            filepath = os.path.join(weight_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filepath)
        
        if missing_files:
            raise FileNotFoundError(
                f"❌ 缺少模型权重文件，请先运行 'python train.py' 训练模型！\n"
                f"缺失的文件:\n" + "\n".join(missing_files) + "\n\n"
                f"💡 提示: 确保 '{weight_dir}/' 目录下包含以下文件:\n"
                f"  - unet_final.pth\n"
                f"  - pspnet_final.pth\n"
                f"  - deeplabv3_final.pth"
            )
    
    def _load_models(self):
        """加载三个模型到内存"""
        from models import get_model
        
        device_str = AGENT_CONFIG['device']
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
        
        print(f"[INFO] 使用设备: {device}")
        
        model_names = ENSEMBLE_CONFIG['models']
        weight_dir = ENSEMBLE_CONFIG['weight_dir']
        
        for model_name in model_names:
            try:
                # 创建模型实例
                model = get_model(model_name)
                
                # 加载权重
                weight_path = os.path.join(weight_dir, f'{model_name}_final.pth')
                model.load_state_dict(torch.load(weight_path, map_location=device))
                
                # 设置为评估模式
                model.eval()
                model.to(device)
                
                self.models[model_name] = {
                    'model': model,
                    'device': device
                }
                
                print(f"✅ {model_name.upper()} 模型加载成功")
                
            except Exception as e:
                raise RuntimeError(
                    f"❌ 加载 {model_name} 模型失败: {str(e)}\n"
                    f"请确认权重文件 '{weight_path}' 是否存在且未损坏"
                )
