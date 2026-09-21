"""
RescueNet应急响应智能体 - CLI入口脚本

使用示例：
    python agent_cli.py --image-folder ./RescueNet/test/test-org-img
    python agent_cli.py --image-folder ./images --question "哪些区域最紧急？" --output my_report.html
"""
import argparse
import os
import sys
from typing import List


def check_prerequisites() -> bool:
    """
    检查Agent运行的前置条件
    
    Returns:
        bool: 是否满足所有条件
    """
    print("🔍 检查Agent运行前置条件...")
    print("=" * 60)
    
    # Check 1: 模型权重文件
    from configs.agent_config import ENSEMBLE_CONFIG

    required_weights = [
        os.path.join(ENSEMBLE_CONFIG['weight_dir'], f'{model_name}_final.pth')
        for model_name in ENSEMBLE_CONFIG['models']
    ]
    missing_weights = [f for f in required_weights if not os.path.exists(f)]

    if missing_weights:
        print("❌ 缺少模型权重文件:")
        for f in missing_weights:
            print(f"   - {f}")
        print("\n💡 请先运行以下命令训练模型:")
        print("   python train.py")
        return False

    print("✅ 模型权重文件齐全")
    
    # Check 2: LLM API Key
    from configs.agent_config import LLM_CONFIG, get_llm_api_key

    api_key = get_llm_api_key()

    if not api_key:
        api_key_env = LLM_CONFIG.get('api_key_env', 'OPENAI_API_KEY')
        if not api_key_env.isidentifier():
            api_key_env = 'OPENAI_API_KEY'
        print("⚠️  警告: LLM API Key未配置")
        print("   请在 config.yaml 中设置 llm.api_key，")
        print(f"   或设置 {api_key_env} 环境变量")
        return False
    
    print("✅ LLM API Key已配置")
    
    print("=" * 60)
    print("✅ 所有前置条件满足，可以启动Agent！\n")
    return True


def batch_process_images(image_folder: str) -> List[str]:
    """
    批量获取图像路径，支持目录或单张图像路径
    
    Args:
        image_folder: 图像文件夹路径或单张图像路径
    
    Returns:
        图像路径列表
    """
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    if os.path.isfile(image_folder):
        if image_folder.lower().endswith(supported_extensions):
            print("📁 处理单张图像")
            return [image_folder]
        raise ValueError(f"文件 '{image_folder}' 不是支持的图像格式")
    
    if not os.path.isdir(image_folder):
        raise ValueError(f"路径 '{image_folder}' 既不是文件也不是文件夹")
    
    image_paths = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(supported_extensions):
            image_paths.append(os.path.join(image_folder, filename))
    
    if not image_paths:
        raise ValueError(f"在 '{image_folder}' 中未找到支持的图像文件")
    
    print(f"📁 找到 {len(image_paths)} 张图像")
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description='🏗️ RescueNet应急响应智能体 (RERA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python agent_cli.py --image-folder ./RescueNet/test/test-org-img
  
  # 自定义问题
  python agent_cli.py --image-folder ./images --question "哪些区域建筑物损坏最严重？"
  
  # 指定输出文件名
  python agent_cli.py --image-folder ./images --output disaster_report.html
  
        """
    )
    
    parser.add_argument('--image-folder', type=str, required=True,
                        help='灾区图像文件夹路径或单张图像路径')
    parser.add_argument('--question', type=str, 
                        default='分析这批灾区的损坏情况并生成救援建议',
                        help='用户问题（默认：分析这批灾区的损坏情况并生成救援建议）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出报告文件名（默认：自动生成）')
    parser.add_argument('--skip-check', action='store_true',
                        help='跳过前置条件检查（不推荐）')
    
    args = parser.parse_args()
    
    # 前置条件检查
    if not args.skip_check:
        if not check_prerequisites():
            sys.exit(1)
    
    # 批量处理图像
    try:
        image_paths = batch_process_images(args.image_folder)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    
    # 初始化工具
    print("\n🔧 初始化Agent工具...")
    from modules.agent.tools import (
        EnsembleSegmentationTool,
        StatisticsAnalyzerTool,
        PriorityCalculatorTool,
        ReportGeneratorTool
    )
    tools = {
        'ensemble_segment': EnsembleSegmentationTool(),
        'analyze_statistics': StatisticsAnalyzerTool(),
        'calculate_priority': PriorityCalculatorTool(),
        'generate_report': ReportGeneratorTool(),
    }
    
    print(f"✅ 已加载 {len(tools)} 个工具\n")
    
    # 初始化ReAct控制器
    from modules.agent.react_controller import ReActController
    from configs.agent_config import get_llm_api_key

    controller = ReActController(
        tools=tools,
        api_key=get_llm_api_key()
    )
    
    # 执行Agent
    context = {
        'image_folder': args.image_folder,
        'image_count': len(image_paths),
        'image_list': image_paths,
        'output_filename': args.output,
    }
    
    print("="*60)
    print(f"📝 用户问题: {args.question}")
    print("="*60 + "\n")
    
    result = controller.run(args.question, context)
    
    # 输出结果
    print("\n" + "="*60)
    print("🤖 Agent推理过程摘要:")
    print("="*60)
    
    for step in result['reasoning_trace']:
        status_icon = {
            'completed': '✅',
            'failed': '❌',
            'pending': '⏳'
        }.get(step['status'], '❓')
        
        print(f"\n{status_icon} [Step {step['iteration']+1}]")
        print(f"   Thought: {step['thought'][:100]}...")
        print(f"   Action: {step['action']}")
    
    print("\n" + "="*60)
    print("📋 Final Answer:")
    print("="*60)
    print(result['final_answer'])
    
    # 如果有报告路径，显示
    if result['result_data'] and 'report_path' in result['result_data']:
        report_path = result['result_data']['report_path']
        print(f"\n✅ 报告已保存到: {report_path}")
        print(f"💡 提示: 在浏览器中打开查看完整报告")
    
    print("\n" + "="*60)
    print("🎉 Agent执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
