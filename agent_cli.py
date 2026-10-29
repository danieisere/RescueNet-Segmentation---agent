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


def build_report_filename(image_path: str, requested_output: str, image_count: int) -> str:
    """为每张图生成独立报告文件名，批量运行时避免相互覆盖。"""
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    if not requested_output:
        return f'{image_stem}_report.html'

    requested_name = os.path.basename(requested_output)
    requested_stem, requested_ext = os.path.splitext(requested_name)
    extension = requested_ext or '.html'
    if image_count == 1:
        return requested_stem + extension
    return f'{requested_stem}_{image_stem}{extension}'


def print_agent_result(result: dict, image_path: str) -> None:
    """输出单张图像的 Agent 推理摘要与报告路径。"""
    print("\n" + "=" * 60)
    print(f"🤖 单图推理摘要: {os.path.basename(image_path)}")
    print("=" * 60)

    for step in result['reasoning_trace']:
        status_icon = {
            'completed': '✅',
            'failed': '❌',
            'pending': '⏳'
        }.get(step['status'], '❓')
        print(f"\n{status_icon} [Step {step['iteration'] + 1}]")
        print(f"   Thought: {step['thought'][:100]}...")
        print(f"   Action: {step['action']}")

    print("\n" + "=" * 60)
    print("📋 当前图像应急救援结论:")
    print("=" * 60)
    print(result['final_answer'])

    if result['result_data'] and 'report_path' in result['result_data']:
        print(f"\n✅ 单图报告已保存到: {result['result_data']['report_path']}")


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
                        default='分析当前灾区图像的损坏情况并生成救援建议',
                        help='用户问题（默认：分析当前灾区图像并生成救援建议）')
    parser.add_argument('--output', type=str, default=None,
                        help='单图报告文件名；批量模式会自动追加图片名')
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
        ReportGeneratorTool,
        MultimodalAnalysisTool,
    )
    tools = {
        'ensemble_segment': EnsembleSegmentationTool(),
        'analyze_statistics': StatisticsAnalyzerTool(),
        'calculate_priority': PriorityCalculatorTool(),
        'multimodal_analyze': MultimodalAnalysisTool(),
        'generate_report': ReportGeneratorTool(),
    }
    
    print(f"✅ 已加载 {len(tools)} 个工具\n")
    
    # 初始化ReAct控制器
    from modules.agent.react_controller import ReActController
    from configs.agent_config import get_llm_api_key

    results = []
    for index, image_path in enumerate(image_paths, 1):
        controller = ReActController(tools=tools, api_key=get_llm_api_key())
        output_filename = build_report_filename(image_path, args.output, len(image_paths))
        context = {
            'image_path': image_path,
            'image_id': os.path.basename(image_path),
            'output_filename': output_filename,
            'analysis_scope': 'single_image',
        }

        print("\n" + "=" * 60)
        print(f"🖼️  单图分析 [{index}/{len(image_paths)}]: {os.path.basename(image_path)}")
        print(f"📝 用户问题: {args.question}")
        print("=" * 60 + "\n")

        result = controller.run(args.question, context)
        results.append(result)
        print_agent_result(result, image_path)

    successful_reports = sum(
        1 for result in results
        if result.get('result_data') and result['result_data'].get('report_path')
    )
    
    print("\n" + "="*60)
    print(f"🎉 批量执行完成：{len(image_paths)} 张图像，生成 {successful_reports} 份独立报告！")
    print("="*60)

    # API/分析失败必须反映在进程退出码中，避免自动化运行把失败误判为成功。
    if any(result.get('status') == 'failed' for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
