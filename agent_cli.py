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

    for step in result.get('reasoning_trace', []):
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

    if result.get('result_data') and 'report_path' in result['result_data']:
        print(f"\n✅ 单图报告已保存到: {result['result_data']['report_path']}")
    print(f"状态：{result.get('status', 'completed')} | 产物目录：{result.get('item_dir', '')}")


def main():
    # Windows 的 NumPy/MKL 与 PyTorch 可能各带一份 Intel OpenMP，首次推理会直接终止进程。
    # 在加载数值库前使用 MKL 的串行线程层，避免重复初始化；保留用户显式环境设置。
    if sys.platform == 'win32':
        os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
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
    parser.add_argument('--engine', choices=['langgraph', 'react'], default=None,
                        help='执行引擎；默认使用配置中的 langgraph')
    
    args = parser.parse_args()
    
    from configs.agent_config import AGENT_CONFIG, ARTIFACTS_CONFIG, LLM_CONFIG, REPORT_CONFIG
    engine = args.engine or AGENT_CONFIG.get('engine', 'langgraph')
    if engine not in {'langgraph', 'react'}:
        parser.error('agent.engine 只能为 langgraph 或 react')
    # 仅做提示检查；模型权重由真正需要分割的节点校验，澄清路径不受权重影响。
    if not args.skip_check:
        from configs.agent_config import get_llm_api_key
        print('LLM 凭据：' + ('已配置' if get_llm_api_key() else '未配置，将由实际调用返回失败'))
    
    # 批量处理图像
    try:
        image_paths = batch_process_images(args.image_folder)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    
    from modules.agent.artifact_store import ArtifactStore
    from modules.agent.runtime_services import RuntimeServices
    from modules.agent.execution_events import event, safe_text
    store = ArtifactStore(ARTIFACTS_CONFIG['output_dir'], engine, args.question, LLM_CONFIG)
    services = RuntimeServices()
    print(f'执行引擎：{engine} | 本次运行目录：{store.run_dir}', flush=True)
    results = []
    for index, image_path in enumerate(image_paths, 1):
        item = store.create_item(image_path, args.output)
        output_filename = build_report_filename(image_path, args.output, len(image_paths))
        # 内部建议文件名不会成为报告意图；只有原始 --output 属于用户显式要求。
        context = {
            'image_path': image_path,
            'image_id': os.path.basename(image_path),
            'output_filename': os.path.splitext(output_filename)[0],
            'explicit_output': args.output,
            'analysis_scope': 'single_image',
        }

        print("\n" + "=" * 60)
        print(f"🖼️  单图分析 [{index}/{len(image_paths)}]: {os.path.basename(image_path)}")
        print(f"📝 用户问题: {args.question}")
        print("=" * 60 + "\n")

        try:
            if engine == 'langgraph':
                from modules.agent.graph_controller import LangGraphController
                controller = LangGraphController(services, item)
                # 直接消费 LangGraph 原生流，开始事件在长耗时节点执行前输出。
                for progress in controller.stream(args.question, context):
                    print(f"[LangGraph #{progress['sequence']:03d}] {progress['node']} | {progress['event']} | {progress['status']} | {progress['summary']}", flush=True)
                result = controller.last_result
            else:
                from modules.agent.react_controller import ReActController
                suffix = os.path.splitext(args.output or '')[1].lower()
                report_format = 'markdown' if suffix in {'.md', '.markdown'} else 'html' if suffix in {'.html', '.htm'} else REPORT_CONFIG.get('format', 'html')
                context.update(report_output_dir=str(item.item_dir / 'report'), report_format=report_format)

                def record_react(node, kind, status, summary):
                    item.append_event(event(store.run_id, item.item_id, node, kind, status, summary))

                controller = ReActController(services.react_tools())
                raw = controller.run(args.question, context, artifact_item=item, on_event=record_react)
                # 旧控制器内部数组保持在内存，统一结果文件只保留公开 trace 与文件引用。
                result = {key: raw[key] for key in ('status', 'final_answer', 'reasoning_trace', 'result_data', 'failure_stage', 'error') if key in raw}
                result.update(status=raw.get('status', 'completed'), engine=engine, run_id=store.run_id,
                              item_id=item.item_id, item_dir=str(item.item_dir), artifacts=list(item.manifest['artifacts']))
                if result['status'] == 'completed' and not item.verify():
                    result.update(status='failed', failure_stage='artifact_verification', error='产物哈希验证失败')
                item.finish(result)
        except Exception as exc:
            result = {'status': 'failed', 'engine': engine, 'run_id': store.run_id, 'item_id': item.item_id,
                      'failure_stage': 'cli_runtime', 'error': safe_text(exc), 'final_answer': '任务失败：' + safe_text(exc),
                      'item_dir': str(item.item_dir), 'artifacts': list(item.manifest['artifacts']), 'result_data': None}
            try:
                item.finish(result)
            except OSError:
                pass
        results.append(result)
        print_agent_result(result, image_path)

    successful_reports = sum(
        1 for result in results
        if result.get('result_data') and result['result_data'].get('report_path')
    )
    
    try:
        store.finish(results)
    except OSError as exc:
        print('最终运行清单写入失败：' + safe_text(exc))
        return 1
    counts = store.manifest['counts']
    print(f"\n运行结束：完成 {counts['completed']}，失败 {counts['failed']}，需澄清 {counts['needs_clarification']}，不支持 {counts['unsupported']}；报告 {successful_reports} 份。")
    print(f'完整结果：{store.run_dir}')
    # 停止执行与成功完成是不同状态，失败/澄清均不能返回成功码。
    return 1 if counts['failed'] else 2 if counts['needs_clarification'] or counts['unsupported'] else 0


if __name__ == "__main__":
    sys.exit(main())
