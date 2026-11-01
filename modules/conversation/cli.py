"""会话终端入口；输入循环与图执行分离，便于未来接入网页。"""
from pathlib import Path

from configs.agent_config import CONVERSATION_CONFIG
from modules.agent.execution_events import safe_text
from modules.agent.runtime_services import RuntimeServices
from .controller import ConversationController
from .schema import DialoguePlan
from .store import SessionStore


HELP = '''会话命令：
  /exit 或 退出       保存并退出，下次可按 session_id 恢复
  /new              创建新会话，保留当前会话全部记录
  /image 图片路径    切换当前图片（支持含空格路径）
  /sessions         查看历史会话
  /help             查看命令
直接输入文字即为当前会话的下一轮。'''


def run_chat(args):
    root = Path(args.message_root or CONVERSATION_CONFIG['message_root']).expanduser().resolve()
    if args.list_sessions:
        for meta in SessionStore.list_sessions(root):
            print(f"{meta['session_id']} | {meta['last_turn']} 轮 | {meta.get('current_image_id') or '未绑定图片'}")
        return 0
    if args.engine and args.engine != 'langgraph':
        raise ValueError('当前多轮入口使用 LangGraph；运行 ReAct 请添加 --single-turn --engine react。')
    if args.output:
        raise ValueError('会话模式请在消息中要求报告及格式；使用 --output 请添加 --single-turn。')
    image_path = args.image_folder
    if image_path and not Path(image_path).is_file():
        raise ValueError('会话模式每次接收一张图片；请提供图片文件，批量目录请添加 --single-turn。')
    store = SessionStore(root, args.session_id) if args.session_id else SessionStore.create(root)
    # 模型实例在进程中跨轮和新会话复用，实际需要图像分析时才加载。
    services = RuntimeServices()
    controller = ConversationController(store, services=services)

    def announce():
        print(f'会话：{store.session_id}\n记忆与成果：{store.session_dir}', flush=True)
    announce()
    print(HELP, flush=True)
    pending = args.question or ('分析当前灾区图像的损坏情况并生成救援建议' if image_path else None)
    exit_code = 0
    while True:
        try:
            text = pending if pending is not None else input('\n你 > ')
            pending = None
            text = text.strip()
            if not text:
                continue
            if text.lower() in {'/exit', 'exit', 'quit', '退出', '退出会话'}:
                print(f'已退出交互，保留会话 {store.session_id}。恢复：python agent_cli.py --session-id {store.session_id}', flush=True)
                return exit_code
            if text == '/help':
                print(HELP)
                continue
            if text == '/sessions':
                for meta in SessionStore.list_sessions(root):
                    print(f"{meta['session_id']} | {meta['last_turn']} 轮")
                continue
            override = None
            if text == '/new':
                override = DialoguePlan(intent='reset_session')
                text = '重新开始一个新会话。'
                image_path = None
            elif text.startswith('/image '):
                image_path = text[len('/image '):].strip().strip('"').strip("'")
                if not image_path:
                    print('请提供图片路径。')
                    continue
                override = DialoguePlan(intent='switch_image')
                text = '切换到本次提供的图片。'
            elif text.startswith('/'):
                print('未知命令，请输入 /help。')
                continue
            for progress in controller.stream(text, image_path=image_path, plan_override=override):
                print(f"[会话 {progress['turn_id']}] {progress['node']} | {progress['event']} | {progress['status']} | {progress['summary']}", flush=True)
            image_path = None
            result = controller.last_result
            print('\n助手 > ' + result['answer'], flush=True)
            print(f"本轮状态：{result['status']} | 归档：{result['turn_dir']}", flush=True)
            exit_code = 1 if result['status'] == 'failed' else 0
            if result.get('next_session_id'):
                store = SessionStore(root, result['next_session_id'])
                controller = ConversationController(store, services=services)
                announce()
        except EOFError:
            print(f'\n输入已结束，会话 {store.session_id} 已保留。', flush=True)
            return exit_code
        except KeyboardInterrupt:
            print(f'\n已暂停。会话 {store.session_id} 可恢复；中断轮次会在下次加载时标记。', flush=True)
            return 130
        except (ValueError, OSError, RuntimeError) as exc:
            print('会话操作未完成：' + safe_text(exc), flush=True)
            exit_code = 1
            image_path = None
