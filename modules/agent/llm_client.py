"""公共 LLM 客户端：统一文字/图片消息与两种 OpenAI 兼容接口。"""
from copy import deepcopy
from typing import Any, Dict, List

from openai import OpenAI

from configs.agent_config import LLM_CONFIG, get_llm_api_key


class LLMRequestError(RuntimeError):
    """请求失败或输出不完整；错误信息不携带请求图片和凭据。"""


class LLMClient:
    """使用统一消息格式；接口差异仅在发送前转换。"""

    def __init__(self, api_key: str = None, config: Dict = None, sdk_client: Any = None):
        self.config = dict(LLM_CONFIG if config is None else config)
        self.sdk = sdk_client if sdk_client is not None else OpenAI(
            api_key=api_key or get_llm_api_key(), base_url=self.config['base_url'],
            timeout=self.config.get('timeout', 90), max_retries=self.config.get('max_retries', 1),
        )

    def _wire_messages(self, messages: List[Dict], responses: bool) -> List[Dict]:
        """将内部图片内容块转换为接口字段，不把 data URL 当作普通文字。"""
        result = deepcopy(messages)
        for message in result:
            if not isinstance(message['content'], list):
                continue
            parts = []
            for part in message['content']:
                if part['type'] == 'text':
                    parts.append({'type': 'input_text' if responses else 'text', 'text': part['text']})
                elif part['type'] == 'image':
                    detail = part.get('detail', 'high')
                    parts.append(
                        {'type': 'input_image', 'image_url': part['data_url'], 'detail': detail}
                        if responses else
                        {'type': 'image_url', 'image_url': {'url': part['data_url'], 'detail': detail}}
                    )
                else:
                    raise ValueError('不支持的内部消息内容类型')
            message['content'] = parts
        return result

    def complete(
        self, messages: List[Dict], max_tokens: int, temperature: float = 0.7,
        timeout: float = None,
    ) -> str:
        """执行一次调用；拒绝空输出和截断内容，避免将半份分析视为完成。"""
        wire_api = self.config.get('wire_api', 'chat_completions')
        if wire_api not in ('responses', 'chat_completions'):
            raise ValueError('wire_api 必须是 responses 或 chat_completions')
        responses = wire_api == 'responses'
        request = {'model': self.config['model']}
        request['input' if responses else 'messages'] = self._wire_messages(messages, responses)
        request['max_output_tokens' if responses else 'max_tokens'] = max_tokens
        if timeout is not None:
            request['timeout'] = timeout
        if self.config.get('supports_temperature', not responses):
            request['temperature'] = temperature
        if responses:
            request['store'] = self.config.get('store', False)
            if self.config.get('reasoning_effort'):
                request['reasoning'] = {'effort': self.config['reasoning_effort']}
            if self.config.get('verbosity'):
                request['text'] = {'verbosity': self.config['verbosity']}
        try:
            if responses:
                response = self.sdk.responses.create(**request)
                if getattr(response, 'status', 'completed') != 'completed':
                    raise LLMRequestError('LLM 输出未完整完成，可能达到输出上限或请求被拒绝')
                content = response.output_text
            else:
                response = self.sdk.chat.completions.create(**request)
                choice = response.choices[0]
                if choice.finish_reason != 'stop':
                    raise LLMRequestError('LLM 输出被截断或未正常结束')
                content = choice.message.content
        except LLMRequestError:
            raise
        except Exception as exc:
            # SDK 异常可能包含请求正文；只暴露类型和 HTTP 状态，不打印原始异常内容。
            status = getattr(exc, 'status_code', None)
            raise LLMRequestError(f'LLM 请求失败（{type(exc).__name__}，HTTP {status}）') from None
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError('LLM 返回空内容')
        return content.strip()
