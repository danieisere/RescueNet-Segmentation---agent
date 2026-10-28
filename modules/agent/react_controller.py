"""
ReAct控制器 - LLM驱动的智能体核心

实现完整的ReAct (Reasoning + Acting) 范式：
- Thought: 推理下一步行动
- Action: 调用工具
- Observation: 处理工具返回结果
"""
import ast
import inspect
import json
import os
import re
from typing import Dict, Any, List

from openai import OpenAI
from modules.agent.base_tool import BaseTool
from configs.agent_config import REACT_CONFIG, LLM_CONFIG, get_llm_api_key


class ReActController:
    """
    基于DeepSeek LLM的ReAct控制器
    
    工作流程：
    1. 接收用户查询和上下文
    2. 构造System Prompt（包含可用工具描述）
    3. 进入ReAct循环（最多max_iterations步）
    4. 解析LLM输出的Thought-Action-Observation
    5. 执行对应Tool
    6. 将结果反馈给LLM
    7. 直到LLM输出Final Answer
    """
    
    def __init__(self, tools: Dict[str, BaseTool], api_key: str = None):
        """
        初始化ReAct控制器
        
        Args:
            tools: 可用工具字典 {tool_name: tool_instance}
            api_key: LLM API Key，默认从配置或环境变量读取
        """
        self.tools = tools
        self.api_key = api_key or get_llm_api_key()
        
        # 初始化OpenAI客户端（DeepSeek兼容OpenAI SDK）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=LLM_CONFIG['base_url']
        )
        
        self.max_iterations = REACT_CONFIG['max_iterations']
        self.temperature = REACT_CONFIG['temperature']
        self.max_tokens = REACT_CONFIG['max_tokens']

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the wire API selected by the LLM configuration."""
        wire_api = LLM_CONFIG.get('wire_api', 'chat_completions')

        if wire_api == 'responses':
            request = {
                'model': LLM_CONFIG['model'],
                'input': messages,
                'max_output_tokens': self.max_tokens,
                'store': LLM_CONFIG.get('store', False),
            }
            reasoning_effort = LLM_CONFIG.get('reasoning_effort')
            if reasoning_effort:
                request['reasoning'] = {'effort': reasoning_effort}
            verbosity = LLM_CONFIG.get('verbosity')
            if verbosity:
                request['text'] = {'verbosity': verbosity}
            if LLM_CONFIG.get('supports_temperature', False):
                request['temperature'] = self.temperature

            response = self.client.responses.create(**request)
            return response.output_text

        request = {
            'model': LLM_CONFIG['model'],
            'messages': messages,
            'max_tokens': self.max_tokens,
        }
        if LLM_CONFIG.get('supports_temperature', True):
            request['temperature'] = self.temperature

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content
    
    def run(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行ReAct循环
        
        Args:
            user_query: 用户问题，如"分析这批灾区图像并生成救援建议"
            context: 上下文信息，如{'image_folder': '/path/to/images'}
        
        Returns:
            {
                'final_answer': str,              # LLM生成的最终回答
                'reasoning_trace': List[Dict],    # 完整的Thought-Action-Observation历史
                'result_data': Dict               # 最终的结构化数据（用于报告生成等）
            }
        """
        # 构造System Prompt
        system_prompt = self._build_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {json.dumps(context, ensure_ascii=False)}\n\nQuestion: {user_query}"}
        ]
        
        reasoning_trace = []
        last_result_data = None
        pipeline_state = {
            'segmentation_results': {},
            'statistics': None,
            'priority': None,
            'report': None,
        }
        
        print("\n" + "="*60)
        print("🤖 Agent开始执行ReAct循环...")
        print("="*60 + "\n")
        
        for iteration in range(self.max_iterations):
            print(f"[Step {iteration + 1}/{self.max_iterations}]")
            
            # Step 1: 调用LLM
            try:
                llm_output = self._call_llm(messages)
                
            except Exception as e:
                error_msg = f"LLM调用失败: {str(e)}"
                print(f"❌ {error_msg}")
                
                # 如果LLM失败，使用规则引擎生成基础回答
                return self._fallback_response(context, last_result_data)
            
            # Step 2: 解析LLM输出
            thought, action, action_input = self._parse_llm_response(llm_output)
            
            print(f"💭 Thought: {thought}")
            print(f"🔧 Action: {action}")
            
            # Step 3: 记录Thought
            trace_step = {
                'iteration': iteration,
                'thought': thought,
                'action': action,
                'status': 'pending'
            }
            
            # Step 4: 检查是否是Final Answer
            if action == 'Final Answer':
                missing_stages = self._missing_pipeline_stages(pipeline_state)
                if missing_stages:
                    observation_str = (
                        "Error: 不能提前生成 Final Answer；尚未完成流水线阶段: "
                        + ", ".join(missing_stages)
                    )
                    print(f"❌ {observation_str}")
                    trace_step['status'] = 'failed'
                    trace_step['error'] = observation_str
                    reasoning_trace.append(trace_step)
                    messages.append({"role": "assistant", "content": llm_output})
                    messages.append({"role": "user", "content": f"Observation: {observation_str}"})
                    continue

                print(f"✅ Final Answer received")
                print(f"\n{action_input}\n")

                report_tool = self.tools.get('generate_report')
                report_path = pipeline_state['report'].get('report_path') if pipeline_state['report'] else None
                if report_tool and report_path and hasattr(report_tool, 'update_conclusion'):
                    report_tool.update_conclusion(report_path, action_input)
                
                trace_step['status'] = 'completed'
                reasoning_trace.append(trace_step)
                
                return {
                    'final_answer': action_input,
                    'reasoning_trace': reasoning_trace,
                    'result_data': pipeline_state['report']
                }
            
            # Step 5: 执行Tool
            try:
                if action not in self.tools:
                    raise ValueError(f"Unknown tool: {action}")
                
                # 解析Action Input并构建工具调用参数
                action_input_dict = self._build_action_input_dict(
                    action,
                    action_input,
                    last_result_data,
                    pipeline_state,
                    context,
                )

                # 执行工具
                observation = self.tools[action].execute(**action_input_dict)
                self._update_pipeline_state(action, observation, pipeline_state)
                observation_str = self._summarize_observation(action, observation)
                
                print(f"📊 Observation: {observation_str[:200]}...")
                
                # 更新状态
                trace_step['status'] = 'completed'
                trace_step['observation'] = observation_str
                
                # 保存结果数据（用于后续步骤）
                last_result_data = observation
                
            except Exception as e:
                observation_str = f"Error: {str(e)}"
                print(f"❌ Tool执行失败: {observation_str}")
                
                trace_step['status'] = 'failed'
                trace_step['error'] = str(e)
            
            reasoning_trace.append(trace_step)
            
            # Step 6: 反馈给LLM
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation_str}"})
        
        # 超过最大迭代次数
        error_msg = f"Task incomplete after {self.max_iterations} iterations"
        print(f"⚠️ {error_msg}")
        
        return {
            'final_answer': f"任务未完成：{error_msg}",
            'reasoning_trace': reasoning_trace,
            'result_data': last_result_data
        }
    
    def _build_system_prompt(self) -> str:
        """构造System Prompt"""
        tools_desc = "\n".join([
            f"- **{tool.name}**: {tool.description}"
            for tool in self.tools.values()
        ])
        
        return f"""你是一个灾害应急响应专家助手。你有以下工具可用：

{tools_desc}

请按照以下步骤解决问题：
1. **Thought**: 分析当前情况，决定下一步做什么
2. **Action**: 选择一个工具并给出参数（必须是JSON格式）
3. **Observation**: 等待工具返回结果
4. 重复以上步骤，直到能够给出**Final Answer**

**输出格式示例：**

Thought: 我需要先对图像进行分割处理
Action: ensemble_segment
Action Input: {{"image_path": "/path/to/image.jpg"}}

Observation: {{...}}

Thought: 现在需要统计损坏情况
Action: analyze_statistics
Action Input: {{"seg_results": [...]}}

...

Final Answer: [你的最终回答，用自然语言总结分析结果和建议，包含具体数据和可执行的救援建议]

**重要规则：**
- 每次只执行一个Action
- Action Input必须是有效的JSON格式
- 当前任务只分析 Context 中 image_path 指定的单张图像，不得进行跨图汇总或图像间排名
- 工具之间的数据由系统自动传递；不要复制 Observation 中的结构化结果
- 必须依次完成 ensemble_segment、analyze_statistics、calculate_priority、generate_report 后才能输出 Final Answer
- Final Answer应该包含当前图像的具体数据、单图受灾等级和可执行的救援建议，该结论会写入当前图像报告
- 如果某个工具失败，尝试其他方法或说明局限性
"""
    
    def _parse_llm_response(self, llm_output: str) -> tuple:
        """
        解析LLM输出，提取Thought、Action和Action Input
        
        Args:
            llm_output: LLM的原始输出
        
        Returns:
            (thought, action, action_input)
        """
        # 提取Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', llm_output, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        
        # 提取Action
        action_match = re.search(r'Action:\s*([^\n]+)', llm_output)
        if not action_match:
            final_match = re.search(r'Final Answer:\s*(.*)', llm_output, re.DOTALL)
            final_answer = final_match.group(1).strip() if final_match else llm_output.strip()
            return thought, "Final Answer", final_answer

        action = action_match.group(1).strip()
        
        # 提取Final Answer内容（如果是Final Answer）
        if action.lower().startswith('final'):
            final_match = re.search(r'Final Answer:\s*(.*)', llm_output, re.DOTALL)
            if final_match:
                action_input = final_match.group(1).strip()
            else:
                # 有时LLM会直接在Action: Final Answer之后给出文本，而不写Final Answer:标签
                parts = re.split(r'Action:\s*Final Answer', llm_output, maxsplit=1, flags=re.IGNORECASE)
                action_input = parts[1].strip() if len(parts) > 1 else ""
            return thought, action, action_input
        
        # 提取Action Input（支持多行JSON或Python字面量）
        action_input_match = re.search(
            r'Action Input:\s*(.+?)(?:\n(?:Thought:|Action:|Observation:)|$)',
            llm_output,
            re.DOTALL
        )
        
        if action_input_match:
            action_input_str = action_input_match.group(1).strip()
            action_input = self._parse_action_input_value(action_input_str)
        else:
            action_input = ""
        
        return thought, action, action_input

    @staticmethod
    def _parse_action_input_value(action_input_str: str) -> Any:
        """解析 Action Input 开头的第一个结构化值，忽略 LLM 追加的尾随文本。"""
        try:
            value, _ = json.JSONDecoder().raw_decode(action_input_str)
            return value
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(action_input_str)
        except (ValueError, SyntaxError):
            return action_input_str.splitlines()[0].strip()

    def _build_action_input_dict(
        self,
        action: str,
        action_input: Any,
        last_result_data: Any,
        pipeline_state: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        将解析后的Action Input转换为工具可接受的参数字典。
        """
        pipeline_state = pipeline_state or {
            'segmentation_results': {},
            'statistics': None,
            'priority': None,
            'report': None,
        }
        context = context or {}

        # 大型工具结果始终通过内存状态传递，不接受 LLM 回传的文本副本。
        if action == 'ensemble_segment':
            image_path = context.get('image_path')
            if not image_path:
                raise ValueError("缺少当前图像路径")
            return {'image_path': image_path}

        if action == 'analyze_statistics':
            seg_results = list(pipeline_state['segmentation_results'].values())
            if not seg_results:
                raise ValueError("缺少真实分割结果，请先调用 ensemble_segment")
            return {'seg_results': seg_results}

        if action == 'calculate_priority':
            if pipeline_state['statistics'] is None:
                raise ValueError("缺少统计结果，请先调用 analyze_statistics")
            return {'stats_result': pipeline_state['statistics']}

        if action == 'generate_report':
            if pipeline_state['priority'] is None:
                raise ValueError("缺少优先级结果，请先调用 calculate_priority")
            result = {'priority_result': pipeline_state['priority']}
            segmentation_results = pipeline_state['segmentation_results']
            if not segmentation_results:
                raise ValueError("缺少语义分割结果，请先调用 ensemble_segment")
            result['segmentation_result'] = next(iter(segmentation_results.values()))
            result['statistics_result'] = pipeline_state['statistics']
            output_filename = self._normalize_output_filename(context.get('output_filename'))
            if output_filename:
                result['output_filename'] = output_filename
            return result

        tool = self.tools[action]
        params = [
            p.name for p in inspect.signature(tool.execute).parameters.values()
            if p.name != 'self' and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        if isinstance(action_input, dict):
            result = action_input.copy()
            return self._filter_tool_params(action, result, params)

        if isinstance(action_input, list):
            if len(params) == 1:
                result = {params[0]: action_input}
                return self._filter_tool_params(action, result, params)
            raise ValueError(f"工具 {action} 期望命名参数，但收到列表输入")

        if isinstance(action_input, str):
            # 解析字符串值，处理可转换的结构
            try:
                parsed = json.loads(action_input)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(action_input)
                except (ValueError, SyntaxError):
                    parsed = action_input

            if isinstance(parsed, dict):
                result = parsed
                return self._filter_tool_params(action, result, params)
            if isinstance(parsed, list) and len(params) == 1:
                result = {params[0]: parsed}
                return self._filter_tool_params(action, result, params)
            if len(params) == 1:
                result = {params[0]: parsed}
                return self._filter_tool_params(action, result, params)

            result = {'raw_input': action_input}
            return self._filter_tool_params(action, result, params)

        if action_input is None:
            result = {}
            return self._filter_tool_params(action, result, params)

        if len(params) == 1:
            result = {params[0]: action_input}
            return self._filter_tool_params(action, result, params)

        result = {'raw_input': str(action_input)}
        return self._filter_tool_params(action, result, params)

    def _update_pipeline_state(
        self,
        action: str,
        observation: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> None:
        """保存工具的真实返回值，供后续工具直接使用。"""
        if action == 'ensemble_segment':
            pipeline_state['segmentation_results'][observation['image_path']] = observation
        elif action == 'analyze_statistics':
            pipeline_state['statistics'] = observation
            pipeline_state['priority'] = None
            pipeline_state['report'] = None
        elif action == 'calculate_priority':
            pipeline_state['priority'] = observation
            pipeline_state['report'] = None
        elif action == 'generate_report':
            pipeline_state['report'] = observation

    def _summarize_observation(self, action: str, observation: Dict[str, Any]) -> str:
        """生成适合 LLM 的小型、无损业务语义摘要。"""
        if action == 'ensemble_segment':
            mask = observation['best_mask']
            summary = {
                'best_model': observation['best_model'],
                'scores': observation['scores'],
                'image_path': observation['image_path'],
                'mask_shape': list(mask.shape),
            }
        else:
            summary = observation

        return json.dumps(summary, ensure_ascii=False, default=self._json_default)[:4000]

    @staticmethod
    def _json_default(value: Any) -> Any:
        """将小型数值类型转为 JSON，拒绝把数组隐式转为摘要字符串。"""
        if hasattr(value, 'item'):
            return value.item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    @staticmethod
    def _normalize_output_filename(output_filename: str) -> str:
        if not output_filename:
            return ''
        filename = os.path.basename(output_filename)
        stem, extension = os.path.splitext(filename)
        if extension.lower() in {'.html', '.htm', '.md', '.markdown'}:
            return stem
        return filename

    @staticmethod
    def _missing_pipeline_stages(pipeline_state: Dict[str, Any]) -> List[str]:
        stages = []
        if not pipeline_state['segmentation_results']:
            stages.append('ensemble_segment')
        if pipeline_state['statistics'] is None:
            stages.append('analyze_statistics')
        if pipeline_state['priority'] is None:
            stages.append('calculate_priority')
        if pipeline_state['report'] is None:
            stages.append('generate_report')
        return stages

    def _filter_tool_params(self, action: str, action_input_dict: Dict[str, Any], params: List[str]) -> Dict[str, Any]:
        """
        过滤掉工具不支持的参数，避免 execute() 收到 unexpected keyword argument。
        """
        return {k: v for k, v in action_input_dict.items() if k in params}

    def _fallback_response(self, context: Dict, result_data: Dict) -> Dict[str, Any]:
        """
        LLM失败时的降级响应（规则引擎）
        
        Args:
            context: 上下文信息
            result_data: 已有的结果数据
        
        Returns:
            降级响应
        """
        fallback_answer = """
⚠️ **注意**: AI助手暂时不可用，以下是基于规则的基础分析：

根据系统自动分析：
- 已完成图像分割和统计
- 建议查看生成的报告文件获取详细信息

请稍后重试或联系技术支持。
"""
        
        return {
            'final_answer': fallback_answer,
            'reasoning_trace': [],
            'result_data': result_data
        }
