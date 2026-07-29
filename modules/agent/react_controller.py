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
import re
from typing import Dict, Any, List

from openai import OpenAI
from modules.agent.base_tool import BaseTool
from configs.agent_config import REACT_CONFIG, LLM_CONFIG


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
            api_key: DeepSeek API Key，默认从配置读取
        """
        self.tools = tools
        self.api_key = api_key or LLM_CONFIG['api_key']
        
        # 初始化OpenAI客户端（DeepSeek兼容OpenAI SDK）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=LLM_CONFIG['base_url']
        )
        
        self.max_iterations = REACT_CONFIG['max_iterations']
        self.temperature = REACT_CONFIG['temperature']
        self.max_tokens = REACT_CONFIG['max_tokens']
    
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
        
        print("\n" + "="*60)
        print("🤖 Agent开始执行ReAct循环...")
        print("="*60 + "\n")
        
        for iteration in range(self.max_iterations):
            print(f"[Step {iteration + 1}/{self.max_iterations}]")
            
            # Step 1: 调用LLM
            try:
                response = self.client.chat.completions.create(
                    model=LLM_CONFIG['model'],
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                llm_output = response.choices[0].message.content
                
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
                print(f"✅ Final Answer received")
                print(f"\n{action_input}\n")
                
                trace_step['status'] = 'completed'
                reasoning_trace.append(trace_step)
                
                return {
                    'final_answer': action_input,
                    'reasoning_trace': reasoning_trace,
                    'result_data': last_result_data
                }
            
            # Step 5: 执行Tool
            try:
                if action not in self.tools:
                    raise ValueError(f"Unknown tool: {action}")
                
                # 解析Action Input并构建工具调用参数
                action_input_dict = self._build_action_input_dict(action, action_input, last_result_data)

                # 执行工具
                observation = self.tools[action].execute(**action_input_dict)
                observation_str = json.dumps(observation, ensure_ascii=False, default=str)[:1000]  # 限制长度
                
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
- Final Answer应该包含具体的数据、优先级排序和 actionable 的救援建议
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
        action = action_match.group(1).strip() if action_match else "Final Answer"
        
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
            
            # 尝试解析JSON或Python字面量
            try:
                action_input = json.loads(action_input_str)
            except json.JSONDecodeError:
                try:
                    action_input = ast.literal_eval(action_input_str)
                except (ValueError, SyntaxError):
                    action_input = action_input_str
        else:
            action_input = ""
        
        return thought, action, action_input

    def _build_action_input_dict(self, action: str, action_input: Any, last_result_data: Any) -> Dict[str, Any]:
        """
        将解析后的Action Input转换为工具可接受的参数字典。
        """
        tool = self.tools[action]
        params = [
            p.name for p in inspect.signature(tool.execute).parameters.values()
            if p.name != 'self' and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        if isinstance(action_input, dict):
            result = action_input.copy()
            return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)

        if isinstance(action_input, list):
            if len(params) == 1:
                result = {params[0]: action_input}
                return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)
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
                return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)
            if isinstance(parsed, list) and len(params) == 1:
                result = {params[0]: parsed}
                return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)
            if len(params) == 1:
                result = {params[0]: parsed}
                return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)

            result = {'raw_input': action_input}
            return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)

        if action_input is None:
            result = {}
            return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)

        if len(params) == 1:
            result = {params[0]: action_input}
            return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)

        result = {'raw_input': str(action_input)}
        return self._filter_tool_params(action, self._fill_missing_tool_params(action, result, last_result_data, params), params)

    def _fill_missing_tool_params(self, action: str, action_input_dict: Dict[str, Any], last_result_data: Any, params: List[str]) -> Dict[str, Any]:
        """
        如果Action Input中缺少必需参数，则尝试从上一步工具结果中补充。
        """
        if action == 'analyze_statistics':
            # If caller didn't provide seg_results, try to fill from last_result_data
            if 'seg_results' not in action_input_dict:
                if isinstance(last_result_data, dict) and 'best_mask' in last_result_data:
                    action_input_dict['seg_results'] = [last_result_data]
                elif isinstance(last_result_data, dict) and 'per_image_stats' in last_result_data:
                    action_input_dict['seg_results'] = last_result_data
            else:
                # If seg_results was provided but is a string representation (e.g. numpy dump),
                # prefer using last_result_data when available to provide a real ndarray.
                seg_val = action_input_dict.get('seg_results')
                if isinstance(seg_val, str) and isinstance(last_result_data, dict) and 'best_mask' in last_result_data:
                    action_input_dict['seg_results'] = [last_result_data]

        if action == 'calculate_priority':
            if 'stats_result' not in action_input_dict:
                if isinstance(last_result_data, dict) and 'per_image_stats' in last_result_data:
                    action_input_dict['stats_result'] = last_result_data
            else:
                # Replace stringified stats_result with actual last_result_data when possible
                stats_val = action_input_dict.get('stats_result')
                if isinstance(stats_val, str) and isinstance(last_result_data, dict) and 'per_image_stats' in last_result_data:
                    action_input_dict['stats_result'] = last_result_data

        if action == 'generate_report':
            if 'priority_result' not in action_input_dict:
                if isinstance(last_result_data, dict) and 'priority_zones' in last_result_data:
                    action_input_dict['priority_result'] = last_result_data
            else:
                # Replace stringified priority_result with actual last_result_data when possible
                pr_val = action_input_dict.get('priority_result')
                if isinstance(pr_val, str) and isinstance(last_result_data, dict) and 'priority_zones' in last_result_data:
                    action_input_dict['priority_result'] = last_result_data

        # 如果工具只有一个必需参数且未提供，则尝试用 last_result_data 补充
        if len(params) == 1 and params[0] not in action_input_dict:
            if params[0] == 'seg_results' and isinstance(last_result_data, dict) and 'best_mask' in last_result_data:
                action_input_dict[params[0]] = [last_result_data]
            elif params[0] == 'stats_result' and isinstance(last_result_data, dict) and 'per_image_stats' in last_result_data:
                action_input_dict[params[0]] = last_result_data
            elif params[0] == 'priority_result' and isinstance(last_result_data, dict) and 'priority_zones' in last_result_data:
                action_input_dict[params[0]] = last_result_data

        return action_input_dict

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
