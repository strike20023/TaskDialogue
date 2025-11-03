"""
MultiWOZ User Simulator

模拟用户进行对话，基于任务目标生成用户话语。
"""

from typing import Dict, Any, Optional
import re
import tenacity

from taskdialogue.core.base.user_simulator import BaseUserSimulator
from taskdialogue.core.models.base import BaseModel
from taskdialogue.core.schemas.message import Message
from taskdialogue.core.utils.logger import logger
from taskdialogue.benchmarks.multiwoz.prompts import format_user_simulator_prompt


def prepare_goals_string(goals):
    """格式化用户目标为字符串"""
    if isinstance(goals, str):
        # 处理 WOZxxx.json 格式: "Task 12345: goal1. goal2."
        result = re.match(r'Task \d{5}: (.*)', goals)
        if result:
            goals = result[1].strip()
            goals = re.split(r'\.\s+', goals)
        else:
            goals = [goals]
    
    # 确保每个目标都以句号结尾
    goals_str = [msg if msg.endswith('.') else f'{msg}.' for msg in goals]
    goals_str = '\n'.join(goals_str)
    
    # 去除 HTML 标签 <span>xx</span> => xx
    goals_str = re.sub(r'<span\b[^>]*>(.*?)</span>', r'\1', goals_str)
    
    return goals_str


def tenacity_retry_log(retry_state):
    """重试日志回调"""
    logger.warning(f"Tenacity: Retrying in {retry_state.next_action.sleep}s "
                   f"due to {retry_state.outcome.exception().__class__.__name__}: "
                   f"{retry_state.outcome.exception()}")


class MultiWOZUserSimulator(BaseUserSimulator):
    """MultiWOZ User Simulator
    
    根据对话目标动态生成用户话语，模拟真实用户行为。
    
    Example:
        >>> model = create_model_from_config(config, "user")
        >>> simulator = MultiWOZUserSimulator(model, config)
        >>> task = {"goal": {...}, "log": [...]}
        >>> simulator.reset(task)
        >>> user_input = simulator.generate_response(None)  # 第一句
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(model, config, **kwargs)
        
        # 配置
        self.temperature = config.get("temperature", 0.01)
        self.max_tokens = config.get("max_tokens", 512)
        
        # 状态
        self.goals = ""
        self.ref_dialog = ""
        self.first_user_utter = ""
        self.turn_idx = 0
    
    def generate_response(
        self, 
        agent_output: Optional[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成用户回复"""
        self.turn_idx += 1
        
        # 第一轮：使用固定的第一句话
        if self.turn_idx == 1:
            assert agent_output in ['', None]
            # 添加第一轮用户输入到历史
            user_msg = Message(role="user", content=self.first_user_utter)
            self.add_message(user_msg)
            return self.first_user_utter
        
        # 使用重试机制
        retrying = tenacity.Retrying(
            wait=tenacity.wait_exponential(min=2, max=60),
            stop=tenacity.stop_after_attempt(5),
            before_sleep=tenacity_retry_log,
            retry=tenacity.retry_if_exception_type(Exception)
        )
        
        try:
            user_utter = retrying(self._generate_response, agent_output)
        except Exception as e:
            logger.error(f"⚠️  User Simulator error: {e}")
            user_utter = "Could you please repeat that?"
        
        # 更新历史
        agent_msg = Message(role="assistant", content=agent_output)
        self.add_message(agent_msg)
        
        user_msg = Message(role="user", content=user_utter)
        self.add_message(user_msg)
        
        return user_utter.strip()
    
    def _generate_response(self, agent_output: str) -> str:
        """内部生成方法"""
        # 构建 prompt
        prompt = format_user_simulator_prompt(
            user_goals=self.goals,
            ref_dialog=self.ref_dialog,
            history=self._format_history(),
            input_text=agent_output
        )
        
        # 调用模型
        messages = [
            {"role": "system", "content": "You are simulating a user in a task-oriented dialogue."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.model.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # 提取文本
        user_utter = response.content.strip()
        
        # 去除可能的前缀
        if user_utter.startswith('User:'):
            user_utter = user_utter[5:].strip()
        
        # 在 "AI Assistant:" 处停止
        if 'AI Assistant:' in user_utter:
            user_utter = user_utter.split('AI Assistant:')[0].strip()
        
        return user_utter
    
    def is_dialogue_complete(self) -> bool:
        """判断对话是否完成
        
        检查最后一句用户话语中是否包含结束标记。
        """
        if not self.dialogue_history:
            return False
        
        last_msg = self.dialogue_history[-1]
        if last_msg.role != "user":
            return False
        
        content = last_msg.content or ""
        content_lower = content.lower()
        
        # 检查明确的结束标记
        end_markers = [
            "dialogue ends",
            "dialogue end",
            "conversation ends",
            "goodbye",
            "bye bye",
            "bye",
            "that's all",
            "that will be all",
            "thank you, goodbye",
            "thanks, goodbye",
            "that's everything",
            "that's all i need"
        ]
        
        # 检查是否包含任何结束标记
        if any(marker in content_lower for marker in end_markers):
            return True
        
        # 检查感谢结束模式："thank you for..." 并且是消息的结尾部分
        if "thank you for" in content_lower and (
            "all your help" in content_lower or
            "your help" in content_lower or
            "everything" in content_lower
        ):
            # 确保这是消息的最后一部分（不是中间的感谢）
            sentences = content.split('.')
            if sentences:
                last_sentence = sentences[-1].lower().strip()
                if "thank you" in last_sentence or "thanks" in last_sentence:
                    return True
        
        return False
    
    def reset(self, task_config: Dict[str, Any]) -> None:
        """重置并加载新任务"""
        self.task_config = task_config
        self.dialogue_history = []
        self.turn_idx = 0
        
        # 准备用户目标
        goal_data = task_config.get('goal', {})
        goal_message = goal_data.get('message', '')
        
        # goal_message 可能是字符串、列表或字典
        if isinstance(goal_message, dict):
            # 如果是字典，提取所有领域的目标
            goal_parts = []
            for domain, domain_goal in goal_message.items():
                if isinstance(domain_goal, str):
                    goal_parts.append(f"{domain}: {domain_goal}")
            goal_message = ' '.join(goal_parts)
        elif isinstance(goal_message, list):
            # 如果是列表，直接使用
            pass
        elif isinstance(goal_message, str):
            # 如果是字符串，直接使用
            pass
        else:
            # 其他情况，使用空字符串
            goal_message = ''
        
        self.goals = prepare_goals_string(goal_message)
        
        # 准备参考对话
        ref_dialog = []
        log = task_config.get('log', [])
        for i, turn in enumerate(log):
            role = 'User' if i % 2 == 0 else 'AI Assistant'
            text = turn.get('text', '')
            ref_dialog.append(f'{role}: {text}')
        self.ref_dialog = '\n'.join(ref_dialog)
        
        # 第一句用户话语
        self.first_user_utter = log[0]['text'] if log else "Hello, I need help with something."
    
    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.dialogue_history:
            return ""
        
        lines = []
        for msg in self.dialogue_history:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"AI Assistant: {msg.content}")
        
        return '\n'.join(lines)

