"""
预测模块，用于实现与语言模型的交互和推理
"""
import logging
import os
from typing import Any, Dict, Optional
import json
import re

# 导入我们的LM模块替代直接使用litellm
from .lm import chat as lm_chat, complete as lm_complete

from .module import Module
from .signature import Signature, ensure_signature
from .prompt import predict_prompts

logger = logging.getLogger(__name__)


class Prediction(Dict[str, Any]):
    """
    预测结果类，用于存储语言模型的预测结果
    
    这个类本质上是一个字典，但提供了通过属性访问值的功能。
    """
    
    def __getattr__(self, key: str) -> Any:
        """
        通过属性访问字典中的值
        
        参数:
            key: 键名
            
        返回:
            对应的值
            
        异常:
            AttributeError: 当键不存在时
        """
        try:
            return self[key]
        except KeyError:
            # 对于缺失的属性，记录警告并返回None而不是抛出异常
            logger.warning(f"访问了不存在的属性: '{key}'")
            return None


class ChatAdapter:
    """
    聊天适配器，用于格式化与大语言模型的交互
    """
    
    def format_user_message_content(self, signature: Signature, inputs: Dict[str, Any]) -> str:
        """
        格式化用户消息内容
        
        参数:
            signature: 签名对象
            inputs: 输入参数
            
        返回:
            格式化后的用户消息内容
        """
        parts = []
        
        # 添加指令（如果有）
        if signature.instructions:
            parts.append(signature.instructions)
        
        # 添加输入字段
        for name, value in inputs.items():
            # 对于复杂对象，使用其字符串表示
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = str(value)
            parts.append(f"{name}: {value}")
        
        return "\n".join(parts)
    
    def create_messages(self, signature: Signature, inputs: Dict[str, Any]) -> list:
        """
        创建消息列表，用于调用大语言模型API
        
        参数:
            signature: 签名对象
            inputs: 输入参数
            
        返回:
            消息列表
        """
        content = self.format_user_message_content(signature, inputs)
        
        # 检查是否需要添加系统提示
        if "next_tool_args" in signature.output_fields:
            # 这是ReAct框架的请求，添加系统提示以指导正确的格式
            system_content = predict_prompts["react_system_prompt"]
        else:
            # 这是普通字段提取任务，添加相应的系统提示
            output_fields = ", ".join(signature.output_fields.keys())
            system_content = predict_prompts["field_extraction_system_prompt"].format(
                output_fields=output_fields
            )
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": content}
        ]


class Predict(Module):
    """
    预测模块，用于调用大语言模型进行预测
    """
    
    def __init__(
        self, 
        signature: Any,
        model: Optional[str] = None,
        chat_adapter: Optional[ChatAdapter] = None,
    ):
        """
        初始化预测模块
        
        参数:
            signature: 签名定义
            model: 要使用的语言模型名称
            chat_adapter: 聊天适配器实例
        """
        super().__init__()
        self.signature = ensure_signature(signature)
        self.model = model  # 这里只保存模型名称，具体模型通过LM模块获取
        self.chat_adapter = chat_adapter or ChatAdapter()
    
    def forward(self, **kwargs: Any) -> Prediction:
        """
        执行预测
        
        参数:
            **kwargs: 输入参数
            
        返回:
            预测结果
        """
        # 准备输入
        inputs = {k: v for k, v in kwargs.items() if k in self.signature.input_fields}
        
        # 创建消息
        messages = self.chat_adapter.create_messages(self.signature, inputs)
        
        try:
            # 调用语言模型
            # 使用我们的LM模块替代直接调用litellm
            params = {"temperature": 0.1}  # 使用较低的温度以获得更确定性的回答
            if self.model:
                params["model"] = self.model
                
            response = lm_chat(messages, **params)
            
            # 提取回答内容
            content = response["content"]
            
            # 解析回答，提取输出字段
            # 这里采用简单方法，假设模型按照要求返回了每个输出字段
            # 在实际应用中，可能需要更复杂的解析逻辑
            outputs = {}
            
            # 处理JSON格式的工具参数
            for field_name in self.signature.output_fields:
                # 寻找字段名在内容中的位置
                field_marker = f"{field_name}:"
                if field_marker in content:
                    # 提取字段值
                    start = content.find(field_marker) + len(field_marker)
                    end = content.find("\n", start)
                    if end == -1:  # 如果是最后一个字段
                        end = len(content)
                    
                    value = content[start:end].strip()
                    
                    # 特殊处理next_tool_args字段，确保它是一个字典
                    if field_name == "next_tool_args":
                        try:
                            # 尝试解析为JSON对象
                            if value.startswith('{') and value.endswith('}'):
                                value = json.loads(value)
                            else:
                                # 尝试从文本中提取JSON格式的参数
                                json_pattern = r'\{.*\}'
                                json_match = re.search(json_pattern, value, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    value = json.loads(json_str)
                                else:
                                    # 如果无法提取JSON，创建一个空字典
                                    logger.warning(f"无法解析工具参数: {value}")
                                    value = {}
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析工具参数为JSON: {value}")
                            value = {}
                    
                    # 特殊处理next_tool_name字段，确保它只是工具名称
                    elif field_name == "next_tool_name":
                        # 去除可能的额外字符
                        value = value.strip()
                        # 如果工具名被其他字符包围，如'search'或[search]
                        if (value.startswith("'") and value.endswith("'")) or \
                           (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("[") and value.endswith("]")):
                            value = value[1:-1].strip()
                        # 尝试匹配一个单词（为工具名）
                        tool_name_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', value)
                        if tool_name_match:
                            value = tool_name_match.group(1)
                        logger.info(f"解析工具名称为: {value}")
                    
                    outputs[field_name] = value
                else:
                    # 如果找不到特定字段，使用整个内容
                    outputs[field_name] = content
            
            # 如果没有提取到任何字段，使用整个内容作为结果
            if not outputs and self.signature.output_fields:
                field_name = next(iter(self.signature.output_fields))
                outputs[field_name] = content
            
            return Prediction(**outputs)
            
        except Exception as e:
            logger.error(f"预测时发生错误: {e}")
            # 返回空预测结果
            return Prediction()


class ChainOfThought(Predict):
    """
    思维链预测，在请求中指示模型展示思维过程
    """
    
    def __init__(
        self, 
        signature: Any,
        model: Optional[str] = None,
        chat_adapter: Optional[ChatAdapter] = None,
    ):
        """
        初始化思维链预测模块
        
        参数:
            signature: 签名定义
            model: 要使用的语言模型名称
            chat_adapter: 聊天适配器实例
        """
        # 添加思维链提示到指令
        enhanced_signature = ensure_signature(signature)
        cot_instructions = predict_prompts["chain_of_thought"]
        
        if enhanced_signature.instructions:
            enhanced_signature.instructions = f"{enhanced_signature.instructions}\n\n{cot_instructions}"
        else:
            enhanced_signature.instructions = cot_instructions
        
        super().__init__(enhanced_signature, model, chat_adapter)