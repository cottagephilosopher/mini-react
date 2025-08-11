"""
预测模块，用于实现与语言模型的交互和推理
"""
import logging
import os
from typing import Any, Dict, Optional
import json
import re
import hashlib
import pickle
from pathlib import Path

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
        system_content = predict_prompts["react_system_prompt"]
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": content}
        ]

class PredictionCache:
    """简单的预测结果缓存"""
    
    def __init__(self, cache_dir: str = ".cache", max_entries: int = 100, enabled: bool = True):
        """
        初始化缓存
        
        参数:
            cache_dir: 缓存目录
            max_entries: 最大缓存条目数
            enabled: 是否启用缓存
        """
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.enabled = enabled
        
        if enabled:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_key(self, messages: list) -> str:
        """生成缓存键"""
        # 使用消息内容的哈希作为键
        content = str(messages).encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def get(self, key: str) -> Optional[dict]:
        """获取缓存项"""
        if not self.enabled:
            return None
            
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        return None
    
    def set(self, key: str, value: dict):
        """设置缓存项"""
        if not self.enabled:
            return
            
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"写入缓存失败: {e}")
        
        # 清理过多的缓存文件
        self._cleanup()
    
    def _cleanup(self):
        """清理过多的缓存文件"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        if len(cache_files) > self.max_entries:
            # 按修改时间排序，删除最旧的文件
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            for file in cache_files[:len(cache_files) - self.max_entries]:
                try:
                    file.unlink()
                except Exception:
                    pass

# 全局缓存实例
prediction_cache = PredictionCache()

class Predict(Module):
    """
    预测模块，用于调用大语言模型进行预测
    """
    
    def __init__(
        self, 
        signature: Any,
        model: Optional[str] = None,
        chat_adapter: Optional[ChatAdapter] = None,
        use_cache: bool = True
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
        self.use_cache = use_cache
    
    def forward(self, **kwargs: Any) -> Prediction:
        """
        执行预测
        
        参数:
            **kwargs: 输入参数
            
        返回:
            预测结果
        """
        print(f"[PREDICT FORWARD] 开始执行，接收参数: {list(kwargs.keys())}")
        
        # 从kwargs中提取lm参数（如果有的话）
        lm = kwargs.pop("lm", None)
        print(f"[PREDICT FORWARD] 提取的lm: {lm.model_name if lm else 'None'} @ {lm.api_base if lm else 'None'}")
        
        logger.debug(f"Predict.forward 接收的lm参数: {lm}")
        if lm:
            logger.debug(f"LM实例详情: 模型={lm.model_name}, API基址={lm.api_base}")
        
        # 准备输入
        inputs = {k: v for k, v in kwargs.items() if k in self.signature.input_fields}
        
        # 创建消息
        messages = self.chat_adapter.create_messages(self.signature, inputs)
        
        # 检查缓存
        if self.use_cache:
            cache_key = prediction_cache.get_key(messages)
            cached_response = prediction_cache.get(cache_key)
            if cached_response:
                logger.info("使用缓存的预测结果")
                return Prediction(**cached_response)

        try:
            # 调用语言模型
            params = {"temperature": 0.1}  # 使用较低的温度以获得更确定性的回答
            
            if lm:
                # 如果传递了lm实例，使用它
                logger.debug(f"使用传递的LM实例: {lm.model_name}, API基址: {lm.api_base}")
                logger.debug(f"LM实例配置: {getattr(lm, 'config', {})}")
                logger.debug(f"即将调用lm.chat，传递参数: {params}")
                response = lm.chat(messages, **params)
                logger.debug(f"lm.chat返回响应: {response}")
            else:
                # 否则使用全局配置
                if self.model:
                    params["model"] = self.model
                logger.debug("使用全局LM配置")
                logger.debug(f"即将调用lm_chat，传递参数: {params}")
                response = lm_chat(messages, **params)
                logger.debug(f"lm_chat返回响应: {response}")
            
            # 提取回答内容
            content = response["content"]
            
            # 解析回答，提取输出字段
            outputs = {}
            
            # 首先检查是否是错误消息
            if "调用语言模型时出错" in content or "请求超时" in content or "网络连接" in content:
                logger.error(f"检测到LLM调用错误: {content}")
                # 返回默认的错误处理结果
                outputs = {
                    "next_tool_name": "finish",
                    "next_tool_args": {
                        "reasoning": "系统遇到网络问题，请稍后重试",
                        "answer": "抱歉，系统暂时无法处理您的请求，请稍后重试。"
                    }
                }
                for field_name in self.signature.output_fields:
                    if field_name == "reasoning":
                        outputs[field_name] = "系统遇到网络问题"
                    elif field_name == "answer":
                        outputs[field_name] = "抱歉，系统暂时无法处理您的请求，请稍后重试。"
                    elif field_name not in outputs:
                        outputs[field_name] = content
            # 检查是否是完整的工具调用格式文本（如日志显示的问题）
            elif "思考：" in content and "工具：" in content and "参数：" in content:
                logger.debug(f"检测到完整的工具调用格式，进行解析: {content[:200]}...")
                
                # 提取工具名：寻找"工具："后的内容
                tool_pattern = r'工具[:：]\s*([^\s\n]+)'
                tool_match = re.search(tool_pattern, content)
                
                if tool_match:
                    tool_name = tool_match.group(1).strip()
                    outputs["next_tool_name"] = tool_name
                    logger.debug(f"提取工具名: {tool_name}")
                    
                    # 提取参数：寻找"参数："后的JSON
                    args_pattern = r'参数[:：]\s*(\{.*?\})'
                    args_match = re.search(args_pattern, content, re.DOTALL)
                    
                    if args_match:
                        args_text = args_match.group(1)
                        try:
                            tool_args = json.loads(args_text.strip())
                            outputs["next_tool_args"] = tool_args
                            logger.debug(f"提取工具参数: {tool_args}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON 解析失败: {args_text}, 错误: {e}")
                            outputs["next_tool_args"] = {}
                    else:
                        outputs["next_tool_args"] = {}
                
                # 提取其他输出字段
                for field_name in self.signature.output_fields:
                    if field_name not in outputs:
                        if field_name == "reasoning":
                            # 提取"思考："后的内容
                            thought_pattern = r'思考[:：]\s*([^\n]+)'
                            thought_match = re.search(thought_pattern, content)
                            if thought_match:
                                outputs[field_name] = thought_match.group(1).strip()
                            else:
                                outputs[field_name] = "正在处理..."
                        elif field_name == "answer":
                            # 对于answer字段，从参数中提取
                            if "next_tool_args" in outputs and isinstance(outputs["next_tool_args"], dict):
                                outputs[field_name] = outputs["next_tool_args"].get("answer", "正在处理...")
                            else:
                                outputs[field_name] = "正在处理..."
                        else:
                            outputs[field_name] = content
            else:
                # 使用原有的正则表达式提取字段逻辑
                for field_name in self.signature.output_fields:
                    # 使用更健壮的正则表达式匹配字段
                    field_pattern = rf"{re.escape(field_name)}:(.*?)(?:\n\w+:|$)"
                    field_match = re.search(field_pattern, content, re.DOTALL)
                    
                    if field_match:
                        value = field_match.group(1).strip()
                        
                        # 特殊处理next_tool_args字段，确保它是一个字典
                        if field_name == "next_tool_args":
                            try:
                                # 尝试从文本中提取JSON格式的参数
                                json_pattern = r'\{.*\}'
                                json_match = re.search(json_pattern, value, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    value = json.loads(json_str)
                                elif value.strip().startswith('{') and value.strip().endswith('}'):
                                    value = json.loads(value)
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
                            # 尝试匹配工具名称（支持连字符和完整名称）
                            # 首先尝试匹配完整的工具名称（包含连字符）
                            tool_name_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_-]+)\b', value)
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
                
            # 缓存结果
            if self.use_cache:
                prediction_cache.set(cache_key, outputs)

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