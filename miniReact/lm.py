"""
语言模型管理模块，基于标准OpenAI协议支持多种LLM
"""
import os
import json
import httpx
from loguru import logger
from typing import Any, Dict, List, Optional, Union, AsyncIterable
from urllib.parse import urljoin



class ContextWindowExceededError(Exception):
    """上下文窗口超出限制异常"""
    pass


class LMConfig:
    """
    语言模型配置管理器
    """
    
    _instance = None  # 单例模式
    
    def __new__(cls):
        """创建或返回单例实例"""
        if cls._instance is None:
            cls._instance = super(LMConfig, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._debug = False
            # 从环境变量加载初始配置
            cls._instance.from_env()
        return cls._instance
    
    def set_model(self, model_name: str):
        """
        设置当前使用的模型名称
        
        参数:
            model_name: 模型名称
        """
        self._config["model"] = model_name
    
    def get_model(self) -> str:
        """
        获取当前模型名称
        
        返回:
            模型名称
        """
        return self._config.get("model", "gpt-3.5-turbo")
    
    def set_api_base(self, api_base: str):
        """设置API基础URL"""
        # 确保URL格式正确
        if not api_base.endswith('/'):
            api_base += '/'
        
        # 只对标准OpenAI格式的URL添加v1，Azure OpenAI有自己的格式
        if not api_base.endswith('v1/') and 'azure.com' not in api_base:
            if api_base.endswith('/'):
                api_base += 'v1/'
            else:
                api_base += '/v1/'
        
        self._config["api_base"] = api_base
        logger.info(f"设置API基址: {api_base}")
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self._config["api_key"] = api_key
    
    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self._config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)
    
    def enable_debug(self):
        """启用调试模式"""
        self._debug = True
        logger.info("已启用调试模式")
    
    def disable_debug(self):
        """禁用调试模式"""
        self._debug = False
        logger.info("已禁用调试模式")
        
    def is_debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        return self._debug
    
    def from_env(self):
        """从环境变量加载配置"""
        # 加载模型名称
        model_name = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        self.set_model(model_name)
        
        # 加载API基础URL和密钥
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if api_key:
            self.set_api_key(api_key)
            
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_API_BASE")
        if api_base:
            self.set_api_base(api_base)
        
        # 加载其他配置
        for key, value in os.environ.items():
            if key.startswith("LLM_") and key not in ["LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE"]:
                config_key = key[4:].lower()
                self.set_config(config_key, value)
        
        # 检查是否启用调试模式
        if os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes"):
            self.enable_debug()
        
        return self


# 全局配置实例
config = LMConfig()


def get_model() -> str:
    """获取当前使用的模型名称"""
    return config.get_model()


def set_model(model_name: str):
    """设置当前使用的模型名称"""
    config.set_model(model_name)


def enable_debug():
    """启用调试模式"""
    config.enable_debug()


def disable_debug():
    """禁用调试模式"""
    config.disable_debug()


class OpenAIClient:
    """
    标准OpenAI协议客户端
    """
    
    def __init__(self, api_base: str = None, api_key: str = None, timeout: float = 60.0):
        """
        初始化OpenAI客户端
        
        参数:
            api_base: API基础URL
            api_key: API密钥
            timeout: 请求超时时间
        """
        self.api_base = api_base or config.get_config("api_base", "https://api.openai.com/v1/")
        self.api_key = api_key or config.get_config("api_key", "")
        self.timeout = timeout
        
        # 确保api_base以/结尾
        if not self.api_base.endswith('/'):
            self.api_base += '/'
            
        # 创建HTTP客户端
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"
        
        self.client = httpx.Client(
            base_url=self.api_base,
            headers=headers,
            timeout=timeout
        )
    
    def chat_completion(self, 
                       model: str,
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """
        调用聊天完成API
        
        参数:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大令牌数
            stream: 是否使用流式响应
            **kwargs: 其他参数
            
        返回:
            API响应
        """
        # 提取Azure API版本参数（如果有）
        api_version = kwargs.pop("api_version", None)
        
        # 构建请求数据
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        # 构建请求URL
        url = "chat/completions"
        if api_version:
            # Azure OpenAI需要api-version查询参数
            url += f"?api-version={api_version}"
        
        if config.is_debug_enabled():
            logger.info(f"请求URL: {self.api_base}{url}")
            logger.info(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        
        try:
            response = self.client.post(url, json=data)
            
            if config.is_debug_enabled():
                logger.info(f"响应状态: {response.status_code}")
                logger.info(f"响应头: {dict(response.headers)}")
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                if config.is_debug_enabled():
                    logger.info(f"响应内容: {json.dumps(result, ensure_ascii=False)}")
                return result
            elif response.status_code == 400:
                # 检查是否是上下文窗口超出异常
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "")
                if "context length" in error_message.lower() or "maximum context length" in error_message.lower():
                    raise ContextWindowExceededError(error_message)
                else:
                    raise Exception(f"API调用失败: {error_message}")
            else:
                error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}
                error_message = error_data.get("error", {}).get("message", error_data.get("error", "未知错误"))
                raise Exception(f"API调用失败 (状态码: {response.status_code}): {error_message}")
                
        except httpx.TimeoutException:
            raise Exception("请求超时，请检查网络连接或增加超时时间")
        except httpx.ConnectError:
            raise Exception("连接失败，请检查API基础URL和网络连接")
        except Exception as e:
            if isinstance(e, ContextWindowExceededError):
                raise e
            logger.error(f"API调用异常: {e}")
            raise Exception(f"API调用异常: {str(e)}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'client'):
            self.client.close()


def setup_openrouter(api_key: str, model: str = "qwen/qwq-32b-preview"):
    """
    快速设置OpenRouter配置
    
    参数:
        api_key: OpenRouter API密钥
        model: 要使用的模型，默认为qwen/qwq-32b-preview
    """
    config.set_api_key(api_key)
    config.set_api_base("https://openrouter.ai/api/v1/")
    set_model(model)
    logger.info(f"已设置OpenRouter，使用模型: {model}")
    return config


def setup_ollama(model: str = "qwen3:8b", api_base: str = "http://localhost:11434"):
    """
    快速设置Ollama配置
    
    参数:
        model: 要使用的Ollama模型，默认为qwen2.5:7b
        api_base: Ollama API基础URL，默认为http://localhost:11434
    """
    # Ollama不需要API密钥
    config.set_api_key("")
    config.set_api_base(api_base)
    set_model(model)
    logger.info(f"已设置Ollama，使用模型: {model}，API基址: {api_base}")
    return config


def setup_openai(api_key: str, model: str = "gpt-3.5-turbo", api_base: str = "https://api.openai.com/v1/"):
    """
    快速设置OpenAI配置
    
    参数:
        api_key: OpenAI API密钥
        model: 要使用的模型，默认为gpt-3.5-turbo
        api_base: API基础URL，默认为官方API地址
    """
    config.set_api_key(api_key)
    config.set_api_base(api_base)
    set_model(model)
    logger.info(f"已设置OpenAI，使用模型: {model}")
    return config


def chat(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """
    使用标准OpenAI协议进行聊天
    
    参数:
        messages: 消息列表，格式为[{"role": "user", "content": "hello"}]
        **kwargs: 其他参数，如temperature、max_tokens等
        
    返回:
        包含回复内容的字典
    """
    model = kwargs.pop("model", config.get_model())
    
    # 显示调试日志
    if config.is_debug_enabled():
        logger.info(f"使用模型: {model}")
        logger.info(f"API基础URL: {config.get_config('api_base')}")
        logger.info(f"消息内容: {messages}")
    else:
        logger.info(f"使用模型: {model}")
    
    try:
        # 创建客户端并调用API
        client = OpenAIClient()
        response = client.chat_completion(model=model, messages=messages, **kwargs)
        
        # 解析响应
        choice = response["choices"][0]
        content = choice["message"]["content"]
        
        # 构建标准响应格式
        result = {
            "content": content,
            "model": response.get("model", model),
            "usage": response.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
        }
        
        return result
        
    except Exception as e:
        logger.error(f"聊天API调用失败: {e}")
        return {"content": f"调用语言模型时出错: {str(e)}", "error": str(e)}


def complete(prompt: str, **kwargs) -> str:
    """
    使用标准OpenAI协议完成文本
    
    参数:
        prompt: 输入提示
        **kwargs: 其他参数
        
    返回:
        完成的文本
    """
    messages = [{"role": "user", "content": prompt}]
    response = chat(messages, **kwargs)
    return response["content"]


class LM:
    """
    语言模型类，兼容DSPy的LM实现
    """
    
    def __init__(self, model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        """
        初始化语言模型
        
        参数:
            model_name: 模型名称
            api_base: API基础URL
            api_key: API密钥
            **kwargs: 其他配置参数
        """
        self.model_name = model_name
        
        # 保存实例特定的配置，不修改全局配置
        self.api_base = api_base
        self.api_key = api_key
        self.config = kwargs.copy()
        
        # 只有在实例化时才更新全局配置（保持向后兼容性）
        if api_base:
            config.set_api_base(api_base)
        if api_key is not None:  # 允许空字符串作为有效值
            config.set_api_key(api_key)
            
        # 设置其他配置项
        for key, value in kwargs.items():
            config.set_config(key, value)
        
        # 设置当前模型
        set_model(self.model_name)
        
        logger.info(f"已初始化LM: {self.model_name}, API Base: {api_base or '(使用全局配置)'}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        使用语言模型进行聊天
        
        参数:
            messages: 消息列表
            **kwargs: 其他参数
            
        返回:
            聊天响应
        """
        # 使用实例特定的配置
        client = OpenAIClient(
            api_base=self.api_base,
            api_key=self.api_key
        )
        
        params = {"temperature": 0.7, **kwargs}
        
        # 如果配置中有api_version，传递给chat_completion
        if hasattr(self, 'config') and 'api_version' in self.config:
            params["api_version"] = self.config["api_version"]
        
        try:
            response = client.chat_completion(
                model=self.model_name,
                messages=messages,
                **params
            )
            
            # 解析响应
            choice = response["choices"][0]
            content = choice["message"]["content"]
            
            # 构建标准响应格式
            result = {
                "content": content,
                "model": response.get("model", self.model_name),
                "usage": response.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                })
            }
            
            return result
            
        except Exception as e:
            logger.error(f"LM实例聊天调用失败: {e}")
            return {"content": f"调用语言模型时出错: {str(e)}", "error": str(e)}
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        使用语言模型完成文本
        
        参数:
            prompt: 输入提示
            **kwargs: 其他参数
            
        返回:
            完成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, **kwargs)
        return response["content"]
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        直接调用语言模型完成文本
        
        参数:
            prompt: 输入提示
            **kwargs: 其他参数
            
        返回:
            完成的文本
        """
        return self.complete(prompt, **kwargs) 