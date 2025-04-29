"""
语言模型管理模块，通过litellm支持多种LLM
"""
import os
from loguru import logger
from typing import Any, Dict, List, Optional, Union



try:
    import litellm
    from litellm import completion as litellm_completion
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    raise ImportError("miniReAct依赖litellm库。请安装: pip install litellm")



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
        return self._config.get("model", "qwen/qwq-32b:free")
    
    def set_api_base(self, api_base: str):
        """设置API基础URL"""
        # 处理OpenRouter的特殊情况
        if "openrouter" in api_base.lower():
            # 保留完整URL格式，包括/v1，因为OpenRouter需要它
            self._config["api_base"] = api_base
            logger.info(f"设置OpenRouter API基址: {api_base}")
            
            # 同时更新litellm的设置
            if hasattr(litellm, "api_base"):
                litellm.api_base = api_base
            return
            
        # 确保API base不以/v1结尾，避免litellm重复添加
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]
            logger.info(f"API base已修改为: {api_base} (删除了尾部的/v1)")
            
        self._config["api_base"] = api_base
        # 同时更新litellm的设置
        if hasattr(litellm, "api_base"):
            litellm.api_base = api_base
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self._config["api_key"] = api_key
        # 同时更新litellm的设置
        if hasattr(litellm, "api_key"):
            litellm.api_key = api_key
    
    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self._config[key] = value
        # 尝试更新litellm的相应设置
        if hasattr(litellm, key):
            setattr(litellm, key, value)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)
    
    def enable_debug(self):
        """启用调试模式"""
        self._debug = True
        if HAS_LITELLM:
            # 启用litellm的调试模式
            litellm._turn_on_debug()
            logger.info("已启用LiteLLM调试模式")
    
    def disable_debug(self):
        """禁用调试模式"""
        self._debug = False
        # litellm没有直接的方法关闭调试，所以我们只能设置本地状态
        
    def is_debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        return self._debug
    
    def from_env(self):
        """从环境变量加载配置"""
        # 加载模型名称
        model_name = os.environ.get("LLM_MODEL", "qwen/qwq-32b:free")
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
            if key.startswith("LLM_") and key != "LLM_MODEL":
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


def setup_openrouter(api_key: str, model: str = "qwen/qwq-32b:free"):
    """
    快速设置OpenRouter配置
    
    参数:
        api_key: OpenRouter API密钥
        model: 要使用的模型，默认为qwen/qwq-32b:free
    """
    config.set_api_key(api_key)
    # 确保使用完整的API基址，包括/v1
    config.set_api_base("https://openrouter.ai/api/v1")
    
    # 确保模型名称有正确的前缀
    if not model.startswith("openrouter/") and "/" in model:
        model = f"openrouter/{model}"
        logger.info(f"已将模型名称更新为: {model}")
        
    set_model(model)
    logger.info(f"已设置OpenRouter，使用模型: {model}")
    
    return config


def setup_ollama(model: str = "qwen3:8b", api_base: str = "http://localhost:11434"):
    """
    快速设置Ollama配置
    
    参数:
        model: 要使用的Ollama模型，默认为qwen3:8b
        api_base: Ollama API基础URL，默认为http://localhost:11434
    """
    # Ollama不需要API密钥
    config.set_api_key("")
    config.set_api_base(api_base)
    # 确保模型名称格式正确
    model_name = format_model_name(model)
    set_model(model_name)
    logger.info(f"已设置Ollama，使用模型: {model_name}，API基址: {api_base}")
    
    return config


def format_model_name(model: str) -> str:
    """
    格式化模型名称，使其适合litellm使用
    
    参数:
        model: 原始模型名称
        
    返回:
        格式化后的模型名称
    """
    # 检查是否已经有提供商前缀
    if model.startswith(("openai/", "anthropic/", "huggingface/", "openrouter/", "ollama/")):
        return model
    
    # 特定模型的映射
    model_mappings = {
        # OpenAI 模型
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        
        # Anthropic 模型
        "claude-3-opus": "anthropic/claude-3-opus-20240229",
        "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        
        # Ollama 模型
        "qwq": "ollama/qwen3:8b",
        "gemma3": "ollama/gemma3:12b",
        "deepseek-r1": "ollama/gemma3:12b",
        "qwen2.5": "ollama/qwen2.5:1.5b",
    }
    
    # 如果是已知模型，直接返回映射
    if model in model_mappings:
        return model_mappings[model]
    
    # 根据特殊格式处理模型名称
    if "/" in model:
        # 处理格式如 "qwen/qwq-32b" 或 "qwen/qwq-32b:free"
        provider, rest = model.split("/", 1)
        
        # 去掉版本号等后缀，如 ":free"
        if ":" in rest:
            model_id = rest.split(":", 1)[0]
        else:
            model_id = rest
            
        # 为不同提供商添加正确的前缀
        if provider.lower() == "qwen":
            return f"qwen/{model_id}"
        elif provider.lower() == "anthropic":
            return f"anthropic/{model_id}"
        elif provider.lower() == "openai":
            return f"openai/{model_id}"
        elif provider.lower() == "huggingface":
            return f"huggingface/{model_id}"
        elif provider.lower() == "ollama":
            return f"ollama/{model_id}"
        else:
            # 其他模型使用 OpenRouter 前缀
            return f"openrouter/{provider}/{model_id}"
    
    # 检查是否为Ollama模型简写
    if model.startswith(("llama", "mistral", "gemma", "phi", "qwen")):
        return f"ollama/{model}"
    
    # 默认使用 OpenRouter 前缀
    return f"openrouter/{model}"


def chat(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """
    使用litellm进行聊天
    
    参数:
        messages: 消息列表，格式为[{"role": "user", "content": "hello"}]
        **kwargs: 其他参数，如temperature、max_tokens等
        
    返回:
        包含回复内容的字典
    """
    # 获取模型名称，不再进行格式化处理，让litellm自己处理
    model = kwargs.pop("model", config.get_model())
    
    # 显示调试日志
    if config.is_debug_enabled():
        logger.info(f"使用模型: {model}")
        logger.info(f"API基础URL: {config.get_config('api_base')}")
        logger.info(f"消息内容: {messages}")
    else:
        logger.info(f"使用模型: {model}")
    
    try:
        # 直接调用litellm，让它处理模型适配
        response = litellm_completion(model=model, messages=messages, **kwargs)
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0,
            } if hasattr(response, "usage") else {}
        }
    except Exception as e:
        logger.error(f"litellm调用失败: {e}")
        # 返回错误信息
        return {"content": f"调用语言模型时出错: {str(e)}", "error": str(e)}


def complete(prompt: str, **kwargs) -> str:
    """
    使用litellm完成文本
    
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
    语言模型类，类似于DSPy的LM实现
    """
    
    def __init__(self, model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        """
        初始化语言模型
        
        参数:
            model_name: 模型名称，如'ollama/llama3.2'或'ollama_chat/llama3.2'
            api_base: API基础URL，如'http://localhost:11434'
            api_key: API密钥
            **kwargs: 其他配置参数
        """
        # 处理DSPy风格的模型名称
        if "_chat/" in model_name:
            # 处理如'ollama_chat/llama3.2'格式
            provider_type, model_id = model_name.split("_chat/", 1)
            if provider_type.lower() == "ollama":
                self.model_name = f"ollama/{model_id}"
            else:
                # 其他情况尝试将_chat替换为/
                self.model_name = model_name.replace("_chat/", "/")
        else:
            # 直接使用原始模型名称，让litellm处理它
            self.model_name = model_name
        
        # 为Ollama特殊处理
        if api_base and "localhost" in api_base and "ollama" in self.model_name.lower():
            logger.info(f"检测到Ollama本地部署，使用模型: {self.model_name}")
            # litellm直接支持ollama格式
            if not self.model_name.startswith("ollama/"):
                if "/" in self.model_name:
                    _, model_id = self.model_name.split("/", 1)
                    self.model_name = f"ollama/{model_id}"
                else:
                    self.model_name = f"ollama/{self.model_name}"
        
        # 为OpenRouter特殊处理
        elif api_base and "openrouter" in api_base.lower():
            logger.info(f"检测到OpenRouter API，使用模型: {self.model_name}")
            # 确保有正确的提供商前缀
            if not self.model_name.startswith("openrouter/") and "/" in self.model_name:
                # 如果是提供商/模型的格式，但没有openrouter前缀
                self.model_name = f"openrouter/{self.model_name}"
                logger.info(f"已将模型名称更新为: {self.model_name}")
                    
        # 设置API基础URL和密钥
        if api_base:
            config.set_api_base(api_base)
        if api_key is not None:  # 允许空字符串作为有效值
            config.set_api_key(api_key)
            
        # 设置其他配置项
        for key, value in kwargs.items():
            config.set_config(key, value)
        
        # 设置当前模型
        set_model(self.model_name)
        
        logger.info(f"已初始化LM: {self.model_name}, API Base: {api_base or '(默认)'}")
    
    def _parse_model_name(self, model_name: str) -> str:
        """
        已弃用：现在在__init__中直接处理模型名称
        仅保留此方法以保持兼容性
        """
        return model_name
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        使用语言模型进行聊天
        
        参数:
            messages: 消息列表
            **kwargs: 其他参数
            
        返回:
            聊天响应
        """
        # 传递当前设置的模型
        return chat(messages, model=self.model_name, **kwargs)
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        使用语言模型完成文本
        
        参数:
            prompt: 输入提示
            **kwargs: 其他参数
            
        返回:
            完成的文本
        """
        return complete(prompt, model=self.model_name, **kwargs)
    
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