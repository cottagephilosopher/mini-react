"""
综合测试多种LLM的类，包括Azure OpenAI、本地Ollama和OpenRouter等
"""
import os
import sys
from loguru import logger
from typing import Dict, Any, Optional
import dotenv
dotenv.load_dotenv(override=True)

# 添加上级目录到模块搜索路径，以便导入minireact
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入minireact
import minireact as mr

class MultiLLMHub:
    """
    综合测试多种LLM的类
    支持测试Azure OpenAI、本地Ollama、OpenRouter等多种LLM服务
    """
    
    def __init__(self, debug: bool = False):
        """
        初始化多LLM测试器
        
        参数:
            debug: 是否启用调试模式
        """
        self.lm_instances = {}  # 存储不同的LM实例
        
        # 启用调试模式
        if debug:
            mr.enable_debug()
            logger.info("已启用调试模式")
    
    def setup_openrouter(self, 
                        api_key: str = None, 
                        model: str = None,
                        api_base: str = None) -> mr.LM:
        """
        设置OpenRouter LLM
        
        参数:
            api_key: OpenRouter API密钥
            model: 要使用的模型
            api_base: API基础URL
            
        返回:
            LM实例
        """
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTE_API_KEY")
            if not api_key:
                logger.warning("未提供OpenRouter API密钥，也未在环境变量中找到")
        if not api_base:
            api_base = os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENROUTE_BASE_URL", "https://openrouter.ai/api/v1/")
        if not model:
            model = os.environ.get("OPENROUTER_MODEL_NAME") or os.environ.get("OPENROUTE_MODEL_NAME", "qwen/qwq-32b-preview")

        logger.info(f"设置OpenRouter LLM，模型: {model}")
        
        # 直接使用模型名称，不添加前缀
        lm = mr.LM(
            model,
            api_base=api_base,
            api_key=api_key
        )
        
        # 存储实例
        self.lm_instances["openrouter"] = lm
        return lm
    
    def setup_openai(self, 
                    api_key: Optional[str] = None, 
                    model: str = None,
                    api_base: Optional[str] = None) -> mr.LM:
        """
        设置OpenAI LLM
        
        参数:
            api_key: OpenAI API密钥，如果为None则尝试从环境变量获取
            model: 要使用的模型
            api_base: API基础URL，如果为None则使用默认值
            
        返回:
            LM实例
        """
        # 如果未提供API密钥，尝试从环境变量获取
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("未提供OpenAI API密钥，也未在环境变量中找到")
        if not api_base:
            api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
        if not model:
            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        
        logger.info(f"设置OpenAI LLM，模型: {model}")
        
        # 直接使用模型名称，标准OpenAI协议不需要前缀
        lm = mr.LM(
            model,
            api_base=api_base,
            api_key=api_key
        )
        
        # 存储实例
        self.lm_instances["openai"] = lm
        return lm
    
    def setup_dashscope(self, 
                    api_key: Optional[str] = None, 
                    model: str = None,
                    api_base: Optional[str] = None) -> mr.LM:
        """
        设置阿里云DashScope LLM（兼容OpenAI协议）
        
        参数:
            api_key: DashScope API密钥，如果为None则尝试从环境变量获取
            model: 要使用的模型
            api_base: API基础URL，如果为None则使用默认值
            
        返回:
            LM实例
        """
        # 如果未提供API密钥，尝试从环境变量获取
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALI_API_KEY")
            if not api_key:
                logger.warning("未提供DashScope API密钥，也未在环境变量中找到")
        if not api_base:
            api_base = os.environ.get("DASHSCOPE_BASE_URL") or os.environ.get("ALI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/")
        if not model:
            model = os.environ.get("DASHSCOPE_MODEL_NAME") or os.environ.get("ALI_MODEL_NAME", "qwen-turbo")
        
        logger.info(f"设置阿里云DashScope LLM，模型: {model}")
        
        # 直接使用模型名称，不添加前缀
        lm = mr.LM(
            model,
            api_base=api_base,
            api_key=api_key
        )
        
        # 存储实例
        self.lm_instances["dashscope"] = lm
        return lm
    
    def setup_azure_openai(self, 
                          api_key: str = None, 
                          api_base: str = None,
                          deployment_name: str = "gpt-4o",
                          api_version: str = "2024-05-01-preview") -> mr.LM:
        """
        设置Azure OpenAI LLM
        
        注意：Azure OpenAI 使用不同的API格式，需要特殊处理
        
        参数:
            api_key: Azure OpenAI API密钥
            api_base: Azure OpenAI资源端点（如：https://your-resource.openai.azure.com）
            deployment_name: Azure部署名称
            api_version: API版本
            
        返回:
            LM实例
        """
        if not api_key:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                logger.warning("未提供Azure OpenAI API密钥，也未在环境变量中找到")
        if not api_base:
            api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not api_base:
                logger.warning("未提供Azure OpenAI API基础URL，也未在环境变量中找到")
        if not deployment_name:
            deployment_name = os.environ.get("AZURE_MODEL_NAME", "gpt-4o")

        # 构建正确的Azure OpenAI API端点
        # Azure格式: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}
        if api_base and not api_base.endswith('/'):
            api_base += '/'
        
        azure_api_base = f"{api_base}openai/deployments/{deployment_name}/"
        
        logger.info(f"设置Azure OpenAI LLM，部署: {deployment_name}")
        logger.info(f"Azure API端点: {azure_api_base}")
        
        # 对于Azure，我们直接使用deployment_name作为模型名
        lm = mr.LM(
            deployment_name,
            api_base=azure_api_base,
            api_key=api_key,
            api_version=api_version
        )
        
        # 存储实例
        self.lm_instances["azure"] = lm
        return lm
    
    def setup_ollama(self, 
                    model: str = None, 
                    api_base: str = "http://localhost:11434") -> mr.LM:
        """
        设置Ollama本地LLM
        
        参数:
            model: 要使用的Ollama模型
            api_base: Ollama API基础URL
            
        返回:
            LM实例
        """
        if not model:
            model = os.environ.get("OLLAMA_MODEL_NAME", "qwen3:8b")

        logger.info(f"设置Ollama LLM，模型: {model}")
        
        try:
            # Ollama兼容OpenAI协议，直接使用模型名称
            lm = mr.LM(
                model,
                api_base=api_base,
                api_key=""  # Ollama通常不需要API密钥
            )
            
            # 存储实例
            self.lm_instances["ollama"] = lm
            return lm
        except Exception as e:
            logger.error(f"设置Ollama出错: {e}")
            logger.info("要使用Ollama，请先安装并运行Ollama服务")
            return None
    
    def setup_custom_openai_compatible(self,
                                     provider_name: str,
                                     api_key: str,
                                     api_base: str,
                                     model: str) -> mr.LM:
        """
        设置自定义的兼容OpenAI协议的LLM提供商
        
        参数:
            provider_name: 提供商名称（用于标识）
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            
        返回:
            LM实例
        """
        logger.info(f"设置自定义OpenAI兼容提供商: {provider_name}，模型: {model}")
        
        lm = mr.LM(
            model,
            api_base=api_base,
            api_key=api_key
        )
        
        # 存储实例
        self.lm_instances[provider_name.lower()] = lm
        return lm
    
    def check_llm(self, 
                llm_type: str, 
                prompt: str = "请用中文写一首关于人工智能的短诗", 
                chat_message: str = "你好，请简单介绍一下自己",
                temperature: float = 0.7) -> Dict[str, Any]:
        """
        测试指定类型的LLM
        
        参数:
            llm_type: LLM类型，如'openrouter', 'openai', 'azure', 'ollama', 'dashscope'
            prompt: 用于complete方法的提示
            chat_message: 用于chat方法的消息
            temperature: 温度参数
            
        返回:
            测试结果字典
        """
        results = {}
        
        if llm_type not in self.lm_instances:
            logger.error(f"未找到类型为 {llm_type} 的LLM实例，请先设置")
            return {"error": f"未找到类型为 {llm_type} 的LLM实例"}
        
        lm = self.lm_instances[llm_type]
        
        # 测试complete方法
        print(f"\n=== 测试 {llm_type} 的complete方法 ===")
        print(f"发送提示: {prompt}")
        
        try:
            response = lm.complete(prompt, temperature=temperature)
            print("\n收到回复:\n", response)
            results["complete"] = response
        except Exception as e:
            error_msg = f"complete方法出错: {e}"
            print(f"\n{error_msg}")
            results["complete_error"] = error_msg
        
        # 测试chat方法
        print(f"\n=== 测试 {llm_type} 的chat方法 ===")
        messages = [
            {"role": "user", "content": chat_message}
        ]
        
        print(f"发送消息: {chat_message}")
        
        try:
            response = lm.chat(messages, temperature=temperature)
            print("\n收到回复:", response["content"])
            results["chat"] = response
        except Exception as e:
            error_msg = f"chat方法出错: {e}"
            print(f"\n{error_msg}")
            results["chat_error"] = error_msg
        
        return results
    
    def check_all(self, 
                prompt: str = "请用中文写一首关于人工智能的短诗", 
                chat_message: str = "你好，请简单介绍一下自己",
                temperature: float = 0.7) -> Dict[str, Dict[str, Any]]:
        """
        测试所有已设置的LLM
        
        参数:
            prompt: 用于complete方法的提示
            chat_message: 用于chat方法的消息
            temperature: 温度参数
            
        返回:
            所有测试结果的字典
        """
        all_results = {}
        
        if not self.lm_instances:
            logger.warning("没有设置任何LLM实例，请先设置")
            return {"error": "没有设置任何LLM实例"}
        
        for llm_type in self.lm_instances:
            print(f"\n\n{'='*50}")
            print(f"测试 {llm_type} LLM")
            print(f"{'='*50}")
            
            results = self.check_llm(llm_type, prompt, chat_message, temperature)
            all_results[llm_type] = results
        
        return all_results
    
    def list_available_providers(self):
        """列出所有可用的提供商设置方法"""
        providers = {
            "openai": "setup_openai() - OpenAI官方API",
            "openrouter": "setup_openrouter() - OpenRouter多模型平台", 
            "dashscope": "setup_dashscope() - 阿里云DashScope",
            "azure": "setup_azure_openai() - Azure OpenAI服务",
            "ollama": "setup_ollama() - Ollama本地部署",
            "custom": "setup_custom_openai_compatible() - 自定义OpenAI兼容提供商"
        }
        
        print("支持的LLM提供商:")
        for key, desc in providers.items():
            print(f"  {key}: {desc}")
        
        return providers