import os
import sys

# 添加上级目录到模块搜索路径，以便导入minireact
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入minireact
from llm_hub import MultiLLMHub


def example_llm_function() :
    # 方式1：使用老方法
    # mr.set_model("openai/gpt-3.5-turbo")  # 使用标准模型名称格式
    # mr.lm_config.set_api_key("sk-or-v1-36f833775045")
    # mr.lm_config.set_api_base("https://openrouter.ai/api")

     # 方式2：使用便捷函数
    # mr.setup_openrouter(
    #    api_key="sk-or-v1-36f8337750452a78b",
    #    model="openai/gpt-3.5-turbo"
    # )

    pass

if __name__ == "__main__":
    """主函数，演示如何使用MultiLLMllm_checker类"""
    
    # 创建测试器实例
    llm_checker = MultiLLMHub(debug=False)
    
    # 设置OpenRouter
    # llm_checker.setup_openrouter()
    # llm_checker.check_llm("openrouter")

    # 设置Ollama
    llm_checker.setup_ollama()
    # llm_checker.check_llm("ollama")

    # 设置Azure OpenAI
    # llm_checker.setup_azure_openai()
    # llm_checker.check_llm("azure")

    # 设置阿里dashscope
    # llm_checker.setup_dashscope()
    # llm_checker.check_llm("dashscope")

    # # 设置OpenAI (如果有API密钥)
    # try:
    #     llm_checker.setup_openai(
    #         api_key="sk-f2c0224566a94b0",
    #         model="qwen-max-2025-01-25",
    #         api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"           
    #     )
    # except Exception as e:
    #     logger.warning(f"设置OpenAI失败: {e}")
    
    # # 测试所有已设置的LLM
    llm_checker.check_all()
