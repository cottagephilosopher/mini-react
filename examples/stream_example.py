"""
使用streamify流式返回功能的示例
"""

import asyncio
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_hub import MultiLLMHub
from minireact import (
    ReAct, Signature, InputField, OutputField, Tool,
    streamify, streaming_response
)

# 定义一个简单的搜索工具
def search(query: str) -> str:
    """模拟搜索引擎，根据查询返回结果"""
    if "天气" in query:
        return "今天天气晴朗，气温25度，适合户外活动。"
    elif "时间" in query:
        return "当前北京时间是14:30。"
    elif "日期" in query:
        return "今天是2023年6月15日。"
    else:
        return f"没有找到关于'{query}'的相关信息。"

# 定义一个计算器工具
def calculate(expression: str) -> str:
    """执行简单的数学计算"""
    try:
        # 安全地评估表达式
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

# 创建任务签名
signature = (
    Signature(
        {
            "question": InputField(desc="用户的问题"),
        },
        {
            "answer": OutputField(desc="问题的最终答案"),
            "reasoning": OutputField(desc="解决问题的推理过程"),
        },
        "你是一个智能助手，能够回答用户的问题。请思考如何使用工具解决问题，然后给出推理过程和最终答案。"
    )
)

lm = MultiLLMHub().setup_azure_openai()
# 创建ReAct实例
react = ReAct(
    signature=signature,
    tools=[search,calculate],
    max_iters=5,
    lm=lm
)

# 使用streamify包装ReAct实例
stream_react = streamify(react)

async def process_stream_to_console(generator):
    """将流式响应输出到控制台"""
    async for chunk in generator:
        if hasattr(chunk, '__str__'):
            print(f"\n{chunk}")
        else:
            print(f"\n未知数据类型: {type(chunk)}")

async def process_stream_to_api(generator):
    """将流式响应转换为API格式并输出"""
    async for chunk in streaming_response(generator):
        print(chunk.strip())

async def main():
    # 要提问的问题
    questions = [
        "今天北京的天气怎么样？",
        "计算125乘以37是多少？",
        "今天是几号？"
    ]
    
    for question in questions:
        print(f"\n\n===== 问题: {question} =====")
        print("\n--- 控制台输出格式 ---")
        # 创建流式响应并输出到控制台
        await process_stream_to_console(stream_react(question=question))
        
        print("\n--- API输出格式 ---")
        # 创建流式响应并以API格式输出
        await process_stream_to_api(stream_react(question=question))

if __name__ == "__main__":
    asyncio.run(main()) 