"""
简单示例程序，展示如何使用miniReAct框架创建一个计算器智能体
"""
import os
import sys
from loguru import logger

from llm_hub import MultiLLMHub
# 添加上级目录到模块搜索路径，以便导入miniReAct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniReact import ReAct,streamify,Signature,InputField,OutputField,streaming_response


# 定义一些计算工具
def add(a: float, b: float) -> float:
    """将两个数相加"""
    return a + b

def subtract(a: float, b: float) -> float:
    """从第一个数中减去第二个数"""
    return a - b

def multiply(a: float, b: float) -> float:
    """将两个数相乘"""
    return a * b

def divide(a: float, b: float) -> float:
    """将第一个数除以第二个数"""
    if b == 0:
        return "错误：除数不能为零"
    return a / b

# 定义任务签名
calculator_signature = Signature(
    {"expression": InputField(desc="要计算的数学表达式")},
    {"result": OutputField(desc="计算结果"), "explanation": OutputField(desc="计算过程的解释")},
    instructions="你是一个计算器智能体，能够解析并计算数学表达式。你需要把复杂的表达式分解成简单的步骤，并使用提供的工具进行计算。"
)


def main():
    # 创建工具列表
    tools = [add, subtract, multiply, divide]
    lm = MultiLLMHub().setup_azure_openai()
    # 创建ReAct智能体
    agent = ReAct(signature=calculator_signature, tools=tools,max_iters=10,lm=lm)
    
    logger.info("欢迎使用计算器智能体！")
    logger.info("输入数学表达式，如 '3 + 4 * 2'，或输入 'exit' 退出")
    
    # while True:
    user_input = input("\n表达式> ")
    # if user_input.lower() in ('exit', 'quit', 'q'):
    #     break
    
    # 使用智能体处理表达式
    try:
        result = agent(expression=user_input)
        logger.info(f"\n问题:{user_input}")
        logger.info(f"\n结果: {result.result}")
        logger.info(f"解释: {result.explanation}")
        
        # 可选：打印轨迹信息，用于调试
        if '--debug' in sys.argv:
            logger.info("\n轨迹详情:")
            for key, value in result.trajectory.items():
                if key.startswith('thought'):
                    logger.info(f"\n思考: {value}")
                elif key.startswith('tool_name'):
                    idx = key.split('_')[-1]
                    logger.info(f"工具: {value}")
                    logger.info(f"参数: {result.trajectory.get(f'tool_args_{idx}', {})}")
                    logger.info(f"观察: {result.trajectory.get(f'observation_{idx}', '')}")
            
    except Exception as e:
        logger.error(f"处理表达式时出错: {e}")


async def process_stream_to_console(generator):
    """将流式响应输出到控制台"""
    async for chunk in generator:
        if hasattr(chunk, '__str__'):
            logger.info("-"*50)
            logger.warning(f"\n{chunk}")
        else:
            logger.info("&"*50)
            logger.warning(f"\n未知数据类型: {type(chunk)}")

async def process_stream_to_api(generator):
    """将流式响应转换为API格式并输出"""
    async for chunk in streaming_response(generator):
        logger.info("-"*50)
        logger.warning(chunk.strip())

async def stream_main():
    tools = [add, subtract, multiply, divide]
    lm = MultiLLMHub().setup_azure_openai()
    # 创建ReAct智能体
    react = ReAct(signature=calculator_signature, tools=tools,max_iters=10,lm=lm)
    stream_react = streamify(react)
    question = "30/(5+1) + 4 * 2-10"
    await process_stream_to_api(stream_react(expression=question))
    
            
if __name__ == "__main__":
    import asyncio
    
    # 启动主程序
    # main() 

    asyncio.run(stream_main())
