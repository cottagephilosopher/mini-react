"""
简单示例程序，展示如何使用miniReAct框架创建一个计算器智能体
"""
import os
import sys
import logging
# 设置语言模型配置
from llm_hub import MultiLLMHub
# 添加上级目录到模块搜索路径，以便导入miniReAct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import miniReact as mr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
calculator_signature = mr.Signature(
    {"expression": mr.InputField(desc="要计算的数学表达式")},
    {"result": mr.OutputField(desc="计算结果"), "explanation": mr.OutputField(desc="计算过程的解释")},
    instructions="你是一个计算器智能体，能够解析并计算数学表达式。你需要把复杂的表达式分解成简单的步骤，并使用提供的工具进行计算。"
)

def main():
    # 创建工具列表
    tools = [add, subtract, multiply, divide]
    lm = MultiLLMHub().setup_azure_openai()
    # 创建ReAct智能体
    agent = mr.ReAct(signature=calculator_signature, tools=tools,lm=lm)
    
    print("欢迎使用计算器智能体！")
    print("输入数学表达式，如 '3 + 4 * 2'，或输入 'exit' 退出")
    
    # while True:
    user_input = input("\n表达式> ")
    # if user_input.lower() in ('exit', 'quit', 'q'):
    #     break
    
    # 使用智能体处理表达式
    try:
        result = agent(expression=user_input)
        print(f"\n问题:{user_input}")
        print(f"\n结果: {result.result}")
        print(f"解释: {result.explanation}")
        
        # 可选：打印轨迹信息，用于调试
        if '--debug' in sys.argv:
            print("\n轨迹详情:")
            for key, value in result.trajectory.items():
                if key.startswith('thought'):
                    print(f"\n思考: {value}")
                elif key.startswith('tool_name'):
                    idx = key.split('_')[-1]
                    print(f"工具: {value}")
                    print(f"参数: {result.trajectory.get(f'tool_args_{idx}', {})}")
                    print(f"观察: {result.trajectory.get(f'observation_{idx}', '')}")
            
    except Exception as e:
        print(f"处理表达式时出错: {e}")
            
if __name__ == "__main__":
    
    
    # 启动主程序
    main() 