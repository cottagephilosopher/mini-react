"""
集成测试，测试优化后的框架
"""
import os
import sys
from pathlib import Path
from llm_hub import MultiLLMHub

# 添加项目根目录到模块搜索路径
# sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import miniReact as mr
from  miniReact.predict import prediction_cache

def test_calculator_agent():
    """测试计算器智能体"""
    # 定义计算工具
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
    
    # 创建工具列表
    tools = [add, subtract, multiply, divide]
    lm = MultiLLMHub().setup_azure_openai()
    # 创建ReAct智能体（中文模式）
    agent = mr.ReAct(
        signature=calculator_signature, 
        tools=tools,
        max_iters=10,
        lm=lm
    )
    
    # 测试简单表达式
    result = agent(expression="3 + 9 * 2 - 4")
    print(f"表达式: 3 + 9 * 2 - 4")
    print(f"结果: {result.result}")
    print(f"解释: {result.explanation}")
    print("\n轨迹详情:")
    for key, value in result.trajectory.items():
        if key.startswith('thought'):
            print(f"\n思考: {value}")
        elif key.startswith('tool_name'):
            idx = key.split('_')[-1]
            print(f"工具: {value}")
            print(f"参数: {result.trajectory.get(f'tool_args_{idx}', {})}")
            print(f"观察: {result.trajectory.get(f'observation_{idx}', '')}")
    
    # 测试复杂表达式
    result = agent(expression="6+3*(2/1+1)")
    print(f"\n表达式: 6+3*(2/1+1)")
    print(f"结果: {result.result}")
    print(f"解释: {result.explanation}")
    print("*"*30)
    result = agent(expression="(10 + 2) * 3 / 2 - 5")
    print(f"\n表达式: (10 + 2) * 3 / 2 - 5")
    print(f"结果: {result.result}")
    print(f"解释: {result.explanation}")
    

if __name__ == "__main__":
    # 启用调试模式
    # mr.enable_debug()
    
    # 运行测试
    test_calculator_agent()