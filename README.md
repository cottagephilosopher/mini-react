


# miniReAct

轻量级的 ReAct (Reasoning and Acting) 智能体框架实现。

## 项目概述

miniReact 是一个轻量级的推理和行动（Reasoning and Acting）框架实现，提供了创建能够思考、使用工具和采取行动的智能体所需的基本组件。这个框架基于 ReAct 范式，允许智能体通过交替的思考-行动-观察循环来解决复杂问题。

## 简介

miniReact 是一个简化的 ReAct 框架实现，允许您创建能够推理和执行操作的智能体。该框架基于思维链（Chain-of-Thought）的概念，使智能体能够逐步思考问题，选择和执行适当的工具，并根据结果做出决策。

## 安装

```bash
pip install -e .
```

## 基本用法

```python
import miniReact as mr

# 定义一个签名（任务描述和输入输出规范）
signature = mr.Signature(
    {"query": mr.InputField()},
    {"answer": mr.OutputField()},
    instructions="回答用户的问题"
)

# 定义工具函数
def search_tool(query: str):
    """搜索工具"""
    # 实际实现会连接到搜索API
    return f"搜索结果: {query}相关信息"

# 创建ReAct智能体
agent = mr.ReAct(signature, tools=[search_tool])

# 使用智能体
result = agent(query="今天的天气如何？")
print(result.answer)
```

## 核心组件

1. **模块系统**
   - `Module`: 所有组件的基类，提供基本的调用接口

2. **工具系统**
   - `Tool`: 封装可调用函数的类，自动处理参数和文档

3. **签名系统**
   - `Signature`: 定义任务的输入、输出和指令
   - `InputField`/`OutputField`: 表示签名中的字段

4. **预测系统**
   - `Predict`: 基本的预测模块，调用语言模型生成响应
   - `ChainOfThought`: 增强版预测，鼓励模型展示思维过程

5. **ReAct 系统**
   - `ReAct`: 核心类，实现推理-行动-观察循环

## 工程结构

```
mini-react/
├── miniReact/
│   ├── __init__.py      # 包初始化和导出
│   ├── module.py        # 模块基类
│   ├── tool.py          # 工具类
│   ├── signature.py     # 签名系统
│   ├── predict.py       # 预测模块
│   └── react.py         # ReAct核心实现
├── examples/
│   ├── simple_demo.py   # 简单计算器示例
│   └── search_demo.py   # 搜索问答示例
├── setup.py             # 安装配置
├── requirements.txt     # 依赖需求
└── README.md            # 项目说明
```

## 主要特性

1. **轻量级设计**: 整个框架保持简洁，核心功能仅依赖 `litellm` 库进行 LLM 交互。

2. **灵活的工具系统**: 自动从函数中提取参数定义和文档，用户可以轻松创建和集成自定义工具。

3. **轨迹管理**: 自动管理和格式化智能体的思考和行动过程，并在轨迹过长时进行智能截断。

4. **错误处理**: 对工具调用错误和上下文长度超出等情况进行处理，提高系统鲁棒性。

5. **完整中文注释**: 所有代码均提供了详细的中文注释，便于理解和学习。

## 使用示例

1. **计算器智能体**: 展示了如何创建一个简单的计算工具和使用 ReAct 框架进行数学表达式计算。

2. **搜索问答智能体**: 展示了如何创建更复杂的多步骤工具使用流程，包括搜索和阅读文档。

## 扩展方向

1. **增加更多适配器**: 支持更多的语言模型和接口。

2. **添加记忆功能**: 为智能体添加长期记忆能力。

3. **增加工具发现**: 允许智能体自动发现和选择可用工具。

4. **实现更多高级功能**: 如工具组合、自我反思等能力。

## 许可证

MIT

        