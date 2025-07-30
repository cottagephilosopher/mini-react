
# minireact

一个轻量级的 ReAct（Reasoning and Acting）智能体框架，让您轻松构建能够思考、使用工具并采取行动的智能代理。 [1](#0-0) 

## ✨ 特性

- **🚀 轻量级设计**：核心功能仅依赖少量库，保持框架简洁高效
- **🔧 灵活工具系统**：自动从函数中提取参数和文档，轻松集成自定义工具
- **🧠 智能轨迹管理**：自动管理推理过程，支持上下文窗口智能截断
- **🌊 流式响应**：支持实时流式输出，提供更好的用户体验
- **🔌 多模型支持**：支持 OpenAI、Ollama、OpenRouter 等多种语言模型提供商
- **💾 智能缓存**：内置预测结果缓存，提升性能并降低 API 成本
- **🛡️ 错误恢复**：完善的错误处理机制，确保系统稳定运行

## 🏗️ 核心架构

minireact 基于模块化架构设计，所有组件都继承自统一的 `Module` 基类： [2](#0-1) 

```
minireact/
├── Module          # 所有组件的基类
├── ReAct           # 核心推理-行动循环
├── Tool            # 工具封装和执行
├── Signature       # 输入输出规范定义
├── Predict         # 语言模型交互
├── LM              # 多提供商语言模型接口
└── streamify       # 流式响应包装器
```

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

### 基础用法

```python
import minireact as mr

# 定义任务签名
signature = mr.Signature(
    {"query": mr.InputField()},
    {"answer": mr.OutputField()},
    instructions="回答用户的问题"
)

# 定义工具函数
def search_tool(query: str):
    """搜索相关信息"""
    return f"搜索到关于 {query} 的相关信息"

def calculator(expression: str):
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 创建 ReAct 智能体
agent = mr.ReAct(signature, tools=[search_tool, calculator])

# 执行任务
result = agent(query="计算 15 * 23 + 45")
print(result.answer)
```

### 流式响应

```python
# 使用流式包装器
streaming_agent = mr.streamify(agent)

async for response in streaming_agent(query="今天天气如何？"):
    if isinstance(response, mr.ThoughtResponse):
        print(f"💭 思考: {response.thought}")
    elif isinstance(response, mr.ToolCallResponse):
        print(f"🔧 调用工具: {response.tool_name}")
    elif isinstance(response, mr.ObservationResponse):
        print(f"👀 观察: {response.observation}")
    elif isinstance(response, mr.FinishResponse):
        print(f"✅ 完成: {response.outputs}")
```

## 🔧 配置语言模型

### OpenAI

```python
import minireact as mr

# 设置 OpenAI 模型
mr.set_model("gpt-4o-mini")
mr.lm_config.set_api_key("your-openai-api-key")
```

### Ollama（本地部署）

```python
# 设置 Ollama
mr.setup_ollama(model="qwen3:8b", api_base="http://localhost:11434")
```

### OpenRouter

```python
# 设置 OpenRouter
mr.setup_openrouter(api_key="your-openrouter-key", model="qwen/qwq-32b-preview")
```

## 📚 核心组件详解

### ReAct 智能体

ReAct 类实现了推理-行动-观察的核心循环： [3](#0-2) 

```python
# 创建智能体时的关键参数
agent = mr.ReAct(
    signature=signature,      # 任务定义
    tools=[tool1, tool2],    # 可用工具列表
    max_iters=5,             # 最大迭代次数
    lm=custom_lm             # 自定义语言模型
)
```

### 工具系统

工具系统自动处理函数封装和参数验证： [4](#0-3) 

```python
# 函数会自动转换为 Tool 对象
def my_tool(param1: str, param2: int = 10):
    """工具描述"""
    return f"处理 {param1} 和 {param2}"

# 或者手动创建 Tool
tool = mr.Tool(
    func=my_function,
    name="custom_name",
    desc="自定义描述"
)
```

### 签名系统

签名定义了任务的输入输出规范： [5](#0-4) 

```python
signature = mr.Signature(
    input_fields={"question": mr.InputField()},
    output_fields={"answer": mr.OutputField()},
    instructions="根据问题提供准确答案"
)
```

## 🔄 执行流程

1. **输入验证**：根据签名验证输入参数
2. **推理循环**：执行思考-行动-观察循环
3. **工具调用**：选择并执行合适的工具
4. **轨迹管理**：记录完整的推理过程
5. **结果提取**：从轨迹中提取最终答案
6. **错误处理**：处理各种异常情况 [6](#0-5) 

## 🛠️ 高级功能

### 轨迹截断

当上下文窗口超出限制时，系统会智能截断轨迹： [7](#0-6) 

### 预测缓存

内置缓存系统提升性能： [8](#0-7) 

### 错误恢复

完善的错误处理和重试机制： [9](#0-8) 

## 📁 项目结构

```
mini-react/
├── minireact/
│   ├── __init__.py      # 包初始化和导出
│   ├── module.py        # 模块基类
│   ├── react.py         # ReAct 核心实现
│   ├── tool.py          # 工具系统
│   ├── signature.py     # 签名系统
│   ├── predict.py       # 预测模块
│   ├── lm.py           # 语言模型接口
│   ├── streamify.py    # 流式响应
│   └── prompt.py       # 提示模板
├── examples/           # 使用示例
├── setup.py           # 安装配置
└── README.md          # 项目说明
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

MIT License

---

**minireact** - 让智能体开发变得简单而强大 🚀

## Notes

这个 README 基于项目的实际代码结构和功能编写，突出了 minireact 框架的核心特性：轻量级设计、模块化架构、流式响应支持和多语言模型集成。 [10](#0-9)  相比原有的 README，新版本更加结构化，增加了更多实用的代码示例和配置说明，同时保持了中文文档的特色。

Wiki pages you might want to explore:
- [Overview (cottagephilosopher/mini-react)](/wiki/cottagephilosopher/mini-react#1)
- [Core Framework (cottagephilosopher/mini-react)](/wiki/cottagephilosopher/mini-react#2)
- [Prediction and Language Model Integration (cottagephilosopher/mini-react)](/wiki/cottagephilosopher/mini-react#2.3)