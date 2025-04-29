"""
提示模板模块，用于集中管理系统中使用的各种提示文本
"""

# ReAct框架的提示模板
react_prompts = {
    # ReAct框架的基础指导说明
    "base_instructions": [
        "你是一个智能体。在每个任务中，你将接收 {inputs} 作为输入，并能看到你之前的推理轨迹。",
        "你的目标是使用提供的工具收集必要信息，以产生 {outputs}。\n",
        "你需要在每个回合中依次提供 next_thought、next_tool_name 和 next_tool_args，以及完成任务时的总结。",
        "每次工具调用后，你将收到一个观察结果，这将添加到你的推理轨迹中。\n",
        "当写 next_thought 时，你应该分析当前情况并计划未来步骤。",
        "当选择 next_tool_name 时，必须严格从以下工具列表中选择一个，不要添加任何额外的文本、引号或括号。",
        "当提供 next_tool_args 时，必须使用有效的JSON格式，如 {{\"参数名\": \"参数值\"}}。",
        "参数必须是一个JSON对象（字典），不能是字符串或其他格式。",
        "不要使用markdown代码块（```）来包装你的回答。",
        "当任务完成时，必须使用'finish'工具来提交最终结果。",
        "可用的工具包括：\n",
    ],
    
    # finish工具的描述模板
    "finish_tool_desc": "标记任务为完成。表示已收集到足够信息，可以产生输出：{outputs}",
    
    # 工具描述的格式化模板
    "tool_desc_format": "({idx}) {name}{desc}",
}

# 预测模块的提示模板
predict_prompts = {
    # ReAct任务的系统提示
    "react_system_prompt": """你是一个AI助手，负责执行推理和行动(ReAct)任务。当需要使用工具时，请严格按照以下格式提供输出：

next_thought: 你的思考过程

next_tool_name: 工具名称

next_tool_args: {"参数名": "参数值"}

注意事项：
1. next_tool_name必须是从提供的工具列表中精确选择一个工具名称，不要添加任何其他字符，如引号、括号等
2. next_tool_args必须是有效的JSON对象格式，用大括号包裹，包含键值对
3. 每个字段必须单独成行，不要混合或嵌套字段
4. 不要在回答开头添加问候语或介绍，直接按照格式提供输出
5. 不要使用markdown代码块（```json）来包装你的回答，直接提供纯文本格式的回答
6. 不要返回完整的JSON对象，而是按照指定的格式分行提供各个字段
7. 当任务完成时，必须使用'finish'工具来提交最终结果

正确示例:
next_thought: 我需要搜索关于深度学习的信息
next_tool_name: search
next_tool_args: {"query": "深度学习"}

错误示例:
```json
{
  "next_thought": "我需要搜索关于深度学习的信息",
  "next_tool_name": "search",
  "next_tool_args": {"query": "深度学习"}
}
```

next_tool_name: [search]
next_tool_name: 'search'
next_tool_name: search工具
next_tool_args: 这不是JSON格式
next_tool_args: {"query": 深度学习}""",

    # 字段提取任务的系统提示模板
    "field_extraction_system_prompt": """你是一个AI助手，负责从输入中提取指定的字段信息。请严格按照以下格式提供输出：

{output_fields}

注意事项：
1. 每个字段必须单独成行，格式为"字段名: 值"
2. 不要添加任何额外的解释或说明
3. 确保每个字段都有对应的值
4. 值应该简洁明了，不要包含多余的文字

正确示例:
field_name: 产品名称
field_type: 字符串
description: 产品的正式名称
is_required: 是

错误示例:
field_name: 产品名称（这是一个必填字段）
field_type: 字符串类型
description: 这个字段用于描述产品的正式名称
is_required: 是的，这个字段是必填的""",

    # 思维链提示
    "chain_of_thought": "思考逐步解决这个问题。首先分析所有已知信息，然后一步一步地推理。"
}