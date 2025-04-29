"""
Prompt template module, used for centralized management of various prompt texts used in the system
"""

# Prompt templates for ReAct framework
react_prompts = {
    # Basic instructions for ReAct framework
    "base_instructions": [
        "You are an agent. In each task, you will receive {inputs} as input and can see your previous reasoning trace.",
        "Your goal is to use the provided tools to collect necessary information to produce {outputs}.\n",
        "You need to provide next_thought, next_tool_name, and next_tool_args in each round, as well as a summary when completing the task.",
        "After each tool call, you will receive an observation, which will be added to your reasoning trace.\n",
        "When writing next_thought, you should analyze the current situation and plan future steps.",
        "When selecting next_tool_name, you must strictly choose one from the following tool list, without adding any additional text, quotes, or brackets.",
        "When providing next_tool_args, you must use valid JSON format, such as {{\"parameter_name\": \"parameter_value\"}}.",
        "Parameters must be a JSON object (dictionary), not a string or other format.",
        "Do not use markdown code blocks (```) to wrap your answer.",
        "When the task is completed, you must use the 'finish' tool to submit the final result.",
        "Available tools include:\n",
    ],
    
    # Template for finish tool description
    "finish_tool_desc": "Mark the task as completed. Indicates that sufficient information has been collected to produce output: {outputs}",
    
    # Formatting template for tool descriptions
    "tool_desc_format": "({idx}) {name}{desc}",
}

# Prompt templates for prediction module
predict_prompts = {
    # System prompt for ReAct tasks
    "react_system_prompt": """You are an AI assistant responsible for executing Reasoning and Acting (ReAct) tasks. When you need to use tools, please provide output strictly according to the following format:

    next_thought: Your thinking process

    next_tool_name: Tool name

    next_tool_args: {"parameter_name": "parameter_value"}

    Notes:
    1. next_tool_name must be precisely selected from the provided tool list, without adding any other characters such as quotes, brackets, etc.
    2. next_tool_args must be in valid JSON object format, wrapped in curly braces, containing key-value pairs
    3. Each field must be on a separate line, do not mix or nest fields
    4. Do not add greetings or introductions at the beginning of your answer, provide output directly in the specified format
    5. Do not use markdown code blocks (```json) to wrap your answer, provide the answer in plain text format
    6. Do not return a complete JSON object, but provide each field on a separate line according to the specified format
    7. When the task is completed, you must use the 'finish' tool to submit the final result

    Correct example:
    next_thought: I need to search for information about deep learning
    next_tool_name: search
    next_tool_args: {"query": "deep learning"}

    Incorrect example:
    ```json
    {
    "next_thought": "I need to search for information about deep learning",
    "next_tool_name": "search",
    "next_tool_args": {"query": "deep learning"}
    }
    ```

    next_tool_name: [search]
    next_tool_name: 'search'
    next_tool_name: search tool
    next_tool_args: This is not JSON format
    next_tool_args: {"query": deep learning}""",

    # System prompt template for field extraction tasks
    "field_extraction_system_prompt": """You are an AI assistant responsible for extracting specified field information from input. Please provide output strictly according to the following format:

    {output_fields}

    Notes:
    1. Each field must be on a separate line, in the format "field_name: value"
    2. Do not add any additional explanations or descriptions
    3. Ensure that each field has a corresponding value
    4. Values should be concise and clear, without containing superfluous text

    Correct example:
    field_name: Product Name
    field_type: String
    description: The official name of the product
    is_required: Yes

    Incorrect example:
    field_name: Product Name (this is a required field)
    field_type: String type
    description: This field is used to describe the official name of the product
    is_required: Yes, this field is required""",

    # Chain of thought prompt
    "chain_of_thought": "Think step by step to solve this problem. First analyze all known information, then reason one step at a time."
}