"""
miniReAct - 轻量级的 ReAct 框架实现

该模块提供了 ReAct（Reasoning and Acting）智能体框架的简化实现。
"""

from .module import Module
from .tool import Tool
from .signature import Signature, InputField, OutputField
from .predict import Predict, ChainOfThought,PredictionCache
from .react import ReAct
from .lm import (
    chat, complete, 
    set_model, get_model,
    enable_debug, disable_debug,
    config as lm_config,
    LM,
    setup_openrouter,
    setup_ollama,
    setup_openai
)
from .streamify import (
    streamify, 
    streaming_response,
    StreamResponse,
    ThoughtResponse,
    ToolCallResponse,
    ObservationResponse,
    FinishResponse
)

__all__ = [
    "Module",
    "Tool",
    "Signature",
    "InputField",
    "OutputField",
    "Predict",
    "PredictionCache",
    "ChainOfThought",
    "ReAct",
    # LM相关
    "chat",
    "complete",
    "set_model",
    "get_model",
    "enable_debug",
    "disable_debug",
    "lm_config",
    "LM",
    "setup_openrouter",
    "setup_ollama",
    "setup_openai",
    # 流式返回相关
    "streamify",
    "streaming_response",
    "StreamResponse",
    "ThoughtResponse",
    "ToolCallResponse",
    "ObservationResponse",
    "FinishResponse",
] 