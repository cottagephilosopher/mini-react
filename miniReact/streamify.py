"""
流式返回模块，为ReAct框架提供流式输出功能
"""
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union, Callable, Awaitable
import inspect
from pathlib import Path
import json
from loguru import logger

# 导入React核心类
from .react import ReAct
from .predict import Prediction


class StreamResponse:
    """
    流式响应基类，用于标识不同类型的流式响应
    """
    pass


class ThoughtResponse(StreamResponse):
    """
    思考过程的流式响应
    """
    def __init__(self, thought: str, index: int):
        self.thought = thought
        self.index = index
        
    def __str__(self):
        return f"思考 #{self.index}: {self.thought}"
    
    def message(self,type:str="reasoning"):
        return json.dumps({
                "type": type,
                "content": self.thought 
            }, ensure_ascii=False) + "\n"


class ToolCallResponse(StreamResponse):
    """
    工具调用的流式响应
    """
    def __init__(self, tool_name: str, tool_args: Dict[str, Any], index: int):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.index = index
        
    def __str__(self):
        return f"调用工具 #{self.index}: {self.tool_name}({json.dumps(self.tool_args, ensure_ascii=False)})"

    def message(self,type:str="status"):
        return json.dumps({
                "type": type,
                "content": f"Step {self.index+1} :Using tools: {self.tool_name}"
            }, ensure_ascii=False) + "\n"


class ObservationResponse(StreamResponse):
    """
    观察结果的流式响应
    """
    def __init__(self, observation: Any, index: int):
        self.observation = observation
        self.index = index
        
    def __str__(self):
        return f"观察 #{self.index}: {self.observation}"

    def message(self,type:str="status"):
        return json.dumps({
                "type": type,
                "content": f"Observing step {self.index+1} Return: {self.observation}"
            }, ensure_ascii=False) + "\n"


class FinishResponse(StreamResponse):
    """
    完成信号的流式响应
    """
    def __init__(self, outputs: Dict[str, Any]):
        self.outputs = outputs
        
    def __str__(self):
        return f"完成: {json.dumps(self.outputs, ensure_ascii=False)}"

    def message(self,output_field:str=None):
        if output_field is not None:
            output = self.outputs.get(output_field, "")
        else:
            output = json.dumps(self.outputs, ensure_ascii=False)
        return json.dumps({
                "type": "content",
                "content": output
            }, ensure_ascii=False) + "\n"


async def _asyncify(func: Callable, *args, **kwargs) -> Any:
    """
    将同步函数转换为异步函数
    
    参数:
        func: 需要转换的函数
        *args, **kwargs: 函数的参数
        
    返回:
        函数的返回值
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def streamify(react_instance: ReAct) -> Callable[..., AsyncGenerator]:
    """
    将ReAct实例转换为支持流式返回的函数
    
    参数:
        react_instance: ReAct实例
        
    返回:
        一个函数，接收与原函数相同的参数，但返回一个异步生成器
    """
    async def stream_wrapper(**kwargs) -> AsyncGenerator:
        """
        包装ReAct的forward方法，将其转换为一个异步生成器
        
        参数:
            **kwargs: 传递给ReAct实例的参数
            
        返回:
            异步生成器，产生流式响应
        """
        # 创建一个内部类来监控和捕获ReAct执行过程
        class StreamReActMonitor:
            def __init__(self):
                self.trajectory = {}
                self.outputs = {}
            
            async def execute(self, **exec_kwargs):
                """
                执行ReAct并捕获流式输出
                
                参数:
                    **exec_kwargs: 传递给ReAct的参数
                """
                # 保存原有的ReAct.forward方法
                original_forward = react_instance.forward
                
                async def monitored_forward(**forward_kwargs):
                    """
                    监控版本的forward方法
                    
                    参数:
                        **forward_kwargs: 传递给ReAct的参数
                    """
                    # 重置状态
                    self.trajectory = {}
                    self.outputs = {}
                    
                    # 创建轨迹字典，用于存储推理过程
                    trajectory = {}
                    
                    # 获取最大迭代次数，可在调用时覆盖默认值
                    max_iters = forward_kwargs.pop("max_iters", react_instance.max_iters)
                    lm = forward_kwargs.pop("lm", react_instance.lm)
                    
                    # 迭代执行推理-行动-观察循环
                    for idx in range(max_iters):
                        try:
                            # 调用react预测模块进行下一步预测
                            pred = react_instance._call_with_potential_trajectory_truncation(
                                react_instance.react, trajectory, lm=lm, **forward_kwargs
                            )
                            
                            # 添加调试信息
                            logger.debug(f"思考: {pred.next_thought}")
                            logger.debug(f"选择工具: {pred.next_tool_name}")
                            logger.debug(f"工具参数: {pred.next_tool_args}")
                            
                            # 流式返回思考过程
                            yield ThoughtResponse(pred.next_thought, idx)
                            
                            # 记录思考、工具名称和参数
                            trajectory[f"thought_{idx}"] = pred.next_thought
                            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
                            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
                            
                            # 流式返回工具调用信息
                            yield ToolCallResponse(pred.next_tool_name, pred.next_tool_args, idx)
                            
                            try:
                                # 调用选定的工具并记录结果
                                tool = react_instance.tools[pred.next_tool_name]
                                trajectory[f"observation_{idx}"] = tool(**pred.next_tool_args)
                            except Exception as err:
                                # 记录工具执行错误
                                trajectory[f"observation_{idx}"] = f"执行错误 {pred.next_tool_name}: {err}"
                            
                            # 流式返回观察结果
                            yield ObservationResponse(trajectory[f"observation_{idx}"], idx)
                            
                            # 如果选择了finish工具，表示推理完成
                            if pred.next_tool_name == "finish":
                                # 检查是否提供了输出字段参数
                                if pred.next_tool_args and len(pred.next_tool_args) > 0:
                                    # 使用提供的参数作为输出
                                    outputs = pred.next_tool_args
                                    # 确保所有必要的输出字段都存在
                                    for field_name in react_instance.signature.output_fields:
                                        if field_name not in outputs:
                                            outputs[field_name] = ""
                                else:
                                    # 尝试从轨迹中提取结果
                                    outputs = {}
                                    
                                    # 首先检查轨迹中是否已经有结果和解释
                                    for field_name in react_instance.signature.output_fields:
                                        if field_name in trajectory:
                                            outputs[field_name] = trajectory[field_name]
                                
                                # 流式返回完成信号
                                yield FinishResponse(outputs)
                                
                                # 将输出添加到轨迹中
                                for field_name, value in outputs.items():
                                    trajectory[field_name] = value
                                
                                break
                        except Exception as e:
                            logger.error(f"执行过程中发生错误: {e}")
                            break
                    
                    # 从最终轨迹中提取结果
                    try:
                        # 首先检查轨迹中是否已经有结果字段
                        outputs = {}
                        for field_name in react_instance.signature.output_fields:
                            if field_name in trajectory:
                                outputs[field_name] = trajectory[field_name]
                        
                        # 如果所有必要的输出字段都已存在，直接返回结果
                        if all(field_name in outputs for field_name in react_instance.signature.output_fields):
                            # 创建预测结果
                            final_prediction = Prediction(trajectory=trajectory, **outputs)
                            # 使用yield而不是return
                            yield final_prediction
                            return  # 使用不带值的return终止生成器
                        
                        # 否则，调用extract模块提取结果
                        extract = react_instance._call_with_potential_trajectory_truncation(
                            react_instance.extract, trajectory, lm=lm, **forward_kwargs
                        )
                        # 合并提取的结果和已有的结果
                        for field_name in react_instance.signature.output_fields:
                            if field_name not in outputs and hasattr(extract, field_name):
                                outputs[field_name] = getattr(extract, field_name)
                        # 创建预测结果并使用yield
                        final_prediction = Prediction(trajectory=trajectory, **outputs)
                        yield final_prediction
                    except Exception as err:
                        logger.error(f"提取结果时发生错误: {err}")
                        # 如果提取失败，创建一个包含默认值的结果
                        default_outputs = {}
                        for field_name in react_instance.signature.output_fields:
                            default_outputs[field_name] = f"无法生成{field_name}，处理过程中出现错误"
                        
                        # 使用yield而不是return
                        yield Prediction(trajectory=trajectory, **default_outputs)
                
                # 执行监控版本的forward方法
                final_prediction = None
                
                # 使用异步for循环收集所有生成的项
                async for item in monitored_forward(**exec_kwargs):
                    if isinstance(item, Prediction):
                        final_prediction = item
                    yield item
                
                # 如果没有生成最终预测结果，使用原始forward方法获取
                if final_prediction is None:
                    try:
                        final_prediction = await _asyncify(original_forward, **exec_kwargs)
                        yield final_prediction
                    except Exception as e:
                        logger.error(f"获取最终预测结果时发生错误: {e}")
            
        # 创建监控实例并执行
        monitor = StreamReActMonitor()
        async for response in monitor.execute(**kwargs):
            yield response
    
    return stream_wrapper


async def streaming_response(streamer: AsyncGenerator) -> AsyncGenerator:
    """
    将流式响应转换为兼容OpenAI格式的API响应
    
    参数:
        streamer: 流式响应生成器
        
    返回:
        兼容OpenAI格式的API响应生成器
    """
    async for value in streamer:
        if isinstance(value, Prediction):
            data = {"prediction": {k: v for k, v in value.items() if k != "trajectory"}}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        elif isinstance(value, StreamResponse):
            data = {"chunk": str(value)}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        elif isinstance(value, str) and value.startswith("data:"):
            # 已经是兼容OpenAI格式的数据，直接返回
            yield value
        else:
            # 未知数据类型，转换为字符串
            data = {"chunk": str(value)}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    # 添加完成标记
    yield "data: [DONE]\n\n" 