"""
ReAct模块，实现推理和行动框架的核心逻辑
"""
from loguru import logger
from typing import Any, Callable, Dict, List, Literal, Optional

# 导入litellm中的上下文窗口异常处理
try:
    from litellm import ContextWindowExceededError
except ImportError:
    # 如果无法导入，创建一个兼容的异常类
    class ContextWindowExceededError(Exception):
        """上下文窗口超出限制异常"""
        pass

from .module import Module
from .predict import ChainOfThought, Prediction, Predict
from .signature import Signature, InputField, OutputField, ensure_signature
from .tool import Tool
from .prompt import react_prompts


class ReAct(Module):
    """
    ReAct类实现了推理和行动（Reasoning and Acting）的框架
    该框架允许智能体在执行任务时进行推理并采取行动。
    它使用一系列工具来交互，并通过推理和观察来决定下一步行动。
    """
    
    def __init__(self, signature: Any, tools: List[Callable], max_iters: int = 5):
        """
        初始化ReAct实例
        
        参数:
            signature: 任务签名，定义了输入和输出
            tools: 工具列表，可以是函数、可调用类或Tool实例
            max_iters: 最大迭代次数，默认为5
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        
        # 将所有工具转换为Tool对象，并构建工具字典
        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}
        
        # 构建指令信息
        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []
        
        # 添加ReAct框架的指导说明，使用prompt.py中的模板
        base_instructions = [instruction.format(inputs=inputs, outputs=outputs) 
                            for instruction in react_prompts["base_instructions"]]
        instr.extend(base_instructions)
        
        # 添加"finish"工具用于标记任务完成
        tools["finish"] = Tool(
            func=lambda: "Done",
            name="finish",
            desc=react_prompts["finish_tool_desc"].format(outputs=outputs),
            args={},
        )
        
        # 添加各个工具的描述
        for idx, tool in enumerate(tools.values()):
            args = getattr(tool, "args")
            desc = (f"，其描述为 <desc>{tool.desc}</desc>。" if tool.desc else "。").replace("\n", "  ")
            desc += f" 它接受JSON格式的参数 {args}。"
            instr.append(react_prompts["tool_desc_format"].format(idx=idx + 1, name=tool.name, desc=desc))
        
        # 创建ReAct签名
        react_signature = (
            Signature({**signature.input_fields}, {}, "\n".join(instr))
            .append("trajectory", InputField(), type_=str)
            .append("next_thought", OutputField(), type_=str)
            .append("next_tool_name", OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", OutputField(), type_=Dict[str, Any])
        )
        
        # 创建提取结果的签名（当轨迹完成后）
        fallback_signature = Signature(
            {**signature.input_fields}, 
            {**signature.output_fields},
            signature.instructions,
        ).append("trajectory", InputField(), type_=str)
        
        # 保存配置
        self.tools = tools
        self.react = Predict(react_signature)  # 用于每次迭代的预测
        self.extract = ChainOfThought(fallback_signature)  # 用于从轨迹提取最终结果
    
    def _format_trajectory(self, trajectory: Dict[str, Any]) -> str:
        """
        格式化轨迹信息
        
        参数:
            trajectory: 轨迹字典
            
        返回:
            格式化后的轨迹字符串
        """
        # 创建一个临时的聊天适配器和签名
        from .predict import ChatAdapter
        adapter = ChatAdapter()
        trajectory_signature = Signature({}, {}, f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)
    
    def forward(self, **input_args: Any) -> Prediction:
        """
        执行ReAct推理过程
        
        参数:
            **input_args: 输入参数
            
        返回:
            包含轨迹和输出的预测结果
        """
        # 创建轨迹字典，用于存储推理过程
        trajectory = {}
        
        # 获取最大迭代次数，可在调用时覆盖默认值
        max_iters = input_args.pop("max_iters", self.max_iters)
        
        # 迭代执行推理-行动-观察循环
        for idx in range(max_iters):
            try:
                # 调用react预测模块进行下一步预测
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
                
                # 确保pred包含所需的属性
                if not hasattr(pred, "next_thought") or not hasattr(pred, "next_tool_name") or not hasattr(pred, "next_tool_args"):
                    logger.error("预测结果缺少必要属性，无法继续执行")
                    break
                
                # 添加调试信息
                logger.info(f"思考: {pred.next_thought}")
                logger.info(f"选择工具: {pred.next_tool_name}")
                logger.info(f"工具参数: {pred.next_tool_args}")
                
                # 验证工具名称是否有效
                if pred.next_tool_name not in self.tools:
                    available_tools = list(self.tools.keys())
                    logger.error(f"工具名称'{pred.next_tool_name}'无效。可用工具: {available_tools}")
                    # 尝试找到最接近的工具名称
                    import difflib
                    closest_match = difflib.get_close_matches(pred.next_tool_name, available_tools, n=1)
                    if closest_match:
                        logger.info(f"使用最接近的工具: {closest_match[0]}")
                        pred.next_tool_name = closest_match[0]
                    else:
                        # 如果找不到接近的匹配，使用finish工具
                        logger.info("找不到接近的工具，使用finish工具")
                        pred.next_tool_name = "finish"
                        pred.next_tool_args = {}
                        
            except ValueError as err:
                logger.warning(f"结束轨迹: 智能体未能选择有效工具: {_fmt_exc(err)}")
                break
            except Exception as err:
                logger.error(f"预测过程中发生错误: {_fmt_exc(err)}")
                break
            
            # 记录思考、工具名称和参数
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            
            try:
                # 调用选定的工具并记录结果
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                # 记录工具执行错误
                trajectory[f"observation_{idx}"] = f"执行错误 {pred.next_tool_name}: {_fmt_exc(err)}"
            
            # 如果选择了finish工具，表示推理完成
            if pred.next_tool_name == "finish":
                break
        
        # 从最终轨迹中提取结果
        try:
            extract = self._call_with_potential_trajectory_truncation(
                self.extract, trajectory, **input_args
            )
            
            # 返回包含轨迹和输出的预测结果
            return Prediction(trajectory=trajectory, **extract)
        except Exception as err:
            logger.error(f"提取结果时发生错误: {_fmt_exc(err)}")
            # 如果提取失败，创建一个包含默认值的结果
            default_outputs = {}
            for field_name in self.signature.output_fields:
                default_outputs[field_name] = f"无法生成{field_name}，处理过程中出现错误"
            
            return Prediction(trajectory=trajectory, **default_outputs)
    
    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        """
        调用模块，当轨迹过长时进行截断处理
        
        参数:
            module: 要调用的模块
            trajectory: 当前轨迹
            **input_args: 输入参数
            
        返回:
            模块调用结果
        """
        # 尝试最多3次，如果遇到上下文长度超出，则截断轨迹
        for attempt in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("轨迹超出上下文窗口限制，截断最早的工具调用信息。")
                try:
                    trajectory = self.truncate_trajectory(trajectory)
                except ValueError as e:
                    logger.error(f"无法截断轨迹: {e}")
                    # 返回一个空的Prediction对象
                    from .predict import Prediction
                    return Prediction()
            except Exception as e:
                logger.error(f"调用模块时发生错误: {e}")
                # 如果是最后一次尝试，返回一个空的Prediction对象
                if attempt == 2:  # 最后一次尝试
                    from .predict import Prediction
                    return Prediction()
        
        # 如果所有尝试都失败，返回一个空的Prediction对象
        from .predict import Prediction
        return Prediction()
    
    def truncate_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        截断轨迹，使其适合上下文窗口
        
        用户可以重写此方法以实现自定义截断逻辑
        
        参数:
            trajectory: 当前轨迹字典
            
        返回:
            截断后的轨迹字典
        """
        keys = list(trajectory.keys())
        if len(keys) < 4:
            # 每个工具调用有4个键：thought, tool_name, tool_args, observation
            raise ValueError(
                "轨迹过长导致你的提示超出了上下文窗口，但轨迹不能被截断，因为它只包含一个工具调用。"
            )
        
        # 删除最早的工具调用（前4个键）
        for key in keys[:4]:
            trajectory.pop(key)
        
        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    返回一个异常的简短字符串表示
    
    参数:
        err: 异常对象
        limit: 保留的堆栈帧数量（从最内层向外）
        
    返回:
        格式化后的异常字符串
    """
    import traceback
    
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()