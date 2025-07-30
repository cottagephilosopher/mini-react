"""
ReAct模块，实现推理和行动框架的核心逻辑
"""
from loguru import logger
from typing import Any, Callable, Dict, List, Literal, Optional

# 导入上下文窗口异常处理
from .lm import ContextWindowExceededError

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
    
    def __init__(self, signature: Any, tools: List[Callable], max_iters: int = 5,lm=None):
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
        self.lm = lm
        
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
        
        # 创建与输出字段对应的args
        finish_args = {field_name: {"type": Any} for field_name in signature.output_fields}
        tools["finish"] = Tool(
            func=lambda **kwargs: "Done",
            name="finish",
            desc=react_prompts["finish_tool_desc"].format(outputs=outputs),
            args=finish_args,  # 提供输出字段作为参数
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
        if lm:
            self.react.lm = lm
        self.extract = ChainOfThought(fallback_signature)  # 用于从轨迹提取最终结果
        if lm:
            self.extract.lm = lm
    
    def _format_trajectory(self, trajectory: Dict[str, Any]) -> str:
        """
        格式化轨迹信息，确保格式清晰
        
        参数:
            trajectory: 轨迹字典
            
        返回:
            格式化后的轨迹字符串
        """
        # 创建一个临时的聊天适配器和签名
        from .predict import ChatAdapter
        adapter = ChatAdapter()
        
        # 处理特殊的错误反馈字段
        error_feedbacks = {}
        normal_fields = {}
        
        for key, value in trajectory.items():
            if key.startswith("error_feedback_"):
                error_feedbacks[key] = value
            else:
                normal_fields[key] = value
        
        # 首先格式化常规字段
        trajectory_signature = Signature({}, {}, f"{', '.join(normal_fields.keys())} -> x")
        formatted_trajectory = adapter.format_user_message_content(trajectory_signature, normal_fields)
        
        # 然后添加错误反馈（如果有的话）
        if error_feedbacks:
            feedback_parts = []
            for key in sorted(error_feedbacks.keys()):
                feedback_parts.append(error_feedbacks[key])
            
            formatted_trajectory += "\n\n" + "\n".join(feedback_parts)
        
        return formatted_trajectory
    
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

        lm = input_args.pop("lm", self.lm)
        
        # 迭代执行推理-行动-观察循环
        for idx in range(max_iters):
            try:
                # 调用react预测模块进行下一步预测
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory,lm=lm, **input_args
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
                    
                    # 最多重试3次
                    max_retries = 3
                    retry_attempt = 1
                    
                    while retry_attempt <= max_retries:
                        # 尝试重新预测
                        retry_pred = self._retry_prediction(
                            self.react, trajectory, retry_attempt, input_args, lm
                        )
                        
                        # 检查新的预测结果是否有效
                        if hasattr(retry_pred, "next_tool_name") and retry_pred.next_tool_name in self.tools:
                            logger.info(f"重试成功：获得有效的工具名称 {retry_pred.next_tool_name}")
                            pred = retry_pred  # 使用新的有效预测结果
                            break
                        
                        retry_attempt += 1
                    
                    # 如果所有重试都失败，才回退到最接近匹配或finish工具
                    if pred.next_tool_name not in self.tools:
                        import difflib
                        # 改进匹配逻辑：优先匹配包含连字符的完整名称
                        closest_match = difflib.get_close_matches(pred.next_tool_name, available_tools, n=3)
                        
                        # 如果有完全匹配的（忽略大小写），使用它
                        exact_match = None
                        for tool in available_tools:
                            if tool.lower() == pred.next_tool_name.lower():
                                exact_match = tool
                                break
                        
                        if exact_match:
                            logger.info(f"找到完全匹配的工具（忽略大小写）: {exact_match}")
                            pred.next_tool_name = exact_match
                        elif closest_match:
                            logger.info(f"使用最接近的工具: {closest_match[0]} (原始: {pred.next_tool_name})")
                            pred.next_tool_name = closest_match[0]
                        else:
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
                tool = self.tools[pred.next_tool_name]
                # 验证工具参数
                required_args = set(getattr(tool, "args", {}).keys())
                provided_args = set(pred.next_tool_args.keys())
                missing_args = required_args - provided_args
                
                if missing_args and pred.next_tool_name != "finish":
                    # 如果缺少必要参数，记录错误并继续
                    error_msg = f"工具 {pred.next_tool_name} 缺少必要参数: {missing_args}"
                    logger.error(error_msg)
                    trajectory[f"observation_{idx}"] = f"执行错误: {error_msg}"
                else:
                    # 调用工具
                    trajectory[f"observation_{idx}"] = tool(**pred.next_tool_args)
            except Exception as err:
                # 记录工具执行错误
                trajectory[f"observation_{idx}"] = f"执行错误 {pred.next_tool_name}: {_fmt_exc(err)}"
            
            # 如果选择了finish工具，表示推理完成
            if pred.next_tool_name == "finish":
                # 检查是否提供了输出字段参数
                if pred.next_tool_args and len(pred.next_tool_args) > 0:
                    # 使用提供的参数作为输出
                    outputs = pred.next_tool_args
                    # 确保所有必要的输出字段都存在
                    for field_name in self.signature.output_fields:
                        if field_name not in outputs:
                            outputs[field_name] = ""
                else:
                    # 尝试从轨迹中提取结果
                    outputs = {}
                    
                    # 首先检查轨迹中是否已经有结果和解释
                    for field_name in self.signature.output_fields:
                        if field_name in trajectory:
                            outputs[field_name] = trajectory[field_name]
                    
                    # 如果输出字段未完全填充，尝试从最后一个工具调用的observation中获取结果
                    if len(outputs) < len(self.signature.output_fields):
                        # 找到最后一个observation
                        last_obs_idx = -1
                        last_obs = None
                        
                        for i in range(idx-1, -1, -1):
                            if f"observation_{i}" in trajectory:
                                last_obs = trajectory[f"observation_{i}"]
                                last_obs_idx = i
                                last_tool = trajectory.get(f"tool_name_{i}", "")
                                break
                        
                        # 如果有最后一个观察结果且输出字段只有一个，直接使用观察结果
                        if last_obs is not None and len(self.signature.output_fields) == 1:
                            field_name = next(iter(self.signature.output_fields))
                            if field_name not in outputs:
                                try:
                                    # 尝试转换为数值(如果是计算任务)，否则保持原样
                                    try:
                                        outputs[field_name] = float(last_obs)
                                    except (ValueError, TypeError):
                                        outputs[field_name] = last_obs
                                    logger.info(f"从最后一个观察中获取'{field_name}'：{outputs[field_name]}")
                                except Exception as e:
                                    logger.error(f"处理最后一个观察时出错：{e}")
                
                # 确保所有必要的输出字段都存在
                for field_name in self.signature.output_fields:
                    if field_name not in outputs:
                        outputs[field_name] = f"无法生成{field_name}"
                
                # 将输出添加到轨迹中
                for field_name, value in outputs.items():
                    trajectory[field_name] = value
                
                break
        
        # 从最终轨迹中提取结果
        try:
            # 首先检查轨迹中是否已经有结果字段
            outputs = {}
            for field_name in self.signature.output_fields:
                if field_name in trajectory:
                    outputs[field_name] = trajectory[field_name]
            
            # 如果所有必要的输出字段都已存在，直接返回结果
            if all(field_name in outputs for field_name in self.signature.output_fields):
                return Prediction(trajectory=trajectory, **outputs)
            
            # 否则，调用extract模块提取结果
            extract = self._call_with_potential_trajectory_truncation(
                self.extract, trajectory, lm=lm, **input_args
            )
            # 合并提取的结果和已有的结果
            for field_name in self.signature.output_fields:
                if field_name not in outputs and hasattr(extract, field_name):
                    outputs[field_name] = getattr(extract, field_name)
            # 返回包含轨迹和输出的预测结果
            return Prediction(trajectory=trajectory, **outputs)
        except Exception as err:
            logger.error(f"提取结果时发生错误: {_fmt_exc(err)}")
            # 如果提取失败，创建一个包含默认值的结果
            default_outputs = {}
            for field_name in self.signature.output_fields:
                default_outputs[field_name] = f"无法生成{field_name}，处理过程中出现错误"
            
            return Prediction(trajectory=trajectory, **default_outputs)
    
    def _call_with_potential_trajectory_truncation(self, module, trajectory, lm=None, **input_args):
        """
        调用模块，当轨迹过长时进行截断处理
        
        参数:
            module: 要调用的模块
            trajectory: 当前轨迹
            lm: 语言模型实例
            **input_args: 输入参数
            
        返回:
            模块调用结果
        """
        # 尝试最多3次，如果遇到上下文长度超出，则截断轨迹
        for attempt in range(3):
            try:
                logger.debug(f"_call_with_potential_trajectory_truncation attempt {attempt}: calling module {module}")
                logger.debug(f"传递的参数: lm={lm.model_name if lm else 'None'}")
                print(f"[REACT DEBUG] 即将调用模块: {module.__class__.__name__}")
                print(f"[REACT DEBUG] 模块类型: {type(module)}")
                print(f"[REACT DEBUG] 模块有forward方法: {hasattr(module, 'forward')}")
                print(f"[REACT DEBUG] 传递的轨迹: {self._format_trajectory(trajectory)}")
                
                result = module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                    lm=lm,
                )
                print(f"[REACT DEBUG] 模块调用返回结果: {result}")
                print(f"[REACT DEBUG] 结果类型: {type(result)}")
                print(f"[REACT DEBUG] 结果内容: {dict(result) if hasattr(result, 'items') else str(result)}")
                
                logger.debug(f"模块调用成功返回: {type(result)}")
                return result
            except ContextWindowExceededError:
                logger.warning("轨迹超出上下文窗口限制，截断最早的工具调用信息。")
                try:
                    trajectory = self.truncate_trajectory(trajectory)
                except ValueError as e:
                    logger.error(f"无法截断轨迹: {e}")
                    # 返回一个错误的Prediction对象
                    from .predict import Prediction
                    return Prediction(next_thought=f"调用语言模型时出错: 无法截断轨迹: {e}", 
                                    next_tool_name=f"调用语言模型时出错: 无法截断轨迹: {e}",
                                    next_tool_args=f"调用语言模型时出错: 无法截断轨迹: {e}")
            except Exception as e:
                logger.error(f"调用模块时发生错误: {e}")
                import traceback
                logger.error(f"完整异常信息: {traceback.format_exc()}")
                # 如果是最后一次尝试，返回一个错误的Prediction对象
                if attempt == 2:  # 最后一次尝试
                    from .predict import Prediction
                    return Prediction(next_thought=f"调用语言模型时出错: {e}", 
                                    next_tool_name=f"调用语言模型时出错: {e}",
                                    next_tool_args=f"调用语言模型时出错: {e}")
        
        # 如果所有尝试都失败，返回一个错误的Prediction对象
        from .predict import Prediction
        return Prediction(next_thought="调用语言模型时出错: 所有尝试都失败了", 
                        next_tool_name="调用语言模型时出错: 所有尝试都失败了",
                        next_tool_args="调用语言模型时出错: 所有尝试都失败了")
    
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
        
        # 计算轨迹中的工具调用次数
        tool_calls = set()
        for key in keys:
            if key.startswith("thought_"):
                idx = key.split("_")[1]
                tool_calls.add(idx)
        
        # 如果只有一个工具调用，无法截断
        if len(tool_calls) <= 1:
            raise ValueError(
                "轨迹过长导致你的提示超出了上下文窗口，但轨迹不能被截断，因为它只包含一个工具调用。"
            )
        
        # 保留最近的工具调用，删除最早的一个
        earliest_idx = min(tool_calls)
        keys_to_remove = [k for k in keys if k.endswith(f"_{earliest_idx}")]
        
        # 删除最早的工具调用相关的键
        for key in keys_to_remove:
            trajectory.pop(key)
        
        logger.info(f"已截断轨迹，删除了工具调用 {earliest_idx}")
        return trajectory

    def _retry_prediction(self, react_predictor, trajectory, attempt, input_args, lm=None):
        """
        当检测到格式问题时，尝试重新发起预测
        
        参数:
            react_predictor: Predict对象
            trajectory: 当前轨迹
            attempt: 当前尝试次数
            input_args: 输入参数
            lm: 语言模型
            
        返回:
            重新预测的结果
        """
        # 构建新的提示，包含错误反馈
        retry_trajectory = dict(trajectory)
        
        # 添加错误反馈到轨迹中
        retry_trajectory[f"error_feedback_{attempt}"] = """
错误：您的输出格式不正确。请严格按照以下格式输出：

next_thought: 您的思考过程（只在这一行）

next_tool_name: 工具名称（只写工具名，不要包含其他文本）

next_tool_args: {"参数名": "参数值"}

请注意每个字段必须单独成行，不要混合字段内容。
"""
        
        # 重新调用预测
        logger.info(f"尝试重新预测 (第{attempt}次尝试)")
        return react_predictor(
            **input_args,
            trajectory=self._format_trajectory(retry_trajectory),
            lm=lm
        )


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