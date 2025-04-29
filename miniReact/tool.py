"""
工具类模块，用于封装智能体可调用的函数或方法
"""
import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints


class Tool:
    """
    工具类，用于封装可被智能体调用的函数或方法
    
    工具是ReAct框架中的基本操作单元，智能体可以选择并调用这些工具来完成任务。
    每个工具都有一个名称、描述和一组参数。
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        desc: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化工具
        
        参数:
            func: 可调用的函数或方法
            name: 工具名称，默认使用函数名
            desc: 工具描述，默认使用函数的文档字符串
            args: 工具参数定义，默认从函数签名提取
        """
        self.func = func
        self.name = name or getattr(func, "__name__", "未命名工具")
        self.desc = desc or inspect.getdoc(func) or "无描述"
        
        # 如果没有提供参数定义，则从函数签名中提取
        if args is None:
            self.args = self._extract_args_from_func(func)
        else:
            self.args = args
    
    def __call__(self, **kwargs: Any) -> Any:
        """
        调用工具函数
        
        参数:
            **kwargs: 传递给工具函数的参数
            
        返回:
            工具函数的返回值
        """
        return self.func(**kwargs)
    
    def _extract_args_from_func(self, func: Callable) -> Dict[str, Any]:
        """
        从函数中提取参数定义
        
        参数:
            func: 待分析的函数
            
        返回:
            参数定义字典，键为参数名，值为参数类型或默认值
        """
        # 获取函数签名
        sig = inspect.signature(func)
        # 获取类型提示
        type_hints = get_type_hints(func)
        
        args = {}
        # 处理每个参数
        for name, param in sig.parameters.items():
            # 跳过self参数（对于方法）
            if name == "self":
                continue
                
            # 尝试获取类型
            arg_type = type_hints.get(name, Any)
            
            # 检查是否有默认值
            if param.default is not inspect.Parameter.empty:
                args[name] = {"type": arg_type, "default": param.default}
            else:
                args[name] = {"type": arg_type}
        
        return args 