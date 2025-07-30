"""
模块基类，所有处理组件都继承自该类
"""
from typing import Any, Dict


class Module:
    """
    模块基类
    
    所有框架中的处理组件都应继承自该基类。
    这个类提供了基本的调用接口和功能，使组件能够像函数一样被调用。
    """
    
    def __init__(self):
        """初始化模块"""
        pass
        
    def __call__(self, **kwargs: Any) -> Any:
        """
        调用模块处理函数
        
        允许将模块实例作为函数调用，内部会调用forward方法进行处理
        
        参数:
            **kwargs: 关键字参数，传递给forward方法
            
        返回:
            forward方法的返回值
        """
        return self.forward(**kwargs)
    
    def forward(self, **kwargs: Any) -> Any:
        """
        前向处理函数
        
        该方法需要被子类重写，定义模块的核心处理逻辑
        
        参数:
            **kwargs: 关键字参数
            
        返回:
            处理结果
        """
        raise NotImplementedError("子类必须实现forward方法") 