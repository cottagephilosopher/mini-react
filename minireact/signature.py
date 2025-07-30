"""
签名模块，用于定义任务的输入和输出规范
"""
from typing import Any, Dict, Optional, Union, Type


class Field:
    """
    字段基类，用于表示签名中的输入或输出字段
    """
    def __init__(self, desc: Optional[str] = None, type_: Optional[Type] = None):
        """
        初始化字段
        
        参数:
            desc: 字段描述
            type_: 字段类型
        """
        self.desc = desc
        self.type = type_


class InputField(Field):
    """
    输入字段，用于表示签名中的输入参数
    """
    pass


class OutputField(Field):
    """
    输出字段，用于表示签名中的输出参数
    """
    pass


class Signature:
    """
    签名类，用于定义任务的输入、输出和指令
    
    签名定义了一个任务的接口，包括它接受什么输入、产生什么输出，以及如何处理这些数据。
    """
    
    def __init__(
        self, 
        input_fields: Optional[Dict[str, Union[InputField, Any]]] = None,
        output_fields: Optional[Dict[str, Union[OutputField, Any]]] = None,
        instructions: Optional[str] = None,
    ):
        """
        初始化签名
        
        参数:
            input_fields: 输入字段定义，键为字段名，值为InputField实例或其他值
            output_fields: 输出字段定义，键为字段名，值为OutputField实例或其他值
            instructions: 任务指令描述
        """
        # 处理输入字段
        self.input_fields = {}
        if input_fields:
            for name, field in input_fields.items():
                if not isinstance(field, InputField):
                    field = InputField()
                self.input_fields[name] = field
        
        # 处理输出字段
        self.output_fields = {}
        if output_fields:
            for name, field in output_fields.items():
                if not isinstance(field, OutputField):
                    field = OutputField()
                self.output_fields[name] = field
        
        self.instructions = instructions

    def append(self, name: str, field: Union[Field, Any], type_: Optional[Type] = None) -> 'Signature':
        """
        添加新字段到签名
        
        参数:
            name: 字段名
            field: 字段定义，Field实例或其他值
            type_: 字段类型
            
        返回:
            更新后的签名对象（自身）
        """
        # 确定是InputField还是OutputField
        if isinstance(field, InputField):
            if type_ is not None:
                field.type = type_
            self.input_fields[name] = field
        elif isinstance(field, OutputField):
            if type_ is not None:
                field.type = type_
            self.output_fields[name] = field
        else:
            # 默认作为输入字段
            self.input_fields[name] = InputField(type_=type_)
        
        return self


def ensure_signature(signature: Union[Signature, Dict, Any]) -> Signature:
    """
    确保参数是一个有效的签名对象
    
    如果参数不是Signature实例，尝试将其转换为签名
    
    参数:
        signature: 签名对象、字典或其他值
        
    返回:
        Signature实例
    """
    if isinstance(signature, Signature):
        return signature
    elif isinstance(signature, dict):
        return Signature(input_fields=signature)
    else:
        # 尝试从其他类型创建签名
        return Signature({}, {}, instructions=str(signature)) 