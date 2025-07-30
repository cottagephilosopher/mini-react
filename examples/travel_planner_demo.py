"""
旅行规划助手示例程序，展示如何使用minireact框架创建一个旅行规划智能体
"""
import os
import sys
import logging
import random
from datetime import datetime, timedelta
# 设置语言模型配置
from llm_hub import MultiLLMHub
# 添加上级目录到模块搜索路径，以便导入minireact
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minireact as mr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 定义一些旅行规划工具
def search_destination(keywords: str, budget_limit: float = None) -> str:
    """
    根据关键词和预算搜索合适的旅行目的地
    
    参数:
        keywords: 搜索关键词，例如"海滩"、"历史"、"自然"等
        budget_limit: 预算限制（可选）
    
    返回:
        推荐目的地列表
    """
    destinations = {
        "海滩": ["三亚", "普吉岛", "巴厘岛", "马尔代夫", "夏威夷"],
        "历史": ["西安", "北京", "罗马", "雅典", "开罗"],
        "自然": ["张家界", "黄石公园", "阿尔卑斯山", "亚马逊雨林", "大堡礁"],
        "美食": ["成都", "广州", "东京", "巴黎", "曼谷"],
        "购物": ["上海", "香港", "纽约", "迪拜", "巴黎"]
    }
    
    results = []
    for key, places in destinations.items():
        if key in keywords.lower():
            results.extend(places)
    
    if not results:
        results = random.sample([item for sublist in destinations.values() for item in sublist], 3)
    
    if budget_limit:
        # 模拟根据预算过滤目的地
        budget_tier = {
            "低": ["西安", "成都", "张家界", "北京", "广州"],
            "中": ["上海", "东京", "曼谷", "三亚", "普吉岛"],
            "高": ["纽约", "巴黎", "马尔代夫", "夏威夷", "迪拜"]
        }
        
        if budget_limit < 5000:
            filtered = [d for d in results if d in budget_tier["低"]]
        elif budget_limit < 15000:
            filtered = [d for d in results if d in budget_tier["低"] or d in budget_tier["中"]]
        else:
            filtered = results
            
        results = filtered if filtered else results
    
    return "、".join(results[:3]) if results else "未找到符合条件的目的地"

def get_attractions(destination: str) -> str:
    """
    获取指定目的地的主要景点
    
    参数:
        destination: 目的地名称
    
    返回:
        主要景点列表
    """
    attractions = {
        "三亚": ["亚龙湾", "天涯海角", "南山文化旅游区", "蜈支洲岛"],
        "西安": ["兵马俑", "古城墙", "大雁塔", "华山"],
        "成都": ["大熊猫繁育研究基地", "锦里古街", "都江堰", "宽窄巷子"],
        "北京": ["故宫", "长城", "天坛", "颐和园"],
        "上海": ["外滩", "东方明珠", "豫园", "迪士尼乐园"],
        "张家界": ["天门山", "张家界国家森林公园", "玻璃桥", "宝峰湖"],
        "普吉岛": ["芭东海滩", "普吉老城", "大佛", "攀牙湾"],
        "东京": ["东京塔", "浅草寺", "迪士尼乐园", "新宿御苑"],
        "巴黎": ["埃菲尔铁塔", "卢浮宫", "凯旋门", "巴黎圣母院"]
    }
    
    if destination in attractions:
        return "、".join(attractions[destination])
    else:
        return f"暂无{destination}的景点信息"

def estimate_budget(destination: str, days: int, luxury_level: str = "标准") -> str:
    """
    估算旅行预算
    
    参数:
        destination: 目的地
        days: 旅行天数
        luxury_level: 奢华程度，可以是"经济"、"标准"或"豪华"
    
    返回:
        预算估算
    """
    # 不同城市的基础日均消费
    base_costs = {
        "三亚": 800, "西安": 500, "成都": 600, "北京": 700, "上海": 900,
        "张家界": 650, "普吉岛": 1000, "东京": 1500, "巴黎": 2000,
        "广州": 600, "香港": 1200, "纽约": 2500, "迪拜": 2200, "曼谷": 800,
        "马尔代夫": 3000, "夏威夷": 2800, "巴厘岛": 1000
    }
    
    # 奢华程度倍数
    luxury_multiplier = {
        "经济": 0.7,
        "标准": 1.0,
        "豪华": 2.0
    }
    
    # 获取基础成本或使用默认值
    base_cost = base_costs.get(destination, 1000)
    multiplier = luxury_multiplier.get(luxury_level, 1.0)
    
    # 计算总预算
    total_budget = base_cost * days * multiplier
    
    # 返回详细的预算明细
    return f"""
预计{destination}旅行{days}天({luxury_level}级别)的预算明细:
- 住宿: {int(base_cost * 0.4 * multiplier * days)}元
- 餐饮: {int(base_cost * 0.3 * multiplier * days)}元
- 交通: {int(base_cost * 0.2 * multiplier * days)}元
- 景点及其他: {int(base_cost * 0.1 * multiplier * days)}元
总预算: 约{int(total_budget)}元
"""

def check_weather(destination: str, travel_date: str = None) -> str:
    """
    查询目的地的天气情况
    
    参数:
        destination: 目的地名称
        travel_date: 旅行日期 (可选，格式: YYYY-MM-DD)
    
    返回:
        天气预报信息
    """
    # 不同城市的季节性天气特点
    weather_patterns = {
        "三亚": {"春": "晴朗，20-25℃", "夏": "炎热，28-35℃", "秋": "晴朗，23-28℃", "冬": "温暖，18-24℃"},
        "西安": {"春": "变化多，10-20℃", "夏": "炎热，25-35℃", "秋": "凉爽，15-25℃", "冬": "寒冷，-5-8℃"},
        "成都": {"春": "多雨，15-22℃", "夏": "闷热，25-32℃", "秋": "凉爽，18-25℃", "冬": "阴冷，5-12℃"},
        "北京": {"春": "多风，8-20℃", "夏": "炎热，25-35℃", "秋": "晴朗，15-25℃", "冬": "寒冷，-10-5℃"},
        "上海": {"春": "多雨，12-20℃", "夏": "潮湿炎热，25-35℃", "秋": "凉爽，18-25℃", "冬": "寒冷，0-10℃"}
    }
    
    # 使用当前日期或指定日期
    if not travel_date:
        current_date = datetime.now()
    else:
        try:
            current_date = datetime.strptime(travel_date, "%Y-%m-%d")
        except ValueError:
            return f"日期格式错误，请使用YYYY-MM-DD格式"
    
    # 确定季节
    month = current_date.month
    if 3 <= month <= 5:
        season = "春"
    elif 6 <= month <= 8:
        season = "夏"
    elif 9 <= month <= 11:
        season = "秋"
    else:
        season = "冬"
    
    # 获取天气信息
    if destination in weather_patterns:
        weather = weather_patterns[destination][season]
        return f"{destination}在{current_date.strftime('%Y-%m-%d')}({season}季)的天气预计: {weather}"
    else:
        return f"暂无{destination}的天气信息"

def generate_itinerary(destination: str, days: int) -> str:
    """
    生成旅行行程建议
    
    参数:
        destination: 目的地
        days: 旅行天数
    
    返回:
        行程安排建议
    """
    # 简化版行程生成
    itinerary = f"{destination}{days}天行程建议:\n\n"
    
    attractions_text = get_attractions(destination)
    attractions_list = attractions_text.split("、")
    
    if attractions_list[0] == f"暂无{destination}的景点信息":
        # 如果没有景点信息，创建一个通用行程
        for day in range(1, days + 1):
            itinerary += f"第{day}天: "
            if day == 1:
                itinerary += f"抵达{destination}，办理入住，熟悉周边环境\n"
            elif day == days:
                itinerary += f"自由活动，购买纪念品，返程\n"
            else:
                itinerary += f"{destination}自由探索和体验当地文化\n"
    else:
        # 根据已知景点创建行程
        attractions_per_day = max(1, min(len(attractions_list) // days, 2))  # 每天1-2个景点
        
        for day in range(1, days + 1):
            itinerary += f"第{day}天: "
            
            if day == 1:
                itinerary += f"抵达{destination}，办理入住，"
                if len(attractions_list) > 0:
                    itinerary += f"下午游览{attractions_list[0]}\n"
                else:
                    itinerary += "熟悉周边环境\n"
            elif day == days:
                itinerary += "上午自由活动，购买纪念品，下午返程\n"
            else:
                day_attractions = []
                for i in range(attractions_per_day):
                    idx = 1 + (day-2)*attractions_per_day + i
                    if idx < len(attractions_list):
                        day_attractions.append(attractions_list[idx])
                
                if day_attractions:
                    itinerary += f"游览{' 和 '.join(day_attractions)}\n"
                else:
                    itinerary += f"继续探索{destination}的其他景点和体验当地文化\n"
    
    return itinerary

# 定义任务签名
travel_planner_signature = mr.Signature(
    {
        "destination_query": mr.InputField(desc="旅行目的地或意向，例如'想去海边'或'历史文化游'"),
        "days": mr.InputField(desc="计划旅行的天数"),
        "budget": mr.InputField(desc="旅行预算上限"),
        "travel_date": mr.InputField(desc="计划出行的日期，格式YYYY-MM-DD"),
        "luxury_level": mr.InputField(desc="旅行的奢华程度：经济、标准或豪华")
    },
    {
        "recommended_destination": mr.OutputField(desc="推荐的旅行目的地"),
        "budget_estimate": mr.OutputField(desc="预算估算"),
        "weather_info": mr.OutputField(desc="目的地天气信息"),
        "attractions": mr.OutputField(desc="主要景点"),
        "itinerary": mr.OutputField(desc="推荐行程")
    },
    instructions="你是一个旅行规划助手，能够帮助用户规划旅行。根据用户提供的信息，你需要推荐合适的目的地、估算预算、提供天气信息、介绍主要景点并生成行程建议。请合理调用工具来完成这些任务，并确保最终提供完整的旅行计划。"
)

def main():
    # 创建工具列表
    tools = [search_destination, check_weather, estimate_budget, get_attractions, generate_itinerary]
    lm = MultiLLMHub().setup_azure_openai()
    # 创建ReAct智能体
    agent = mr.ReAct(signature=travel_planner_signature, tools=tools, max_iters=10, lm=lm)
    
    print("欢迎使用旅行规划助手！")
    print("请告诉我你想去什么样的地方，旅行天数等信息")
    
    destination_query = input("\n目的地或意向> ")
    days = int(input("计划旅行天数> "))
    budget = input("预算上限(可选)> ")
    budget = float(budget) if budget else None
    travel_date = input("出行日期(YYYY-MM-DD)(可选)> ")
    travel_date = travel_date if travel_date else None
    luxury_level = input("奢华程度(经济/标准/豪华)(可选)> ")
    luxury_level = luxury_level if luxury_level else "标准"
    
    # 使用智能体处理旅行规划
    try:
        result = agent(
            destination_query=destination_query,
            days=days,
            budget=budget,
            travel_date=travel_date,
            luxury_level=luxury_level
        )
        
        print("\n===== 您的旅行计划 =====")
        print(f"\n推荐目的地: {result.recommended_destination}")
        print(f"\n预算估算: {result.budget_estimate}")
        print(f"\n天气信息: {result.weather_info}")
        print(f"\n主要景点: {result.attractions}")
        print(f"\n推荐行程: \n{result.itinerary}")
        
        # 可选：打印轨迹信息，用于调试
        if '--debug' in sys.argv:
            print("\n轨迹详情:")
            for key, value in result.trajectory.items():
                if key.startswith('thought'):
                    print(f"\n思考: {value}")
                elif key.startswith('tool_name'):
                    idx = key.split('_')[-1]
                    print(f"工具: {value}")
                    print(f"参数: {result.trajectory.get(f'tool_args_{idx}', {})}")
                    print(f"观察: {result.trajectory.get(f'observation_{idx}', '')}")
            
    except Exception as e:
        print(f"处理旅行规划时出错: {e}")
            
if __name__ == "__main__":
    # 启动主程序
    main() 