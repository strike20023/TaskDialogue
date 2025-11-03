"""
智能槽位值匹配模块
解决评估中格式不匹配导致的误判问题

参考 TD-Eval 的 normalization.py 进行规范化处理
"""

import re
from typing import Any, Optional
from difflib import SequenceMatcher


class SlotValueNormalizer:
    """槽位值标准化器"""
    
    @staticmethod
    def normalize_time(value: str) -> str:
        """
        标准化时间值，统一转换为24小时制
        
        支持的格式:
        - "after 17:15" → "17:15"
        - "by 11:30" → "11:30"
        - "5:50" → "05:50"
        - "12:45 am" → "00:45" (12小时制转24小时制)
        - "1:30 pm" → "13:30" (12小时制转24小时制)
        - "24:45" → "00:45" (24小时制特殊情况)
        """
        if not isinstance(value, str):
            return str(value)
        
        # 转小写
        value = value.lower().strip()
        
        # 去除时间前缀
        time_prefixes = [
            'after ', 'by ', 'before ', 'at ', 'around ',
            'arrive by ', 'leave at ', 'depart at ', 'departing at ',
            'arriving by ', 'arriving at '
        ]
        for prefix in time_prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
        
        # 检查是否是12小时制 (包含 am/pm)
        is_12h = 'am' in value or 'pm' in value
        is_pm = 'pm' in value
        
        # 提取时间 (HH:MM 格式)
        time_match = re.search(r'(\d{1,2}):(\d{2})', value)
        if not time_match:
            return value
        
        hour, minute = time_match.groups()
        hour = int(hour)
        
        # 处理12小时制转24小时制
        if is_12h:
            if is_pm:
                # PM: 12:00 PM = 12:00, 1:00 PM = 13:00
                if hour != 12:
                    hour += 12
            else:
                # AM: 12:00 AM = 00:00, 1:00 AM = 01:00
                if hour == 12:
                    hour = 0
        else:
            # 处理24小时制的特殊情况 (24:xx -> 00:xx)
            if hour >= 24:
                hour = hour - 24
        
        # 标准化为 24小时制 HH:MM 格式
        return f"{hour:02d}:{minute}"
    
    @staticmethod
    def normalize_price(value: str) -> str:
        """
        标准化价格值
        - "£17.60" → "17.60"
        - "17.60 pounds" → "17.60"
        - "$100" → "100"
        """
        if not isinstance(value, str):
            return str(value)
        
        value = value.strip()
        
        # 去除货币符号
        value = re.sub(r'[£$€¥]', '', value)
        
        # 提取数字 (包括小数点)
        price_match = re.search(r'(\d+\.?\d*)', value)
        if price_match:
            return price_match.group(1)
        
        return value
    
    @staticmethod
    def normalize_duration(value: str) -> str:
        """
        标准化时长值
        - "1 hour 30 minutes" → "90 minutes"
        - "1.5 hours" → "90 minutes"
        - "90 mins" → "90 minutes"
        """
        if not isinstance(value, str):
            return str(value)
        
        value = value.lower().strip()
        
        # 提取分钟数
        minutes = 0
        
        # 匹配 "X hours Y minutes"
        hours_match = re.search(r'(\d+\.?\d*)\s*hours?', value)
        if hours_match:
            minutes += int(float(hours_match.group(1)) * 60)
        
        mins_match = re.search(r'(\d+)\s*(?:minutes?|mins?)', value)
        if mins_match:
            minutes += int(mins_match.group(1))
        
        if minutes > 0:
            return f"{minutes} minutes"
        
        # 如果只有数字，假设是分钟
        num_match = re.search(r'(\d+)', value)
        if num_match:
            return f"{num_match.group(1)} minutes"
        
        return value
    
    @staticmethod
    def normalize_name(value: str, domain: Optional[str] = None) -> str:
        """
        标准化名称（参考 TD-Eval 的 name_to_canonical）
        
        处理常见的名称变体，如：
        - "hotel du vin bistro" → "hotel du vin and bistro"
        - "nando's" → "nandos"
        - "&" → "and"
        """
        if not isinstance(value, str):
            return str(value)
        
        # 基础规范化
        value = value.strip().lower()
        value = value.replace(" & ", " and ")
        value = value.replace("&", " and ")
        value = value.replace(" '", "'")
        value = value.replace("bed and breakfast", "b and b")
        
        # 标准化空格
        value = re.sub(r'\s+', ' ', value)
        
        # 领域特定的规范化（参考 TD-Eval）
        if domain == "restaurant" or domain is None:
            venue_mapping = {
                "hotel du vin bistro": "hotel du vin and bistro",
                "the river bar and grill": "the river bar steakhouse and grill",
                "nando's": "nandos",
                "city center b and b": "city center north b and b",
                "acorn house": "acorn guest house",
                "caffee uno": "caffe uno",
                "cafe uno": "caffe uno",
                "rosa's": "rosas bed and breakfast",
                "restaurant called two two": "restaurant two two",
                "restaurant 2 two": "restaurant two two",
                "restaurant two 2": "restaurant two two",
                "restaurant 2 2": "restaurant two two",
                "restaurant 1 7": "restaurant one seven",
                "restaurant 17": "restaurant one seven",
            }
            if value in venue_mapping:
                value = venue_mapping[value]
        
        if domain == "hotel" or domain is None:
            hotel_mapping = {
                "lime house": "limehouse",
                "cityrooms": "cityroomz",
                "whale of time": "whale of a time",
                "huntingdon hotel": "huntingdon marriott hotel",
                "holiday inn exlpress, cambridge": "express by holiday inn cambridge",
                "university hotel": "university arms hotel",
                "arbury guesthouse and lodge": "arbury lodge guesthouse",
                "bridge house": "bridge guest house",
                "arbury guesthouse": "arbury lodge guesthouse",
                "nandos in the city centre": "nandos city centre",
                "a and b guest house": "a and b guesthouse",
                "acorn guesthouse": "acorn guest house",
            }
            if value in hotel_mapping:
                value = hotel_mapping[value]
        
        if domain == "attraction" or domain is None:
            attraction_mapping = {
                "broughton gallery": "broughton house gallery",
                "scudamores punt co": "scudamores punting co",
                "cambridge botanic gardens": "cambridge university botanic gardens",
                "the junction": "junction theatre",
                "trinity street college": "trinity college",
                "christ college": "christ's college",
                "christs": "christ's college",
                "history of science museum": "whipple museum of the history of science",
                "parkside pools": "parkside swimming pool",
                "the botanical gardens at cambridge university": "cambridge university botanic gardens",
                "cafe jello museum": "cafe jello gallery",
            }
            if value in attraction_mapping:
                value = attraction_mapping[value]
        
        return value
    
    @staticmethod
    def normalize_boolean(value: str) -> str:
        """
        标准化布尔值
        - "yes"/"Yes"/"YES" → "yes"
        - "true"/"True" → "yes"
        - "no"/"No"/"NO" → "no"
        - "false"/"False" → "no"
        """
        if not isinstance(value, str):
            return str(value)
        
        value = value.lower().strip()
        
        # 正值
        if value in ['yes', 'true', '1', 'y', 't']:
            return 'yes'
        
        # 负值
        if value in ['no', 'false', '0', 'n', 'f']:
            return 'no'
        
        return value
    
    @staticmethod
    def normalize_generic(value: str) -> str:
        """
        通用标准化
        - 统一小写
        - 去除多余空格
        """
        if not isinstance(value, str):
            return str(value)
        
        return value.lower().strip()


class SmartSlotMatcher:
    """智能槽位匹配器"""
    
    # 槽位类型映射
    TIME_SLOTS = ['leaveat', 'arriveby', 'time', 'leave time', 'arrival time', 'arrive time']
    PRICE_SLOTS = ['price', 'pricerange']
    DURATION_SLOTS = ['duration']
    BOOLEAN_SLOTS = ['parking', 'internet', 'wifi']
    NAME_SLOTS = ['name', 'destination', 'departure', 'area', 'type', 'food', 'day']
    CAR_TYPE_SLOTS = ['car type', 'car_type', 'cartype']  # Taxi car type (color + brand)
    PHONE_SLOTS = ['phone', 'phone number', 'phone_number', 'contact number']  # Phone numbers
    POSTCODE_SLOTS = ['postcode', 'post code', 'post', 'zip', 'zipcode']  # Postcodes
    REFERENCE_SLOTS = ['reference', 'reference number', 'refer number', 'confirmation', 'booking code']  # Reference numbers
    
    def __init__(self, fuzzy_threshold: float = 0.95):
        """
        Args:
            fuzzy_threshold: 模糊匹配阈值 (0-1)，越高越严格
                            默认 0.95 对齐 TD-Eval 的 fuzzy_ratio=95
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.normalizer = SlotValueNormalizer()
    
    def match(self, slot_name: str, value1: str, value2: str) -> bool:
        """
        智能匹配两个槽位值
        
        Args:
            slot_name: 槽位名称，用于判断类型
            value1: 值1 (通常来自goal)
            value2: 值2 (通常来自LLM提取)
        
        Returns:
            是否匹配
        """
        # 处理 None 和 'none'
        if value1 in [None, 'none', ''] or value2 in [None, 'none', '']:
            return value1 == value2 or (value1 in ['none', ''] and value2 in ['none', ''])
        
        # 转换为字符串
        value1 = str(value1)
        value2 = str(value2)
        
        # 精确匹配（标准化后）
        slot_name_lower = slot_name.lower().replace('_', ' ')
        
        # 根据槽位类型选择标准化方法
        if any(time_slot in slot_name_lower for time_slot in self.TIME_SLOTS):
            norm1 = self.normalizer.normalize_time(value1)
            norm2 = self.normalizer.normalize_time(value2)
        elif any(price_slot in slot_name_lower for price_slot in self.PRICE_SLOTS):
            norm1 = self.normalizer.normalize_price(value1)
            norm2 = self.normalizer.normalize_price(value2)
        elif any(dur_slot in slot_name_lower for dur_slot in self.DURATION_SLOTS):
            norm1 = self.normalizer.normalize_duration(value1)
            norm2 = self.normalizer.normalize_duration(value2)
        elif any(bool_slot in slot_name_lower for bool_slot in self.BOOLEAN_SLOTS):
            norm1 = self.normalizer.normalize_boolean(value1)
            norm2 = self.normalizer.normalize_boolean(value2)
        elif any(car_slot in slot_name_lower for car_slot in self.CAR_TYPE_SLOTS):
            # Car type: 直接比较，不区分大小写
            norm1 = self.normalizer.normalize_generic(value1)
            norm2 = self.normalizer.normalize_generic(value2)
        elif any(phone_slot in slot_name_lower for phone_slot in self.PHONE_SLOTS):
            # Phone: 只比较数字部分
            norm1 = re.sub(r'[^\d]', '', str(value1))
            norm2 = re.sub(r'[^\d]', '', str(value2))
        elif any(postcode_slot in slot_name_lower for postcode_slot in self.POSTCODE_SLOTS):
            # Postcode: 移除空格，统一大小写
            norm1 = re.sub(r'\s+', '', str(value1)).upper()
            norm2 = re.sub(r'\s+', '', str(value2)).upper()
        elif any(ref_slot in slot_name_lower for ref_slot in self.REFERENCE_SLOTS):
            # Reference number: 移除空格、统一大小写、只保留字母数字
            norm1 = re.sub(r'[^A-Z0-9]', '', str(value1).upper())
            norm2 = re.sub(r'[^A-Z0-9]', '', str(value2).upper())
        elif any(name_slot in slot_name_lower for name_slot in self.NAME_SLOTS):
            norm1 = self.normalizer.normalize_name(value1)
            norm2 = self.normalizer.normalize_name(value2)
        else:
            # 通用标准化
            norm1 = self.normalizer.normalize_generic(value1)
            norm2 = self.normalizer.normalize_generic(value2)
        
        # 精确匹配
        if norm1 == norm2:
            return True
        
        # 模糊匹配 (用于处理拼写差异)
        similarity = self._string_similarity(norm1, norm2)
        return similarity >= self.fuzzy_threshold
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """计算两个字符串的相似度 (0-1)"""
        return SequenceMatcher(None, s1, s2).ratio()


# 全局实例
_matcher = SmartSlotMatcher()


def smart_match(slot_name: str, value1: Any, value2: Any) -> bool:
    """
    智能匹配两个槽位值（便捷函数）
    
    Args:
        slot_name: 槽位名称
        value1: 值1 (通常来自goal)
        value2: 值2 (通常来自LLM提取)
    
    Returns:
        是否匹配
    """
    return _matcher.match(slot_name, value1, value2)


def normalize_slot_value(slot_name: str, value: Any) -> str:
    """
    标准化槽位值（便捷函数）
    
    Args:
        slot_name: 槽位名称
        value: 原始值
    
    Returns:
        标准化后的值
    """
    if value in [None, 'none', '']:
        return 'none'
    
    value = str(value)
    slot_name_lower = slot_name.lower().replace('_', ' ')
    normalizer = SlotValueNormalizer()
    
    # 根据槽位类型选择标准化方法
    if any(time_slot in slot_name_lower for time_slot in SmartSlotMatcher.TIME_SLOTS):
        return normalizer.normalize_time(value)
    elif any(price_slot in slot_name_lower for price_slot in SmartSlotMatcher.PRICE_SLOTS):
        return normalizer.normalize_price(value)
    elif any(dur_slot in slot_name_lower for dur_slot in SmartSlotMatcher.DURATION_SLOTS):
        return normalizer.normalize_duration(value)
    elif any(bool_slot in slot_name_lower for bool_slot in SmartSlotMatcher.BOOLEAN_SLOTS):
        return normalizer.normalize_boolean(value)
    elif any(car_slot in slot_name_lower for car_slot in SmartSlotMatcher.CAR_TYPE_SLOTS):
        return normalizer.normalize_generic(value)
    elif any(phone_slot in slot_name_lower for phone_slot in SmartSlotMatcher.PHONE_SLOTS):
        # Phone: 只保留数字
        import re
        return re.sub(r'[^\d]', '', str(value))
    elif any(name_slot in slot_name_lower for name_slot in SmartSlotMatcher.NAME_SLOTS):
        return normalizer.normalize_name(value)
    else:
        return normalizer.normalize_generic(value)

