"""
数据库格式检测模块

自动从数据库中检测字段格式，用于生成评估prompt中的格式要求
"""

import re
from typing import Dict, List, Optional, Set
from collections import Counter


class FormatDetector:
    """格式检测器 - 从数据库样本中检测格式模式"""
    
    @staticmethod
    def detect_price_format(samples: List[str]) -> str:
        """
        检测价格格式
        
        Returns:
            格式描述，如 "X.XX pounds" 或 "$X.XX"
        """
        if not samples:
            return "numeric value with currency"
        
        # 统计格式模式
        patterns = []
        for sample in samples[:10]:  # 只看前10个样本
            sample = str(sample).strip()
            
            if 'pounds' in sample.lower():
                patterns.append('X.XX pounds')
            elif 'dollar' in sample.lower():
                patterns.append('X.XX dollars')
            elif sample.startswith('£'):
                patterns.append('£X.XX')
            elif sample.startswith('$'):
                patterns.append('$X.XX')
            else:
                patterns.append('numeric')
        
        # 返回最常见的格式
        if patterns:
            most_common = Counter(patterns).most_common(1)[0][0]
            return most_common
        
        return "numeric value with currency"
    
    @staticmethod
    def detect_duration_format(samples: List[str]) -> str:
        """
        检测时长格式
        
        Returns:
            格式描述，如 "X minutes" 或 "X hours Y minutes"
        """
        if not samples:
            return "duration with unit"
        
        patterns = []
        for sample in samples[:10]:
            sample = str(sample).strip().lower()
            
            if 'minutes' in sample or 'mins' in sample:
                if 'hour' in sample:
                    patterns.append('X hours Y minutes')
                else:
                    patterns.append('X minutes')
            elif 'hour' in sample:
                patterns.append('X hours')
            else:
                patterns.append('numeric')
        
        if patterns:
            most_common = Counter(patterns).most_common(1)[0][0]
            return most_common
        
        return "duration with unit (minutes/hours)"
    
    @staticmethod
    def detect_time_format(samples: List[str]) -> str:
        """
        检测时间格式
        
        Returns:
            格式描述，如 "HH:MM" 或 "H:MM AM/PM"
        """
        if not samples:
            return "time in HH:MM format"
        
        patterns = []
        for sample in samples[:10]:
            sample = str(sample).strip()
            
            # 检测是否有前导零
            time_match = re.match(r'(\d{1,2}):(\d{2})', sample)
            if time_match:
                hour = time_match.group(1)
                if len(hour) == 2 and hour.startswith('0'):
                    patterns.append('HH:MM')  # 有前导零
                else:
                    patterns.append('H:MM')   # 没有前导零
            
            # 检测是否有 AM/PM
            if 'AM' in sample.upper() or 'PM' in sample.upper():
                patterns[-1] = patterns[-1] + ' AM/PM' if patterns else 'H:MM AM/PM'
        
        if patterns:
            most_common = Counter(patterns).most_common(1)[0][0]
            return most_common
        
        return "HH:MM (24-hour format)"
    
    @staticmethod
    def detect_boolean_format(samples: List[str]) -> str:
        """
        检测布尔值格式
        
        Returns:
            格式描述，如 "yes/no" 或 "true/false"
        """
        if not samples:
            return "yes/no"
        
        formats = set()
        for sample in samples[:10]:
            sample = str(sample).strip().lower()
            if sample in ['yes', 'no']:
                formats.add('yes/no')
            elif sample in ['true', 'false']:
                formats.add('true/false')
            elif sample in ['1', '0']:
                formats.add('1/0')
        
        if 'yes/no' in formats:
            return 'yes/no'
        elif 'true/false' in formats:
            return 'true/false'
        elif '1/0' in formats:
            return '1/0'
        
        return 'yes/no'


class DatabaseFormatAnalyzer:
    """数据库格式分析器 - 分析整个数据库的格式"""
    
    def __init__(self, db_module):
        """
        初始化
        
        Args:
            db_module: 数据库模块 (autodia.tools.db)
        """
        self.db = db_module
        self._format_cache = {}
    
    def analyze_train_formats(self) -> Dict[str, str]:
        """分析 Train 表的格式"""
        if 'train' in self._format_cache:
            return self._format_cache['train']
        
        # 获取样本数据
        sample_query = {
            'departure': 'norwich',
            'destination': 'cambridge',
            'day': 'monday'
        }
        samples = self.db.query_trains(sample_query)[:20]
        
        if not samples:
            return {}
        
        formats = {}
        
        # 价格格式
        price_samples = [train.price for train in samples if hasattr(train, 'price')]
        if price_samples:
            formats['price'] = FormatDetector.detect_price_format(price_samples)
        
        # 时长格式
        duration_samples = [train.duration for train in samples if hasattr(train, 'duration')]
        if duration_samples:
            formats['duration'] = FormatDetector.detect_duration_format(duration_samples)
        
        # 时间格式
        time_samples = []
        for train in samples:
            if hasattr(train, 'leaveAt'):
                time_samples.append(train.leaveAt)
            if hasattr(train, 'arriveBy'):
                time_samples.append(train.arriveBy)
        if time_samples:
            formats['time'] = FormatDetector.detect_time_format(time_samples)
        
        self._format_cache['train'] = formats
        return formats
    
    def analyze_restaurant_formats(self) -> Dict[str, str]:
        """分析 Restaurant 表的格式"""
        if 'restaurant' in self._format_cache:
            return self._format_cache['restaurant']
        
        # 获取样本
        samples = []
        for area in ['centre', 'north', 'south']:
            try:
                results = self.db.query_venue_by_name('restaurant', 'cheap')  # 使用简单查询
                if results:
                    samples.append(results)
                    if len(samples) >= 5:
                        break
            except:
                pass
        
        formats = {}
        
        # 价格范围格式
        if samples:
            pricerange_samples = [r.pricerange for r in samples if hasattr(r, 'pricerange')]
            if pricerange_samples:
                # 价格范围通常是 cheap/moderate/expensive
                formats['pricerange'] = 'price range (cheap/moderate/expensive)'
        
        self._format_cache['restaurant'] = formats
        return formats
    
    def analyze_hotel_formats(self) -> Dict[str, str]:
        """分析 Hotel 表的格式"""
        if 'hotel' in self._format_cache:
            return self._format_cache['hotel']
        
        # 直接查询一个已知的 hotel
        try:
            result = self.db.query_venue_by_name('hotel', 'cambridge belfry')
            if result:
                samples = [result]
            else:
                samples = []
        except:
            samples = []
        
        formats = {}
        
        if samples:
            # 布尔字段格式
            internet_samples = [str(h.internet) for h in samples if hasattr(h, 'internet')]
            if internet_samples:
                formats['internet'] = FormatDetector.detect_boolean_format(internet_samples)
            
            parking_samples = [str(h.parking) for h in samples if hasattr(h, 'parking')]
            if parking_samples:
                formats['parking'] = FormatDetector.detect_boolean_format(parking_samples)
        
        self._format_cache['hotel'] = formats
        return formats
    
    def analyze_all_formats(self) -> Dict[str, Dict[str, str]]:
        """分析所有表的格式"""
        return {
            'train': self.analyze_train_formats(),
            'restaurant': self.analyze_restaurant_formats(),
            'hotel': self.analyze_hotel_formats(),
            # attraction 和 taxi 可以继续添加
        }
    
    def generate_format_instructions(self, domain: str) -> str:
        """
        生成该领域的格式说明
        
        Args:
            domain: 领域名称
        
        Returns:
            格式说明文本
        """
        formats = self._format_cache.get(domain, {})
        
        if not formats:
            # 尝试分析
            if domain == 'train':
                formats = self.analyze_train_formats()
            elif domain == 'restaurant':
                formats = self.analyze_restaurant_formats()
            elif domain == 'hotel':
                formats = self.analyze_hotel_formats()
        
        if not formats:
            return ""
        
        instructions = []
        
        for field, format_desc in formats.items():
            if field == 'price':
                instructions.append(f"  - Prices: Use format \"{format_desc}\" (e.g., \"17.60 pounds\")")
            elif field == 'duration':
                instructions.append(f"  - Duration: Use format \"{format_desc}\" (e.g., \"79 minutes\")")
            elif field == 'time':
                instructions.append(f"  - Times: Use format \"{format_desc}\" (e.g., \"05:16\")")
            elif field in ['internet', 'parking']:
                instructions.append(f"  - {field.capitalize()}: Use format \"{format_desc}\"")
        
        if instructions:
            return "- **Format Requirements (based on database):**\n" + "\n".join(instructions)
        
        return ""


# 全局分析器实例（懒加载）
_global_analyzer = None

def get_format_analyzer():
    """获取全局格式分析器"""
    global _global_analyzer
    if _global_analyzer is None:
        from taskdialogue.benchmarks.multiwoz.tools import db
        _global_analyzer = DatabaseFormatAnalyzer(db)
    return _global_analyzer


def get_format_instructions_for_domain(domain: str) -> str:
    """
    获取指定领域的格式说明
    
    Args:
        domain: 领域名称 (train, restaurant, hotel, attraction, taxi)
    
    Returns:
        格式说明文本，用于插入到评估prompt中
    """
    analyzer = get_format_analyzer()
    return analyzer.generate_format_instructions(domain)


def preload_all_formats():
    """预加载所有格式（避免评估时的延迟）"""
    analyzer = get_format_analyzer()
    analyzer.analyze_all_formats()
    print("✅ 数据库格式分析完成")
    
    # 打印分析结果
    print("\n📊 检测到的格式:")
    for domain, formats in analyzer._format_cache.items():
        if formats:
            print(f"\n  {domain.upper()}:")
            for field, format_desc in formats.items():
                print(f"    - {field}: {format_desc}")

