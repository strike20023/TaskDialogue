"""
数据库 Schema 和枚举值提取模块
从实际数据库中动态提取可选值，用于评估提示和验证
"""

from typing import Dict, Set, List, Optional
from collections import defaultdict
from taskdialogue.benchmarks.multiwoz.tools import db

# SQLite 连接配置（用于多进程环境）
SQLITE_CONNECT_ARGS = {
    'timeout': 30,  # 30 秒超时
    'check_same_thread': False,  # 允许多线程访问
}


class DatabaseSchema:
    """数据库 Schema 分析器 - 从实际数据中提取枚举值"""
    
    def __init__(self):
        self._enum_cache = {}
        self._initialized = False
    
    def initialize(self):
        """初始化：扫描数据库获取所有枚举值"""
        if self._initialized:
            return
        
        print("📊 正在分析数据库 Schema...")
        
        # Restaurant
        self._enum_cache['restaurant'] = self._extract_restaurant_enums()
        
        # Hotel
        self._enum_cache['hotel'] = self._extract_hotel_enums()
        
        # Attraction
        self._enum_cache['attraction'] = self._extract_attraction_enums()
        
        # Train
        self._enum_cache['train'] = self._extract_train_enums()
        
        self._initialized = True
        print("✅ 数据库 Schema 分析完成")
    
    def _extract_restaurant_enums(self) -> Dict[str, Set[str]]:
        """提取 Restaurant 的枚举值"""
        enums = defaultdict(set)
        
        # 采样查询不同 area
        # 使用 SQL 直接查询以获取所有可能值
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from taskdialogue.benchmarks.multiwoz.tools.utils import DB_PATH
            
            engine = create_engine(f'sqlite:///{DB_PATH}', connect_args=SQLITE_CONNECT_ARGS)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            # 导入 Restaurant 类
            from taskdialogue.benchmarks.multiwoz.tools.db import Restaurant
            
            # 查询所有餐厅
            restaurants = session.query(Restaurant).limit(200).all()
            for venue in restaurants:
                if venue.area:
                    enums['area'].add(venue.area.lower())
                if venue.food:
                    enums['food'].add(venue.food.lower())
                if venue.pricerange:
                    enums['pricerange'].add(venue.pricerange.lower())
            
            session.close()
        except Exception as e:
            print(f"  ⚠️  Restaurant Schema 提取失败: {e}")
        
        return dict(enums)
    
    def _extract_hotel_enums(self) -> Dict[str, Set[str]]:
        """提取 Hotel 的枚举值"""
        enums = defaultdict(set)
        
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from taskdialogue.benchmarks.multiwoz.tools.utils import DB_PATH
            from taskdialogue.benchmarks.multiwoz.tools.db import Hotel
            
            engine = create_engine(f'sqlite:///{DB_PATH}', connect_args=SQLITE_CONNECT_ARGS)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            hotels = session.query(Hotel).limit(200).all()
            for venue in hotels:
                if venue.area:
                    enums['area'].add(venue.area.lower())
                if venue.pricerange:
                    enums['pricerange'].add(venue.pricerange.lower())
                if venue.type:
                    enums['type'].add(venue.type.lower())
                if venue.stars:
                    enums['stars'].add(str(venue.stars))
                if venue.internet:
                    enums['internet'].add(venue.internet.lower())
                if venue.parking:
                    enums['parking'].add(venue.parking.lower())
            
            session.close()
        except Exception as e:
            print(f"  ⚠️  Hotel Schema 提取失败: {e}")
        
        return dict(enums)
    
    def _extract_attraction_enums(self) -> Dict[str, Set[str]]:
        """提取 Attraction 的枚举值"""
        enums = defaultdict(set)
        
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from taskdialogue.benchmarks.multiwoz.tools.utils import DB_PATH
            from taskdialogue.benchmarks.multiwoz.tools.db import Attraction
            
            engine = create_engine(f'sqlite:///{DB_PATH}', connect_args=SQLITE_CONNECT_ARGS)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            attractions = session.query(Attraction).limit(200).all()
            for venue in attractions:
                if venue.area:
                    enums['area'].add(venue.area.lower())
                if venue.type:
                    enums['type'].add(venue.type.lower())
                if venue.entrance_fee:
                    # 保留原始值（free, 5 pounds 等）
                    enums['entrance_fee'].add(venue.entrance_fee.lower())
            
            session.close()
        except Exception as e:
            print(f"  ⚠️  Attraction Schema 提取失败: {e}")
        
        return dict(enums)
    
    def _extract_train_enums(self) -> Dict[str, Set[str]]:
        """提取 Train 的枚举值（主要是格式）"""
        enums = {}
        
        # Train 的槽位主要是格式，不是枚举
        # 但可以提供示例
        try:
            sample_trains = db.query_trains({
                'departure': 'cambridge',
                'destination': 'london kings cross',
                'day': 'monday'
            })
            
            if sample_trains:
                train = sample_trains[0]
                enums['price_format'] = getattr(train, 'price', '17.60 pounds')
                enums['duration_format'] = getattr(train, 'duration', '79 minutes')
                enums['trainID_format'] = getattr(train, 'trainID', 'TR1234')
        except Exception as e:
            print(f"  ⚠️  Train 查询失败: {e}")
        
        return enums
    
    def get_enum_values(self, domain: str, slot: str) -> Optional[Set[str]]:
        """
        获取指定领域和槽位的枚举值
        
        Args:
            domain: 领域名称
            slot: 槽位名称
        
        Returns:
            枚举值集合，如果没有则返回 None
        """
        if not self._initialized:
            self.initialize()
        
        # 标准化槽位名称
        slot_normalized = slot.replace(' ', '_').lower()
        
        if domain in self._enum_cache:
            return self._enum_cache[domain].get(slot_normalized)
        
        return None
    
    def get_enum_hint(self, domain: str, slot: str, max_items: int = 15) -> str:
        """
        获取枚举值的提示文本
        
        Args:
            domain: 领域名称
            slot: 槽位名称
            max_items: 最多显示的枚举项数
        
        Returns:
            格式化的提示文本
        """
        enum_values = self.get_enum_values(domain, slot)
        
        if not enum_values:
            return ''
        
        # 转换为列表并排序
        values_list = sorted(list(enum_values))
        
        if len(values_list) > max_items:
            # 太多了，只显示部分 + "..."
            shown = values_list[:max_items]
            return ', '.join(shown) + ', ...'
        else:
            return ', '.join(values_list)
    
    def format_slot_question_with_enum(self, question: str, domain: str, slot: str) -> str:
        """
        为问题添加枚举值提示
        
        Args:
            question: 原始问题
            domain: 领域名称
            slot: 槽位名称
        
        Returns:
            增强后的问题
        """
        enum_hint = self.get_enum_hint(domain, slot)
        
        if enum_hint:
            # 特殊处理 entrance_fee（保留原始格式）
            if slot == 'entrance fee' or slot == 'entrance_fee':
                return question + f'\n   NOTE: Extract EXACTLY as mentioned. Common values: {enum_hint}'
            else:
                return question + f' (Options: {enum_hint})'
        
        return question
    
    def print_schema_summary(self):
        """打印 Schema 摘要"""
        if not self._initialized:
            self.initialize()
        
        print("\n" + "="*80)
        print("📊 数据库 Schema 摘要")
        print("="*80)
        
        for domain, enums in self._enum_cache.items():
            print(f"\n{domain.upper()}:")
            for slot, values in enums.items():
                if isinstance(values, set):
                    count = len(values)
                    sample = list(sorted(values))[:5]
                    print(f"  • {slot:<20s}: {count:>3d} 个值 (示例: {', '.join(sample)})")
                else:
                    print(f"  • {slot:<20s}: {values}")


# 全局单例
_db_schema = None


def get_db_schema() -> DatabaseSchema:
    """获取数据库 Schema 单例"""
    global _db_schema
    if _db_schema is None:
        _db_schema = DatabaseSchema()
    return _db_schema


def get_slot_enum_hint(domain: str, slot: str) -> str:
    """
    获取槽位的枚举提示（便捷函数）
    
    Args:
        domain: 领域名称
        slot: 槽位名称
    
    Returns:
        枚举值提示文本
    """
    schema = get_db_schema()
    return schema.get_enum_hint(domain, slot)


def preload_db_schema():
    """预加载数据库 Schema（避免评估时的延迟）"""
    schema = get_db_schema()
    schema.initialize()
    schema.print_schema_summary()

