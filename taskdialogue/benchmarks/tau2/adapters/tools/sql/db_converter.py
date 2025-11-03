"""数据库转换工具：将 Tau2 的 JSON 数据库转换为 SQLite。

参考 MultiWOZ 的数据库实现，将 Pydantic 模型转换为 SQLite 表。
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from taskdialogue.core.utils.logger import get_logger

logger = get_logger(__name__)


def convert_flightdb_to_sqlite(json_path: Path, sqlite_path: Path) -> None:
    """将 FlightDB JSON 转换为 SQLite 数据库。
    
    Args:
        json_path: 输入 JSON 文件路径
        sqlite_path: 输出 SQLite 文件路径
    """
    logger.info(f"📊 转换 Airline 数据库: {json_path} → {sqlite_path}")
    
    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保输出目录存在
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建 SQLite 数据库
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    try:
        # 创建表结构
        _create_airline_tables(cursor)
        
        # 转换 flights（需要处理嵌套的 dates）
        flights = data.get('flights', {})
        _convert_flights(cursor, flights)
        logger.info(f"  ✓ 转换 {len(flights)} 个航班")
        
        # 转换 users
        users = data.get('users', {})
        _convert_users(cursor, users)
        logger.info(f"  ✓ 转换 {len(users)} 个用户")
        
        # 转换 reservations
        reservations = data.get('reservations', {})
        _convert_reservations(cursor, reservations)
        logger.info(f"  ✓ 转换 {len(reservations)} 个预订")
        
        # 创建索引
        _create_airline_indexes(cursor)
        
        conn.commit()
        logger.info(f"✅ 数据库转换完成: {sqlite_path}")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 数据库转换失败: {e}")
        raise
    finally:
        conn.close()


def _create_airline_tables(cursor: sqlite3.Cursor) -> None:
    """创建 Airline 数据库表结构。"""
    
    # flights 表（主表，存储航班基本信息）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            flight_number TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            scheduled_departure_time_est TEXT NOT NULL,
            scheduled_arrival_time_est TEXT NOT NULL
        )
    """)
    
    # flight_dates 表（存储航班按日期的状态信息）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flight_dates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_number TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            available_seats_business INTEGER,
            available_seats_economy INTEGER,
            available_seats_basic_economy INTEGER,
            price_business INTEGER,
            price_economy INTEGER,
            price_basic_economy INTEGER,
            estimated_departure_time_est TEXT,
            estimated_arrival_time_est TEXT,
            actual_departure_time_est TEXT,
            actual_arrival_time_est TEXT,
            FOREIGN KEY (flight_number) REFERENCES flights(flight_number),
            UNIQUE(flight_number, date)
        )
    """)
    
    # users 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL,
            dob TEXT NOT NULL,
            membership TEXT NOT NULL,
            address1 TEXT,
            address2 TEXT,
            city TEXT,
            state TEXT,
            country TEXT,
            zip TEXT
        )
    """)
    
    # user_payment_methods 表（用户支付方式）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_payment_methods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            payment_id TEXT NOT NULL,
            source TEXT NOT NULL,
            brand TEXT,
            last_four TEXT,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, payment_id)
        )
    """)
    
    # user_passengers 表（用户保存的乘客信息）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_passengers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dob TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # reservations 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            reservation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            flight_type TEXT NOT NULL,
            cabin TEXT NOT NULL,
            total_baggages INTEGER NOT NULL,
            nonfree_baggages INTEGER NOT NULL,
            insurance TEXT NOT NULL,
            status TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # reservation_flights 表（预订中的航班）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservation_flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reservation_id TEXT NOT NULL,
            flight_number TEXT NOT NULL,
            date TEXT NOT NULL,
            price INTEGER NOT NULL,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id)
        )
    """)
    
    # reservation_passengers 表（预订中的乘客）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservation_passengers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reservation_id TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dob TEXT NOT NULL,
            FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id)
        )
    """)
    
    # reservation_payments 表（预订的支付记录）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservation_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reservation_id TEXT NOT NULL,
            payment_id TEXT NOT NULL,
            amount INTEGER NOT NULL,
            FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id)
        )
    """)
    
    # user_reservations 表（用户-预订关联表）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_reservations (
            user_id TEXT NOT NULL,
            reservation_id TEXT NOT NULL,
            PRIMARY KEY (user_id, reservation_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id)
        )
    """)


def _convert_flights(cursor: sqlite3.Cursor, flights: Dict[str, Any]) -> None:
    """转换 flights 数据。"""
    for flight_number, flight in flights.items():
        # 插入主航班记录
        cursor.execute("""
            INSERT OR REPLACE INTO flights (
                flight_number, origin, destination,
                scheduled_departure_time_est, scheduled_arrival_time_est
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            flight.get('flight_number', flight_number),
            flight.get('origin'),
            flight.get('destination'),
            flight.get('scheduled_departure_time_est'),
            flight.get('scheduled_arrival_time_est'),
        ))
        
        # 处理 dates（嵌套字典）
        dates = flight.get('dates', {})
        for date, date_status in dates.items():
            status = date_status.get('status')
            
            # 提取可用座位和价格（如果状态是 available）
            available_seats = date_status.get('available_seats', {}) if isinstance(date_status, dict) else {}
            prices = date_status.get('prices', {}) if isinstance(date_status, dict) else {}
            
            # 提取时间信息
            estimated_departure = date_status.get('estimated_departure_time_est')
            estimated_arrival = date_status.get('estimated_arrival_time_est')
            actual_departure = date_status.get('actual_departure_time_est')
            actual_arrival = date_status.get('actual_arrival_time_est')
            
            cursor.execute("""
                INSERT OR REPLACE INTO flight_dates (
                    flight_number, date, status,
                    available_seats_business, available_seats_economy, available_seats_basic_economy,
                    price_business, price_economy, price_basic_economy,
                    estimated_departure_time_est, estimated_arrival_time_est,
                    actual_departure_time_est, actual_arrival_time_est
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flight_number, date, status,
                available_seats.get('business'),
                available_seats.get('economy'),
                available_seats.get('basic_economy'),
                prices.get('business'),
                prices.get('economy'),
                prices.get('basic_economy'),
                estimated_departure,
                estimated_arrival,
                actual_departure,
                actual_arrival,
            ))


def _convert_users(cursor: sqlite3.Cursor, users: Dict[str, Any]) -> None:
    """转换 users 数据。"""
    for user_id, user in users.items():
        # 提取姓名和地址
        name = user.get('name', {})
        address = user.get('address', {})
        
        # 插入用户主记录
        cursor.execute("""
            INSERT OR REPLACE INTO users (
                user_id, first_name, last_name, email, dob, membership,
                address1, address2, city, state, country, zip
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            name.get('first_name'),
            name.get('last_name'),
            user.get('email'),
            user.get('dob'),
            user.get('membership'),
            address.get('address1'),
            address.get('address2'),
            address.get('city'),
            address.get('state'),
            address.get('country'),
            address.get('zip'),
        ))
        
        # 处理支付方式
        payment_methods = user.get('payment_methods', {})
        for payment_id, payment_method in payment_methods.items():
            source = payment_method.get('source')
            cursor.execute("""
                INSERT OR REPLACE INTO user_payment_methods (
                    user_id, payment_id, source, brand, last_four, amount
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                payment_id,
                source,
                payment_method.get('brand'),
                payment_method.get('last_four'),
                payment_method.get('amount'),
            ))
        
        # 处理保存的乘客信息
        saved_passengers = user.get('saved_passengers', [])
        for passenger in saved_passengers:
            cursor.execute("""
                INSERT INTO user_passengers (
                    user_id, first_name, last_name, dob
                ) VALUES (?, ?, ?, ?)
            """, (
                user_id,
                passenger.get('first_name'),
                passenger.get('last_name'),
                passenger.get('dob'),
            ))
        
        # 处理用户预订关联
        user_reservations = user.get('reservations', [])
        for reservation_id in user_reservations:
            cursor.execute("""
                INSERT OR IGNORE INTO user_reservations (
                    user_id, reservation_id
                ) VALUES (?, ?)
            """, (user_id, reservation_id))


def _convert_reservations(cursor: sqlite3.Cursor, reservations: Dict[str, Any]) -> None:
    """转换 reservations 数据。"""
    for reservation_id, reservation in reservations.items():
        # 插入预订主记录
        cursor.execute("""
            INSERT OR REPLACE INTO reservations (
                reservation_id, user_id, origin, destination, flight_type, cabin,
                total_baggages, nonfree_baggages, insurance, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reservation_id,
            reservation.get('user_id'),
            reservation.get('origin'),
            reservation.get('destination'),
            reservation.get('flight_type'),
            reservation.get('cabin'),
            reservation.get('total_baggages'),
            reservation.get('nonfree_baggages'),
            reservation.get('insurance'),
            reservation.get('status'),
            reservation.get('created_at'),
        ))
        
        # 处理预订中的航班
        flights = reservation.get('flights', [])
        for flight in flights:
            cursor.execute("""
                INSERT INTO reservation_flights (
                    reservation_id, flight_number, date, price, origin, destination
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                reservation_id,
                flight.get('flight_number'),
                flight.get('date'),
                flight.get('price'),
                flight.get('origin'),
                flight.get('destination'),
            ))
        
        # 处理预订中的乘客
        passengers = reservation.get('passengers', [])
        for passenger in passengers:
            cursor.execute("""
                INSERT INTO reservation_passengers (
                    reservation_id, first_name, last_name, dob
                ) VALUES (?, ?, ?, ?)
            """, (
                reservation_id,
                passenger.get('first_name'),
                passenger.get('last_name'),
                passenger.get('dob'),
            ))
        
        # 处理预订的支付记录
        payment_history = reservation.get('payment_history', [])
        for payment in payment_history:
            cursor.execute("""
                INSERT INTO reservation_payments (
                    reservation_id, payment_id, amount
                ) VALUES (?, ?, ?)
            """, (
                reservation_id,
                payment.get('payment_id'),
                payment.get('amount'),
            ))


def _create_airline_indexes(cursor: sqlite3.Cursor) -> None:
    """创建索引以提高查询性能。"""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_flight_dates_flight_number ON flight_dates(flight_number)",
        "CREATE INDEX IF NOT EXISTS idx_flight_dates_date ON flight_dates(date)",
        "CREATE INDEX IF NOT EXISTS idx_flight_dates_status ON flight_dates(status)",
        "CREATE INDEX IF NOT EXISTS idx_user_payment_methods_user_id ON user_payment_methods(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_user_passengers_user_id ON user_passengers(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_reservations_user_id ON reservations(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_reservation_flights_reservation_id ON reservation_flights(reservation_id)",
        "CREATE INDEX IF NOT EXISTS idx_reservation_flights_flight_number ON reservation_flights(flight_number)",
        "CREATE INDEX IF NOT EXISTS idx_user_reservations_user_id ON user_reservations(user_id)",
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)


def convert_retaildb_to_sqlite(json_path: Path, sqlite_path: Path) -> None:
    """将 RetailDB JSON 转换为 SQLite 数据库。
    
    Args:
        json_path: 输入 JSON 文件路径
        sqlite_path: 输出 SQLite 文件路径
    """
    logger.warning("RetailDB 转换功能尚未实现")
    raise NotImplementedError("RetailDB conversion not implemented yet")


def convert_telecomdb_to_sqlite(json_path: Path, sqlite_path: Path) -> None:
    """将 TelecomDB JSON 转换为 SQLite 数据库。
    
    Args:
        json_path: 输入 JSON 文件路径
        sqlite_path: 输出 SQLite 文件路径
    """
    logger.warning("TelecomDB 转换功能尚未实现")
    raise NotImplementedError("TelecomDB conversion not implemented yet")


def convert_domain_db_to_sqlite(domain: str, json_path: Optional[Path] = None, sqlite_path: Optional[Path] = None) -> Path:
    """转换指定 domain 的数据库（自动检测路径）。
    
    Args:
        domain: 域名 (airline, retail, telecom)
        json_path: 输入 JSON 文件路径（可选，默认从 DATA_DIR 读取）
        sqlite_path: 输出 SQLite 文件路径（可选，默认保存到 data/tau2/sql/）
    
    Returns:
        生成的 SQLite 文件路径
    """
    from taskdialogue.benchmarks.tau2.utils.utils import DATA_DIR
    
    # 自动检测输入路径
    if json_path is None:
        json_path = DATA_DIR / "tau2" / "domains" / domain / "db.json"
    
    # 自动生成输出路径
    if sqlite_path is None:
        sqlite_path = DATA_DIR / "tau2" / "sql" / f"{domain}.db"
    
    # 确保输出目录存在
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 调用对应的转换函数
    if domain == "airline":
        convert_flightdb_to_sqlite(json_path, sqlite_path)
    elif domain == "retail":
        convert_retaildb_to_sqlite(json_path, sqlite_path)
    elif domain == "telecom":
        convert_telecomdb_to_sqlite(json_path, sqlite_path)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    return sqlite_path

