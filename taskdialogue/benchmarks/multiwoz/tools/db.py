import sqlite3
from collections import OrderedDict

from sqlalchemy import Column, Integer, String, create_engine, or_
from sqlalchemy.orm import declarative_base, sessionmaker

from taskdialogue.benchmarks.multiwoz.tools.utils import DB_PATH, TableItem

# DB_PATH = 'multiwoz.db'

# SQLite 连接配置（用于多进程环境）
SQLITE_CONNECT_ARGS = {
    'timeout': 30,  # 30 秒超时
    'check_same_thread': False,  # 允许多线程访问
}


def clean_name(name):
    """Clean venue name for database queries."""
    if not name:
        return name
    return name.strip().lower()


class Veune(TableItem):

    def satisfying(self, constraint):
        """
        检查venue是否满足约束条件
        使用智能槽位匹配，支持格式变化
        """
        # Import smart matcher
        try:
            from taskdialogue.benchmarks.multiwoz.evaluators.matching.slot_matcher import smart_match
        except ImportError:
            # Fallback to old logic if import fails
            smart_match = None
        
        # Clean
        cons = {}
        for slot, value in constraint.items():
            if value in ['dontcare', '', 'none', 'not mentioned']:
                continue
            if slot in ['postcode', 'phone']:
                value = ''.join(x for x in value if x != ' ')
            if slot == 'entrance fee':
                cons['entrance_fee'] = value
            else:
                cons[slot] = value

        # Check
        for slot, cons_value in cons.items():
            db_value = getattr(self, slot, None)
            if db_value is None:
                return False
            
            # Special handling for address (partial match)
            if slot == 'address':
                if not (str(db_value).lower() in str(cons_value).lower()):
                    return False
            else:
                # Use smart match if available
                if smart_match:
                    if not smart_match(slot, db_value, cons_value):
                        return False
                else:
                    # Fallback: case-insensitive comparison
                    if str(db_value).lower() != str(cons_value).lower():
                        return False
        
        return True


Base = declarative_base()


class Restaurant(Base, Veune):
    __tablename__ = 'restaurant'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    area = Column(String)
    pricerange = Column(String)
    food = Column(String)
    type = Column(String)
    phone = Column(String)
    postcode = Column(String)
    address = Column(String)

    def items(self):
        return OrderedDict(
            name=self.name,
            area=self.area,
            pricerange=self.pricerange,
            food=self.food,
            type=self.type,
            phone=self.phone,
            postcode=self.postcode,
            address=self.address,
        ).items()


class Hotel(Base, Veune):
    __tablename__ = 'hotel'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    area = Column(String)
    internet = Column(String)
    parking = Column(String)
    pricerange = Column(String)
    stars = Column(String)
    phone = Column(String)
    address = Column(String)
    postcode = Column(String)

    def items(self):
        return OrderedDict(
            name=self.name,
            type=self.type,
            area=self.area,
            internet=self.internet,
            parking=self.parking,
            pricerange=self.pricerange,
            stars=self.stars,
            phone=self.phone,
            address=self.address,
            postcode=self.postcode,
        ).items()


class Attraction(Base, Veune):
    __tablename__ = 'attraction'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    area = Column(String)
    phone = Column(String)
    address = Column(String)
    postcode = Column(String)
    entrance_fee = Column(String)

    def items(self):
        return OrderedDict(
            name=self.name,
            type=self.type,
            area=self.area,
            phone=self.phone,
            address=self.address,
            postcode=self.postcode,
            entrance_fee=self.entrance_fee,
        ).items()


class Train(Base, Veune):
    __tablename__ = 'train'

    id = Column(Integer, primary_key=True)
    arriveBy = Column(String)
    day = Column(String)
    departure = Column(String)
    destination = Column(String)
    leaveAt = Column(String)
    price = Column(String)
    trainID = Column(String)
    duration = Column(String)

    def items(self):
        return OrderedDict(
            trainID=self.trainID,
            departure=self.departure,
            destination=self.destination,
            day=self.day,
            leaveAt=self.leaveAt,
            arriveBy=self.arriveBy,
            price=self.price,
            duration=self.duration,
        ).items()

    def satisfying(self, constraint: dict):
        """
        检查train是否满足约束条件
        使用智能槽位匹配，支持格式变化
        """
        # Import smart matcher
        try:
            from taskdialogue.benchmarks.multiwoz.evaluators.slot_matcher import smart_match
        except ImportError:
            smart_match = None
        
        for slot, cons_value in constraint.items():
            train_value = getattr(self, slot, None)
            if train_value is None:
                return False
            elif slot == 'leaveAt':
                # 时间比较：train的出发时间要晚于或等于要求
                if train_value < cons_value:
                    return False
            elif slot == 'arriveBy':
                # 时间比较：train的到达时间要早于或等于要求
                if train_value > cons_value:
                    return False
            else:
                # Use smart match for other slots
                if smart_match:
                    if not smart_match(slot, train_value, cons_value):
                        return False
                else:
                    # Fallback: case-insensitive comparison
                    if isinstance(train_value, str) and isinstance(cons_value, str):
                        if train_value.lower() != cons_value.lower():
                            return False
                    else:
                        if train_value != cons_value:
                            return False
        return True
    

DOMAIN_CLASS_MAP = {
    'restaurant': Restaurant,
    'hotel': Hotel,
    'attraction': Attraction,
    'train': Train,
}


def query_venue_by_name(domain, name, db_path=DB_PATH):
    '''Return one Venue object or None.'''
    assert domain in DOMAIN_CLASS_MAP

    engine = create_engine(f'sqlite:///{db_path}', connect_args=SQLITE_CONNECT_ARGS)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(DOMAIN_CLASS_MAP[domain])
    name = clean_name(name)
    items = items.filter_by(name=name)
    item = items.first()
    return item


def query_venue_by_name_or_address(domain, place, db_path=DB_PATH):
    '''Return one Venue object or None.'''
    assert domain in DOMAIN_CLASS_MAP
    engine = create_engine(f'sqlite:///{db_path}', connect_args=SQLITE_CONNECT_ARGS)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    Venue = DOMAIN_CLASS_MAP[domain]
    items = session.query(Venue)
    items = items.filter(or_(Venue.name == clean_name(place), Venue.address == place))
    item = items.first()
    return item


def query_train_by_id(id, db_path=DB_PATH):
    '''Return one Train object or None.'''
    engine = create_engine(f'sqlite:///{db_path}', connect_args=SQLITE_CONNECT_ARGS)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(Train)
    items = items.filter_by(trainID=id)
    item = items.first()
    return item


def query_trains(info, db_path=DB_PATH):
    engine = create_engine(f'sqlite:///{db_path}', connect_args=SQLITE_CONNECT_ARGS)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(Train)

    sub_info = {s: info[s] for s in ['day', 'departure', 'destination', 'trainID'] if s in info}
    items = items.filter_by(**sub_info)
    if time := info.get('leaveAt'):
        items = items.filter(Train.leaveAt >= time)
    if time := info.get('arriveBy'):
        items = items.filter(Train.arriveBy <= time)

    return items.all()


def query_by_sql(sql, db_path=DB_PATH):
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.execute(sql)
    records = cursor.fetchall()
    return records
