import random
import re
import sqlite3

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from taskdialogue.benchmarks.multiwoz.tools.db import query_venue_by_name_or_address, SQLITE_CONNECT_ARGS
from taskdialogue.benchmarks.multiwoz.tools.utils import DB_PATH, BOOK_DB_PATH, TableItem, clean_time


# region: DB Define

class BookRecord(TableItem):

    def satisfying(self, constraint):
        """
        检查booking记录是否满足约束条件
        使用智能槽位匹配，支持格式变化
        """
        # Import smart matcher
        try:
            from taskdialogue.evaluators.matching.slot_matcher import smart_match
        except ImportError:
            smart_match = None
        
        # Clean
        cons = {}
        for slot, value in constraint.items():
            if 'invalid' in slot:
                continue
            cons[slot] = value

        # Check
        for slot, cons_value in cons.items():
            db_value = getattr(self, slot, None)
            if db_value is None:
                return False
            
            # Use smart match if available
            if smart_match:
                if not smart_match(slot, db_value, cons_value):
                    return False
            else:
                # Fallback: case-insensitive comparison
                if isinstance(db_value, str) and isinstance(cons_value, str):
                    if db_value.lower() != cons_value.lower():
                        return False
                else:
                    if db_value != cons_value:
                        return False
        
        return True
    

Base = declarative_base()


class RestaurantBook(Base, BookRecord):
    __tablename__ = 'restaurant_book'

    id = Column(Integer, primary_key=True)
    refer_number = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    people = Column(String, nullable=False)
    day = Column(String, nullable=False)
    time = Column(String, nullable=False)


class HotelBook(Base, BookRecord):
    __tablename__ = 'hotel_book'

    id = Column(Integer, primary_key=True)
    refer_number = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    people = Column(String, nullable=False)
    day = Column(String, nullable=False)
    stay = Column(String, nullable=False)


class TrainBook(Base, BookRecord):
    __tablename__ = 'train_book'

    id = Column(Integer, primary_key=True)
    refer_number = Column(String, nullable=False, unique=True)
    trainID = Column(String, nullable=False)
    tickets = Column(String, nullable=False)


def generate_reference_num():
    # genereate a 8 character long reference number with random lower letters and numbers
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for i in range(8))


DOMAIN_BOOK_CLASS_MAP = {
    'restaurant': RestaurantBook,
    'hotel': HotelBook,
    'train': TrainBook,
}


def check_db_exist(table, column, value, db_path=None):
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)
    sql = f'SELECT {column} FROM {table} WHERE {column} = "{value}"'
    result = conn.execute(sql)
    if result.fetchone():
        return True
    else:
        return False


def query_booking_by_refer_num(domain, refer_number, book_db_path=BOOK_DB_PATH):
    '''Return one Book object or None.'''
    assert domain in DOMAIN_BOOK_CLASS_MAP

    engine = create_engine(f'sqlite:///{book_db_path}', connect_args=SQLITE_CONNECT_ARGS)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(DOMAIN_BOOK_CLASS_MAP[domain])
    items = items.filter_by(refer_number=refer_number)
    item = items.first()
    return item
    
# endregion    

# region: DB Booking: restaurant, hotel, train

def make_booking_db(domain, info, book_db_path=BOOK_DB_PATH, db_path=None):  # TODO: Refactor: split by domain
    assert domain in DOMAIN_BOOK_CLASS_MAP
    
    # Use default db_path if not provided
    if db_path is None:
        db_path = DB_PATH

    # Clean (lower) - 但保持 train ID 的大小写
    if domain == 'train' and 'train id' in info:
        train_id = info['train id']  # 保存原始 train ID
        info = {k: v.lower() for k, v in info.items()}
        info['train id'] = train_id  # 恢复 train ID 的原始大小写
    else:
        info = {k: v.lower() for k, v in info.items()}

    # Check slots missing
    DOMAIN_BOOK_SLOT_DESC = {
        'restaurant': {
            'name': 'the restaurant name',
            'people': 'the number of people',
            'day': 'the booking day',
            'time': 'the booking time',
        },
        'hotel': {
            'name': 'the hotel name',
            'people': 'the number of people',
            'day': 'the booking day',
            'stay': 'the days to stay',
        },
        'train': {
            'train id': 'the trian id',
            'tickets': 'the number of tickects',
        },
    }
    book_slot_desc = DOMAIN_BOOK_SLOT_DESC[domain]
    missing_slots = [slot for slot in book_slot_desc if slot not in info]
    if missing_slots != []:
        slots_str = ', '.join(book_slot_desc[s] for s in missing_slots)
        return False, f'Booking failed. Please provide {slots_str} for reservation.'
    
    # Check value
    if domain == 'restaurant':
        if info['name'] == '[restaurant name]':
            return False, f'Booking failed. Please provide the restaurant name to book.'
        missing_slots = [s for s in ['people', 'day', 'time'] if info[s] == '[value]']
        if len(missing_slots) > 0:
            missing_slots = ', '.join(missing_slots)
            return False, f'Booking failed. Please provide the values for {missing_slots}.'
        if not info['people'].isdigit() or int(info['people']) <= 0:
            return False, f'Booking failed. The value of people should be a positive integer.'
        DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if info['day'] not in DAYS:
            return False, f'Booking failed. The value of day should be a day in a week.'
        info['time'] = clean_time(info['time'])
        if not re.fullmatch(r'\d\d:\d\d', info['time']):
            return False, f'Booking failed. please provide a valid time, like "08:30".'

    elif domain == 'hotel':
        if info['name'] == '[hotel name]':
            return False, f'Booking failed. Please provide the hotel name to book.'
        missing_slots = [s for s in ['people', 'day', 'stay'] if info[s] == '[value]']
        if len(missing_slots) > 0:
            missing_slots = ', '.join(missing_slots)
            return False, f'Booking failed. Please provide the values for {missing_slots}.'
        if not info['people'].isdigit() or int(info['people']) <= 0:
            return False, f'Booking failed. The value of people should be a positive integer.'
        DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if info['day'] not in DAYS:
            return False, f'Booking failed. The value of day should be a day in a week.'
        if not info['stay'].isdigit() or int(info['people']) <= 0:
            return False, f'Booking failed. The value of stay should be a positive integer.'

    elif domain == 'train':
        if info['train id'] == '[train id]':
            return False, f'Booking failed. Please provide the train id to book.'
        if info['tickets'] == '[value]':
            return False, f'Booking failed. Please the number of tickets to book.'
        if not info['tickets'].isdigit() or int(info['tickets']) <= 0:
            return False, f'Booking failed. The value of tickets should be a positive integer.'
    else:
        raise ValueError(f'{domain = }')

    
    # Check name
    if domain == 'restaurant':
        if not check_db_exist('restaurant', 'name', info['name'], db_path=db_path):
            return False, f'Booking failed. "{info["name"]}" is not found in the restaurant database. Please provide a valid restaurant name.'
    elif domain == 'hotel':
        if not check_db_exist('hotel', 'name', info['name'], db_path=db_path):
            return False, f'Booking failed. "{info["name"]}" is not found in the hotel database. Please provide a valid hotel name.'
    elif domain == 'train':
        if not check_db_exist('train', 'trainID', info['train id'], db_path=db_path):
            return False, f'Booking failed. "{info["train id"]}" is not found in the train databse. Please provide a valid train id.'
    else:
        raise ValueError(f'{domain = }')

    # DB Operation
    engine = create_engine(f'sqlite:///{book_db_path}', connect_args=SQLITE_CONNECT_ARGS)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    refer_number = generate_reference_num()
    if domain == 'train':
        info['trainID'] = info.pop('train id')
    book = DOMAIN_BOOK_CLASS_MAP[domain](refer_number=refer_number, **info)

    session.add(book)
    session.commit()

    return True, f'Booking succeed. The reference number is {refer_number}.'

# endregion


# region: Taxi Booking

def pick_taxi():
    taxi_colors = ["black","white","red","yellow","blue",'grey']
    taxi_types = ["toyota","skoda","bmw",'honda','ford','audi','lexus','volvo','volkswagen','tesla']

    color = random.choice(taxi_colors)
    brand = random.choice(taxi_types)
    phone = ''.join(random.choice('0123456789') for i in range(10))

    return color, brand, phone


def make_booking_taxi(info, db_path=None):
    # Use default db_path if not provided
    if db_path is None:
        db_path = DB_PATH
    
    # Check 1. 'departure', 'destination'
    place_slots = {'departure': None, 'destination': None}

    # 1.1 Both departure and destination must be provided
    missing_slots = [slot for slot in place_slots if info.get(slot) in [None, '[value]']]
    if missing_slots != []:
        slots_str = ' and '.join(missing_slots)
        return False, f'Booking failed. The {slots_str} is missing.'

    # 1.2 Validate departure and destination - 放宽限制，允许任何地点名称
    # 尝试在数据库中查找，但不强制要求找到
    for slot in place_slots:
        place = info[slot]
        found = False
        for domain in ['restaurant', 'hotel', 'attraction']:
            if venue := query_venue_by_name_or_address(domain, place, db_path=db_path):
                place_slots[slot] = venue
                found = True
                break
        # 如果在数据库中找不到，创建一个简单的对象来存储地点名称
        if not found:
            from collections import namedtuple
            SimplePlace = namedtuple('SimplePlace', ['name'])
            place_slots[slot] = SimplePlace(name=place)
    
    # 1.3 Validate relation between departure and destination
    if place_slots['departure'].name == place_slots['destination'].name:
        return False, f'Booking failed. The departure and destination can not be the same place.'

    # Check 2. 'leave time', 'arrive time'
    # time_slots = ['leave time', 'arrive time']  # NOTE: for gpt-3.5-turbo
    info2 = {}
    for k, v in info.items():
        if k == 'leave time':
            k = 'leave'
        elif k == 'arrive time':
            k = 'arrive'
        info2[k] = v
    info = info2
    time_slots = ['leave', 'arrive']

    # 2.1 leave time and arrive time must be provided one
    present_slots = [slot for slot in time_slots if info.get(slot) not in [None, '[value]']]
    if len(present_slots) == 0:
        return False, f'Booking failed. The leave time or arrive time is missing.'
    
    # Not allow both leave time and arrive time
    # elif len(present_slots) == 2:
    #     return False, f'Booking failed. Only one of the leave time and arrive time can be provided.'
    # present_time_slot = present_slots[0]
    # time = info[present_time_slot]
    # time = clean_time(time)
    # if not re.fullmatch(r'\d\d:\d\d', time):
    #     return False, f'Booking failed. Please provide valid time format for the {present_time_slot}, like "07:30".'

    # 2.2 Validate time and
    invalid_slots = [slot for slot in present_slots if not re.fullmatch(r'\d\d:\d\d', clean_time(info[slot]))]
    if invalid_slots != []:
        slots_str = ' and '.join(invalid_slots)
        return False, f'Booking failed. Please provide valid time format for the {slots_str}, like "07:30".'

    # Book  
    color, brand, phone = pick_taxi()
    return True, f'Booking succeed. There is a {color} {brand} taxi. Contact number is {phone}.'

# endregion


# region: Language Interface

def extract_book_info(text):
    text = text.lower()
    slot_values = re.findall(r'\b([\w ]+):\s+(.+?)(?=,|$)', text)
    info = dict(slot_values)
    return info


def book_restaurant(text):
    '''Expected: name: pizza hut city centre, people: 2, day: saturday, time: 18:00'''
    info = extract_book_info(text)
    flag, msg = make_booking_db('restaurant', info)
    return msg


def book_hotel(text):
    '''Expected: name: sleeperz hotel, people: 2, stay: 2'''
    info = extract_book_info(text)
    flag, msg = make_booking_db('hotel', info)
    return msg


def book_train(text):
    '''Expected: train id: tr1234, ticket: 1'''
    info = extract_book_info(text)
    flag, msg = make_booking_db('train', info)
    return msg


def book_taxi(text):
    '''Expected: departure: xx, destination: xx, leave: xx, arrive: xx'''
    info = extract_book_info(text)
    flag, msg = make_booking_taxi(info)
    return msg

# endregion
