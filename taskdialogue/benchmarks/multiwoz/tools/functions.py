"""
Function definitions and schemas for the agent's tool calling capabilities.

This module contains all function preparation logic and schemas for:
- Database query functions (restaurants, hotels, attractions, trains)
- Booking functions (restaurants, hotels, trains, taxis)

CHANGE LOG:
- 2025-10-28: Enhanced function schemas with detailed usage guidelines
  * Moved domain-specific instructions from system prompt to function descriptions
  * Added proactive booking patterns and common error reminders
  * Backup saved as functions.py.backup_20251028
"""

import sqlite3
from functools import partial
from pathlib import Path
from langchain_community.utilities import SQLDatabase

from taskdialogue.benchmarks.multiwoz.tools.booking import make_booking_db, make_booking_taxi
from taskdialogue.benchmarks.multiwoz.tools.database import get_default_db_path


class FunctionConfig:
    """Configuration for function behaviors."""
    
    # Query result display limits
    TRAIN_MAX_ITEMS = 10
    TRAIN_MAX_CHARS = 1000
    
    SORTED_MAX_ITEMS = 8
    SORTED_MAX_CHARS = 800
    
    FILTERED_MAX_ITEMS = 7
    FILTERED_MAX_CHARS = 700
    
    DEFAULT_MAX_ITEMS = 5
    DEFAULT_MAX_CHARS = 500
    
    # Days of week for bookings
    DAYS_OF_WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    # =========================================================================
    # Database field enums (extracted from actual data)
    # Added: 2025-10-28
    # Purpose: Provide enum constraints for function parameters to guide LLM
    # =========================================================================
    
    # Common fields
    AREA_ENUM = ['centre', 'east', 'north', 'south', 'west']
    PRICERANGE_ENUM = ['cheap', 'moderate', 'expensive']
    
    # Restaurant specific
    # food 有23个值，太多不适合枚举
    
    # Hotel specific
    HOTEL_TYPE_ENUM = ['hotel', 'guesthouse']
    HOTEL_STARS_ENUM = ['0', '2', '3', '4']
    HOTEL_INTERNET_ENUM = ['yes', 'no']
    HOTEL_PARKING_ENUM = ['yes', 'no']
    
    # Attraction specific  
    ATTRACTION_TYPE_ENUM = ['architecture', 'boat', 'cinema', 'college', 'concerthall', 
                           'entertainment', 'museum', 'mutliple sports', 'nightclub', 
                           'park', 'swimmingpool', 'theatre']
    
    @classmethod
    def load_from_config(cls, config_dict=None):
        """Load configuration from a dictionary (for future extensibility)."""
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(cls, key.upper()):
                    setattr(cls, key.upper(), value)


def prepare_query_db_functions(domain, db_path):
    """
    Prepare database query functions for a specific domain.
    
    Args:
        domain: One of 'restaurant', 'hotel', 'attraction', 'train'
        db_path: Path to the SQLite database
        
    Returns:
        dict: Contains 'name', 'function', and 'schema' keys
    """
    
    def query_db(sql, table=None, db_path=None):
        """Execute an SQL query on the database and format results."""
        if table and table not in sql:
            return f'Please query the {table} table in the database.'

        conn = sqlite3.connect(db_path, timeout=30)
        try:
            cursor = conn.execute(sql)
            records = cursor.fetchall()
        except Exception as e:
            return str(e)
        finally:
            conn.close()

        if len(records) == 0:
            return 'No results found.'
        
        # Determine display limits based on query type
        total_records = len(records)
        
        if table == 'train':
            max_items = FunctionConfig.TRAIN_MAX_ITEMS
            max_chars = FunctionConfig.TRAIN_MAX_CHARS
        elif 'ORDER BY' in sql.upper():
            max_items = FunctionConfig.SORTED_MAX_ITEMS
            max_chars = FunctionConfig.SORTED_MAX_CHARS
        elif 'WHERE' in sql.upper() and sql.upper().count('AND') >= 2:
            max_items = FunctionConfig.FILTERED_MAX_ITEMS
            max_chars = FunctionConfig.FILTERED_MAX_CHARS
        else:
            max_items = FunctionConfig.DEFAULT_MAX_ITEMS
            max_chars = FunctionConfig.DEFAULT_MAX_CHARS

        result = []
        result.append(f'Found {total_records} record(s) in total. Showing up to {max_items} records:')
        result.append('')
        
        # Build table header
        n_chars = 0
        line = '| ' + ' | '.join(desc[0] for desc in cursor.description) + ' |'
        n_chars += len(line) + 1
        result.append(line)
        line = '| ' + ' | '.join(['---'] * len(cursor.description)) + ' |'
        n_chars += len(line) + 1
        result.append(line)
        
        # Add records
        displayed_count = 0
        for i, record in enumerate(records, start=1):
            line = '| ' + ' | '.join(str(v) for v in record) + ' |'
            n_chars += len(line) + 1
            if n_chars <= max_chars and i <= max_items:
                result.append(line)
                displayed_count = i
            else:
                break
        
        # Add truncation notice if needed
        if total_records > displayed_count:
            n_left = total_records - displayed_count
            result.append('')
            result.append(f'... and {n_left} more record(s) not shown (total: {total_records} records)')
            result.append('')
            result.append('NOTE: The above results are sufficient for making recommendations to the user.')
            result.append('DO NOT query again just to see more records. Use the information shown above.')
        
        return '\n'.join(result)

    def get_table_info(domain, db_path):
        """Get table schema information from the database."""
        db = SQLDatabase.from_uri(
            database_uri=f'sqlite:///{db_path}',
            include_tables=[domain],
            sample_rows_in_table_info=2,
        )
        table_info = db.get_table_info()
        return table_info
    
    def make_schema(domain, name, table_info):
        """Create function schema for the query function."""
        if domain == 'train':
            max_display = '10'
        else:
            max_display = '5-10'
        
        # 添加领域特定的枚举提示
        enum_hints = ""
        if domain == 'restaurant':
            enum_hints = f'''

COMMON FIELD VALUES (for reference):
- area: {', '.join(FunctionConfig.AREA_ENUM)}
- pricerange: {', '.join(FunctionConfig.PRICERANGE_ENUM)}
- food: many types (e.g., chinese, italian, indian, british, etc.)'''
        elif domain == 'hotel':
            enum_hints = f'''

COMMON FIELD VALUES (for reference):
- area: {', '.join(FunctionConfig.AREA_ENUM)}
- pricerange: {', '.join(FunctionConfig.PRICERANGE_ENUM)}
- type: {', '.join(FunctionConfig.HOTEL_TYPE_ENUM)}
- stars: {', '.join(FunctionConfig.HOTEL_STARS_ENUM)}
- internet/parking: {', '.join(FunctionConfig.HOTEL_INTERNET_ENUM)}'''
        elif domain == 'attraction':
            enum_hints = f'''

COMMON FIELD VALUES (for reference):
- area: {', '.join(FunctionConfig.AREA_ENUM)}
- type: {', '.join(FunctionConfig.ATTRACTION_TYPE_ENUM)}'''
        
        func_desc_temp = f'''Query {{domain}} database using SQL.

Table Schema:
{{table_info}}{enum_hints}

Notes: Results auto-limited to {max_display} records. Don't query repeatedly with different LIMITs.

⚠️ IMPORTANT - SQL String Escaping:
• If a value contains single quotes (e.g., "bishop's stortford"), escape them by doubling the quote
• Correct: WHERE destination = 'bishop''s stortford'  
• Wrong: WHERE destination = 'bishop's stortford'  (will cause syntax error)
• This applies to all string values in SQL: names, locations, addresses, etc.'''

        param_desc_temp = f'SQL query for {{domain}} table'
        
        schema = {
            'name': name,
            'description': func_desc_temp.format(table_info=table_info, domain=domain),
            'parameters': {
                'type': 'object',
                'properties': {
                    'sql': {
                        'type': 'string',
                        'description': param_desc_temp.format(domain=domain),
                    }
                },
                'required': ['sql'],
            }
        }
        return schema

    assert domain in ['restaurant', 'hotel', 'attraction', 'train'], f"Invalid domain: {domain}"
    
    name = f'query_{domain}s'
    function = partial(query_db, table=domain, db_path=db_path)
    table_info = get_table_info(domain, db_path)
    schema = make_schema(domain, name, table_info)

    return {'name': name, 'function': function, 'schema': schema}


def prepare_book_functions(domain, db_path=None, book_db_path=None):
    """
    Prepare booking functions for a specific domain.
    
    Args:
        domain: One of 'restaurant', 'hotel', 'train', 'taxi'
        db_path: Path to main database (for validation)
        book_db_path: Path to booking database
        
    Returns:
        dict: Contains 'name', 'function', and 'schema' keys
    """
    # Use default paths if not provided
    if db_path is None:
        from taskdialogue.benchmarks.multiwoz.tools.utils import get_db_path
        db_path = get_db_path()
    if book_db_path is None:
        from taskdialogue.benchmarks.multiwoz.tools.utils import get_book_db_path
        book_db_path = get_book_db_path()
    
    if domain == 'restaurant':
        def book_restaurant(name, people, day, time):
            """Book a restaurant with specified parameters."""
            info = {'name': name, 'people': str(people), 'day': day, 'time': time}
            flag, msg = make_booking_db('restaurant', info, book_db_path=book_db_path, db_path=db_path)
            return msg

        name = 'book_restaurant'
        schema = {
            'name': name,
            'description': '''Book restaurant reservation. 

⚠️ Use EXACT name from query results, not paraphrases.
✓ Book directly when all params available.
✗ Never fabricate success - this returns real booking result.''',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'The EXACT name of the restaurant from database (e.g., "The Oak Bistro", not "the bistro")',
                    },
                    'people': {
                        'type': 'integer',
                        'description': 'the number of people',
                    },
                    'day': {
                        'type': 'string',
                        'enum': FunctionConfig.DAYS_OF_WEEK,
                        'description': 'the day when the people go to the restaurant',
                    },
                    'time': {
                        'type': 'string',
                        'description': 'the time of the reservation',
                    },
                },
                'required': ['name', 'people', 'day', 'time'],
            }
        }
        return {'name': name, 'function': book_restaurant, 'schema': schema}

    elif domain == 'hotel':
        def book_hotel(name, people, day, stay):
            """Book a hotel with specified parameters."""
            info = {'name': name, 'people': str(people), 'day': day, 'stay': str(stay)}
            flag, msg = make_booking_db('hotel', info, book_db_path=book_db_path, db_path=db_path)
            return msg

        name = 'book_hotel'
        schema = {
            'name': name,
            'description': '''Book hotel reservation. 

⚠️ Use EXACT name from query results, not paraphrases.
✓ Book directly when all params available.
Note: "stay" = number of NIGHTS (e.g., "2 nights" → stay=2)
✗ Never fabricate success - this returns real booking result.''',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'The EXACT name of the hotel from database (e.g., "Express by Holiday Inn Cambridge")',
                    },
                    'people': {
                        'type': 'integer',
                        'description': 'the number of people',
                    },
                    'day': {
                        'type': 'string',
                        'enum': FunctionConfig.DAYS_OF_WEEK,
                        'description': 'the day when the reservation starts',
                    },
                    'stay': {
                        'type': 'integer',
                        'description': 'the number of days of the reservation',
                    },
                },
                'required': ['name', 'people', 'day', 'stay'],
            }
        }
        return {'name': name, 'function': book_hotel, 'schema': schema}

    elif domain == 'train':
        def buy_train_tickets(train_id, tickets):
            """Buy train tickets with specified parameters."""
            info = {'train id': train_id, 'tickets': str(tickets)}
            flag, msg = make_booking_db('train', info, book_db_path=book_db_path, db_path=db_path)
            return msg

        name = 'buy_train_tickets'
        schema = {
            'name': name,
            'description': '''Book train tickets.

⚠️ train_id from query_trains (e.g., "TR1328")
✓ PROACTIVE: Book BEST matching train directly (closest to user's time preference)
✓ After booking: Share train ID, times, price, reference number
✗ Never say "booked" without calling this function''',
            'parameters': {
                'type': 'object',
                'properties': {
                    'train_id': {
                        'type': 'string',
                        'description': 'the unique id of the train',
                    },
                    'tickets': {
                        'type': 'integer',
                        'description': 'the number of tickets to buy',
                    },
                },
                'required': ['train_id', 'tickets'],
            }
        }
        return {'name': name, 'function': buy_train_tickets, 'schema': schema}

    elif domain == 'taxi':
        def book_taxi(departure, destination, leave_time=None, arrive_time=None, **kwargs):
            """Book a taxi with specified parameters.
            
            Note: taxi booking doesn't need 'day' parameter (immediate service),
            but we accept it via **kwargs to avoid errors if model provides it.
            """
            info = {'departure': departure, 'destination': destination}
            if leave_time:
                info['leave time'] = leave_time
            if arrive_time:
                info['arrive time'] = arrive_time
            # Ignore extra parameters like 'day' (taxis are immediate service)
            flag, msg = make_booking_taxi(info, db_path=db_path)
            return msg

        name = 'book_taxi'
        schema = {
            'name': name,
            'description': '''Book taxi. Returns REAL car type (color+brand) and phone number.

🚨 CRITICAL ERRORS TO AVOID:
✗ NEVER say "I've booked" without calling this
✗ NEVER fabricate car details or phone numbers
✓ MUST call this FIRST to get real booking info

Smart inference:
• "taxi to the restaurant" → use restaurant name from conversation
• "from the hotel" → use hotel name mentioned earlier
• Don't ask "which?" if only ONE venue of that type discussed

Required: departure + destination + (leave_time OR arrive_time)
After calling: Share the car type and phone returned.''',
            'parameters': {
                'type': 'object',
                'properties': {
                    'departure': {
                        'type': 'string',
                        'description': 'Pickup location. Use EXACT venue name from conversation (e.g., "The Cambridge Belfry", "Pizza Hut City Centre", "Christ\'s College", "city centre")',
                    },
                    'destination': {
                        'type': 'string',
                        'description': 'Drop-off location. Use EXACT venue name from conversation (e.g., "The Oak Bistro", "Gonville Hotel", "ADC Theatre")',
                    },
                    'leave_time': {
                        'type': 'string',
                        'description': 'The departure time in HH:MM format (e.g., "14:30", "09:00"). Use this when user specifies "leave at" or "depart at" time. Must be explicitly provided by the user.',
                    },
                    'arrive_time': {
                        'type': 'string',
                        'description': 'The arrival time in HH:MM format (e.g., "14:30", "09:00"). Use this when user specifies "arrive by" time. Must be explicitly provided by the user.',
                    },
                },
                'required': ['departure', 'destination'],
            }
        }
        return {'name': name, 'function': book_taxi, 'schema': schema}
    
    else:
        raise ValueError(f'Invalid domain: {domain}')


def get_all_function_factories(db_path=None, book_db_path=None):
    """
    Get all function factories for the agent.
    
    Args:
        db_path: Path to the SQLite database (uses default if None)
        book_db_path: Path to booking database (uses default if None)
        
    Returns:
        list: List of function factory callables
    """
    if db_path is None:
        db_path = get_default_db_path()
    
    if book_db_path is None:
        from taskdialogue.benchmarks.multiwoz.tools.utils import get_book_db_path
        book_db_path = get_book_db_path()
    
    return [
        partial(prepare_query_db_functions, domain='restaurant', db_path=db_path),
        partial(prepare_book_functions, domain='restaurant', db_path=db_path, book_db_path=book_db_path),
        partial(prepare_query_db_functions, domain='hotel', db_path=db_path),
        partial(prepare_book_functions, domain='hotel', db_path=db_path, book_db_path=book_db_path),
        partial(prepare_query_db_functions, domain='attraction', db_path=db_path),
        partial(prepare_query_db_functions, domain='train', db_path=db_path),
        partial(prepare_book_functions, domain='train', db_path=db_path, book_db_path=book_db_path),
        partial(prepare_book_functions, domain='taxi', db_path=db_path, book_db_path=book_db_path),
    ]

