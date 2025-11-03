import json
from pprint import pprint

import openai
import tenacity

from taskdialogue.benchmarks.multiwoz.tools import booking, db
from taskdialogue.benchmarks.multiwoz.tools.utils import (
    DOMAINS, 
    clean_time, 
    tenacity_retry_log, 
    prepare_goals_string
)
from taskdialogue.core.models.factory import create_model_from_config
from taskdialogue.core.utils.config import load_config
from taskdialogue.benchmarks.multiwoz.prompts import (
    get_evaluation_system_prompt,
    get_evaluation_human_template,
    get_evaluation_answer_format_template,
)

# 初始化评估模型
# Note: OpenAI API key is now handled by the Model class
_eval_model = None

def get_eval_model(config=None):
    """获取评估模型单例（从配置文件读取）"""
    global _eval_model
    if _eval_model is None:
        if config is None:
            config = load_config()
        
        # 从配置读取评估模型名称和类型
        eval_model_name = config.get('evaluation.eval_model', 'deepseek-chat')
        eval_model_type = config.get('evaluation.eval_model_type', 'deepseek')
        
        # 从配置读取评估模型参数
        temperature = config.get('evaluation.model_params.temperature', 0.01)
        max_tokens = config.get('evaluation.model_params.max_tokens', 4096)
        
        # 使用 taskdialogue 的模型工厂创建模型
        from taskdialogue.core.models.factory import create_model
        _eval_model = create_model(
            provider=eval_model_type,
            model_name=eval_model_name,
            config=config,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print(f"📊 评估模型已初始化: {eval_model_name} (type={eval_model_type}, temperature={temperature}, max_tokens={max_tokens})")
    return _eval_model


# Prompts are now imported from taskdialogue.benchmarks.multiwoz.prompts module


def prepare_dialog_string(dialog):
    dialog_str = []
    for turn in dialog:
        dialog_str.append(f'User: {turn["user"]}')
        dialog_str.append(f'AI Assistant: {turn["agent"]}')
    dialog_str = '\n'.join(dialog_str)
    return dialog_str


def extract_venue_name_from_history(dialog, domain):
    """
    从对话历史中提取venue名称（备用方法）
    当LLM回答失败时使用
    
    Args:
        dialog: 对话历史列表
        domain: 领域名称 (restaurant, hotel, attraction)
    
    Returns:
        提取的venue名称，如果未找到返回'none'
    """
    import re
    
    # 定义不同场景的匹配模式
    patterns = [
        # 预订场景
        rf"(?:book|booked|reservation at|reserve)\s+([A-Z][a-zA-Z0-9\s&'\-\.]+?)(?:\s+(?:for|on|at|with)|[,\.]|$)",
        # 推荐场景
        rf"(?:recommend|suggest)\s+([A-Z][a-zA-Z0-9\s&'\-\.]+?)(?:\s+(?:for|as|which)|[,\.]|$)",
        # 选择场景
        rf"(?:choose|go with|pick|select)\s+([A-Z][a-zA-Z0-9\s&'\-\.]+?)(?:\s+(?:for|as)|[,\.\?]|$)",
        # 信息查询场景
        rf"(?:about|information about|details about)\s+([A-Z][a-zA-Z0-9\s&'\-\.]+?)(?:\s+(?:is|are)|[,\.\?]|$)",
        # 直接提及
        rf"(?:the\s+)?([A-Z][a-zA-Z0-9\s&'\-\.]+?)\s+(?:is a|is an|is the)\s+{domain}",
    ]
    
    # 从后往前搜索（最近的提及最相关）
    for turn in reversed(dialog):
        # 检查agent和user的消息
        for role in ['agent', 'user']:
            text = turn.get(role, '')
            if not text:
                continue
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.group(1).strip()
                    # 清理名称
                    name = name.rstrip('.,!?;:')
                    # 移除尾部的常见词
                    name = re.sub(r'\s+(is|are|has|have|was|were)$', '', name, flags=re.IGNORECASE)
                    
                    # 验证名称合理性
                    if len(name) > 3 and len(name) < 50:
                        # 检查是否包含太多停用词
                        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at']
                        words = name.lower().split()
                        if len([w for w in words if w not in stop_words]) > 0:
                            return name
    
    return 'none'


# region: Taxi

TAXI_SLOT_MAP = {
    'departure': 'departure',
    'destination': 'destination',
    'leaveAt': 'leave time',
    'arriveBy': 'arrival time',
    'car type': 'car type',
    'phone': 'phone number',
}


def prepare_taxi_questions(goal):
    '''
    Goal:
        'info': {'arriveBy', 'departure', 'destination', 'leaveAt'}
        'reqt': {'car type', 'phone'}
    '''
    questions = []
    answer_formats = []
    q_idx = 1

    # Find Venue (info): arriveBy, departure, destination, leaveAt
    for slot in goal.get('info', {}):
        slot_mapped = TAXI_SLOT_MAP[slot]
        q = f'{q_idx}. What is the {slot_mapped} of the taxi that the user books?'
        a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    # Request Slots (reqt): car type, phone
    for slot in goal.get('reqt', []):
        slot_mapped = TAXI_SLOT_MAP[slot]
        if slot == 'car type':
            q = f'''{q_idx}. What is the car type (color and brand) of the taxi?
   IMPORTANT: Carefully search through ALL the AI Assistant's responses about the taxi.
   - Look for phrases like: "There is a [color] [brand] taxi", "your taxi ([color] [brand])", or "[color] [brand], contact"
   - Examples: "grey BMW", "blue Toyota", "red Honda", "white Audi", "black Lexus"
   - The car type information is often mentioned together with the contact number in parentheses
   - Extract the FULL car type including both color and brand
   - If no car type is mentioned anywhere in the conversation, answer "none"'''
        elif slot == 'phone':
            q = f'''{q_idx}. What is the phone number (contact number) of the taxi?
   IMPORTANT: Carefully search through ALL the AI Assistant's responses about the taxi.
   - Look for phrases like: "Contact number is [number]", "Phone: [number]", "contact [number]", or numbers in parentheses after car type
   - The phone number is typically 10-11 digits
   - Examples: "07123456789", "01223456789", "6309886173"
   - Extract the COMPLETE phone number (all digits)
   - If no phone number is mentioned anywhere in the conversation, answer "none"'''
        else:
            q = f'{q_idx}. What is the {slot_mapped} of the taxi?'
        a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = get_evaluation_answer_format_template().format(answer_formats=answer_formats)

    return questions, answer_formats


def evaluate_by_domain_taxi(goal, llm_answer):
    '''
    Taxi 领域评估
    
    使用优化的评估逻辑 (参考 TD-Eval):
    - Inform: 检查是否提供了所有 info slots (departure, destination, time)
    - Success: 在 Inform 成功的前提下，检查是否提供了所有 reqt slots (car type, phone)
    '''
    # Clean time
    for slot in ['leave time', 'arrival time']:
        if time := llm_answer.get(slot):
            llm_answer[slot] = clean_time(time)
    
    # 使用新的评估逻辑
    from taskdialogue.benchmarks.multiwoz.evaluators.evaluation_logic import get_domain_evaluator
    evaluator = get_domain_evaluator()
    
    result = evaluator.evaluate_taxi_domain(
        goal=goal,
        llm_extracted=llm_answer
    )
    
    return result

# endregion: Taxi

# region: Train

TRAIN_SLOT_MAP = {
    'trainID': 'train id',
    'price': 'price',
    'duration': 'duration',
    'leaveAt': 'leave time',
    'arriveBy': 'arrive time',
}


def prepare_train_questions(goal, add_format_hints=True):
    '''
    Goal:
        'info': {'arriveBy', 'day', 'departure', 'destination', 'leaveAt'},
        'book': {'invalid', 'people'},
        'reqt': {'arriveBy', 'duration', 'leaveAt', 'price', 'trainID'},

    Asserts:
        - assert goal.get('reqt') XOR goal.get('book')
    
    Args:
        add_format_hints: 是否添加数据库格式提示
    '''

    questions = []
    answer_formats = []
    q_idx = 1

    # 🆕 提取 Info Slots (用于 JGA/Slot-F1 评估)
    # Train 的 info slots 是用户的约束条件
    INFO_SLOT_MAP = {
        'departure': 'departure station',
        'destination': 'destination station',
        'day': 'day of travel',
        'leaveAt': 'departure time',
        'arriveBy': 'arrival time'
    }
    
    for slot in goal.get('info', {}):
        if slot in INFO_SLOT_MAP:
            slot_desc = INFO_SLOT_MAP[slot]
            q = f'{q_idx}. What {slot_desc} did the user request for the train? Extract from the user\'s requirements.'
            a = f'"info_{slot}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    # Book (book)
    if goal.get('book'):
        assert not goal.get('reqt')
        q = f'''{q_idx}. What is the reference number of the booked train tickets?
   IMPORTANT: Extract the EXACT reference number.
   - Format: Usually 8 characters (e.g., "TR12AB34", "XY78WZ90")
   - Extract ONLY the code'''
        a = f'"reference number": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1
    
    # Request Slots (reqt)
    else:  
        assert not goal.get('book')
        for slot in goal.get('reqt', []):
            slot_mapped = TRAIN_SLOT_MAP[slot]
            if slot == 'trainID':
                q = f'{q_idx}. What is the id of the train?'
                a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
            else:
                # 添加格式提示
                if add_format_hints and slot in ['price', 'duration']:
                    if slot == 'price':
                        q = f'{q_idx}. What is the {slot_mapped} of the train? (Format: X.XX pounds)'
                    elif slot == 'duration':
                        q = f'{q_idx}. What is the {slot_mapped} of the train? (Format: X minutes)'
                    else:
                        q = f'{q_idx}. What is the {slot_mapped} of the train?'
                else:
                    q = f'{q_idx}. What is the {slot_mapped} of the train?'
                a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = get_evaluation_answer_format_template().format(answer_formats=answer_formats)

    return questions, answer_formats


def evaluate_by_domain_train(goal, llm_answer, success_mode='relaxed'):
    '''
    Train 领域评估
    
    使用优化的评估逻辑 (参考 TD-Eval):
    - Inform: 检查是否找到满足约束的火车
    - Success: 在 reqt 场景下等同于 Inform
    - Book: 在 Inform 成功的前提下检查预订
    
    Args:
        success_mode: 'relaxed' 或 'strict'
    '''
    # Clean time
    for slot in ['leave time', 'arrival time', 'arrive time']:
        if time := llm_answer.get(slot):
            llm_answer[slot] = clean_time(time)
    
    # 准备数据
    book_record = None
    train_info = None
    matched_items = []
    
    # 场景1: 查询信息 (有 reqt)
    if goal.get('reqt'):
        matched_items = db.query_trains(goal.get('info', {}))
    
    # 场景2: 预订 (有 book)
    elif goal.get('book'):
        refer_number = llm_answer.get('reference number', 'none')
        book_record = booking.query_booking_by_refer_num('train', refer_number)
        
        if book_record:
            train_info = db.query_train_by_id(book_record.trainID)
            if not train_info:
                train_info = f'"{book_record.trainID}" is not found in the "train" table.'
        else:
            book_record = f'"{refer_number}" is not found in the "book_train" table.'
            train_info = f'No train as invalid refer number "{refer_number}".'
    
    # 使用新的评估逻辑
    from taskdialogue.benchmarks.multiwoz.evaluators.evaluation_logic import get_domain_evaluator
    evaluator = get_domain_evaluator()
    
    result = evaluator.evaluate_train_domain(
        goal=goal,
        llm_extracted=llm_answer,
        matched_items=matched_items,
        book_record=book_record,
        train_info=train_info,
        success_mode=success_mode
    )
    
    return result

# endregion: Train

# region: hotel, restaurant, attraction

def prepare_hotel_questions(goal):
    '''
    Goal:
        'info': {'area', 'internet', 'name', 'parking', 'pricerange', 'stars', 'type'},
        'book': {'day', 'invalid', 'people', 'pre_invalid', 'stay'},
        'reqt': {'area', 'pricerange', 'type', 'stars',
                'internet', 'parking',
                'address', 'phone', 'postcode',},

    Asserts:
        - assert fail_book.issubset(book)
        - Book: 'stay', 'people', 'day' all in book
    '''
    questions = []
    answer_formats = []
    q_idx = 1

    # 🆕 提取 Info Slots (用于 JGA/Slot-F1 评估)
    INFO_SLOT_MAP = {
        'area': 'area (east, west, north, south, centre)',
        'pricerange': 'price range (cheap, moderate, expensive)',
        'type': 'type (hotel, guesthouse)',
        'stars': 'star rating',
        'internet': 'free internet/wifi availability (yes/no)',
        'parking': 'free parking availability (yes/no)'
    }
    
    for slot in goal.get('info', {}):
        if slot == 'name':
            continue  # name 单独处理
        if slot in INFO_SLOT_MAP:
            slot_desc = INFO_SLOT_MAP[slot]
            q = f'{q_idx}. What {slot_desc} did the user request for the hotel? Extract from the user\'s requirements in the dialogue.'
            a = f'"info_{slot}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    # Find Venue (info)
    if goal.get('book', []):
        q = f'''{q_idx}. What is the EXACT name of the hotel that the user chooses and books?
   IMPORTANT: Extract the hotel name exactly as it appears in the dialogue.
   - Look for phrases like "I'll book [hotel name]" or "reservation at [hotel name]"
   - If the AI Assistant recommends multiple hotels, extract the FIRST hotel mentioned
   - If user asks questions about the hotels (like address, phone) without explicitly choosing, extract the FIRST hotel from the AI's recommendation list
   - Use the exact name, including capitalization and special characters
   - If no specific hotel is mentioned, answer "none"
   - Answer with ONLY the hotel name, no additional text'''
    else:
        q = f'''{q_idx}. What is the EXACT name of the hotel that the user is interested in?
   IMPORTANT: Extract the hotel name exactly as it appears in the dialogue.
   - Look for the hotel name mentioned by the user or agent
   - If the AI Assistant recommends multiple hotels, extract the FIRST hotel mentioned in the list
   - If user asks questions about the hotels without explicitly choosing, it means they are interested in the FIRST hotel from the recommendation
   - Use the exact name, including capitalization and special characters
   - If no specific hotel is mentioned, answer "none"
   - Answer with ONLY the hotel name, no additional text'''
    a = f'"hotel": "<fill the answer of question {q_idx}>"'
    questions.append(q)
    answer_formats.append(a)
    q_idx += 1

    # Book (book)
    if goal.get('book'):
        q = f'''{q_idx}. What is the reference number of the booked hotel?
   IMPORTANT: Extract the EXACT reference number (usually 8 alphanumeric characters).
   - Look for phrases like: "reference number is [CODE]", "booking code: [CODE]"
   - Format: Usually 8 characters (e.g., "AB12CD34", "XY78WZ90")
   - Extract ONLY the code'''
        a = f'"reference number": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    # Request Slots (reqt)
    for slot in goal.get('reqt', []):
        if slot == 'area':
            q = f'{q_idx}. What is the area of the hotel? (east, west, north, south, centre)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (east, west, north, south, centre)>"'
        elif slot == 'pricerange':
            q = f'{q_idx}. What is the price of the hotel? (cheap, moderate, expensive)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (cheap, moderate, expensive)>"'
        elif slot == 'type':
            q = f'{q_idx}. What is the type of the hotel? (guesthouse, hotel)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (guesthouse, hotel)>"'
        elif slot == 'stars':
            q = f'{q_idx}. What is the stars of the hotel? (1, 2, 3, ...)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (1, 2, 3, ...)>"'
        elif slot == 'internet':
            q = f'{q_idx}. Does the hotel have free internet/wifi? (yes, no)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (yes, no)>"'
        elif slot == 'parking':
            q = f'{q_idx}.  Does the hotel have free parking? (yes, no)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (yes, no)>"'
        elif slot == 'address':
            q = f'{q_idx}. What is the address of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'phone':
            q = f'{q_idx}. What is the phone number of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'postcode':
            q = f'{q_idx}. What is the postcode of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        else:
            q = None
            a = None

        if q and a:
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = get_evaluation_answer_format_template().format(answer_formats=answer_formats)

    return questions, answer_formats


def prepare_restaurant_questions(goal):
    '''
    Goal:
        'info': {'area', 'food', 'name', 'pricerange'},
        'book': {'day', 'invalid', 'people', 'pre_invalid', 'time'},
        'reqt': {'address', 'area', 'food', 'phone', 'postcode', 'pricerange'},

    Asserts:
        - assert fail_book.issubset(book)
        - Book: 'time', 'people', 'day' all in book
    '''

    questions = []
    answer_formats = []
    q_idx = 1

    # 🆕 提取 Info Slots (用于 JGA/Slot-F1 评估)
    # 这些是用户的约束条件，LLM 需要从对话中理解和提取
    INFO_SLOT_MAP = {
        'area': 'area (east, west, north, south, centre)',
        'food': 'food type',
        'pricerange': 'price range (cheap, moderate, expensive)'
    }
    
    for slot in goal.get('info', {}):
        if slot == 'name':
            continue  # name 单独处理
        if slot in INFO_SLOT_MAP:
            slot_desc = INFO_SLOT_MAP[slot]
            q = f'{q_idx}. What {slot_desc} did the user request for the restaurant? Extract from the user\'s requirements in the dialogue.'
            a = f'"info_{slot}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    # Find Venue (info)
    if goal.get('book', []):
        q = f'''{q_idx}. What is the EXACT name of the restaurant that the user chooses and books?
   IMPORTANT: Extract the restaurant name exactly as it appears in the dialogue.
   - Look for phrases like "I'll book [restaurant name]" or "reservation at [restaurant name]"
   - If the AI Assistant recommends multiple restaurants, extract the FIRST restaurant mentioned
   - If user asks questions about the restaurants (like address, phone) without explicitly choosing, extract the FIRST restaurant from the AI's recommendation list
   - Use the exact name, including capitalization and special characters
   - If no specific restaurant is mentioned, answer "none"
   - Answer with ONLY the restaurant name, no additional text'''
    else:
        q = f'''{q_idx}. What is the EXACT name of the restaurant that the user is interested in?
   IMPORTANT: Extract the restaurant name exactly as it appears in the dialogue.
   - Look for the restaurant name mentioned by the user or agent
   - If the AI Assistant recommends multiple restaurants, extract the FIRST restaurant mentioned in the list
   - If user asks questions about the restaurants (like address, phone) without explicitly choosing, it means they are interested in the FIRST restaurant from the recommendation
   - Use the exact name, including capitalization and special characters
   - If no specific restaurant is mentioned, answer "none"
   - Answer with ONLY the restaurant name, no additional text'''
    a = f'"restaurant": "<fill the answer of question {q_idx}>"'
    questions.append(q)
    answer_formats.append(a)
    q_idx += 1

    # Book (book)
    if goal.get('book'):
        q = f'''{q_idx}. What is the reference number of the booked restaurant?
   IMPORTANT: Extract the EXACT reference number (usually 8 alphanumeric characters).
   - Look for phrases like: "reference number is [CODE]", "booking reference: [CODE]", "confirmation code [CODE]"
   - Format: Usually 8 characters, mix of letters and numbers (e.g., "AB12CD34", "XY56ZW78", "9TVDE006")
   - If no reference number is mentioned, answer "none"
   - Extract ONLY the code, no additional text'''
        a = f'"reference number": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    # Request Slots (reqt)
    for slot in goal.get('reqt', []):
        if slot == 'area':
            q = f'{q_idx}. What is the area of the restaurant? (east, west, north, south, centre)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (east, west, north, south, centre)>"'
        elif slot == 'pricerange':
            q = f'{q_idx}. What is the price range of the restaurant? (cheap, moderate, expensive)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (cheap, moderate, expensive)>"'
        elif slot == 'food':
            q = f'{q_idx}. What is the food type of the restaurant?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'address':
            q = f'{q_idx}. What is the address of the restaurant?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'phone':
            q = f'{q_idx}. What is the phone number of the restaurant?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'postcode':
            q = f'{q_idx}. What is the postcode of the restaurant?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        else:
            q = None
            a = None

        if q and a:
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = get_evaluation_answer_format_template().format(answer_formats=answer_formats)

    return questions, answer_formats


def prepare_attraction_questions(goal, provide_db_hints=True):
    '''
    Goal:
        'info': {'area', 'name', 'type'},
        'reqt': {'address', 'area', 'entrance fee', 'phone', 'postcode', 'type'},
    
    Args:
        provide_db_hints: 是否提供数据库可选值提示（严格模式）
    '''

    questions = []
    answer_formats = []
    q_idx = 1

    # 🆕 提取 Info Slots (用于 JGA/Slot-F1 评估)
    INFO_SLOT_MAP = {
        'area': 'area (east, west, north, south, centre)',
        'type': 'type (architecture, boat, cinema, college, concert hall, entertainment, museum, multiple sports, nightclub, park, special, swimming pool, theatre)'
    }
    
    for slot in goal.get('info', {}):
        if slot == 'name':
            continue  # name 单独处理
        if slot in INFO_SLOT_MAP:
            slot_desc = INFO_SLOT_MAP[slot]
            q = f'{q_idx}. What {slot_desc} did the user request for the attraction? Extract from the user\'s requirements in the dialogue.'
            a = f'"info_{slot}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    # Find Venue (info)
    q = f'''{q_idx}. What is the EXACT name of the attraction that the user is interested in?
   IMPORTANT: Extract the attraction name exactly as it appears in the dialogue.
   - Look for the attraction name mentioned by the user or agent
   - If the AI Assistant recommends multiple attractions, extract the FIRST attraction mentioned in the list
   - If user asks questions about the attractions (like address, phone, entrance fee) without explicitly choosing, it means they are interested in the FIRST attraction from the recommendation
   - Use the exact name, including capitalization and special characters
   - If no specific attraction is mentioned, answer "none"
   - Answer with ONLY the attraction name, no additional text'''
    a = f'"attraction": "<fill the answer of question {q_idx}>"'
    questions.append(q)
    answer_formats.append(a)
    q_idx += 1

    # Request Slots (reqt)
    for slot in goal.get('reqt', []):
        if slot == 'area':
            q = f'{q_idx}. What is the area of the attraction? (east, west, north, south, centre)'
        elif slot == 'entrance fee':
            q = f'{q_idx}. What is the entrance fee of the attraction?'
            if provide_db_hints:
                from taskdialogue.benchmarks.multiwoz.evaluators.db_schema import get_slot_enum_hint
                enum_hint = get_slot_enum_hint('attraction', 'entrance_fee')
                if enum_hint:
                    q += f'\n   NOTE: Extract EXACTLY as mentioned. Common values: {enum_hint}'
                else:
                    q += '\n   NOTE: Extract the EXACT entrance fee (e.g., "free", "5 pounds", "3.50 pounds"), NOT price range'
        elif slot == 'type':
            q = f'{q_idx}. What is the type of the attraction?'
            if provide_db_hints:
                from taskdialogue.benchmarks.multiwoz.evaluators.db_schema import get_slot_enum_hint
                enum_hint = get_slot_enum_hint('attraction', 'type')
                if enum_hint:
                    q += f'\n   NOTE: Extract from these types: {enum_hint}'
                else:
                    q += '\n   NOTE: Common types: museum, park, theatre, college, etc.'
        elif slot == 'address':
            q = f'{q_idx}. What is the address of the attraction?'
        elif slot == 'phone':
            q = f'{q_idx}. What is the phone number of the attraction?'
            if provide_db_hints:
                q += '\n   NOTE: Extract complete digits (e.g., "01223334900")'
        elif slot == 'postcode':
            q = f'{q_idx}. What is the postcode of the attraction?'
            if provide_db_hints:
                q += '\n   NOTE: Extract exactly as mentioned (e.g., "CB2 1UJ")'
        else:
            q = None
        
        if q:
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = get_evaluation_answer_format_template().format(answer_formats=answer_formats)

    return questions, answer_formats


def evaluate_by_domain_others(goal, llm_answer, domain, dialog_pred=None, success_mode='relaxed'):  # restaurant, hotel, attraction
    '''
    Domain: restaurant, hotel, attraction
    
    使用优化的评估逻辑 (参考 TD-Eval):
    - Inform: 检查系统提供的实体是否满足用户的约束条件
    - Success: 在 Inform 成功的前提下，检查是否提供了所有请求的信息
    - Book: 在 Inform 成功的前提下，检查预订是否成功
    
    Args:
        success_mode: 'relaxed' 或 'strict'
    '''
    # 提取 venue 名称
    name = llm_answer.get(domain, 'none')
    
    # 备用方案：从对话历史中提取
    extraction_method = 'llm_direct'
    if (name == 'none' or not name or name.lower() == 'none') and dialog_pred:
        name_from_history = extract_venue_name_from_history(dialog_pred, domain)
        if name_from_history != 'none':
            name = name_from_history
            extraction_method = 'history_fallback'
        else:
            extraction_method = 'llm_failed'
    
    # 查询 venue
    venue = db.query_venue_by_name(domain=domain, name=name)
    if not venue:
        venue = f'"{name}" is not found in the "{domain}" table.'
    
    # 查询预订记录 (如果需要)
    book_record = None
    if goal.get('book'):
        refer_number = llm_answer.get('reference number', 'none')
        book_record = booking.query_booking_by_refer_num(domain=domain, refer_number=refer_number)
        if not book_record:
            book_record = f'"{refer_number}" is not found in the "book_{domain}" table.'
    
    # 使用新的评估逻辑
    from taskdialogue.benchmarks.multiwoz.evaluators.evaluation_logic import get_domain_evaluator
    evaluator = get_domain_evaluator()
    
    result = evaluator.evaluate_venue_domain(
        domain=domain,
        venue=venue,
        goal=goal,
        llm_extracted=llm_answer,
        book_record=book_record,
        success_mode=success_mode
    )
    
    # 添加额外信息
    result['inform']['extraction_method'] = extraction_method
    
    return result

# endregion

@tenacity.retry(wait=tenacity.wait_exponential(min=2, max=60),
                    stop=tenacity.stop_after_attempt(8),
                    before_sleep=tenacity_retry_log,
                    retry=tenacity.retry_if_exception_type((Exception, json.JSONDecodeError, ValueError)))
def llm_qa(goal_messages, dialog_pred, questions, answer_formats, model, domain=None):
    goals_str = prepare_goals_string(goal_messages)
    dialog_str = prepare_dialog_string(dialog_pred)
    
    # Get prompts from prompts module
    system_prompt = get_evaluation_system_prompt()
    human_template = get_evaluation_human_template()
    
    # 动态获取格式要求（基于数据库检测）
    format_requirements = ""
    # 暂时禁用格式检测，避免导入错误
    # if domain:
    #     try:
    #         from taskdialogue.benchmarks.multiwoz.evaluators.format_detector import get_format_instructions_for_domain
    #         format_requirements = get_format_instructions_for_domain(domain)
    #         if format_requirements:
    #             format_requirements = "\n" + format_requirements
    #     except Exception as e:
    #         print(f"⚠️  格式检测失败 ({domain}): {e}")
    #         format_requirements = ""
    
    human_prompt = human_template.format(
        goals=goals_str, 
        dialog=dialog_str, 
        questions=questions, 
        answer_formats=answer_formats,
        format_requirements=format_requirements,
    )

    # 检查prompt长度（粗略估计）
    prompt_length = len(system_prompt) + len(human_prompt)

    # 使用传入的评估模型（而非全局的）
    # model 参数由 evaluate_single_dialogue 传入，已包含正确的 API key
    
    try:
        # 使用评估模型初始化时的参数，并获取真实的token使用量
        result = model.chat(
            system_message=system_prompt,
            user_message=human_prompt,
            return_usage=True
        )
        result_origin = result['content']
        token_usage = result['usage']
    except Exception as e:
        print(f"\n⚠️  LLM调用失败: {e}")
        print(f"Prompt长度: {prompt_length}字符")
        raise
    
    # 检查响应是否为空
    if not result_origin or result_origin.strip() == '':
        print(f"\n⚠️  LLM返回了空响应")
        print(f"Prompt长度: {prompt_length}字符")
        print(f"System message长度: {len(system_prompt)}")
        print(f"User message长度: {len(human_prompt)}")
        raise ValueError("LLM返回了空响应，无法进行评估")

    # Clean json string
    def clean_json_string(text):
        text = text.strip()
        
        # 移除markdown代码块标记
        if text.startswith('```'):
            # 查找第一个换行符（代码块开始后）
            start_pos = text.find('\n')
            if start_pos > -1:
                text = text[start_pos + 1:]
            else:
                text = text[3:]  # 移除 ```
        
        # 移除json标记（如果有）
        if text.startswith('json'):
            text = text[4:].lstrip()
        
        # 移除末尾的代码块标记
        if '```' in text:
            text = text[:text.find('```')]
        
        # 移除前后的反引号
        text = text.strip('`').strip()
        
        # 尝试找到JSON对象的开始和结束
        # 支持 {} 和 [] 两种格式
        start_char = None
        end_char = None
        
        if '{' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx > -1 and end_idx > -1 and end_idx > start_idx:
                text = text[start_idx:end_idx + 1]
        elif '[' in text:
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx > -1 and end_idx > -1 and end_idx > start_idx:
                text = text[start_idx:end_idx + 1]
        
        return text.strip()
    
    result_cleann = clean_json_string(result_origin)

    try:
        llm_answer = json.loads(result_cleann)
        
        # 🔍 调试：检查是否提取了 info_ 字段
        info_fields = [k for k in llm_answer.keys() if k.startswith('info_')]
        if info_fields and False:  # 设为 False 关闭调试输出
            print(f"✅ LLM 提取的 info fields: {info_fields}")
        
    except json.JSONDecodeError as e:
        # 打印调试信息
        print(f"\n⚠️  JSON解析失败")
        print(f"原始响应长度: {len(result_origin)}")
        print(f"原始响应前300字符:\n{result_origin[:300]}")
        print(f"\n清理后的JSON长度: {len(result_cleann)}")
        print(f"清理后的JSON前300字符:\n{result_cleann[:300]}")
        print(f"\n错误详情: {e}")
        raise  # 重新抛出异常，让tenacity重试
    
    return llm_answer, token_usage


def evaluate_single_dialogue(dialog, goal, eval_model, success_mode='strict', provide_db_hints=True):
    """
    评估单个对话（完整流程）
    
    Args:
        dialog: 对话历史（格式：[{"user": ..., "agent": ...}, ...]）
        goal: 目标（格式：{domain: {info: {...}, reqt: {...}, book: {...}}}）
        eval_model: 评估模型
        success_mode: 评估模式（strict 或 relaxed）
        provide_db_hints: 是否提供数据库提示
        
    Returns:
        评估结果字典
    """
    result = {
        'domains': {},
        'overall': {}
    }
    
    # 检查 goal 是否有效
    if not goal or not isinstance(goal, dict):
        result['error'] = 'Invalid or empty goal'
        return result
    
    # 遍历每个领域
    for domain, domain_goal in goal.items():
        if domain not in ['restaurant', 'hotel', 'attraction', 'train', 'taxi']:
            continue
        
        # 跳过空goal的域（没有info, reqt, book任何内容的域）
        if not domain_goal or (not domain_goal.get('info') and not domain_goal.get('reqt') and not domain_goal.get('book')):
            continue
        
        # 调用对应领域的评估函数
        if domain == 'taxi':
            from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import (
                prepare_taxi_questions,
                evaluate_by_domain_taxi
            )
            questions, answer_formats = prepare_taxi_questions(domain_goal)
            llm_answer, _ = llm_qa(
                goal_messages=domain_goal,
                dialog_pred=dialog,
                questions=questions,
                answer_formats=answer_formats,
                model=eval_model,
                domain=domain
            )
            domain_result = evaluate_by_domain_taxi(domain_goal, llm_answer)
            
        elif domain == 'train':
            from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import (
                prepare_train_questions,
                evaluate_by_domain_train
            )
            questions, answer_formats = prepare_train_questions(domain_goal, add_format_hints=provide_db_hints)
            llm_answer, _ = llm_qa(
                goal_messages=domain_goal,
                dialog_pred=dialog,
                questions=questions,
                answer_formats=answer_formats,
                model=eval_model,
                domain=domain
            )
            domain_result = evaluate_by_domain_train(domain_goal, llm_answer, success_mode=success_mode)
            
        else:  # restaurant, hotel, attraction
            # 动态选择问题准备函数
            if domain == 'restaurant':
                from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import prepare_restaurant_questions
                questions, answer_formats = prepare_restaurant_questions(domain_goal)
            elif domain == 'hotel':
                from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import prepare_hotel_questions
                questions, answer_formats = prepare_hotel_questions(domain_goal)
            else:  # attraction
                from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import prepare_attraction_questions
                questions, answer_formats = prepare_attraction_questions(domain_goal, provide_db_hints=provide_db_hints)
            
            llm_answer, _ = llm_qa(
                goal_messages=domain_goal,
                dialog_pred=dialog,
                questions=questions,
                answer_formats=answer_formats,
                model=eval_model,
                domain=domain
            )
            domain_result = evaluate_by_domain_others(domain_goal, llm_answer, domain, dialog, success_mode=success_mode)
        
        result['domains'][domain] = domain_result
    
    # 计算整体指标
    total_domains = len(result['domains'])
    if total_domains > 0:
        # 明确检查 None 值，保留 0 和布尔值（参考 metrics.py）
        inform_sum = sum(
            d.get('inform', {}).get('complete') if d.get('inform', {}).get('complete') is not None else 0 
            for d in result['domains'].values()
        )
        success_sum = sum(
            d.get('success', {}).get('complete') if d.get('success', {}).get('complete') is not None else 0 
            for d in result['domains'].values()
        )
        
        inform_rate = inform_sum / total_domains
        success_rate = success_sum / total_domains
        combined_score = 0.5 * inform_rate + 0.5 * success_rate
        
        # 计算 JGA 和 Slot-F1（从所有领域的 slot_values 聚合）
        from taskdialogue.benchmarks.multiwoz.evaluators.metrics import calculate_jga, calculate_slot_f1
        
        all_jga_scores = []
        all_f1_scores = []
        
        for domain, domain_data in result['domains'].items():
            # 获取该领域的 slot_values
            inform_slot_values = domain_data.get('inform', {}).get('slot_values', {})
            
            # 获取 ground truth（从 goal 中）
            domain_goal = goal.get(domain, {})
            gt_slots = domain_goal.get('info', {})
            
            if inform_slot_values and gt_slots:
                # 计算 JGA
                jga_score = calculate_jga(inform_slot_values, gt_slots)
                all_jga_scores.append(jga_score)
                
                # 计算 Slot F1
                f1_result = calculate_slot_f1(inform_slot_values, gt_slots)
                all_f1_scores.append(f1_result['f1'])
        
        # 对话级别的 JGA 和 F1：所有领域的平均值
        jga = sum(all_jga_scores) / len(all_jga_scores) if all_jga_scores else 0.0
        slot_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        
        result['overall'] = {
            'inform_rate': inform_rate,
            'success_rate': success_rate,
            'combined_score': combined_score,
            'jga': jga,
            'slot_f1': slot_f1
        }
    
    return result


def show_eval_result(result):
    RED = '\u001b[1;31m'
    GREEN = '\u001b[1;33m'
    RESET = '\u001b[0m'
    for k, v in result.items():
        if isinstance(v, str) or isinstance(v, float) or v is None:
            print(RED + f'[{k}]' + RESET + f' {v}')

        elif isinstance(v, dict):
            indent = 4
            if 'complete' in v:
                print(RED + f'[{k}] complete: {v["complete"]}' + RESET)
            else:
                print(RED + f'[{k}]' + RESET)
            for kk, vv in v.items():
                if kk == 'complete':
                    continue
                print(' ' * indent + GREEN + f'{kk}: ' + RESET, end='')
                print(vv)

        else:
            print(RED + f'[{k}]' + RESET)
            pprint(v)


def evaluate_by_domain(domain, run_result, model='gpt-3.5-turbo-0301', verbose=True, success_mode='relaxed'):
    """
    按领域评估对话结果
    
    Args:
        domain: 领域名称
        run_result: 运行结果
        model: 模型名称
        verbose: 是否显示详细信息
        success_mode: Success 评估模式
            - 'relaxed': 宽松模式，只检查是否提供（对齐 TD-Eval）
            - 'strict': 严格模式，验证值是否与数据库匹配
    """
    assert run_result['goals'].get(domain)
    
    goal_dict = run_result['goals'][domain]
    dialog_pred = run_result['dialog_pred']
    goal_messages = run_result['goal_messages']

    if domain == 'hotel':
        questions, answer_formats = prepare_hotel_questions(goal_dict)
        llm_answer, token_usage = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model, domain=domain)
        result = evaluate_by_domain_others(goal_dict, llm_answer, domain, dialog_pred, success_mode=success_mode)

    elif domain == 'restaurant':
        questions, answer_formats = prepare_restaurant_questions(goal_dict)
        llm_answer, token_usage = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model, domain=domain)
        result = evaluate_by_domain_others(goal_dict, llm_answer, domain, dialog_pred, success_mode=success_mode)

    elif domain == 'attraction':
        # 读取是否提供数据库提示
        from taskdialogue.core.utils.config import load_config
        config = load_config()
        provide_hints = config.get('evaluation.provide_db_hints', True) and success_mode == 'strict'
        
        questions, answer_formats = prepare_attraction_questions(goal_dict, provide_db_hints=provide_hints)
        llm_answer, token_usage = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model, domain=domain)
        result = evaluate_by_domain_others(goal_dict, llm_answer, domain, dialog_pred, success_mode=success_mode)

    elif domain == 'train':
        questions, answer_formats = prepare_train_questions(goal_dict)
        llm_answer, token_usage = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model, domain=domain)
        result = evaluate_by_domain_train(goal_dict, llm_answer, success_mode=success_mode)

    elif domain == 'taxi':
        questions, answer_formats = prepare_taxi_questions(goal_dict)
        llm_answer, token_usage = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model, domain=domain)
        result = evaluate_by_domain_taxi(goal_dict, llm_answer)

    else:
        raise ValueError(f'{domain = }')
    
    # 添加token使用统计和LLM调用次数到结果中
    result['token_usage'] = token_usage
    result['llm_call_count'] = 1  # 每个domain评估调用一次LLM
    
    if verbose:
        show_eval_result(result)
    return result
