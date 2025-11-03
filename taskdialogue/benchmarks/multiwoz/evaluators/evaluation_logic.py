"""
优化的评估逻辑模块
参考 TD-Eval/MultiWOZ_Evaluation 的严格标准

核心原则:
1. Inform: 系统是否提供了满足用户约束的实体
2. Success: 在 Inform 成功的前提下，是否提供了所有请求的信息
3. 严格的依赖关系: Success ⊆ Inform
"""

from typing import Dict, Any, Optional
from taskdialogue.benchmarks.multiwoz.evaluators.matching.slot_matcher import smart_match


class EvaluationStandard:
    """
    统一的评估标准
    参考 TD-Eval 的 Inform & Success 逻辑
    """
    
    @staticmethod
    def evaluate_inform(venue, goal_info: Dict[str, Any], fail_info: Optional[Dict[str, Any]] = None, 
                       llm_extracted_info: Optional[Dict[str, Any]] = None) -> tuple[bool, Dict[str, str]]:
        """
        评估 Inform 指标
        
        核心逻辑 (参考 TD-Eval metrics.py 第 228-236 行):
        - 检查系统提供的 venue 是否满足用户的 info 约束
        - 系统提供的实体必须是用户目标的子集
        
        Args:
            venue: 系统提供的实体 (来自数据库查询)
            goal_info: 用户目标的 info 约束 (informable slots)
            fail_info: 备用的 info 约束 (用于处理数据库限制)
            llm_extracted_info: LLM 从对话中提取的 info slots (用于 JGA/Slot-F1)
        
        Returns:
            (is_complete, slot_values):
                - is_complete: Inform 是否完成 (True/False)
                - slot_values: LLM 提取的槽位值 (用于 JGA 和 Slot-F1)
        """
        from taskdialogue.benchmarks.multiwoz.tools.db import Veune
        
        # 如果没有找到 venue
        if not isinstance(venue, Veune):
            # 使用 LLM 提取的值（如果有），否则设为 'none'
            slot_values = {}
            if goal_info:
                for slot in goal_info:
                    slot_values[slot] = llm_extracted_info.get(f'info_{slot}', 'none') if llm_extracted_info else 'none'
            return False, slot_values
        
        # 检查 venue 是否满足用户的约束
        # 参考 TD-Eval: offered_venues 必须是 goal_venues 的子集
        is_complete = venue.satisfying(goal_info)
        
        # 如果有 fail_info (备用约束)，也算成功
        if not is_complete and fail_info:
            is_complete = venue.satisfying(fail_info)
        
        # 🆕 使用 LLM 提取的 info slots (用于 JGA 和 Slot-F1 计算)
        # 这才是真正评估系统对用户约束的理解能力
        slot_values = {}
        if goal_info:
            for slot in goal_info:
                info_key = f'info_{slot}'
                if llm_extracted_info and info_key in llm_extracted_info:
                    # 使用 LLM 提取的值
                    slot_values[slot] = llm_extracted_info[info_key]
                else:
                    # 如果 LLM 没提取，设为 'none'（表示 LLM 没能识别这个约束）
                    # 不要用 venue 或 goal 的值，因为那样会失去评估意义
                    slot_values[slot] = 'none'
        
        return is_complete, slot_values
    
    @staticmethod
    def evaluate_success(venue, 
                        llm_extracted_slots: Dict[str, Any], 
                        goal_reqt: Dict[str, Any],
                        inform_complete: bool,
                        mode: str = 'relaxed') -> tuple[bool, Dict[str, str]]:
        """
        评估 Success 指标
        
        核心逻辑 (参考 TD-Eval metrics.py 第 245-254 行):
        - **前提条件**: 只有当 Inform = 1 时才能计算 Success
        - 检查系统是否提供了用户请求的所有信息 (requestable slots)
        - 必须提供所有 reqt slots 才算成功
        
        Args:
            venue: 系统提供的实体
            llm_extracted_slots: LLM 从对话中提取的槽位值
            goal_reqt: 用户目标中的请求槽位 (reqt)
            inform_complete: Inform 是否完成
            mode: 评估模式
                - 'relaxed': 宽松模式，只检查是否提供（对齐 TD-Eval）
                - 'strict': 严格模式，验证值是否与数据库匹配
        
        Returns:
            (is_complete, slot_values):
                - is_complete: Success 是否完成 (True/False)
                - slot_values: 提取的槽位值
        """
        from taskdialogue.benchmarks.multiwoz.tools.db import Veune
        
        # 参考 TD-Eval 第 245 行: if match['total']
        # Success 的前提是 Inform 必须成功
        if not inform_complete:
            slot_values = {slot: llm_extracted_slots.get(slot, 'none') for slot in goal_reqt} if goal_reqt else {}
            return False, slot_values
        
        # 如果没有请求槽位，Success 自动为 True
        if not goal_reqt:
            return True, {}
        
        # 提取 LLM 的槽位值
        slot_values = {slot: llm_extracted_slots.get(slot, 'none') for slot in goal_reqt}
        
        # 根据模式选择评估策略
        if mode == 'relaxed':
            # 宽松模式（对齐 TD-Eval 第 186-198 行）：
            # 只检查系统是否"提到了"这些信息，不验证值是否正确
            # TD-Eval: if requestable_slot in system_utterance
            is_complete = all(
                v and str(v).lower() not in ['none', ''] 
                for v in slot_values.values()
            )
        else:  # strict mode
            # 严格模式：验证提供的值是否与数据库匹配
            if isinstance(venue, Veune):
                is_complete = venue.satisfying(slot_values)
            else:
                # venue 不存在时降级为宽松标准
                is_complete = all(
                    v and str(v).lower() not in ['none', ''] 
                    for v in slot_values.values()
                )
        
        return is_complete, slot_values
    
    @staticmethod
    def evaluate_book(venue,
                     book_record,
                     goal_book: Dict[str, Any],
                     fail_book: Optional[Dict[str, Any]],
                     inform_complete: bool) -> bool:
        """
        评估 Book 指标
        
        核心逻辑:
        - **前提条件**: Inform 必须成功
        - 预订记录必须存在
        - 预订的 venue 必须匹配
        - 预订参数必须满足用户要求
        
        Args:
            venue: 系统提供的实体
            book_record: 预订记录
            goal_book: 用户目标的预订参数
            fail_book: 备用的预订参数
            inform_complete: Inform 是否完成
        
        Returns:
            is_complete: Book 是否完成
        """
        from taskdialogue.benchmarks.multiwoz.tools.db import Veune
        from taskdialogue.benchmarks.multiwoz.tools.booking import BookRecord
        
        # 前提条件: Inform 必须成功
        if not inform_complete:
            return False
        
        # 检查 venue 和 book_record 是否有效
        if not isinstance(venue, Veune) or not isinstance(book_record, BookRecord):
            return False
        
        # 检查预订的 venue 是否匹配 (大小写不敏感)
        if book_record.name.lower() != venue.name.lower():
            return False
        
        # 检查预订参数是否满足要求
        book_match = book_record.satisfying(goal_book)
        if not book_match and fail_book:
            book_match = book_record.satisfying(fail_book)
        
        return book_match


class DomainEvaluator:
    """
    领域级别的评估器
    整合 Inform、Success、Book 的评估逻辑
    """
    
    def __init__(self):
        self.standard = EvaluationStandard()
    
    def evaluate_venue_domain(self,
                             domain: str,
                             venue,
                             goal: Dict[str, Any],
                             llm_extracted: Dict[str, Any],
                             book_record=None,
                             success_mode: str = 'relaxed') -> Dict[str, Any]:
        """
        评估 venue 类型的领域 (restaurant, hotel, attraction)
        
        Args:
            domain: 领域名称
            venue: 查询到的实体
            goal: 用户目标
            llm_extracted: LLM 提取的信息
            book_record: 预订记录 (可选)
        
        Returns:
            评估结果字典
        """
        result = {
            'domain': domain,
            'goal': goal,
            'inform': {
                'complete': None,
                'venue_name': llm_extracted.get(domain, 'none'),
                'venue_info': venue,
                'slot_values': None,
            },
            'success': {
                'complete': None,
                'slot_values': None,
            },
            'book': {
                'complete': None,
                'refer_number': None,
                'book_record': None,
            }
        }
        
        # 1. 评估 Inform
        goal_info = goal.get('info', {})
        fail_info = goal.get('fail_info', {})
        
        inform_complete, inform_slot_values = self.standard.evaluate_inform(
            venue, goal_info, fail_info, llm_extracted_info=llm_extracted
        )
        
        result['inform']['complete'] = int(inform_complete)
        result['inform']['slot_values'] = inform_slot_values
        
        # 2. 评估 Success (依赖于 Inform)
        goal_reqt = goal.get('reqt', {})
        
        if goal_reqt:
            success_complete, success_slot_values = self.standard.evaluate_success(
                venue, llm_extracted, goal_reqt, inform_complete, mode=success_mode
            )
            
            result['success']['complete'] = int(success_complete)
            result['success']['slot_values'] = success_slot_values
        else:
            # 参考 TD-Eval: 如果没有 reqt，Success 自动为 1（没有要求自然满足）
            # 但前提是 Inform 必须成功
            result['success']['complete'] = int(inform_complete)
            result['success']['slot_values'] = {}
        
        # 3. 评估 Book (依赖于 Inform)
        goal_book = goal.get('book', {})
        
        if goal_book:
            refer_number = llm_extracted.get('reference number', 'none')
            
            book_complete = self.standard.evaluate_book(
                venue, book_record, goal_book, 
                goal.get('fail_book', {}), inform_complete
            )
            
            result['book']['refer_number'] = refer_number
            result['book']['complete'] = int(book_complete)
            result['book']['book_record'] = book_record
        
        return result
    
    def evaluate_train_domain(self,
                             goal: Dict[str, Any],
                             llm_extracted: Dict[str, Any],
                             matched_items: list,
                             book_record=None,
                             train_info=None,
                             success_mode: str = 'relaxed') -> Dict[str, Any]:
        """
        评估 train 领域
        
        Train 的特殊性:
        - 没有 venue name，通过 trainID 或约束条件查询
        - Inform: 检查是否找到满足约束的火车
        - Success: 检查提供的信息是否正确 (如果有 reqt)
        - Book: 检查预订是否成功 (如果有 book)
        
        Args:
            goal: 用户目标
            llm_extracted: LLM 提取的信息
            matched_items: 匹配的火车列表
            book_record: 预订记录 (如果有 book)
            train_info: 火车信息 (如果有 book)
        
        Returns:
            评估结果字典
        """
        result = {
            'domain': 'train',
            'goal': goal,
            'inform': {
                'complete': None,
                'slot_values': None,
            },
            'success': {
                'complete': None,
                'slot_values': None,
            },
            'book': {
                'complete': None,
                'refer_number': None,
                'book_record': None,
                'train_info': None,
            }
        }
        
        goal_info = goal.get('info', {})
        goal_reqt = goal.get('reqt', {})
        goal_book = goal.get('book', {})
        
        # 场景1: 只查询信息 (有 reqt，没有 book)
        if goal_reqt and not goal_book:
            # 提取 LLM 的槽位值
            from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import TRAIN_SLOT_MAP
            slot_values = {slot: llm_extracted.get(TRAIN_SLOT_MAP[slot], 'none') for slot in goal_reqt}
            
            # Step 1: 计算 Inform - 检查是否提到了火车 (trainID 或其他 reqt slots)
            # 参考 TD-Eval: Inform 是检查是否提到了 TRAINID 或找到了匹配的火车
            # 这里我们检查是否有火车在数据库中满足 info 约束
            if matched_items:
                # 找到了满足 info 约束的火车，Inform = 1
                inform_complete = True
            else:
                # 没有找到火车，Inform = 0
                inform_complete = False
            
            # Step 2: 计算 Success - 在 Inform=1 的前提下，检查是否提供了所有 reqt slots
            # 参考 TD-Eval 第 245 行: if match['total']: 只有 Inform 成功时才检查 Success
            if inform_complete:
                if success_mode == 'relaxed':
                    # 宽松模式: 只检查是否提供了值
                    success_complete = all(
                        v and str(v).lower() not in ['none', ''] 
                        for v in slot_values.values()
                    )
                else:  # strict mode
                    # 严格模式: 验证值是否正确（至少有一个火车满足）
                    success_complete = any(item.satisfying(slot_values) for item in matched_items)
            else:
                # Inform 失败，Success 自动为 0
                success_complete = False
            
            result['inform']['complete'] = int(inform_complete)
            result['success']['complete'] = int(success_complete)
            result['success']['slot_values'] = slot_values
            
            # 保存 inform slot_values (LLM 提取的约束条件，用于 JGA/Slot-F1)
            if goal_info:
                info_slot_values = {}
                for slot in goal_info:
                    # 使用 LLM 提取的 info slots
                    info_key = f'info_{slot}'
                    if info_key in llm_extracted:
                        info_slot_values[slot] = llm_extracted[info_key]
                    else:
                        # LLM 没提取，设为 'none'（评估 LLM 的理解能力）
                        info_slot_values[slot] = 'none'
                
                result['inform']['slot_values'] = info_slot_values
        
        # 场景2: 预订火车 (有 book)
        elif goal_book:
            refer_number = llm_extracted.get('reference number', 'none')
            
            # 检查预订记录和火车信息是否存在
            from taskdialogue.benchmarks.multiwoz.tools.booking import BookRecord
            from taskdialogue.benchmarks.multiwoz.tools.db import TableItem
            
            if isinstance(book_record, BookRecord) and isinstance(train_info, TableItem):
                # 检查火车是否满足约束
                inform_complete = train_info.satisfying(goal_info)
                
                # 检查预订是否满足要求
                if inform_complete:
                    book_complete = book_record.satisfying({'tickets': goal_book['people']})
                else:
                    book_complete = False
                
                # 🔧 book 场景的 Success: 参考 TD-Eval，如果没有 reqt，Success = Inform
                # 因为 book 场景用户的主要目标是预订，而不是查询信息
                success_complete = inform_complete
                
                # 保存 inform slot_values (LLM 提取的)
                if goal_info:
                    info_slot_values = {}
                    for slot in goal_info:
                        info_key = f'info_{slot}'
                        if info_key in llm_extracted:
                            info_slot_values[slot] = llm_extracted[info_key]
                        else:
                            info_slot_values[slot] = 'none'
                    
                    result['inform']['slot_values'] = info_slot_values
            else:
                inform_complete = False
                book_complete = False
                success_complete = False
                
                if goal_info:
                    # 使用 LLM 提取的 info slots（即使失败也记录）
                    info_slot_values = {}
                    for slot in goal_info:
                        info_key = f'info_{slot}'
                        info_slot_values[slot] = llm_extracted.get(info_key, 'none')
                    result['inform']['slot_values'] = info_slot_values
            
            result['inform']['complete'] = int(inform_complete)
            result['success']['complete'] = int(success_complete)
            result['book']['complete'] = int(book_complete)
            result['book']['refer_number'] = refer_number
            result['book']['book_record'] = book_record
            result['book']['train_info'] = train_info
        
        # 确保 inform_slot_values 总是被设置
        if 'slot_values' not in result['inform'] and goal_info:
            info_slot_values = {}
            for slot in goal_info:
                info_key = f'info_{slot}'
                info_slot_values[slot] = llm_extracted.get(info_key, 'none')
            result['inform']['slot_values'] = info_slot_values
        
        return result
    
    def evaluate_taxi_domain(self,
                            goal: Dict[str, Any],
                            llm_extracted: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估 taxi 领域
        
        Taxi 的特殊性:
        - 没有数据库，无法验证
        - Inform: 检查是否提供了 info slots (departure, destination, time)
        - Success: 检查是否提供了 reqt slots (car type, phone)
        
        Args:
            goal: 用户目标
            llm_extracted: LLM 提取的信息
        
        Returns:
            评估结果字典
        """
        from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import TAXI_SLOT_MAP
        
        result = {
            'domain': 'taxi',
            'goal': goal,
            'inform': {
                'complete': None,
                'slot_values': None,
            },
            'success': {
                'complete': None,
                'slot_values': None,
            },
            'book': {
                'complete': None,
            }
        }
        
        goal_info = goal.get('info', {})
        goal_reqt = goal.get('reqt', {})
        
        # Taxi Inform 的正确理解：
        # - TD-Eval: taxi 自动 MATCHED（第 206-208 行）
        # - End-to-End: 应该检查是否提供了出行信息（info slots: departure, destination, time）
        # - 不是检查是否预订成功（那是 Success 的职责）
        if goal_info:
            slot_values = {slot: llm_extracted.get(TAXI_SLOT_MAP[slot], 'none') for slot in goal_info}
            
            # Taxi Inform: 检查是否所有 info slots 都被提取（表示对话中讨论了出行细节）
            # 对齐其他领域的 Inform 逻辑：检查约束条件是否被提到
            inform_complete = all(
                v and str(v).lower() not in ['none', ''] 
                for v in slot_values.values()
            )
            
            result['inform']['complete'] = int(inform_complete)
            result['inform']['slot_values'] = slot_values
        else:
            # 没有 info 的 taxi（异常情况）
            result['inform']['complete'] = 0
        
        # Success: 检查是否提供了所有 reqt slots (依赖于 Inform)
        # 对齐 TD-Eval 第 248-250 行：检查是否提供了 requestable slots
        if goal_reqt:
            slot_values = {slot: llm_extracted.get(TAXI_SLOT_MAP[slot], 'none') for slot in goal_reqt}
            
            # Success 依赖于 Inform (TD-Eval 第 245 行: if match['total'])
            if result['inform']['complete'] == 1:
                # 对齐 TD-Eval：检查是否所有 reqt slots 都被提供（提到了）
                # TD-Eval 第 186-198 行：if requestable_slot in system_utterance
                success_complete = all(
                    v and str(v).lower() not in ['none', ''] 
                    for v in slot_values.values()
                )
            else:
                success_complete = False
            
            result['success']['complete'] = int(success_complete)
            result['success']['slot_values'] = slot_values
        else:
            # 没有 reqt: Success = Inform (参考 venue 逻辑)
            result['success']['complete'] = int(result['inform']['complete'])
        
        return result


# 全局实例
_domain_evaluator = DomainEvaluator()


def get_domain_evaluator() -> DomainEvaluator:
    """获取领域评估器单例"""
    return _domain_evaluator

