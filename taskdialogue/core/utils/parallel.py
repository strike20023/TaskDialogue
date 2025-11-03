"""
并行处理工具模块
支持多进程和多线程，适用于推理和评估任务
"""
from typing import List, Callable, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import json
from pathlib import Path
from tqdm import tqdm


def parallel_process(
    items: List[Any],
    process_func: Callable[[Any, int], Tuple[int, Any]],
    num_workers: int = 1,
    use_processes: bool = False,
    desc: str = "Processing",
    progress_callback: Optional[Callable] = None,
    show_progress: bool = True
) -> List[Any]:
    """
    并行处理列表项（线程或进程池）
    
    Args:
        items: 待处理的项目列表
        process_func: 处理函数，接收 (item, idx) 返回 (idx, result)
        num_workers: 并发数
        use_processes: True使用进程池，False使用线程池
        desc: 进度条描述
        progress_callback: 进度回调函数
        
    Returns:
        按原始顺序排列的结果列表
    """
    if num_workers <= 1:
        # 单线程/进程模式
        results = []
        iterator = tqdm(items, desc=desc, disable=not show_progress) if show_progress else enumerate(items)
        for idx, item in (enumerate(iterator) if show_progress else iterator):
            try:
                _, result = process_func(item, idx)
                results.append(result)
                if progress_callback:
                    progress_callback(result, idx)
            except Exception as e:
                print(f"\n❌ Item {idx} failed: {e}")
                results.append(None)
        return results
    
    # 多线程/进程模式（保证顺序）
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    results_dict = {}
    
    with executor_class(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_idx = {}
        for idx, item in enumerate(items):
            future = executor.submit(process_func, item, idx)
            future_to_idx[future] = idx
        
        # 收集结果
        pbar = tqdm(total=len(items), desc=desc, disable=not show_progress)
        with pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, result = future.result()
                    results_dict[result_idx] = result
                    if progress_callback:
                        progress_callback(result, idx)
                except Exception as e:
                    print(f"\n❌ Item {idx} failed: {e}")
                    results_dict[idx] = None
                finally:
                    pbar.update(1)
    
    # 按索引排序
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    return results


class BatchSaver:
    """批量保存工具（线程安全）"""
    
    def __init__(self, save_func: Callable[[List[Any]], None], save_batch_size: int = 32, lock: Optional[Lock] = None):
        """
        Args:
            save_func: 保存函数，接收结果列表
            save_batch_size: 批量保存大小
            lock: 线程锁（多线程时需要）
        """
        self.save_func = save_func
        self.save_batch_size = save_batch_size
        self.lock = lock
        self.buffer = []
    
    def add(self, result: Any):
        """添加结果到缓冲区，满时自动保存"""
        self.buffer.append(result)
        if len(self.buffer) >= self.save_batch_size:
            self.flush()
    
    def flush(self):
        """立即保存缓冲区中的所有结果"""
        if not self.buffer:
            return
        
        if self.lock:
            with self.lock:
                self._save()
        else:
            self._save()
        
        self.buffer = []
    
    def _save(self):
        """实际保存操作（内部方法）"""
        self.save_func(self.buffer)


def save_jsonl(results: List[dict], output_path: Path):
    """保存结果为 JSONL 格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 追加模式写入
    with open(output_path, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def save_json(results: List[dict], output_path: Path):
    """保存结果为 JSON 格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 追加模式写入
    with open(output_path, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def parallel_process_with_saving(
    items: List[Any],
    process_func: Callable[[Any, int], Tuple[int, Any]],
    save_func: Callable[[List[Any]], None],
    num_workers: int = 1,
    save_batch_size: int = 32,
    use_processes: bool = False,
    desc: str = "Processing",
    final_save: bool = True,
    show_progress: bool = True
) -> List[Any]:
    """
    并行处理并按顺序批量保存
    
    关键：避免高并发时内存堆积，每当按顺序完成 save_batch_size 个任务就立即保存
    
    Args:
        items: 待处理的项目列表
        process_func: 处理函数，返回 (idx, result)
        save_func: 保存函数，接收结果列表
        num_workers: 并发数（推荐 4-256，高并发时务必设置 save_batch_size）
        save_batch_size: 批量保存大小（每收集到连续的 N 个结果就保存）
        use_processes: 是否使用进程池（默认线程池）
        desc: 进度条描述
        final_save: 是否最后再保存一次未满批次的
        
    Returns:
        按原始顺序排列的结果列表
    """
    results_dict = {}
    next_save_idx = 0  # 下一个要保存的索引
    
    if num_workers <= 1:
        # 单线程模式
        batch_to_save = []
        iterator = tqdm(items, desc=desc, disable=not show_progress) if show_progress else enumerate(items)
        for idx, item in (enumerate(iterator) if show_progress else iterator):
            try:
                _, result = process_func(item, idx)
                results_dict[idx] = result
                batch_to_save.append(result)
                
                # 每满 save_batch_size 就保存一次
                if len(batch_to_save) >= save_batch_size:
                    try:
                        save_func(batch_to_save)
                    except Exception as e:
                        print(f"\n❌ Save function failed in single-thread mode at index {idx - len(batch_to_save) + 1}: {e}")
                    batch_to_save = []
                    
            except Exception as e:
                print(f"\n❌ Item {idx} failed: {e}")
                results_dict[idx] = None
        
        # 保存剩余结果
        if final_save and batch_to_save:
            try:
                save_func(batch_to_save)
                print(f"\n✅ Final save in single-thread completed: {len(batch_to_save)} items")
            except Exception as e:
                print(f"\n❌ Final save in single-thread failed: {e}")
    
    else:
        # 多线程/进程模式（按顺序增量保存）
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=num_workers) as executor:
            future_to_idx = {}
            for idx, item in enumerate(items):
                future = executor.submit(process_func, item, idx)
                future_to_idx[future] = idx
            
            pbar = tqdm(total=len(items), desc=desc, disable=not show_progress)
            with pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result_idx, result = future.result()
                        results_dict[result_idx] = result
                    except Exception as e:
                        print(f"\n❌ Item {idx} failed: {e}")
                        results_dict[idx] = None
                    finally:
                        pbar.update(1)
                    
                    # ✅ 改进：每次检查时，收集并保存所有从 next_save_idx 开始的连续完成结果
                    # 这确保了：1. 按顺序保存  2. 不会遗漏任何批次
                    while next_save_idx in results_dict:
                        batch_to_save = []
                        # 收集一批连续结果
                        while next_save_idx in results_dict and len(batch_to_save) < save_batch_size:
                            batch_to_save.append(results_dict[next_save_idx])
                            next_save_idx += 1
                        
                        # 如果收集满一批，立即保存
                        if len(batch_to_save) >= save_batch_size:
                            try:
                                save_func(batch_to_save)
                                print(f"\n✅ Saved batch: indices {next_save_idx - len(batch_to_save)} to {next_save_idx - 1} ({len(batch_to_save)} items)")
                            except Exception as e:
                                print(f"\n❌ Save function failed at index {next_save_idx - len(batch_to_save)}: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            # 不足一批，留给下次或 final_save
                            # 需要回退 next_save_idx
                            next_save_idx -= len(batch_to_save)
                            break
        
        # 保存最后剩余的连续结果
        if final_save:
            batch_to_save = []
            while next_save_idx in results_dict:
                batch_to_save.append(results_dict[next_save_idx])
                next_save_idx += 1
            
            if batch_to_save:
                try:
                    save_func(batch_to_save)
                    print(f"\n✅ Final save completed: indices {next_save_idx - len(batch_to_save)} to {next_save_idx - 1} ({len(batch_to_save)} items)")
                except Exception as e:
                    print(f"\n❌ Final save failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 返回按索引排序的结果
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    return results

