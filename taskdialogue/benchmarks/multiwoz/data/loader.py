"""
MultiWOZ 数据加载器
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class MultiWOZDataset:
    """MultiWOZ 数据集类"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def get_by_id(self, dialogue_id: str) -> Optional[Dict[str, Any]]:
        """根据对话 ID 获取数据"""
        for item in self.data:
            if item.get("dialogue_idx") == dialogue_id:
                return item
        return None
    
    def filter_by_domains(self, domains: List[str]) -> "MultiWOZDataset":
        """筛选包含指定领域的对话"""
        filtered = []
        for item in self.data:
            goal = item.get("goal", {})
            item_domains = list(goal.keys())
            if any(d in domains for d in item_domains):
                filtered.append(item)
        return MultiWOZDataset(filtered)


def load_multiwoz_data(
    data_path: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    remove_police_hospital: bool = True,
    enabled_domains: Optional[List[str]] = None
) -> MultiWOZDataset:
    """加载 MultiWOZ 数据
    
    Args:
        data_path: 数据文件路径
        split: 数据集划分 (train/dev/test)
        num_samples: 限制样本数量
        remove_police_hospital: 是否移除 police 和 hospital 领域
        enabled_domains: 启用的领域列表（过滤）
        
    Returns:
        MultiWOZDataset 实例
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 处理 MultiWOZ 格式（dialogue_id 作为 key 的字典）
    if isinstance(raw_data, dict):
        # 检查是否是 {train: [...], test: [...]} 格式
        if split and split in raw_data:
            data = raw_data[split]
        else:
            # 是 {dialogue_id: dialogue_data} 格式，转换为列表
            data = []
            for dialogue_id, dialogue_data in raw_data.items():
                dialogue_data['dialogue_idx'] = dialogue_id  # 添加 dialogue_idx 字段
                data.append(dialogue_data)
            
            # 应用 split 过滤（使用外部 split 文件）
            if split and split != 'null' and split.lower() != 'none':
                data_dir = data_path.parent
                split_ids = None
                
                if split == 'test':
                    split_file = data_dir / 'testListFile.json'
                elif split in ['valid', 'val', 'validation']:
                    split_file = data_dir / 'valListFile.json'
                elif split == 'train':
                    # train 是所有不在 test/val 中的数据
                    test_file = data_dir / 'testListFile.json'
                    val_file = data_dir / 'valListFile.json'
                    
                    exclude_ids = set()
                    if test_file.exists():
                        with open(test_file, 'r') as f:
                            exclude_ids.update(line.strip() for line in f if line.strip())
                    if val_file.exists():
                        with open(val_file, 'r') as f:
                            exclude_ids.update(line.strip() for line in f if line.strip())
                    
                    data = [d for d in data if d.get('dialogue_idx') not in exclude_ids]
                    print(f"Loaded {len(data)} train dialogues (excluded {len(exclude_ids)} from test/val)")
                else:
                    print(f"⚠️  Unknown split '{split}', using all data")
                
                # 对于 test/val，加载 split 文件
                if split in ['test', 'valid', 'val', 'validation'] and 'split_file' in locals():
                    if split_file.exists():
                        with open(split_file, 'r') as f:
                            split_ids = set(line.strip() for line in f if line.strip())
                        data = [d for d in data if d.get('dialogue_idx') in split_ids]
                        print(f"Loaded {len(data)} {split} dialogues from split file")
                    else:
                        print(f"⚠️  Split file not found: {split_file}, using all data")
    else:
        data = raw_data
    
    # 移除 police 和 hospital 领域
    if remove_police_hospital:
        original_len = len(data)
        data = [
            d for d in data
            if not any(
                domain in ['police', 'hospital'] and d.get('goal', {}).get(domain)  # 检查领域是否有实际内容
                for domain in d.get('goal', {}).keys()
            )
        ]
        if len(data) < original_len:
            print(f"Removed police/hospital: {original_len} → {len(data)} dialogues")
    
    # 限制样本数量
    if num_samples is not None:
        data = data[:num_samples]
    
    dataset = MultiWOZDataset(data)
    
    # 领域过滤
    if enabled_domains:
        dataset = dataset.filter_by_domains(enabled_domains)
    
    return dataset

