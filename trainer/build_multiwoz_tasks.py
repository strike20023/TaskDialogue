import argparse
import json
import os
from typing import List, Dict

from taskdialogue.core.utils.config import load_config
from taskdialogue.benchmarks.multiwoz.data import load_multiwoz_data


def _ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _write_jsonl(items: List[Dict], out_path: str):
    _ensure_dir(out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _write_parquet(items: List[Dict], out_path: str):
    try:
        import pandas as pd
    except Exception:
        raise RuntimeError("pandas 未安装，无法写入 parquet。请改用 --format jsonl 或安装 pandas/pyarrow。")
    _ensure_dir(out_path)
    df = pd.DataFrame(items)
    # Prefer pyarrow if available
    try:
        df.to_parquet(out_path, engine='pyarrow', index=False)
    except Exception:
        # Fallback to fastparquet
        df.to_parquet(out_path, engine='fastparquet', index=False)


def build_tasks(split: str, config_path: str) -> List[Dict]:
    cfg = load_config(config_path)
    data_cfg = cfg.get('data', {})
    domains_cfg = cfg.get('domains', {})

    data_path = data_cfg.get('path', 'data/multiwoz/data.json')
    remove_ph = bool(data_cfg.get('remove_police_hospital', True))
    enabled_domains = domains_cfg.get('enabled', None)

    # Sanity check: data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}。请先解压 data.tar.gz 或更新配置中的 data.path。")

    dataset = load_multiwoz_data(
        data_path=data_path,
        split=split,
        num_samples=None,
        remove_police_hospital=remove_ph,
        enabled_domains=enabled_domains,
    )
    tasks: List[Dict] = []
    for item in dataset:
        did = item.get('dialogue_idx') or item.get('dialogue_id') or item.get('id')
        if did is None:
            continue
        tasks.append({'dialogue_id': did})
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Build MultiWOZ task files containing dialogue_id for VERL Trainer.')
    parser.add_argument('--config', type=str, default='configs/multiwoz/default.yaml', help='配置文件路径')
    parser.add_argument('--train-output', type=str, default='data/multiwoz_train.jsonl', help='训练集任务输出路径')
    parser.add_argument('--val-output', type=str, default='data/multiwoz_val.jsonl', help='验证集任务输出路径')
    parser.add_argument('--format', type=str, choices=['jsonl', 'parquet'], default='jsonl', help='任务文件格式')
    args = parser.parse_args()

    train_tasks = build_tasks('train', args.config)
    val_tasks = build_tasks('test', args.config)

    if args.format == 'jsonl':
        _write_jsonl(train_tasks, args.train_output)
        _write_jsonl(val_tasks, args.val_output)
    else:
        _write_parquet(train_tasks, args.train_output)
        _write_parquet(val_tasks, args.val_output)

    print(f"Wrote {len(train_tasks)} train tasks to {args.train_output}")
    print(f"Wrote {len(val_tasks)} val tasks to {args.val_output}")


if __name__ == '__main__':
    main()