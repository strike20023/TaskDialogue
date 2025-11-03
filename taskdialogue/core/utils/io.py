"""
I/O utilities for TaskDialogue.

Provides utilities for reading and writing data files.
"""

import json
from typing import Any


def load_json_or_jsonl(path: str) -> Any:
    """
    Load JSON or JSONL file automatically.
    
    This function automatically detects the file format:
    - Files with .jsonl extension are treated as JSONL format
    - Files with .json extension are treated as regular JSON format
    - For other extensions, tries JSON first, then falls back to JSONL
    
    Args:
        path: Path to the JSON or JSONL file
        
    Returns:
        Parsed data (dict, list, or list of objects)
        
    Examples:
        >>> # Load JSON array file
        >>> data = load_json_or_jsonl('results.json')
        >>> 
        >>> # Load JSONL file
        >>> data = load_json_or_jsonl('results.jsonl')
    """
    # Check file extension first (most reliable)
    if path.endswith('.jsonl'):
        # JSONL format - read line by line
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
            return data
    else:
        # Regular JSON format (default for .json files)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_jsonl(data: list, path: str):
    """
    Save data to JSONL format.
    
    Args:
        data: List of objects to save
        path: Output file path
        
    Examples:
        >>> data = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
        >>> save_jsonl(data, 'output.jsonl')
    """
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

