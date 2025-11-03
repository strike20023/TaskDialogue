"""
Utility functions for tools.

This module provides helper functions and path management for the tools package.
"""

import os
import re
from pathlib import Path
from typing import Optional

# Domains
DOMAINS = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']


def get_project_root() -> Path:
    """Get the project root directory (TaskDialogue/).
    
    Path structure:
    TaskDialogue/taskdialogue/benchmarks/multiwoz/tools/utils.py
                                                   ^- __file__
                                              ^- .parent (tools/)
                                         ^- .parent (multiwoz/)
                                    ^- .parent (benchmarks/)
                               ^- .parent (taskdialogue/)
                          ^- .parent (TaskDialogue/)  <- project root
    """
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_db_path() -> str:
    """Get the database path."""
    return str(get_data_dir() / "db" / "multiwoz.db")


def get_book_db_path() -> str:
    """Get the booking database path."""
    return str(get_data_dir() / "db" / "booking.db")


# For backward compatibility
DB_PATH = get_db_path()
BOOK_DB_PATH = get_book_db_path()
DATA_DIR = str(get_data_dir())


# ============================================================================
# Helper Classes and Functions
# ============================================================================

class TableItem:
    """Base class for database table items."""
    
    def as_dict(self):
        """Convert to dictionary."""
        result = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)
                if value is not None:
                    result[key] = value
        return result


def clean_time(time_str: str) -> str:
    """
    Clean and normalize time strings.
    
    Args:
        time_str: Time string (e.g., "13:45", "1:45pm")
        
    Returns:
        Normalized time string
    """
    if not time_str:
        return time_str
    
    # Remove spaces
    time_str = time_str.strip().lower()
    
    # Handle different formats
    if ':' not in time_str:
        return time_str
    
    # Ensure 2-digit format for hours
    parts = time_str.split(':')
    if len(parts[0]) == 1:
        time_str = '0' + time_str
    
    return time_str


# Tenacity retry logging
def tenacity_retry_log(retry_state):
    """Log retry attempts for tenacity."""
    print(f"⚠️  Retrying {retry_state.fn.__name__} "
          f"(attempt {retry_state.attempt_number})...")


# Backward compatibility flags (these will be removed in future versions)
USE_ZHIPUAI = os.getenv('USE_ZHIPUAI', 'false').lower() == 'true'
USE_DEEPSEEK = os.getenv('USE_DEEPSEEK', 'false').lower() == 'true'


def tenacity_retry_log(retry_state):
    """Callback for tenacity retry logging."""
    from termcolor import colored
    t = retry_state.next_action.sleep
    e = retry_state.outcome.exception()
    msg = f'Tenacity: Retrying call in {t:.2f} seconds as it raise {e.__class__.__name__}: '
    msg = colored(msg, 'red', force_color=True) + str(e)
    print(msg)


def prepare_goals_string(goals):
    """Prepare goals string from MultiWOZ format."""
    import re
    goals_str = str(goals)
    # Remove HTML tags
    goals_str = re.sub(r'<span\b[^>]*>(.*?)</span>', r'\1', goals_str)
    return goals_str
