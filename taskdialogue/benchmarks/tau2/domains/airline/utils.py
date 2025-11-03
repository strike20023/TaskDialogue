from pathlib import Path
import os

# 使用统一的 DATA_DIR（从 tau2_bench.utils.utils）
from taskdialogue.benchmarks.tau2.utils.utils import DATA_DIR

AIRLINE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "airline"
AIRLINE_DB_PATH = AIRLINE_DATA_DIR / "db.json"
AIRLINE_POLICY_PATH = AIRLINE_DATA_DIR / "policy.md"
AIRLINE_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks.json"


