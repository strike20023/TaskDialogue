from pathlib import Path
import os


# Base data directory for τ² assets inside TaskDialogue repo  
# autodia/tau2_bench/core/utils.py -> 向上4层到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# 支持环境变量覆盖
DATA_DIR = Path(os.environ.get("TAU2_DATA_DIR", str(PROJECT_ROOT / "data")))


