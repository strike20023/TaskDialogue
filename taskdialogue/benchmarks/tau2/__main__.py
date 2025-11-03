#!/usr/bin/env python3
"""Command-line interface for Tau2-bench.

Usage:
    python -m taskdialogue.tau2_bench --config configs/tau2_default.yaml
    python -m taskdialogue.tau2_bench run --domain airline --num-tasks 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from taskdialogue.core.utils.config import load_config
from taskdialogue.benchmarks.tau2.cli import run_from_config
from taskdialogue.benchmarks.tau2.config import ConfigManager
from taskdialogue.core.utils.logger import get_logger



logger = get_logger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Tau2-bench: Task-Oriented Dialogue Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tau2_default.yaml",
        help="Configuration file path (default: configs/tau2_default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: auto-generated in results/tau2/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress configuration summary",
    )
    
    # Domain and task settings
    task_group = parser.add_argument_group("Task Settings")
    task_group.add_argument(
        "--domain",
        type=str,
        choices=["airline", "retail", "telecom"],
        help="Override domain (airline, retail, or telecom)",
    )
    task_group.add_argument(
        "--num-tasks",
        type=int,
        help="Override number of tasks to run",
    )
    task_group.add_argument(
        "--num-trials",
        type=int,
        help="Override number of trials per task",
    )
    
    # Limits
    limit_group = parser.add_argument_group("Limits")
    limit_group.add_argument(
        "--max-steps",
        type=int,
        help="Override maximum dialogue steps",
    )
    limit_group.add_argument(
        "--max-errors",
        type=int,
        help="Override maximum errors allowed",
    )
    limit_group.add_argument(
        "--seed",
        type=int,
        help="Override random seed",
    )
    
    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--provider",
        type=str,
        choices=["openai", "deepseek", "zhipuai", "vllm"],
        help="Override model provider",
    )
    model_group.add_argument(
        "--model",
        type=str,
        help="Override model name",
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature",
    )
    
    # Tools
    tools_group = parser.add_argument_group("Tools")
    tools_group.add_argument(
        "--tools-profile",
        type=str,
        choices=["original", "sql"],
        help="Override tools profile (original or sql)",
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"📂 Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Merge CLI arguments
        cli_args = {
            "domain": args.domain,
            "num_tasks": args.num_tasks,
            "num_trials": args.num_trials,
            "max_steps": args.max_steps,
            "max_errors": args.max_errors,
            "seed": args.seed,
            "tools_profile": args.tools_profile,
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
        }
        config = ConfigManager.merge_cli_args(config, cli_args)
        
        # Run benchmark
        logger.info("\n🚀 Starting Tau2-bench run...")
        results = run_from_config(config, verbose=not args.quiet)
        
        # Determine output file
        if args.output:
            output_path = Path(args.output)
        else:
            save_dir = Path("results/tau2")
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = config.get("tau2.domain", "airline")
            num_tasks = len(results.get("simulations", []))
            output_path = save_dir / f"tau2_{domain}_{num_tasks}tasks_{timestamp}.json"
        
        # Save results
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ Benchmark Complete!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Tasks completed:  {len(results.get('simulations', []))}")
        logger.info(f"Success rate:     {results.get('metrics', {}).get('success_rate', 0):.2%}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.info(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

