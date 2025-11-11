"""
TaskDialogue 统一命令行接口

提供统一的方式运行不同的 benchmark：
- multiwoz: 运行 MultiWOZ benchmark
- tau2: 运行 Tau2 benchmark
- serve: 启动 API 服务（未来功能）
"""

import click
from pathlib import Path

from taskdialogue.core.utils.config import load_config
from taskdialogue.core.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """TaskDialogue - 统一的多 Benchmark 对话系统框架
    
    支持的 benchmarks:
    - MultiWOZ: 多领域任务导向对话
    - Tau2: 客服场景对话评测
    
    使用 --help 查看每个命令的详细说明。
    """
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="配置文件路径 (YAML)"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["inference", "evaluation", "full"]),
    default="full",
    help="运行模式: inference(仅推理), evaluation(仅评估), full(完整流程)"
)
@click.option(
    "--num-tasks",
    "-n",
    type=int,
    default=None,
    help="限制任务数量（用于测试）"
)
@click.option(
    "--task-ids",
    type=str,
    default=None,
    help="指定任务ID列表（逗号分隔）"
)
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True),
    default=None,
    help="输入文件路径（evaluation模式下指定要评估的predictions.jsonl文件）"
)
@click.option(
    "--output-file",
    type=str,
    default=None,
    help="输出文件基础名（不含扩展名和路径）"
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="覆盖配置项 (格式: key=value，可多次使用)"
)
def multiwoz(config, mode, num_tasks, task_ids, input_file, output_file, override):
    """运行 MultiWOZ benchmark
    
    示例:
    
      # 运行完整流程
      taskdialogue multiwoz -c configs/multiwoz/default.yaml
      
      # 只运行推理
      taskdialogue multiwoz -c configs/multiwoz/default.yaml -m inference
      
      # 限制任务数量
      taskdialogue multiwoz -c configs/multiwoz/default.yaml -n 10
      
      # 指定任务ID
      taskdialogue multiwoz -c configs/multiwoz/default.yaml --task-ids MUL0001,MUL0002
    """
    from taskdialogue.benchmarks.multiwoz import MultiWOZPipeline
    
    # 加载配置
    cfg = load_config(config)
    setup_logging(cfg)
    
    # 覆盖配置
    if num_tasks:
        cfg.set("data.num_tasks", num_tasks)
    
    # 应用命令行覆盖
    if override:
        for item in override:
            if '=' not in item:
                logger.warning(f"忽略无效的覆盖参数: {item} (格式应为 key=value)")
                continue
            key, value = item.split('=', 1)
            # 尝试转换为正确的类型
            try:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('-', '', 1).replace('.', '', 1).isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
            except:
                pass
            
            cfg.set(key, value)
            logger.info(f"配置覆盖: {key} = {value}")
    
    logger.info("="*80)
    logger.info("TaskDialogue MultiWOZ Benchmark")
    logger.info("="*80)
    logger.info(f"Config: {config}")
    logger.info(f"Mode: {mode}")
    
    # 创建 pipeline
    pipeline = MultiWOZPipeline(cfg.to_dict())
    breakpoint()
    # 解析任务ID
    task_id_list = None
    if task_ids:
        task_id_list = [t.strip() for t in task_ids.split(",")]
    
    # 运行
    if mode == "inference":
        results = pipeline.run_inference(task_id_list)
        pipeline.save_inference_results(results)
        logger.info(f"✓ Inference complete: {len(results)} results saved")
        
    elif mode == "evaluation":
        # 加载已有推理结果
        if input_file:
            logger.info(f"Loading inference results from: {input_file}")
            inf_results = pipeline.load_inference_results([input_file])
        else:
            logger.info("Loading latest inference results...")
            inf_results = pipeline.load_inference_results()
        
        if not inf_results:
            logger.error("No inference results found. Run inference first!")
            return
        
        logger.info(f"Loaded {len(inf_results)} inference results")
        eval_results = pipeline.run_evaluation(inf_results)
        
        # 使用自定义输出文件名或默认名
        if output_file:
            pipeline.save_evaluation_results(eval_results, output_basename=output_file)
        else:
            pipeline.save_evaluation_results(eval_results)
        
        logger.info(f"✓ Evaluation complete: {len(eval_results)} results saved")
        
    else:  # full
        inf_results, eval_results = pipeline.run_full_pipeline(task_id_list)
        logger.info(f"✓ Pipeline complete: {len(inf_results)} dialogues evaluated")


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="配置文件路径 (YAML)"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["inference", "evaluation", "full"]),
    default="full",
    help="运行模式: inference(仅推理), evaluation(仅评估), full(完整流程)"
)
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["airline", "retail", "telecom"]),
    default=None,
    help="Tau2 领域 (当 all_domains=true 时可以不指定)"
)
@click.option(
    "--num-tasks",
    "-n",
    type=int,
    default=None,
    help="限制任务数量"
)
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True),
    default=None,
    help="输入文件路径（evaluation模式下指定要评估的predictions.json文件）"
)
@click.option(
    "--output-file",
    type=str,
    default=None,
    help="输出文件基础名（不含扩展名和路径）"
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="覆盖配置项 (格式: key=value，可多次使用)"
)
def tau2(config, mode, domain, num_tasks, input_file, output_file, override):
    """运行 Tau2 benchmark
    
    示例:
    
      # 运行完整流程
      taskdialogue tau2 -c configs/tau2/default.yaml -d airline
      
      # 只运行推理
      taskdialogue tau2 -c configs/tau2/default.yaml -d airline -m inference
      
      # 限制任务数量
      taskdialogue tau2 -c configs/tau2/default.yaml -d airline -n 10
      
      # 覆盖配置项
      taskdialogue tau2 -c configs/tau2/default.yaml -d airline -o inference.num_workers=16
    """
    from taskdialogue.benchmarks.tau2 import Tau2Pipeline
    
    # 加载配置
    cfg = load_config(config)
    setup_logging(cfg)
    
    # 覆盖配置
    all_domains = cfg.get("tau2.all_domains", False)
    if not all_domains and domain:
        cfg.set("tau2.domain", domain)
    if num_tasks:
        cfg.set("tau2.trials.num_tasks", num_tasks)
    
    # 应用命令行覆盖
    if override:
        for item in override:
            if '=' not in item:
                logger.warning(f"忽略无效的覆盖参数: {item} (格式应为 key=value)")
                continue
            key, value = item.split('=', 1)
            # 尝试转换为正确的类型
            try:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('-', '', 1).replace('.', '', 1).isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
            except:
                pass
            
            cfg.set(key, value)
            logger.info(f"配置覆盖: {key} = {value}")
    
    logger.info("="*80)
    logger.info("TaskDialogue Tau2 Benchmark")
    logger.info("="*80)
    logger.info(f"Config: {config}")
    logger.info(f"Mode: {mode}")
    all_domains = cfg.get("tau2.all_domains", False)
    if all_domains:
        logger.info(f"Domain: 全部 (airline, retail, telecom)")
    else:
        logger.info(f"Domain: {domain or cfg.get('tau2.domain', 'airline')}")
    
    # 创建 pipeline
    pipeline = Tau2Pipeline(cfg.to_dict())
    
    # 运行（不再支持task_ids，使用num_tasks限制任务数量）
    if mode == "inference":
        inf_results = pipeline.run_inference(task_ids=None)
        logger.info(f"✓ Inference complete: {len(inf_results)} results")
    elif mode == "evaluation":
        # 加载推理结果
        if input_file:
            # 从指定文件加载（使用 BasePipeline 的实现）
            logger.info(f"Loading inference results from: {input_file}")
            inf_results = pipeline.load_inference_results([input_file])
        else:
            # 加载最新结果
            logger.info("Loading latest inference results...")
            inf_results = pipeline.load_inference_results()
        
        if not inf_results:
            logger.error("No inference results found. Run inference first!")
            return
        
        logger.info(f"Loaded {len(inf_results)} inference results")
        eval_results = pipeline.run_evaluation(inf_results)
        logger.info(f"✓ Evaluation complete: {len(eval_results)} results")
    else:  # full
        inf_results, eval_results = pipeline.run_full_pipeline(task_ids=None)
        logger.info(f"✓ Pipeline complete: {len(inf_results)} simulations evaluated")


@cli.command()
def version():
    """显示版本信息"""
    click.echo("TaskDialogue Framework v2.0.0")
    click.echo("Multi-Benchmark Dialogue System")
    click.echo("")
    click.echo("Supported benchmarks:")
    click.echo("  - MultiWOZ (多领域任务导向对话)")
    click.echo("  - Tau2 (客服场景对话评测)")


@cli.command()
@click.option(
    "--benchmark",
    "-b",
    type=click.Choice(["multiwoz", "tau2"]),
    required=True,
    help="Benchmark 类型"
)
def init_config(benchmark):
    """生成示例配置文件
    
    示例:
    
      taskdialogue init-config -b multiwoz
      taskdialogue init-config -b tau2
    """
    from taskdialogue.cli.config_templates import get_template
    
    template = get_template(benchmark)
    
    # 保存到当前目录
    output_file = f"{benchmark}_config.yaml"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(template)
    
    logger.info(f"✓ Config template saved to: {output_file}")
    logger.info(f"  Edit the file and run: taskdialogue {benchmark} -c {output_file}")


if __name__ == "__main__":
    cli()

