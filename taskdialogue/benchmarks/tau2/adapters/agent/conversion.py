"""Converters between τ² simulation results and TaskDialogue prediction format."""

from typing import Any, Dict, List


def convert_tau2_to_taskdialogue_predictions(tau2_results_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Tau2 simulation results to TaskDialogue prediction format.
    
    Note: This function name was changed from convert_tau2_to_autodia_predictions.
    For backward compatibility, the old name is still available as an alias.
    """
    predictions: List[Dict[str, Any]] = []
    for run in tau2_results_json.get("simulations", []):
        dialogue_id = f"{run.get('domain')}_{run.get('task_id')}_{run.get('trial', 0)}"
        turns: List[Dict[str, Any]] = []
        for m in run.get("trajectory", []):
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant"):
                turns.append({"role": role, "content": content})
        predictions.append({
            "dialogue_id": dialogue_id,
            "turns": turns,
            "goal": run.get("task", {}).get("goal", {}),
        })
    return predictions


# 向后兼容：保留旧的函数名
convert_tau2_to_autodia_predictions = convert_tau2_to_taskdialogue_predictions


