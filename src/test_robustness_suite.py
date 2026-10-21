"""Automated benchmark test suite across multiple seeds and jittered WADs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_engine_state_policy import evaluate_policy, load_policy
from project_paths import PROJECT_ROOT, REPORT_ROOT
from runtime_utils import resolve_device


def run_benchmark(
    checkpoint_path: Path,
    *,
    device: torch.device,
    output_report: Path,
    test_seeds: list[int] = (0, 1, 2, 3, 4, 5, 42, 100, 444, 999),
    jitter_configs: list[Path] | None = None,
    perturbation_tests: list[tuple[str, int]] = (("left", 8), ("right", 8), ("backward", 6)),
    episode_timeout: int = 8000,
) -> dict:
    model, payload = load_policy(checkpoint_path, device)
    model.eval()

    results = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "architecture": payload.get("architecture", "snn"),
            "epoch": payload.get("epoch", -1) + 1,
            "best_score": payload.get("best_score", 0.0),
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "standard_seeds": [],
        "jitter_runs": [],
        "perturbation_runs": [],
    }

    print("=" * 70, flush=True)
    print(f"BENCHMARKING CHECKPOINT: {checkpoint_path.name}", flush=True)
    print(f"Architecture: {payload.get('architecture')} | Epoch: {payload.get('epoch', -1) + 1}", flush=True)
    print("=" * 70, flush=True)

    route_data = PROJECT_ROOT / "data" / "demonstrations" / "map01"

    # 1. Standard Seeds Benchmark
    print("\n[Suite 1/3] Standard Seeds Benchmark (Default MAP01):", flush=True)
    for seed in test_seeds:
        res = evaluate_policy(
            model,
            device=device,
            route_data=route_data,
            seed=seed,
            render=False,
            render_delay=0.0,
            episode_timeout=episode_timeout,
            progress_interval=0,
            use_control_shield=False,
        )
        res_summary = {
            "seed": seed,
            "completed": res["completed"],
            "terminal": res["terminal_reason"],
            "route_progress": res["route_progress"],
            "kills": res["kills"],
            "shots_fired": res["shots_fired"],
            "attack_precision": res["attack_precision"],
            "ammo_remaining": res["ammo_remaining"],
            "stuck_events": res["stuck_events"],
            "steps": res["steps"],
        }
        results["standard_seeds"].append(res_summary)
        status_str = "PASS" if res["completed"] else f"FAIL ({res['terminal_reason']} at {res['route_progress']:.1%})"
        print(f"  Seed {seed:4d}: {status_str:30s} | Kills: {res['kills']:2d}/15 | Prec: {res['attack_precision']:.1%} | Shots: {res['shots_fired']:3d} | Steps: {res['steps']:3d}", flush=True)

    # 2. Jittered WAD Benchmark
    configs_to_test = jitter_configs or []
    if configs_to_test:
        print("\n[Suite 2/3] Random Jittered Enemy WAD Benchmark:", flush=True)
        for cfg in configs_to_test:
            if not cfg.exists():
                continue
            for seed in [0, 1, 2]:
                res = evaluate_policy(
                    model,
                    device=device,
                    route_data=route_data,
                    seed=seed,
                    config_file=cfg,
                    render=False,
                    render_delay=0.0,
                    episode_timeout=episode_timeout,
                    progress_interval=0,
                    use_control_shield=False,
                )
                res_summary = {
                    "config": cfg.name,
                    "seed": seed,
                    "completed": res["completed"],
                    "terminal": res["terminal_reason"],
                    "route_progress": res["route_progress"],
                    "kills": res["kills"],
                    "shots_fired": res["shots_fired"],
                    "attack_precision": res["attack_precision"],
                    "ammo_remaining": res["ammo_remaining"],
                    "stuck_events": res["stuck_events"],
                    "steps": res["steps"],
                }
                results["jitter_runs"].append(res_summary)
                status_str = "PASS" if res["completed"] else f"FAIL ({res['terminal_reason']} at {res['route_progress']:.1%})"
                print(f"  {cfg.name} (Seed {seed}): {status_str:25s} | Kills: {res['kills']:2d}/15 | Prec: {res['attack_precision']:.1%} | Shots: {res['shots_fired']:3d} | Steps: {res['steps']:3d}", flush=True)

    # 3. Altered-Start Perturbation Benchmark
    print("\n[Suite 3/3] Altered-Start Perturbation Benchmark:", flush=True)
    for pert_name, pert_steps in perturbation_tests:
        res = evaluate_policy(
            model,
            device=device,
            route_data=route_data,
            seed=0,
            start_perturbation=pert_name,
            start_perturbation_steps=pert_steps,
            render=False,
            render_delay=0.0,
            episode_timeout=episode_timeout,
            progress_interval=0,
            use_control_shield=False,
        )
        res_summary = {
            "perturbation": f"{pert_name} ({pert_steps} steps)",
            "completed": res["completed"],
            "terminal": res["terminal_reason"],
            "route_progress": res["route_progress"],
            "kills": res["kills"],
            "shots_fired": res["shots_fired"],
            "attack_precision": res["attack_precision"],
            "steps": res["steps"],
        }
        results["perturbation_runs"].append(res_summary)
        status_str = "PASS" if res["completed"] else f"FAIL ({res['terminal_reason']} at {res['route_progress']:.1%})"
        print(f"  Perturbation {pert_name:8s} ({pert_steps} st): {status_str:22s} | Kills: {res['kills']:2d}/15 | Prec: {res['attack_precision']:.1%} | Steps: {res['steps']:3d}", flush=True)

    # Calculate Aggregates
    std_pass_rate = np.mean([r["completed"] for r in results["standard_seeds"]])
    std_mean_route = np.mean([r["route_progress"] for r in results["standard_seeds"]])
    std_mean_kills = np.mean([r["kills"] for r in results["standard_seeds"]])

    jitter_pass_rate = np.mean([r["completed"] for r in results["jitter_runs"]]) if results["jitter_runs"] else 0.0
    jitter_mean_route = np.mean([r["route_progress"] for r in results["jitter_runs"]]) if results["jitter_runs"] else 0.0
    jitter_mean_kills = np.mean([r["kills"] for r in results["jitter_runs"]]) if results["jitter_runs"] else 0.0

    pert_pass_rate = np.mean([r["completed"] for r in results["perturbation_runs"]])

    results["summary"] = {
        "standard_completion_rate": float(std_pass_rate),
        "standard_mean_route": float(std_mean_route),
        "standard_mean_kills": float(std_mean_kills),
        "jitter_completion_rate": float(jitter_pass_rate),
        "jitter_mean_route": float(jitter_mean_route),
        "jitter_mean_kills": float(jitter_mean_kills),
        "perturbation_completion_rate": float(pert_pass_rate),
    }

    # Generate Markdown Report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"""# Autonomous Policy Robustness Benchmark Report

**Date**: {results['metadata']['timestamp']}  
**Checkpoint**: `{checkpoint_path.name}`  
**Architecture**: `{results['metadata']['architecture']}` (Epoch {results['metadata']['epoch']})  
**Evaluation Device**: `{device}`  

---

## Executive Summary

| Test Suite | Total Episodes | Passed (Completed) | Completion Rate | Mean Route Progress | Mean Kills |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Seeds** | {len(test_seeds)} | {sum(1 for r in results['standard_seeds'] if r['completed'])} | **{std_pass_rate:.1%}** | **{std_mean_route:.1%}** | **{std_mean_kills:.1f} / 15** |
| **Jittered Monster WADs** | {len(results['jitter_runs'])} | {sum(1 for r in results['jitter_runs'] if r['completed'])} | **{jitter_pass_rate:.1%}** | **{jitter_mean_route:.1%}** | **{jitter_mean_kills:.1f} / 15** |
| **Altered-Start Perturbations** | {len(perturbation_tests)} | {sum(1 for r in results['perturbation_runs'] if r['completed'])} | **{pert_pass_rate:.1%}** | - | - |

---

## 1. Standard Seeds Breakdown

| Seed | Status | Route Progress | Kills | Attack Precision | Shots Fired | Ammo Left | Steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results["standard_seeds"]:
        stat = "Completed" if r["completed"] else f"Died ({r['terminal']})"
        md_content += f"| {r['seed']} | {stat} | {r['route_progress']:.1%} | {r['kills']} | {r['attack_precision']:.1%} | {r['shots_fired']} | {r['ammo_remaining']:.0f} | {r['steps']} |\n"

    md_content += """
---

## 2. Jittered Enemy WADs Breakdown

| Scenario | Seed | Status | Route Progress | Kills | Attack Precision | Shots Fired | Ammo Left | Steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results["jitter_runs"]:
        stat = "Completed" if r["completed"] else f"Died ({r['terminal']})"
        md_content += f"| {r['config']} | {r['seed']} | {stat} | {r['route_progress']:.1%} | {r['kills']} | {r['attack_precision']:.1%} | {r['shots_fired']} | {r['ammo_remaining']:.0f} | {r['steps']} |\n"

    md_content += """
---

## 3. Altered-Start Perturbation Breakdown

| Perturbation | Status | Route Progress | Kills | Attack Precision | Shots Fired | Steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results["perturbation_runs"]:
        stat = "Completed" if r["completed"] else f"Died ({r['terminal']})"
        md_content += f"| {r['perturbation']} | {stat} | {r['route_progress']:.1%} | {r['kills']} | {r['attack_precision']:.1%} | {r['shots_fired']} | {r['steps']} |\n"

    output_report.write_text(md_content, encoding="utf-8")
    json_path = output_report.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved Markdown Report to: {output_report}", flush=True)
    print(f"Saved JSON Report to: {json_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full robustness benchmark suite.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report-file", type=Path, default=Path("reports/robustness_benchmark_report.md"))
    parser.add_argument("--episode-timeout", type=int, default=6000)
    args = parser.parse_args()

    jitter_cfgs = [
        Path("scenarios/generated/level01_jitter_s10.cfg"),
        Path("scenarios/generated/level01_jitter_s20.cfg"),
        Path("scenarios/generated/level01_jitter_s30.cfg"),
    ]

    run_benchmark(
        args.checkpoint.expanduser().resolve(),
        device=resolve_device(args.device),
        output_report=args.report_file.expanduser().resolve(),
        jitter_configs=jitter_cfgs,
        episode_timeout=args.episode_timeout,
    )


if __name__ == "__main__":
    main()
