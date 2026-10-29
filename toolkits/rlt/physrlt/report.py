# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Publish an evidence ledger without treating development screens as benchmarks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json


def read(path: Path, default):
    """Read an atomically published artifact, allowing not-yet-started jobs."""
    return json.loads(path.read_text()) if path.exists() else default


def table(rows: list[dict], columns: dict[str, str]) -> str:
    """Render a compact, escaped evidence table with explicit column names."""
    header = "".join(f"<th>{html.escape(value)}</th>" for value in columns.values())
    body = []
    for row in rows:
        cells = []
        for key in columns:
            value = row.get(key, "—")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def build(root: Path) -> dict:
    """Aggregate completed and partial artifacts while retaining invalidation."""
    jobs, probes, controls, collections, stress, paired = [], [], [], [], [], []
    for campaign in sorted(root.glob("pilot-v*")):
        plan = read(campaign / "plan.json", {})
        status = read(campaign / "status.json", {})
        invalidated = (campaign / "INVALIDATION.json").exists()
        for job in plan.get("jobs", []):
            state = status.get(job["id"], {"state": "not_started"})
            jobs.append(
                {
                    "campaign": campaign.name,
                    "job": job["id"],
                    "state": state["state"],
                    "attempts": state.get("attempts", 0),
                    "gpu": state.get("gpu"),
                    "exit_code": state.get("exit_code"),
                    "output": job["output"],
                }
            )
        for path in sorted(campaign.glob("task*/collection/result.json")):
            collections.append(
                {
                    "campaign": campaign.name,
                    "task": path.parents[1].name,
                    **read(path, {}),
                }
            )
        for path in sorted(campaign.glob("task*/probes/*/result.json")):
            result = read(path, {})
            horizon = result.get("horizons", {}).get("10")
            if not horizon:
                continue
            response = horizon["response"]["token_and_action"]
            geometry = horizon["geometry"]["token_and_action"]
            probes.append(
                {
                    "campaign": campaign.name,
                    "task": path.parents[2].name,
                    "method": result["method"],
                    "seed": result["seed"],
                    "steps": result["steps"],
                    "response_mse": response["development_scaled_mse"],
                    "geometry_mse": geometry["development_scaled_mse"],
                    "object_rmse_m": geometry["object_rmse_metres"],
                    "source": str(path),
                }
            )
        for path in sorted(campaign.glob("task*/control-seed*/episodes.json")):
            groups = defaultdict(list)
            manifest = read(path.parent / "run_manifest.json", {})
            for row in read(path, []):
                groups[
                    (
                        row["method"],
                        row["stage"],
                        row["split"],
                        row.get("execution_gain", 1.0),
                    )
                ].append(row)
            for (method, stage, split, gain), group in groups.items():
                valid = not (invalidated and stage.startswith("online_"))
                if stage.startswith("online_") and manifest.get(
                    "control_protocol_version"
                ) not in {2, 3}:
                    valid = False
                controls.append(
                    {
                        "campaign": campaign.name,
                        "task": path.parents[1].name,
                        "run": path.parent.name,
                        "method": method,
                        "stage": stage,
                        "split": split,
                        "gain": gain,
                        "successes": sum(bool(r["success"]) for r in group),
                        "episodes": len(group),
                        "unique_layouts": len({r["state_id"] for r in group}),
                        "mean_ticks_all": statistics.mean(r["ticks"] for r in group),
                        "mean_seconds_all": statistics.mean(
                            r["seconds"] for r in group
                        ),
                        "evidence": "development_only"
                        if valid
                        else "INVALIDATED_online_protocol",
                        "run_complete": (path.parent / "result.json").exists(),
                        "source": str(path),
                    }
                )
        for row in read(campaign / "reference_stress/episodes.json", []):
            stress.append(
                {
                    "campaign": campaign.name,
                    **{
                        key: row[key]
                        for key in (
                            "task",
                            "state_id",
                            "execution_gain",
                            "delay_ticks",
                            "success",
                            "ticks",
                        )
                    },
                }
            )
        for path in sorted(campaign.glob("task*/paired_actions/effects.json")):
            for row in read(path, []):
                paired.append(
                    {
                        "campaign": campaign.name,
                        "task": path.parents[1].name,
                        "complete": (path.parent / "result.json").exists(),
                        **row,
                    }
                )
    groups = defaultdict(list)
    for row in probes:
        groups[(row["campaign"], row["task"], row["method"])].append(row)
    predictions = []
    for (campaign, task, method), group in groups.items():
        predictions.append(
            {
                "campaign": campaign,
                "task": task,
                "method": method,
                "seeds": len({r["seed"] for r in group}),
                **{
                    key: statistics.mean(r[key] for r in group)
                    for key in ("response_mse", "geometry_mse", "object_rmse_m")
                },
            }
        )
    payload = {
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs": jobs,
        "collections": collections,
        "probe_fits": probes,
        "prediction_means": predictions,
        "control": controls,
        "stress": stress,
        "paired_action_effects": paired,
        "scope": "Development screens only; repeated layouts and invalidated online runs are not paper evidence.",
    }
    output = root / "overview"
    output.mkdir(exist_ok=True)
    write_json(output / "ledger.json", payload)
    for name, rows in (
        ("prediction_fits", probes),
        ("control", controls),
        ("jobs", jobs),
    ):
        if rows:
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    dev = [
        row
        for row in controls
        if row["split"] == "development" and row["evidence"] == "development_only"
    ]
    invalid = [row for row in controls if row["evidence"].startswith("INVALIDATED")]
    total_chunks = sum(row["chunks"] for row in collections)
    content = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>PhysRLT 开发实验记录</title>
<style>body{{max-width:1450px;margin:32px auto;padding:0 24px;font:16px/1.6 system-ui;color:#223047;background:#f5f7fb}}h1,h2{{color:#12345b}}.card{{background:white;padding:20px 28px;margin:22px 0;border-radius:12px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #dfe5ef;text-align:left}}th{{background:#eaf0f8}}.warning{{border-left:5px solid #cb8724}}code{{background:#edf1f6;padding:2px 5px}}</style>
<h1>PhysRLT · LIBERO-Pro 小规模开发记录</h1><p>更新于 {payload["updated"]}。此页面由原始结果生成，不填入预期成功率。</p>
<div class="card"><h2>研究主线与实现</h2><p>冻结 π₀.₅，从动作专家已有前向计算读取特征，用实际执行后的关节／末端响应，以及米制深度／物体位置变化监督小表征。物理经验沉淀在表征参数中，不加入历史记忆检索、agent 或外部语言模型。</p><p>当前采集 {total_chunks} 个 transition chunks，完成 {len(probes)} 次离线拟合；拟合次数包含重复的简单输入对照，不等于独立创新点。相同数据上的多 seed 也不是新的交互数据。</p><p>实验使用固定投影特征和 128 维小表征，<strong>不是原始 2048 维 RLT 的精确复现</strong>。所有深度／物体真值仅用于训练监督，部署不读取未来测量。几何与执行响应是互补监督，不是两个已证实的创新贡献。</p></div>
<div class="card warning"><h2>先读结论边界</h2><p>预测误差下降不等于控制收益。对照包含 prefix、本体＋动作、打乱标签和相同特征来源的重建模型。成功率、包含失败的平均步数和墙钟开销必须同时看；这里没有人工接管实验，不能报告接管减少。</p><p>首轮 critic 的 replay 与 actor 动作域不一致，相关在线结果全部作废，下面单独保留。协议 v2 已修正为同一 policy 提议动作域；实际接受命令另存用于物理监督。旧数据采集、离线预测及 BC 不受这一错误影响。</p><p>早期每任务只有 2 个开发初态；新增复核使用 6 个新开发初态，但均来自公开布局，不能当作独立论文测试集。命令增益／延迟是自定义软件执行扰动，不是修改摩擦、刚度或官方 LIBERO-Pro 成绩。</p></div>
<div class="card"><h2>1. 队列与每次尝试</h2>"""
    content += table(
        jobs,
        {
            "campaign": "轮次",
            "job": "任务",
            "state": "状态",
            "attempts": "启动次数",
            "gpu": "GPU",
            "exit_code": "退出码",
        },
    )
    content += '</div><div class="card"><h2>2. 冻结表征预测：10 个控制步后</h2><p>MSE 已按训练集尺度归一化；物体 RMSE 单位为米。各方法内按训练 seed 求均值，未合并不同任务。</p>'
    content += table(
        predictions,
        {
            "campaign": "轮次",
            "task": "任务",
            "method": "表征",
            "seeds": "Seed 数",
            "response_mse": "响应 MSE ↓",
            "geometry_mse": "几何 MSE ↓",
            "object_rmse_m": "物体 RMSE ↓",
        },
    )
    content += '</div><div class="card"><h2>3. 有效协议的开发闭环结果</h2><p>每行保留分母、初态数、训练 seed、条件和阶段，不合并重复 reference。online_frozen_eval 才是在线训练后冻结复评；warmup 不是在线 RL。未结束的 run 可能还缺后续方法。</p>'
    control_columns = {
        "campaign": "轮次",
        "task": "任务",
        "run": "运行",
        "method": "方法",
        "stage": "阶段",
        "gain": "增益",
        "successes": "成功",
        "episodes": "回合",
        "unique_layouts": "初态数",
        "mean_ticks_all": "平均步数（含失败）",
        "mean_seconds_all": "平均秒数",
        "run_complete": "运行结束",
    }
    content += table(dev, control_columns)
    content += '</div><div class="card"><h2>4. 训练初态上的执行扰动筛查</h2>'
    content += table(
        stress,
        {
            "task": "任务",
            "state_id": "训练初态",
            "execution_gain": "平移增益",
            "delay_ticks": "延迟步数",
            "success": "成功",
            "ticks": "控制步数",
        },
    )
    content += '</div><div class="card"><h2>5. 相同状态、不同动作的配对诊断</h2><p>直接预测器的动作效果误差应低于零效果对照。检查的是新动作在训练状态上的局部响应，不是泛化成功率；若误差更高，应作为物理解释不足的负证据。</p>'
    content += table(
        paired,
        {
            "task": "任务",
            "case_id": "初态",
            "start_tick": "执行前步数",
            "method": "方法",
            "horizon": "预测步长",
            "response_effect_scaled_mse": "响应效果 MSE ↓",
            "zero_effect_response_scaled_mse": "零效果 MSE",
            "object_effect_rmse_m": "物体效果 RMSE ↓",
            "zero_effect_object_rmse_m": "零效果 RMSE",
            "complete": "运行结束",
        },
    )
    content += '</div><div class="card warning"><h2>6. 作废结果与验收原则</h2><p>以下记录仅供审计，不可纳入论文或简历的在线 RL 成绩。</p>'
    content += table(
        invalid,
        {
            "campaign": "轮次",
            "task": "任务",
            "run": "运行",
            "method": "方法",
            "stage": "阶段",
            "split": "划分",
            "evidence": "作废原因",
        },
    )
    content += "<p>只有物理监督相对其匹配重建对照，在多个 seed／初态上重复改善成功率或交互成本，且没有增加部署真值输入，才进入后续规模实验。暂未达到时应保留否定结果，继续检查动作可辨识性、监督噪声和表征被控制器利用的方式。</p></div></html>"
    temp = output / "index.html.tmp"
    temp.write_text(content)
    temp.replace(output / "index.html")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        result = build(args.root)
        print(
            json.dumps(
                {
                    "updated": result["updated"],
                    "fits": len(result["probe_fits"]),
                    "jobs": len(result["jobs"]),
                }
            ),
            flush=True,
        )
        complete = result["jobs"] and all(
            row["state"] in {"complete", "failed", "blocked"} for row in result["jobs"]
        )
        if not args.watch or complete:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
