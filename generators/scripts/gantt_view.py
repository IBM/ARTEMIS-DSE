# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
from helper import setup_logger

mpl.use('agg')
plt.style.use("seaborn-deep")
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 5
# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.axis'] = 'x'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.edgecolor'] = "lightgray"

width = .5

if not len(sys.argv) == 2:
    print(f"Usage: python3 gantt_view.py <path_to_SoC_phase_work_rate_file>")
    exit(1)

if __name__ == "__main__":
    logger = setup_logger('MyLogger')
    fig, ax1 = plt.subplots(1, figsize=(4,11))
    ax2 = ax1.twinx()
    labels = ["Heterogeneous"]
    colors = ["#fdae61"]
    files = sys.argv[1:]
    # print(files)
    for j, filepath in enumerate(files):
        df = pd.read_csv(filepath)
        # print(df)
        tasks = df["Task/Phase"].iloc[:len(df.index)-2].to_list()
        tasks.reverse()
        # print(tasks)
        phases = df.columns[1:-1].to_list()
        # print(phases)
        phase_lats = df[df["Task/Phase"] == "Cum. Phase Latency"].iloc[0].to_list()[1:-1]
        phase_lat_dict = {}
        c = 0
        for phase_id in phases:
            phase_lat_dict[phase_id] = phase_lats[c]
            c += 1
        # print(phase_lat_dict)

        start_times = {}
        end_times = {}
        for phase_id in phases:
            cols = df[phase_id]
            cur_tasks = df[df[phase_id].notnull()]["Task/Phase"].iloc[:-2].to_list()
            
            # print(cur_tasks)
            for task in cur_tasks:
                # print(task)
                if task not in start_times:
                    if task == "souurce":
                        start_times[task] = phase_lat_dict[str(int(phase_id))]
                    else:
                        start_times[task] = phase_lat_dict[str(int(phase_id)-1)]
                end_times[task] = phase_lat_dict[phase_id]

        task_starts, task_times = [], []
        for task in tasks:
            task_starts.append(start_times[task])
            task_times.append(end_times[task] - start_times[task])

        # index = range(len(tasks))
        is_to_remove = []
        for i in range(len(tasks)):
            if tasks[i] == "souurce":
                tasks[i] = "Entry"
            elif tasks[i] == "siink":
                tasks[i] = "Exit"
            else:
                if "DummyLast" in tasks[i]:
                    is_to_remove.append(i)
                else:
                    dag_id = int(tasks[i].split('_')[-1])
                    task_name = tasks[i].split('_')[0]
                    task_id_in_dag = int(tasks[i].split('_')[2])
                    tasks[i] = f"{task_name}[{dag_id},{task_id_in_dag}]"
        for i in sorted(is_to_remove, reverse=True):
            del tasks[i]
            del task_starts[i]
            del task_times[i]
        # print(tasks)
        x = np.arange(len(tasks))
        # print(labels[j])
        # print(task_starts)
        if j == 0:
            ax2.barh(x, task_times, width, left=task_starts, label=labels[j], edgecolor="black", color=colors[j], alpha=.9, zorder=1)
            ax2.set(yticks=x, yticklabels=tasks, ylim=[2*width - 1, len(tasks)])
        elif j == 1:
            ax1.barh(x, task_times, width, left=task_starts, label=labels[j], edgecolor="black", color=colors[j], alpha=.9)
            ax1.set(yticks=x, yticklabels=tasks, ylim=[2*width - 1, len(tasks)])
    ax1.set_xlabel("DAG Inter-Arrival Time (s)")
    # fig.legend()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    plt.grid(zorder=0, axis='y')
    # plt.show()
    outfile = sys.argv[1].split(".csv")[0] + ".pdf"
    plt.savefig(outfile, bbox_inches='tight')
    logger.info(f"Wrote to file: {outfile}")
