#!/usr/bin/env python3

import glob
import os
import math
from pprint import pprint

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from helper import setup_logger

from typing import Dict, List

mpl.use('agg')
plt.style.use("seaborn-deep")
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.axis'] = 'x'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.edgecolor'] = "lightgray"

width = .5
results_root = "./results"

ndags=50
DLINE=0.021

def get_val(string:str, substr:str):
    string = string.split('_')
    for i, tok in enumerate(string):
        if tok == substr:
            return string[i+1]
    print(string, substr)
    raise ValueError

if __name__ == "__main__":
    logger = setup_logger('MyLogger')
    all_dag_deadline_met_perc = []; first = True
    path = f"{results_root}/04.27.23_PROB/EXPLORE_MODE_staggered_USE_DYN_NDAGS_0_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._0.5_0.5_DEADLINE_iat_NEW_LAT_AMP/ARTEMIS/sim_prob*"
    dirs = glob.glob(path)
    assert dirs, str(path)
    all_dirs = [dir_ for dir_ in dirs if os.path.isdir(dir_)]

    n_bins = int(math.sqrt(100*len(all_dirs)))
    # n_bins = 10
    iat_met_deadlines = []
    iat_total = []
    # bin_width = (max_dag_iat - min_dag_iat)/n_bins
    for i in range(n_bins):
        iat_met_deadlines.append(0)
        iat_total.append(0)
    lats:Dict[int, Dict[float, List[float]]] = {}
    deadlines_met_perc:Dict[int, Dict[float, float]] = {}
    IAT_CAP = .010
    for dir_ in all_dirs:
        dag_iat = float(get_val(dir_, "iataudio"))
        if dag_iat > IAT_CAP:
            continue
        n_traces = len(glob.glob(f"{dir_}/*"))
        for trace_id in range(n_traces):
            if trace_id not in deadlines_met_perc.keys():
                deadlines_met_perc[trace_id] = {}
            if trace_id not in lats.keys():
                lats[trace_id] = {}
            run_log_fname = f"{dir_}/final_{trace_id+1}/run.log"
            lats[trace_id][dag_iat], deadlines_met_perc[trace_id][dag_iat] = [], None
            print(f"Reading file: {run_log_fname}")
            with open(run_log_fname) as f:
                data = f.readlines()
                for line in reversed(data):
                    if line.startswith("@@ "):
                        line = line.rstrip('\n').split('[')[1].split(']')[0].split(' ')
                        lats[trace_id][dag_iat] = [float(el.rstrip(',')) for el in line]
                        assert ndags == len(lats[trace_id][dag_iat])
                        lats[trace_id][dag_iat] = [-1. if (item-DLINE) > 0. else item for item in lats[trace_id][dag_iat]]
                        deadlines_met_perc[trace_id][dag_iat] = 100. * (ndags - lats[trace_id][dag_iat].count(-1.)) / ndags
                        assert len(lats[trace_id][dag_iat]) == ndags, f"Found only {len(lats[trace_id][dag_iat])} DAGs' results, expected {ndags}"
                        for dag_id, lat in enumerate(lats[trace_id][dag_iat]):
                            if dag_id == 0:
                                continue
                            # print(lat, arr_times[dag_id], arr_times[dag_id-1])
                            # exit(1)
                            # iat = (arr_times[dag_id] - arr_times[dag_id-1])*1e3 # interarrival time in ms
                            # assert iat <= max_dag_iat, f"found DAG with iat={iat}, increase max_dag_iat and try again"
                            # iat_bin_id = int(iat/bin_width)
                            # # print(iat, bin_width, n_bins, iat_bin_id)
                            # # iat_total_bins[]
                            # iat_total[iat_bin_id] += 1
                            # if lat != -1.: # deadline was met
                            #     iat_met_deadlines[iat_bin_id] += 1
                        # lats[trace_id][dag_iat] = [np.nan if item == -1. else item for item in lats[trace_id][dag_iat]]
                        break
            if not lats[trace_id][dag_iat]:
                del lats[trace_id][dag_iat]
    pprint(deadlines_met_perc)
    sorted_dag_iats = sorted(deadlines_met_perc[0].keys())
    sorted_dag_rates = [int(round(1/el, 0)) for el in sorted_dag_iats]
    for i, dag_iat in enumerate(sorted_dag_iats): # across all iats
        if first:
            all_dag_deadline_met_perc.append([])
        for trace_id in range(n_traces): # across all iats
            if deadlines_met_perc[trace_id][dag_iat] != np.nan:
                all_dag_deadline_met_perc[i].append(deadlines_met_perc[trace_id][dag_iat])
    first = False
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(2, 1.75))
    axs.boxplot(list(reversed(all_dag_deadline_met_perc)), widths=0.5) # , showmeans=True)
    axs.set_xticklabels(list(reversed(sorted_dag_rates)), rotation=45, ha='right')
    axs.set_xlabel("DAG Arrival Rate (Hz)")
    axs.set_ylabel("DAG Deadlines Met %")
    axs.set_ylim((0, None))
    plt.savefig("./outputs/audio_dec.deadlines_met_boxplot.pdf", bbox_inches='tight')
    logger.info(f"Saved file: ./outputs/audio_dec.deadlines_met_boxplot.pdf")
    # print(f"Deadlines met % for {ncv}{nrad}{int(nvit)}")
    # pprint(deadlines_met_perc)
    # iat_total = np.array(iat_total, dtype=np.float)
    # iat_met_deadlines = np.array(iat_met_deadlines, dtype=np.float)
    # iat_met_perc = 100.*np.divide(iat_met_deadlines, iat_total)
    # # construct histogram of DAG interarrival times and % of DAGs for which deadlines were met
    # x = np.arange(min_dag_iat, max_dag_iat, bin_width, dtype=np.float).tolist()
    # df_dict = {'DAG Inter-Arrival Time (s)': [], '% DAGs that meet deadlines': []}
    # for i in range(len(x)):
    #     df_dict['DAG Inter-Arrival Time (s)'].append(f"({int(x[i])}-{int(x[i]+bin_width)})")
    #     df_dict['% DAGs that meet deadlines'].append(iat_met_perc[i])
    # df = pd.DataFrame.from_dict(df_dict)
    # outfname = f"{root}/results_histogram_{ncv}{nrad}{int(nvit)}.csv"
    # df.to_csv(outfname, index=False)
    # print(f"Wrote to file: {outfname}")

    # # construct heatmap of DAGs and their latencies for each run along the columns
    # for trace_id in range(n_traces):
    #     plt.figure()
    #     df = pd.DataFrame.from_dict(lats[trace_id])
    #     df = df.reindex(sorted(df.columns), axis=1)
    #     plot = sns.heatmap(df, annot=False)
    #     # df.style.background_gradient(cmap='Blues')
    #     fig = plot.get_figure()
    #     plt.tight_layout()
    #     outfname = f"{root}/heatmap_ncv_{ncv}_nrad_{nrad}_nvit_{nvit}_trace_{trace_id}.pdf"
    #     fig.savefig(outfname) 
    #     print(f"Wrote to file: {outfname}")