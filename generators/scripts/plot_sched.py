#!/usr/bin/env python

import glob
import math
import random
from copy import *
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# print(plt.style.available)
# plt.style.use('seaborn-pastel')
import numpy as np
import pandas as pd
from gen_parse_data import def_budgets
from matplotlib import cm
from scipy.stats import gmean
from sympy import factorial
from sympy.functions.combinatorial import numbers

results_root = "./results"

SWEEP="composition"
SWEEP="IAT"
GM=1
TYPE="bar" # "bar"
FIG_TYPE="ut"
FONTSIZE=13
SKIP_OK=False
FILTER_POWER_BASED_ON_LAT=True
FILTER_IF_NOT_MET_BUDGET=False
COLOR_CODE_IF_NOT_MET_BUDGET=False
DRAW_BUDGET_LINE=True
PLOT_NORMALIZED_METRICS=0
TITLEPAD=0
LABELPAD=5
width=.25/1.25# *2
mpl.use('agg')
# plt.style.use("fivethirtyeight")
plt.style.use('seaborn-whitegrid')
plt.rcParams['lines.linewidth'] = .5
# plt.rcParams['lines.markersize'] = 5
# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.axis'] = 'x'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
# plt.rcParams['text.usetex'] = False
plt.rcParams['axes.edgecolor'] = "lightgray"
plt.rcParams['axes.linewidth'] = .5

font_path = 'fonts/JournalSans.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['figure.constrained_layout.use'] = True
# color = ['#00429d', '#b04351', '#73a2c6', '#fe908f', ] # 
color = ['#92c5de', '#437fb5', '#ca0020', '#d1e5f0', '#0571b0', ] # 
# color = ['#bf3a45', '#eab192']
# color = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    # return f"{10**val:<4,.0}"  # remove int() if you don't use MaxNLocator
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation

def get_tok_val(string, srch_tok):
    val = None
    string_split = '.'.join(string.split('.')[:-1]) # remove file extension
    string_split = string_split.split('/')[-1] # remove root dir
    string_split = string_split.split('_')
    for i, tok in enumerate(string_split):
        if tok == srch_tok:
            val = string_split[i+1]
            break
    return val

def filter_if_not_met_budget(filter_flag:bool, val:float, ideal_dict:Dict[str, float], metric:str):
    if not filter_flag:
        return val
    if metric == "lat" and val != ideal_dict[metric]:
        return 0.
    if PLOT_NORMALIZED_METRICS and val < ideal_dict[metric]:
        return 0.
    elif not PLOT_NORMALIZED_METRICS and val > ideal_dict[metric]:
        return 0.
    return val

if __name__ == "__main__":
    log_dir = {
        50: {
            True: { 
                "FARSI"     : f"{results_root}/__11.14.22_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_CONSTRAIN_TOPOLOGY_1_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_newBudgets/FARSI",
                "ARTEMIS"   : f"{results_root}/__11.16.22_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_CONSTRAIN_TOPOLOGY_1_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_newBudgets/RTEX",
            },
            False: {
                "EDF"           : f"{results_root}/11.20.22_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_newBudgets_edf/RTEX",
                "MS_STAT"       : f"{results_root}/11.20.22_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_newBudgets_ms_stat/RTEX",
                "MS_DYN"        : f"{results_root}/11.20.22_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_newBudgets_ms_dyn/RTEX",
                "MS_STAT_ENERGY": f"{results_root}/11.20.22_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_newBudgets_ms_stat_energy/RTEX"
            }
        }
    }
    # frame_all = ["Fixed-Hom", "Fixed-Het", "RTEX-Hom", "ARTEMIS"] #, "RTEX-Q"]
    # frame_all = ["Fixed-Het", "ARTEMIS"] #, "RTEX-Q"]
    frame_all = ["EDF", "MS_STAT", "MS_DYN", "MS_STAT_ENERGY"] #, "RTEX-Q"]
    prefix = {
        "wall_time": "exp",
        "soc": "sim_det",
    }
    if FIG_TYPE == "ct" or FIG_TYPE == "ut":
        IAT_MIN = .0095
    else:
        IAT_MIN = None
    soc_dim = (3,3)
    deadlines = [50] # [50, 200] # in ms
    # deadlines = [50, 100] # in ms
    if FIG_TYPE == "ct":
        constrained_top_all = [True]
    elif FIG_TYPE == "ut":
        constrained_top_all = [False]
    else:
        constrained_top_all = [False]
    if SWEEP == "composition":
        metric_all = ["lat", "pow"]
    elif SWEEP == "IAT":
        metric_all = ["wall_time"]
    else:
        raise NotImplementedError
    z_ideal = {}
    if PLOT_NORMALIZED_METRICS:
        zlabel = { "pow": "DAG Deadlines\nMet % per mW", "area": "DAG Deadlines\nMet % per mm$^2$" }
        z_ideal["pow"] = 100. / (def_budgets["miniera"]["power"]*1e3) # todo for other workloads
        z_ideal["area"] = 100. / (def_budgets["miniera"]["area"]*1e6) # todo for other workloads
    else:
        zlabel = { "lat": "DAG Deadlines\nMet %", "pow": "Power (mW)", "area": "Area (mm$^2$)" }
        z_ideal["pow"] = def_budgets["miniera"]["power"]*1e3 # todo for other workloads
        z_ideal["area"] = def_budgets["miniera"]["area"]*1e6 # todo for other workloads
    zlabel["lat"] = "DAG Deadlines\nMet %"
    zlabel["wall_time"] = "\nExplr. Time (s)"

    z_ideal["lat"] = 100. # def_budgets["miniera"]["latency"] # todo for other workloads

    for dline in deadlines:
        x = {}; z = {}
        for metric in metric_all:
            if metric not in x: x[metric] = {}
            if metric not in z: z[metric] = {}
            fig = plt.figure(figsize=(7,5))
            first_ax = None
            # plt.figure()
            axs = {}
            x_max, x_min = 0, np.inf
            z_max, z_min = 0, np.inf
            for j, constrained_top in enumerate(constrained_top_all):
                axs[constrained_top] = {}
                unique_y_vals = []
                for k, frame in enumerate(frame_all):
                    x[metric][frame] = {}; z[metric][frame] = {}
                    print(f"Parsing for {metric}, {dline}, {constrained_top}, {frame}")
                    path = f"{log_dir[dline][constrained_top][frame]}/results_dagInterArrTime_*.csv"
                    csv_files = sorted(glob.glob(path))
                    # print(csv_files)
                    assert csv_files, f"No CSV files found in path: {path}"
                    csv_files.reverse() # sort interarrival times in descending order
                    for csv_file in csv_files:
                        # print(f"Reading file: {csv_file}")
                        iat = float(get_tok_val(csv_file, "dagInterArrTime"))
                        if IAT_MIN != None and iat < IAT_MIN:
                            continue
                        dar = x_val = int(round(1/iat, 0)) # DAG arrival rate in Hz
                        # print(csv_file, fr)
                        df = pd.read_csv(csv_file)
                        skip = False
                        for col in ["Prefix", "Elapsed Time (s)"]:
                            if col not in df.columns:
                                print(f"Error: column {col} not found in file: {csv_file}, skipping...")
                                if not SKIP_OK:
                                    exit(1)
                                skip = True
                                break
                        if skip:
                            continue
                        
                        # # reverse
                        # df = df.iloc[::-1]
                        if metric == "wall_time":
                            df = df[df["Prefix"] == prefix["wall_time"]]
                        else:
                            df = df[df["Prefix"] == prefix["soc"]]

                        for index, row in df.iterrows(): # iterate over each Ncv,Nvit,Nrad tuple
                            # if row["Best Power (mW)"] > 1000.:
                            #     print("Filtered row: ", row)
                            #     continue
                            # if row["Best Area (mm^2)"] > 6.:
                            #     print("Filtered row: ", row)
                            #     continue
                            # plot_label = (int(row['Num Viterbis']), int(row['Num CVs']))
                            # plot_label = (int(row['Num CVs']), int(row['Num Radars']))
                            plot_label = (int(row['Num CVs']), int(row['Num Radars']), int(row['Num Viterbis']))
                            if plot_label not in x[metric][frame]:
                                x[metric][frame][plot_label] = []
                            if plot_label not in z[metric][frame]:
                                z[metric][frame][plot_label] = []

                            if metric == "lat":
                                z_val = row["Best DAG Deadline Meet %"]
                                z_val = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                            elif metric == "pow":
                                if PLOT_NORMALIZED_METRICS:
                                    z_val = row["Best DAG Deadline Meet %"] / (row["Best Power (mW)"])
                                else:
                                    z_val = row["Best Power (mW)"]
                                z_val = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                            elif metric == "area":
                                if PLOT_NORMALIZED_METRICS:
                                    z_val = row["Best DAG Deadline Meet %"] / (row["Best Area (mm^2)"])
                                else:
                                    z_val = row["Best Area (mm^2)"]
                                z_val = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                            elif metric == "wall_time":
                                z_val = math.log(row["Elapsed Time (s)"], 10)
                            else:
                                raise NotImplementedError
                            
                            y_val = int(row["Num Radars"])
                            # y_val = int(row["Num Viterbis"])
                            ylabel = "$N_{Rad}$" # "$N_{Vit}$"
                            if y_val not in unique_y_vals:
                                unique_y_vals.append(y_val)
                            
                            x[metric][frame][plot_label].append(x_val)
                            z[metric][frame][plot_label].append(z_val)
                                
        for metric in metric_all:
            if SWEEP == "IAT":
                num_plots = len(x[metric][frame].keys())
                # plot the subplot figures
                for i, plot_label in enumerate(x[metric][frame].keys()):
                    idx = num_plots*j+i+1
                    if first_ax == None:
                        if plot_label not in axs[constrained_top]:
                            axs[constrained_top][plot_label] = fig.add_subplot(len(constrained_top_all)*len(unique_y_vals), num_plots//len(unique_y_vals), idx)
                        first_ax = axs[constrained_top][plot_label]
                    else:
                        if plot_label not in axs[constrained_top]:
                            axs[constrained_top][plot_label] = fig.add_subplot(len(constrained_top_all)*len(unique_y_vals), num_plots//len(unique_y_vals), idx, sharey=first_ax)
                    ax = axs[constrained_top][plot_label]

                    for k, frame in enumerate(frame_all):
                        N = len(x[metric][frame][plot_label])
                        x_ind = np.arange(N+1)
                        xticklabs = [x_ if l%2==1 else "" for l, x_ in enumerate(x[metric][frame][plot_label])] + ["GM"]
                        xticklabs = x[metric][frame][plot_label] + ["GM"]
                        assert len(z[metric][frame][plot_label]) == N, f"{len(z[metric][frame][plot_label])} != {N}"
                        gm = gmean(z[metric][frame][plot_label])
                        print(metric, dline, plot_label, frame, gm)
                        col = color[k]
                        ax.bar(x_ind + k*width, z[metric][frame][plot_label] + [gm], width=width, color=col, alpha=.8, lw=1., edgecolor="black") # , color=colors[frame])
                    ax.set_xticks(x_ind + width*(len(frame_all)-1)/2, xticklabs)
                    if metric == "wall_time":
                        ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
                        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    else:
                        if DRAW_BUDGET_LINE:
                            ax.plot(np.arange(ax.get_xlim()[0], ax.get_xlim()[1]+1), [z_ideal[metric]] * len(np.arange(ax.get_xlim()[0], ax.get_xlim()[1]+1)), color="black", ls='--', lw=2)
                    ax.set_xlim()
                    if (idx-1) % (num_plots//len(unique_y_vals)) == 0: 
                        ax.set_ylabel(zlabel[metric], labelpad=LABELPAD, fontsize=FONTSIZE)
                    else:
                        # ax.set_yticklabels([]) # .set_ticks_position("none")
                        pass
                        ax.yaxis.set_tick_params(labelbottom=False)
                    if (idx-1) > (num_plots*len(constrained_top_all))-(num_plots//len(unique_y_vals))-1: 
                        ax.set_xlabel("DAG Arrival Rate (Hz)", labelpad=LABELPAD, fontsize=FONTSIZE)
                    top_str = None
                    # if constrained_top:
                    #     top_str = "[CT]" # "Constrained Topology"
                    # else:
                    #     top_str = "[UT]" # "Unconstrained Topology"
                    subplot_title = "$N_{CV}$,$N_{Rad}$,$N_{Vit}$:" + f"{plot_label[0]},{plot_label[1]},{plot_label[2]}"
                    if top_str:
                        subplot_title += f", {top_str}"
                    ax.set_title(subplot_title, fontsize=FONTSIZE)
                    handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper center')

                figpath = f"./outputs/sched_deadline_{dline}ms.{metric}.pdf"
                plt.savefig(figpath)
                print(f"Plot saved in {figpath}")
                plt.close()

            elif SWEEP == "composition":
                arate = 90
                fig = plt.figure(figsize=(7/2,5/2.5))
                ax = fig.add_subplot(1,1,1)
                N = len(x[metric][frame].keys())
                x_ind = np.arange(N) # +1)
                xticklabs = [x_ if l%2==1 else "" for l, x_ in enumerate(x[metric][frame][plot_label])] + ["GM"]
                xticklabs = list(x[metric][frame].keys()) # + ["GM"]
                metric_for_arate, hatch, annotations = {}, {}, {}
                for k, frame in enumerate(frame_all):
                    metric_for_arate[frame] = []
                    hatch[frame] = []
                    annotations[frame] = []
                    for i, plot_label in enumerate(x[metric][frame].keys()):
                        arate_idx = x[metric][frame][plot_label].index(arate)
                        metric_for_arate[frame].append(z[metric][frame][plot_label][arate_idx])
                        lat = z["lat"][frame][plot_label][arate_idx]
                        if lat != 100.:
                            hatch[frame].append('xxx')
                            annotations[frame].append(str(int(lat)))
                        else:
                            hatch[frame].append(None)
                            annotations[frame].append('')
                    print(frame, metric_for_arate[frame], hatch[frame])
                for k, frame in enumerate(frame_all):
                    # gm = gmean(metric_for_arate[frame])
                    # print(metric, dline, arate, frame, gm)
                    col = color[k]
                    hatches = hatch[frame]
                    print(len(x_ind), len(metric_for_arate[frame]), len(hatches))
                    for x_pt, ann in enumerate(annotations[frame]):
                        ax.text(x_ind[x_pt] + k*width, metric_for_arate[frame][x_pt]*1.05, ann, ha='center', size=9)
                    ax.bar(x_ind + k*width, metric_for_arate[frame], width=width, color=[col if h is None else "white" for h in hatches], alpha=.8, lw=1., edgecolor=["black" if h is None else col for h in hatches], hatch=hatches) # , color=colors[frame])
                    # ax.bar(x_ind + k*width, metric_for_arate[frame] + [gm], width=width, color=col, alpha=.8, lw=1., edgecolor=["black" if h is None else col for h in hatches], hatch=hatches) # , color=colors[frame])
                ax.set_xticks(x_ind + width*(len(frame_all)-1)/2, xticklabs, rotation=45, ha='right')
                ax.set_xlabel("ERA Workload Size ($N_{CV}$,$N_{Rad}$,$N_{Vit}$)")
                ax.set_ylabel(zlabel["pow"])
                # ax.set_ylim(bottom=200)
                figpath = f"./outputs/sched_arate_{arate}_deadline_{dline}ms.{metric}.pdf"
                plt.savefig(figpath, subplots_adjust="tight")
                print(f"Plot saved in {figpath}")
                plt.close()