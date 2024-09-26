#!/usr/bin/env python

import glob
import os
import math
import random
from copy import *
from pprint import pprint
from typing import Dict, List, Tuple
import sys
sys.path.append("../")
from gen_parse_data import power_area_dict
from helper import iprint, wprint

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
import math

results_root = "./results"
FILTER_POWER_BASED_ON_LAT=True
DAR_LIM=164
GM=1
TYPE="bar" # "bar"
USE_LOG_SCALE_FOR_WALL_TIME=True # False
WALL_TIME_YLIM=None # 2500
# SOC_TYPE="fixed_het" # "farsi"
# SOC_TYPE="farsi"
# TOP_TYPE="ut" # "ct"
FONTSIZE=13
SKIP_OK=False
FILTER_IF_NOT_MET_BUDGET=False
COLOR_CODE_IF_NOT_MET_BUDGET=False
DRAW_BUDGET_LINE=0
PLOT_NORMALIZED_METRICS=False # True
TITLEPAD=0
LABELPAD=5
mpl.use('agg')
# plt.style.use("fivethirtyeight")
plt.style.use('seaborn-whitegrid')
plt.rcParams['lines.linewidth'] = .5
# plt.rcParams['lines.markersize'] = 5
# plt.rcParams['axes.grid'] = False
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
# color = {"bar": ['#92c5de', '#437fb5', '#ca0020', '#d1e5f0', '#0571b0', ]} # 
# color = {"bar": ['#00429d', '#b04351', '#73a2c6', '#fe908f', ]} # 
# color = {"bar": ["#ef5675", "#ffa600"], "line": ["#003f5c", "#7a5195"]}

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

def plot_one(log_dir, run_type, plot_stacking, SOC_TYPE, TOP_TYPE, sched_pol, dag_iat_all, ncv_nrad_nvit_all, color):
    # frame_all = ["Fixed-Hom", "Fixed-Het", "RTEX-Hom", "ARTEMIS"] #, "RTEX-Q"]
    # frame_all = ["Fixed-Het", "ARTEMIS"] #, "RTEX-Q"]
    if SOC_TYPE == "fixed_het":
        frame_all = ["Fixed-Het", "ARTEMIS"] #, "RTEX-Q"]
        metric_all = ["lat", "pow", "area"]
        metric_all = ["lat_pow", "lat_area", "lat_wall_time"]
        width=.33# *2
        if plot_stacking == 'v':
            figsize=(3.,3)
        else:
            figsize=(6.,1.5)

    elif SOC_TYPE == "farsi":
        if TOP_TYPE == "ut":
            frame_all = ["FARSI-RR", "FARSI-DYN", "ARTEMIS"]
            width=.25
            if plot_stacking == 'v':
                figsize=(3.,3)
            else:
                figsize=(6.,1.5)
        else:
            frame_all = ["FARSI-RR", "FARSI-DYN", "ARTEMIS"]
            width=.225
            if plot_stacking == 'v':
                figsize=(3.,3)
            else:
                figsize=(6.,1.5)
        # metric_all = ["wall_time", "lat", "pow", "area"]
        metric_all = ["lat_wall_time", "lat_pow", "lat_area"]
    prefix = {
        "wall_time": "exp",
        "soc": "sim_det",
    }
    # if SOC_TYPE == "farsi" and (TOP_TYPE == "ct" or TOP_TYPE == "ut"):
    #     IAT_MIN = .0095
    # else:
    IAT_MIN = None
    soc_dim = (3,3)
    deadlines = [50] # [50, 200] # in ms
    # deadlines = [50, 100] # in ms
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
    zlabel["lat_pow"] = "DAG Deadlines\nMet %"
    zlabel["lat_area"] = "DAG Deadlines\nMet %"
    zlabel["lat_wall_time"] = "DAG Deadlines\nMet %"
    zlabel["wall_time"] = "\nExplr. Time (s)"

    z_ideal["lat"] = 100. # def_budgets["miniera"]["latency"] # todo for other workloads

    for metric in metric_all:
        if metric == "lat_pow": assert not PLOT_NORMALIZED_METRICS, "Unsupported"
        if metric == "pow" or metric == "lat_pow":
            marker = "v"
        elif metric == "lat":
            marker = "."
        else:
            marker = None
        for dline in deadlines:
            x = {}; y = {}; z = {}; z_twin = {}
            unique_y_vals = []
            for iat in dag_iat_all:
                wprint(f"DAG IAT: {iat}")
                if IAT_MIN != None and iat < IAT_MIN:
                    continue
                dar = x_val = int(round(1/iat, 0)) # DAG arrival rate in Hz
                if DAR_LIM is not None and dar > DAR_LIM: continue
                for ncv_nrad_nvit in ncv_nrad_nvit_all:
                    nc, nr, nv = ncv_nrad_nvit
                    wprint(f"\tN_CV, N_rad, N_vit: {ncv_nrad_nvit}")
                    for k, frame in enumerate(frame_all):
                        elaps_time_pref = "Best Exp"
                        if frame not in x: x[frame] = {}
                        if frame not in y: y[frame] = {}
                        if frame not in z: z[frame] = {}
                        if frame not in z_twin: z_twin[frame] = {}
                        if ncv_nrad_nvit not in x[frame]: x[frame][ncv_nrad_nvit] = []
                        if ncv_nrad_nvit not in y[frame]: y[frame][ncv_nrad_nvit] = []
                        if ncv_nrad_nvit not in z[frame]: z[frame][ncv_nrad_nvit] = []
                        if ncv_nrad_nvit not in z_twin[frame]: z_twin[frame][ncv_nrad_nvit] = []
                        print(f"Parsing for {metric}, {dline}, {TOP_TYPE}, {frame}")
                        wprint(f"\t\tFramework: {frame} ({sched_pol})")
                        root = log_dir[dline][TOP_TYPE][frame](sched_pol)
                        if TOP_TYPE == "ct" or frame == "Fixed-Het":
                            csv_file = f"{root}/{soc_dim[0]}_{soc_dim[1]}_{iat}_{nc}_{nr}_{nv}.results.csv"
                        elif TOP_TYPE == "ut":
                            csv_file = f"{root}/{iat}_{nc}_{nr}_{nv}.results.csv"
                        csv_file_full = glob.glob(csv_file)
                        if not len(csv_file_full) == 1: # , f"expected only 1 match for {csv_file}, but found {csv_file_full}"
                            x[frame][ncv_nrad_nvit].append(x_val)
                            z[frame][ncv_nrad_nvit].append(0.)
                            if metric in ["lat_wall_time", "lat_pow", "lat_area"]: # for dual y-axes
                                z_twin[frame][ncv_nrad_nvit].append((0., None))
                        else:
                            csv_file_full = csv_file_full[0]
                            assert os.path.exists(csv_file_full), f"Path not found: {csv_file_full}"
                            iprint(f"Reading file: {csv_file_full}")
                            # print(csv_file_full, fr)
                            df_orig = pd.read_csv(csv_file_full)
                            skip = False
                            for col in ["Prefix", f"{elaps_time_pref} Elapsed Time (s)"]:
                                if col not in df_orig.columns:
                                    print(f"Error: column {col} not found in file: {csv_file_full}, skipping...")
                                    print(df_orig.columns)
                                    if not SKIP_OK:
                                        exit(1)
                                    skip = True
                                    break
                            if skip:
                                continue
                            
                            if metric in ["wall_time"]:
                                df = df_orig[df_orig["Prefix"] == prefix["wall_time"]]
                            else:
                                df = df_orig[df_orig["Prefix"] == prefix["soc"]]

                            if metric == "lat":
                                z_val = df["Best DAG Deadline Meet %"].iat[0]
                                z_val = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                            elif metric == "pow":
                                if PLOT_NORMALIZED_METRICS:
                                    z_val = df["Best DAG Deadline Meet %"].iat[0] / (df["Best Power (mW)"].iat[0])
                                else:
                                    z_val = df["Best Power (mW)"].iat[0]
                                z_val = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                            elif metric == "area":
                                if PLOT_NORMALIZED_METRICS:
                                    z_val = df["Best DAG Deadline Meet %"].iat[0] / (df["Best Area (mm^2)"].iat[0])
                                else:
                                    z_val = df["Best Area (mm^2)"].iat[0]
                                z_val = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                            elif metric == "wall_time":
                                if USE_LOG_SCALE_FOR_WALL_TIME:
                                    z_val = math.log(df[f"{elaps_time_pref} Elapsed Time (s)"], 10)
                                else:
                                    z_val = df[f"{elaps_time_pref} Elapsed Time (s)"]
                            elif metric == "lat_pow": # for dual y-axes
                                z_val      = df["Best DAG Deadline Meet %"].iat[0]
                                z_val      = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                                missing_gpp_power = 0
                                z_val_twin = 1e-3 * df["Best Power (mW)"].iat[0] + missing_gpp_power
                                z_val_twin = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val_twin, z_ideal, "pow")
                                if FILTER_POWER_BASED_ON_LAT:
                                    if z_val != 100:
                                        # z_val_twin = np.nan
                                        hatch = 'xxx'
                                    else:
                                        hatch = None
                                    z_twin[frame][ncv_nrad_nvit].append((z_val_twin, hatch))
                                else:
                                    z_twin[frame][ncv_nrad_nvit].append(z_val_twin)
                            elif metric == "lat_wall_time": # for dual y-axes
                                z_val      = df_orig["Best DAG Deadline Meet %"].iat[0]
                                z_val      = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                                missing_gpp_power = 0
                                assert len(df_orig[df_orig["Prefix"] == prefix["wall_time"]][f"{elaps_time_pref} Elapsed Time (s)"].values) == 1
                                wall_time_val = df_orig[df_orig["Prefix"] == prefix["wall_time"]][f"{elaps_time_pref} Elapsed Time (s)"].values[0]
                                # print(wall_time_val)
                                if USE_LOG_SCALE_FOR_WALL_TIME:
                                    z_val_twin = math.log(wall_time_val, 10)
                                else:
                                    z_val_twin = wall_time_val
                                if FILTER_POWER_BASED_ON_LAT:
                                    if z_val != 100:
                                        # z_val_twin = np.nan
                                        hatch = 'xxx'
                                    else:
                                        hatch = None
                                    z_twin[frame][ncv_nrad_nvit].append((z_val_twin, hatch))
                                else:
                                    z_twin[frame][ncv_nrad_nvit].append(z_val_twin)
                            elif metric == "lat_area": # for dual y-axes
                                z_val      = df["Best DAG Deadline Meet %"].iat[0]
                                z_val      = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val, z_ideal, metric)
                                missing_gpp_area = 0
                                z_val_twin = df["Best Area (mm^2)"].iat[0] + missing_gpp_area
                                z_val_twin = filter_if_not_met_budget(FILTER_IF_NOT_MET_BUDGET, z_val_twin, z_ideal, "area")
                                if FILTER_POWER_BASED_ON_LAT:
                                    if z_val != 100:
                                        # z_val_twin = np.nan
                                        hatch = 'xxx'
                                    else:
                                        hatch = None
                                    z_twin[frame][ncv_nrad_nvit].append((z_val_twin, hatch))
                                else:
                                    z_twin[frame][ncv_nrad_nvit].append(z_val_twin)
                            else:
                                raise NotImplementedError
                            
                            y_val = nr
                            # y_val = int(row["Num Viterbis"])
                            ylabel = "$N_{Rad}$" # "$N_{Vit}$"
                            if y_val not in unique_y_vals:
                                unique_y_vals.append(y_val)
                            
                            x[frame][ncv_nrad_nvit].append(x_val)
                            z[frame][ncv_nrad_nvit].append(z_val)
                        
            pprint(x)
            pprint(z)
            # plot the subplot figures
            axs = {}
            num_subplots = len(ncv_nrad_nvit_all)
            if plot_stacking == "h":
                fig, axs = plt.subplots(1, num_subplots, sharey=True, figsize=figsize)
            else:
                fig, axs = plt.subplots(num_subplots, 1, sharex=False, figsize=figsize)
            for i, ncv_nrad_nvit in enumerate(ncv_nrad_nvit_all):
                ax = axs[i]
                top_str = None
                subplot_title = "$N_{CV}$,$N_{Rad}$,$N_{Vit}$:" + f"{ncv_nrad_nvit[0]},{ncv_nrad_nvit[1]},{int(ncv_nrad_nvit[2])}"
                if top_str:
                    subplot_title += f", {top_str}"
                ax.set_title(subplot_title, fontsize=FONTSIZE) # , bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', pad=.1), alpha=1)
                for k, frame in enumerate(frame_all):
                    num_xpoints = len(x[frame][ncv_nrad_nvit])
                    # xticklabs = [x_ if l%2==1 else "" for l, x_ in enumerate(x[frame][ncv_nrad_nvit])]
                    xticklabs = x[frame][ncv_nrad_nvit] # all DAG IATs (1 over)
                    assert len(z[frame][ncv_nrad_nvit]) == num_xpoints, f"{len(z[frame][ncv_nrad_nvit])} != {num_xpoints}"
                    plot_y = z[frame][ncv_nrad_nvit]
                    if metric in ["lat_wall_time", "lat_pow", "lat_area"]:
                        gm = None
                        x_ind = np.arange(num_xpoints)
                        if FILTER_POWER_BASED_ON_LAT:
                            plot_y_twin, hatches = list(map(list, zip(*z_twin[frame][ncv_nrad_nvit])))
                            if GM and metric == "lat_wall_time":
                                print(f"plot_y_twin: {plot_y_twin}")
                                print([v_ for id, v_ in enumerate(plot_y_twin) if v_ != 0.0 and hatches[id] == None])
                                gm = gmean([v_ for id, v_ in enumerate(plot_y_twin) if v_ != 0.0 and hatches[id] == None])
                                plot_y_twin += [gm]
                                xticklabs += ["GM"]
                                hatches.append(None)
                                x_ind = np.arange(num_xpoints+1)
                        else:
                            plot_y_twin = z_twin[frame][ncv_nrad_nvit]
                    else:
                        if GM:
                            # if any(v_ is None for v_ in z[frame][ncv_nrad_nvit]):
                            #     gm = 0.
                            # else:
                            gm = gmean([v_ for v_ in z[frame][ncv_nrad_nvit] if v_ != 0.0])
                            plot_y += [gm]
                            xticklabs += ["GM"]
                            x_ind = np.arange(num_xpoints+1)
                        else:
                            gm = None
                            x_ind = np.arange(num_xpoints)
                    if metric in ["lat_wall_time", "lat_pow", "lat_area"]:
                        iprint("metric:{}, dline:{}, ncv_nrad_nvit:{}, frame:{}, plot_y_twin[-1]:{}, gm:{}".format(metric, dline, ncv_nrad_nvit, frame, plot_y_twin[-1], gm))
                    else:
                        iprint("metric:{}, dline:{}, ncv_nrad_nvit:{}, frame:{}, gm:{}".format(metric, dline, ncv_nrad_nvit, frame, gm))
                    if TYPE == "bar": # bar for lat, line for pow
                        if metric in ["lat_wall_time", "lat_pow", "lat_area"]:
                            if not FILTER_POWER_BASED_ON_LAT:
                                ax2 = ax.twinx()
                                ax2.grid(zorder=0)
                                ax2.bar(x_ind + k*width, plot_y_twin, width=width, color=color[TYPE][k], alpha=.8, lw=1., edgecolor="black", label=frame) # , color=colors[frame])
                                ax.plot(x_ind, plot_y, color=color["line"][k], lw=1.5, marker=marker, label=frame) # , color=colors[frame])
                                ax.grid(zorder=0)
                                ax.set_zorder(ax2.get_zorder() + 1)
                                ax.patch.set_visible(False)
                            else:
                                # print(len(x_ind))
                                # print(len(plot_y_twin))
                                # print(len(hatches))
                                ax.bar(x_ind + k*width, plot_y_twin, width=width, color=[color[TYPE][k] if h is None else "white" for h in hatches], alpha=.8, lw=1., edgecolor=["black" if h is None else color[TYPE][k] for h in hatches], label=frame, hatch=hatches) # , color=colors[frame])
                                if metric != "lat_wall_time":
                                    annotations = [str(int(plot_y[k])) if h is not None else '' for k, h in enumerate(hatches)]
                                    for x_pt, ann in enumerate(annotations):
                                        ax.text(x_ind[x_pt] + k*width, plot_y_twin[x_pt]*1.05, ann, ha='center', size=8) # , str(v), color='blue', fontweight='bold')
                                # else:
                                #     ax.set_ylim((None, WALL_TIME_YLIM))

                        else:
                            assert len(color[TYPE]) >= len(frame_all), f"Need at least {frame_all} colors"
                            ax.bar(x_ind + k*width, plot_y, width=width, color=color[TYPE][k], alpha=.8, lw=1., edgecolor="black") # , color=colors[frame])
                    elif TYPE == "line":
                        ax.plot(x_ind, plot_y, color=color[TYPE][k], lw=2, marker=marker) # , color=colors[frame])
                    else:
                        raise NotImplementedError
                ax.set_xticks(x_ind + width*(len(frame_all)-1)/2, xticklabs)
                if USE_LOG_SCALE_FOR_WALL_TIME:
                    if metric in ["lat_wall_time", "wall_time"]:
                        ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
                        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax.set_xlim()
                # if (idx-1) % (num_subplots//len(unique_y_vals)) == 0 and not (idx == 2 or idx ==4): 
                if metric == "lat_pow":
                    labl = "Power ($W$)"
                elif metric == "lat_area":
                    labl = "Area ($mm^2$)"
                elif metric == "lat_wall_time":
                    labl = "Explr. Time ($s$)"
                if metric in ["lat_wall_time", "lat_pow", "lat_area"]:
                    if FILTER_POWER_BASED_ON_LAT:
                        if plot_stacking == "v" or i == 0:
                            ax.set_ylabel(labl, labelpad=LABELPAD, fontsize=FONTSIZE)
                    else:
                        ax2.set_ylabel(labl, labelpad=LABELPAD, fontsize=FONTSIZE)
                        ax.set_ylabel(zlabel[metric], labelpad=LABELPAD, fontsize=FONTSIZE)
                else:
                    ax.set_ylabel(zlabel[metric], labelpad=LABELPAD, fontsize=FONTSIZE)
                # else:
                #     ax.yaxis.set_tick_params(labelbottom=False)
                #     if metric == "lat_pow" and not FILTER_POWER_BASED_ON_LAT: ax2.yaxis.set_tick_params(labelbottom=False)
                # if (idx-1) > num_subplots-(num_subplots//len(unique_y_vals))-2: 
                if plot_stacking == "h" or i == num_subplots-1:
                    ax.set_xlabel("DAG Arrival Rate (Hz)", labelpad=LABELPAD, fontsize=FONTSIZE)

                handles, labels = ax.get_legend_handles_labels()

            # plt.legend(handles, labels, loc='upper center')
            os.makedirs("./outputs", exist_ok=True)
            figpath = f"./outputs/{run_type}_deadline_{dline}ms_{SOC_TYPE}_{TOP_TYPE}.{metric}.pdf"
            plt.savefig(figpath)
            print(f"Plot saved in {figpath}")
            plt.close()

if __name__ == "__main__":
    pol_sel = {
        (1,2,1.0): "ms_dyn_energy",
        (1,4,4.0): "ms_dyn_energy",
    }
    log_dir = {
        "real": {
            50: {
                "ct": { # this is with agg budgets
                    "ARTEMIS"     : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_1_mesh_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/ARTEMIS",
                    "FARSI-RR"    : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_1_mesh_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/FARSI-RR",
                    "FARSI-DYN"   : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_1_mesh_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/FARSI-DYN",

                },
                "ut": { 
                    "ARTEMIS"     : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._*_DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/ARTEMIS",
                    "Fixed-Het"   : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_1_bus_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/FIXED_HET",
                    "FARSI-RR"    : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._*_DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/FARSI-RR",
                    "FARSI-DYN"   : lambda pol: f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._*_DEADLINE_0.05_LAT_AMP_NO_REMAP_{pol}/FARSI-DYN",
                },
            },
        },
    }
    dag_iat_all = [0.067, 0.044, 0.035, 0.027, 0.022, 0.019, 0.017, 0.015, 0.013, 0.012, 0.011, 0.01, 0.009]
    ncv_nrad_nvit_all = [(1, 2, 1.0), (1, 4, 4.0)]
    sched_pol = "ms_dyn_energy"
    plot_one(log_dir["real"], "real", 'h', SOC_TYPE="farsi",     TOP_TYPE="ct", sched_pol=sched_pol, dag_iat_all=[0.067, 0.044, 0.035, 0.027, 0.022, 0.019, 0.017, 0.015], ncv_nrad_nvit_all=ncv_nrad_nvit_all, color={"bar": ['#abd9e9', '#d7191c', '#fdae61', '#2c7bb6'], "line": ['#d7191c', '#fdae61', '#2c7bb6']})
    plot_one(log_dir["real"], "real", 'h', SOC_TYPE="farsi",     TOP_TYPE="ut", sched_pol=sched_pol, dag_iat_all=[0.067, 0.044, 0.035, 0.027, 0.022, 0.019, 0.017, 0.015], ncv_nrad_nvit_all=ncv_nrad_nvit_all, color={"bar": ['#d7191c', '#fdae61', '#2c7bb6'], "line": ['#d7191c', '#fdae61', '#2c7bb6']})
    plot_one(log_dir["real"], "real", 'h', SOC_TYPE="fixed_het", TOP_TYPE="ut", sched_pol=sched_pol, dag_iat_all=[0.067, 0.044, 0.035, 0.027, 0.022, 0.019, 0.017, 0.015], ncv_nrad_nvit_all=ncv_nrad_nvit_all, color = {"bar": ["#d7191c", "#2c7bb6"], "line": ["#d7191c", "#2c7bb6"]})
    # plot_one(log_dir, SOC_TYPE="farsi", color={"bar": ['#004c6d', '#9dc6e0'], "line": ['#004c6d', '#9dc6e0']})