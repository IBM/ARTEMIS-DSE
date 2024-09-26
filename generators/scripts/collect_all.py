# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd

sys.path.append("../")
from helper import setup_logger

frac_for_report = .7 # fraction of n_exp that need to finish before the report for it is generated
skip_ok = False # allow to skip experiments that haven't started
# prefix_all = ["exp", "sim_det", "sim_prob"]

def get_matching_line_in_file(fname:str, string:str):
    try:
        with open(fname, 'r') as f:
            for line in f.readlines():
                if string in line:
                    return line.rstrip('\n')
    except:
        logger.error(f"Error in file {fname}")
        exit(1)
    return None

def get_stats_for_workloads(run_root:str, skipped_exps_ok=False):
    # initialize collection
    elapsed_time = {}; latency = {}; latency_budget = {}; power = {}; area = {};  power_budget = {}; area_budget = {}
    design_composition = {}
    ip_count = {}
    gpp_count = {}
    ic_count = {}
    mem_count = {}
    # iterate over all experiments run
    scale_tup_all = set()
    exps_run = []
    exp_dirs = glob.glob(f"{run_root}/final_*")
    assert exp_dirs, f"No paths found in {run_root}/final_*"
    for exp_dir in exp_dirs:
        exp_id = int(exp_dir.split("final_")[-1])
        # initialize data collection
        exp_dir = f"{run_root}/final_{exp_id}/*__*/*"
        # iterate over different budgetted runs
        paths = glob.glob(exp_dir)
        # print(exp_dir, paths)
        if not paths:
            assert skipped_exps_ok, f"No paths found: {exp_dir}"
            continue
        for dir_ in sorted(paths):
            res_filename = f"{dir_}/runs/0/FARSI_simple_run*.csv"
            res_filenames = glob.glob(res_filename)
            if not res_filenames:
                assert skipped_exps_ok, f"{res_filenames} {res_filename}"
                continue
            else:
                assert len(res_filenames) == 1, f"{res_filenames} {res_filename}"
            res_filename = res_filenames[0]

            exps_run.append(exp_id)
            df = pd.read_csv(res_filename)
            lat_budget_scale  = df.budget_scaling_latency.values[0]
            pow_budget_scale  = df.budget_scaling_power.values[0]
            area_budget_scale = df.budget_scaling_area.values[0]
            scale_tup = (lat_budget_scale, pow_budget_scale, area_budget_scale)
            scale_tup_all.add(scale_tup)
            latency_, latency_budget_ = df.latency.values[0], df.latency_budget.values[0]
            if scale_tup not in latency: latency[scale_tup] = {}
            if exp_id not in latency[scale_tup]: latency[scale_tup][exp_id] = {}
            if scale_tup not in latency_budget: latency_budget[scale_tup] = {}
            if exp_id not in latency_budget[scale_tup]: latency_budget[scale_tup][exp_id] = {}
            for tok in latency_budget_.split(';')[:-1]:
                latency_budget[scale_tup][exp_id][tok.split('=')[0]] = float(tok.split('=')[1])
            for tok in latency_.split(';')[:-1]:
                latency[scale_tup][exp_id][tok.split('=')[0]] = float(tok.split('=')[1])
            if scale_tup not in power: power[scale_tup] = {}
            if exp_id not in power[scale_tup]: power[scale_tup][exp_id] = float(df.power.values[0])
            if scale_tup not in power_budget: power_budget[scale_tup] = {}
            if exp_id not in power_budget[scale_tup]: power_budget[scale_tup][exp_id] = float(df.power_budget.values[0])
            if scale_tup not in area: area[scale_tup] = {}
            if exp_id not in area[scale_tup]: area[scale_tup][exp_id] = float(df.area.values[0])
            if scale_tup not in area_budget: area_budget[scale_tup] = {}
            if exp_id not in area_budget[scale_tup]: area_budget[scale_tup][exp_id] = float(df.area_budget.values[0])
            if scale_tup not in elapsed_time: elapsed_time[scale_tup] = {}
            if exp_id not in elapsed_time[scale_tup]: elapsed_time[scale_tup][exp_id] = df.elapsed_time.values[0]
            if scale_tup not in design_composition: design_composition[scale_tup] = {}
            if scale_tup not in ip_count: ip_count[scale_tup] = {}
            if scale_tup not in gpp_count: gpp_count[scale_tup] = {}
            if scale_tup not in ic_count: ic_count[scale_tup] = {}
            if scale_tup not in mem_count: mem_count[scale_tup] = {}

            # for hardware blocks
            res_filename = f"{dir_}/runs/0/hardware_graph_best.csv"
            df = pd.read_csv(res_filename)
            assert exp_id not in design_composition[scale_tup]
            design_composition[scale_tup][exp_id] = df["Block Name"].tolist()
            ip_count[scale_tup][exp_id], gpp_count[scale_tup][exp_id], ic_count[scale_tup][exp_id], mem_count[scale_tup][exp_id] = 0, 0, 0, 0
            for block in design_composition[scale_tup][exp_id]:
                if block.startswith("IP_"):
                    ip_count[scale_tup][exp_id] += 1
                elif block.startswith("LMEM_ic_") or block.startswith("GMEM_ic_"):
                    ic_count[scale_tup][exp_id] += 1
                elif block.startswith("LMEM_") or block.startswith("GMEM_"):
                    mem_count[scale_tup][exp_id] += 1
                elif block.startswith("A53_"):
                    gpp_count[scale_tup][exp_id] += 1

            design_composition[scale_tup][exp_id] = '__'.join(sorted(design_composition[scale_tup][exp_id]))
    
    assert exps_run, f"Not even one experiment finished successfully! paths: {paths}"

    slack_for_dag:Dict[Tuple[float,float,float], Dict[int, Dict[int, float]]] = {}
    num_dags_met_deadline:Dict[Tuple[float,float,float], Dict[int, Dict[int, float]]] = {}
    avg_num_dags_met_deadline:Dict[Tuple[float,float,float], float] = {}

    fin_elapsed_time:Dict[Tuple[float,float,float], Dict[str, float]] = {}
    best_slack_for_dag:Dict[Tuple[float,float,float], Dict[int, float]] = {}
    best_design_composition:Dict[Tuple[float,float,float], str] = {}
    best_ip_count:Dict[Tuple[float,float,float], int] = {}
    best_gpp_count:Dict[Tuple[float,float,float], int] = {}
    best_ic_count:Dict[Tuple[float,float,float], int] = {}
    best_mem_count:Dict[Tuple[float,float,float], int] = {}
    best_num_dags_met_deadline, best_exp_id, best_power, best_area = {}, {}, {}, {}
    avg_power:Dict[Tuple[float,float,float], float] = {}; avg_area:Dict[Tuple[float,float,float], float] = {}
    for scale_tup in scale_tup_all:
        if scale_tup not in num_dags_met_deadline: num_dags_met_deadline[scale_tup] = {}
        if scale_tup not in slack_for_dag: slack_for_dag[scale_tup] = {}
        if scale_tup not in avg_num_dags_met_deadline: avg_num_dags_met_deadline[scale_tup] = 0
        if scale_tup not in avg_power: avg_power[scale_tup] = 0.
        if scale_tup not in avg_area: avg_area[scale_tup] = 0.
        if scale_tup not in best_num_dags_met_deadline: best_num_dags_met_deadline[scale_tup] = -1
        if scale_tup not in best_exp_id: best_exp_id[scale_tup] = exps_run[0]
        if scale_tup not in best_power: best_power[scale_tup] = power[scale_tup][exps_run[0]]
        if scale_tup not in best_area: best_area[scale_tup] = area[scale_tup][exps_run[0]]
        elapsed_time_all_exps = {}
        for exp_id in exps_run:
            num_dags_met_deadline[scale_tup][exp_id] = 0
            slack_for_dag[scale_tup][exp_id] = []

            for _, wrkld_name in enumerate(latency[scale_tup][exp_id].keys()):
                dag_deadline = latency_budget[scale_tup][exp_id][wrkld_name]
                dag_lat = latency[scale_tup][exp_id][wrkld_name]
                # if the dag was dropped because it missed its deadline then we are expecting -1 for its latency
                if dag_lat != -1.:
                    if dag_lat <= dag_deadline:
                        num_dags_met_deadline[scale_tup][exp_id] += 1
                    slack_for_dag[scale_tup][exp_id].append(dag_deadline - dag_lat)
                else:
                    slack_for_dag[scale_tup][exp_id].append(-1.)
            if (num_dags_met_deadline[scale_tup][exp_id] > best_num_dags_met_deadline[scale_tup]) or \
               (num_dags_met_deadline[scale_tup][exp_id] == best_num_dags_met_deadline[scale_tup]) and (power[scale_tup][exp_id] < best_power[scale_tup]) or \
               (num_dags_met_deadline[scale_tup][exp_id] == best_num_dags_met_deadline[scale_tup]) and (power[scale_tup][exp_id] == best_power[scale_tup]) and (area[scale_tup][exp_id] < best_area[scale_tup]):
                best_num_dags_met_deadline[scale_tup] = num_dags_met_deadline[scale_tup][exp_id]
                best_exp_id[scale_tup]              = exp_id
                best_power[scale_tup]               = power[scale_tup][exp_id]
                best_area[scale_tup]                = area[scale_tup][exp_id]
                best_slack_for_dag[scale_tup]       = slack_for_dag[scale_tup][exp_id]
                best_design_composition[scale_tup]  = design_composition[scale_tup][exp_id]
                best_ip_count[scale_tup]            = ip_count[scale_tup][exp_id]
                best_gpp_count[scale_tup]           = gpp_count[scale_tup][exp_id]
                best_ic_count[scale_tup]            = ic_count[scale_tup][exp_id]
                best_mem_count[scale_tup]           = mem_count[scale_tup][exp_id]

            avg_num_dags_met_deadline[scale_tup]    += num_dags_met_deadline[scale_tup][exp_id]
            avg_power[scale_tup]                    += power[scale_tup][exp_id]
            avg_area[scale_tup]                     += area[scale_tup][exp_id]

            elapsed_time_all_exps[exp_id] = elapsed_time[scale_tup][exp_id]

        # record average data across all the experiments
        avg_num_dags_met_deadline[scale_tup]    /= len(exps_run)
        avg_power[scale_tup]                    /= len(exps_run)
        avg_area[scale_tup]                     /= len(exps_run)

        fin_elapsed_time[scale_tup] = {}
        fin_elapsed_time[scale_tup]["max"] = max(elapsed_time_all_exps.values())
        fin_elapsed_time[scale_tup]["sum"] = sum(elapsed_time_all_exps.values())
        fin_elapsed_time[scale_tup]["avg"] = fin_elapsed_time[scale_tup]["sum"]/float(len(exps_run))
        fin_elapsed_time[scale_tup]["best_exp"] = elapsed_time_all_exps[best_exp_id[scale_tup]]

    return avg_num_dags_met_deadline, avg_power, avg_area, best_exp_id, best_num_dags_met_deadline, best_slack_for_dag, best_power, best_area, fin_elapsed_time, best_design_composition, best_ip_count, best_gpp_count, best_ic_count, best_mem_count

def collect_all(outdir:str, soc_dims:Tuple[int, int], dag_inter_arr_time:float, n_dags:List[int], n_cvs:List[int], n_rads:List[int], n_vits:List[float], n_exp:List[int], skipped_exps_ok=False):
    n_exp_dict = {}; n_dags_dict = {}
    for i, prefix in enumerate(prefix_all):
        n_exp_dict[prefix] = n_exp[i]
        n_dags_dict[prefix] = n_dags[i]

    columns = ["SoC Dim", "DAG Int. Arr. Time (s)", "Num DAGs", \
        "Num CVs", "Num Radars", \
        "Num Viterbis", "Lat Budget Scale", "Power Budget Scale", "Area Budget Scale", \
        "Best No. of DAGs met Deadline", "Best DAG Deadline Meet %", "Best Power (mW)", "Best Area (mm^2)", \
        "Avg. of DAGs met Deadline", "Avg. DAG Deadline Meet %", "Avg. Power (mW)", "Avg. Area (mm^2)", \
        "Max. Elapsed Time (s)", "Sum Elapsed Time (s)", "Avg. Elapsed Time (s)", "Best Exp Elapsed Time (s)", \
        "Best Exp ID", "Best Exp Num GPP", "Best Exp Num IP", "Best Exp Num Mem", "Best Exp Num IC", "Best Exp Composition", "Best Exp Slacks"]
    if prefix_all != [""]:
        columns = ["Prefix"] + columns
    df = pd.DataFrame(columns=columns)
    soc_x_dim, soc_y_dim = soc_dims[0], soc_dims[1]
    for prefix in prefix_all:
        n_dags = n_dags_dict[prefix]
        n_exp  = n_exp_dict[prefix]
        for nc in n_cvs:
            for nr in n_rads:
                for nv in n_vits:
                    if prefix == "":
                        log_path=f"{outdir}/soc_{soc_x_dim}x{soc_y_dim}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arr_time}_ncv_{nc}_nrad_{nr}_nvit_{nv}"
                        df_entry = [f"{soc_x_dim}x{soc_y_dim}", dag_inter_arr_time, n_dags, nc, nr, nv]
                    else:
                        log_path=f"{outdir}/{prefix}_soc_{soc_x_dim}x{soc_y_dim}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arr_time}_ncv_{nc}_nrad_{nr}_nvit_{nv}"
                        if "oldDag" not in prefix:
                            df_entry = [prefix, f"{soc_x_dim}x{soc_y_dim}", dag_inter_arr_time, n_dags, nc, nr, nv]
                        else:
                            glob_str = f"{outdir}/{prefix}_soc_{soc_x_dim}x{soc_y_dim}_numDags_*_dagInterArrTime_0.0_ncv_{nc}_nrad_{nr}_nvit_{nv}"
                            log_path = glob.glob(glob_str)
                            if len(log_path) != 1: #, f"Couldn't find any matching paths for: {glob_str}"
                                glob_str = f"{outdir}/{prefix}_soc_{soc_x_dim}x{soc_y_dim}_numDags_*_dagInterArrTime_{dag_inter_arr_time}_ncv_{nc}_nrad_{nr}_nvit_{nv}"
                                log_path = glob.glob(glob_str)
                                assert len(log_path) == 1, f"Couldn't find any matching paths for: {glob_str}"
                            log_path = log_path[0]
                            df_entry = [prefix.split('-')[0], f"{soc_x_dim}x{soc_y_dim}", dag_inter_arr_time, n_dags, nc, nr, nv]
                    # iprint(f"Probing path: {log_path}")
                    if not os.path.exists(log_path):
                        # wprint(log_path + " does not exist!")
                        if skip_ok:
                            continue
                        else:
                            exit(1)
                    avg_n_dags_met, avg_power, avg_area, best_n_met_exp_id, n_dags_met, slacks, best_power, best_area, elapsed_time, best_design_composition, best_ip_count, best_gpp_count, best_ic_count, best_mem_count = get_stats_for_workloads(log_path, skipped_exps_ok)
                    for budget_scale_tup in sorted(avg_n_dags_met.keys()):
                        df_entry_inner = list(budget_scale_tup)
                        if best_n_met_exp_id[budget_scale_tup] == -1 or n_dags_met[budget_scale_tup] == -1:
                            df_entry_inner += ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
                        else:
                            df_entry_inner.append(n_dags_met[budget_scale_tup])
                            df_entry_inner.append(100.*n_dags_met[budget_scale_tup]/n_dags)
                            df_entry_inner.append(best_power[budget_scale_tup]*1e3)
                            df_entry_inner.append(best_area[budget_scale_tup]*1e6)
                            df_entry_inner.append(avg_n_dags_met[budget_scale_tup])
                            df_entry_inner.append(100.*avg_n_dags_met[budget_scale_tup]/n_dags)
                            df_entry_inner.append(avg_power[budget_scale_tup]*1e3)
                            df_entry_inner.append(avg_area[budget_scale_tup]*1e6)
                            df_entry_inner.append(elapsed_time[budget_scale_tup]["max"])
                            df_entry_inner.append(elapsed_time[budget_scale_tup]["sum"])
                            df_entry_inner.append(elapsed_time[budget_scale_tup]["avg"])
                            df_entry_inner.append(elapsed_time[budget_scale_tup]["best_exp"])

                            df_entry_inner.append(best_n_met_exp_id[budget_scale_tup])
                            df_entry_inner.append(best_gpp_count[budget_scale_tup])
                            df_entry_inner.append(best_ip_count[budget_scale_tup])
                            df_entry_inner.append(best_mem_count[budget_scale_tup])
                            df_entry_inner.append(best_ic_count[budget_scale_tup])
                            df_entry_inner.append(best_design_composition[budget_scale_tup])

                            df_entry_inner.append('_'.join([str(s) for s in slacks[budget_scale_tup]]))
                        df.loc[len(df.index)] = df_entry + df_entry_inner
    if df.empty:
        logger.error("Something went wrong, no data was collected")
        exit(1)
    # iprint(f"Collecting data for SoC dims = {soc_dims}, num DAGs = {n_dags}, DAG mean inter-arrival time = {dag_inter_arr_time}, num CVs = {n_cvs}, num radars = {n_rads}, num Viterbis = {n_vits}")
    # outfile = f"{outdir}/results_soc_{soc_x_dim}x{soc_y_dim}_dagInterArrTime_{dag_inter_arr_time}.csv"
    df.to_csv(sys.stdout, index=False)
    # iprint(f"Results:")
    # print(df)
    # iprint(f"Wrote to {outfile}.")

if __name__ == "__main__":
    argc = len(sys.argv)
    assert argc == 12, "Usage: python3 collect_all.py <logs_path> <prefix_list> <soc dim X> <soc dim Y> <DAG inter-arrival time> <n_dags> <n_cvs> <n_vits> <n_rads> <number_of_experiments_for_each_prefix> <skipped_exps_ok> but found " + str(argc) + " args"

    logger = setup_logger('MyLogger')

    LOG_ROOT = sys.argv[1]
    prefix_all = sys.argv[2].split(' ')
    soc_dims = (int(sys.argv[3]), int(sys.argv[4]))
    dag_inter_arr_time = float(sys.argv[5])
    n_dags = [int(n) for n in sys.argv[6].replace('"', '').split(' ')]
    assert len(n_dags) == len(prefix_all)
    n_cvs = [int(n) for n in sys.argv[7].replace('"', '').split(' ')]
    n_rads = [int(n) for n in sys.argv[8].replace('"', '').split(' ')]
    n_vits = [float(n) for n in sys.argv[9].replace('"', '').split(' ')]
    n_exp = [int(n) for n in sys.argv[10].replace('"', '').split(' ')]
    skipped_exps_ok = bool(sys.argv[11])
    assert len(n_exp) == len(prefix_all)
    # print(LOG_ROOT, soc_dims, dag_inter_arr_time, n_dags, n_cvs, n_rads, n_vits, n_exp)
    for n in n_exp:
        assert n > 0
    collect_all(LOG_ROOT, soc_dims, dag_inter_arr_time, n_dags, n_cvs, n_rads, n_vits, n_exp, skipped_exps_ok)
