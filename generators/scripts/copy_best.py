# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
import sys
from typing import Dict, List
sys.path.append('../')
from helper import replace_tok, setup_logger

from collect_all import get_stats_for_workloads

if __name__ == "__main__":
    assert len(sys.argv) == 13, "Usage: ./copy_best.py <path> <farsi_inp_dir> <soc_x_dim> <soc_y_dim> <dag_interarrival_time> <num_dags_exploration> <num_experiments> <dag_interarrival_time_simulation> <num_dags_simulation> <constrain_topology> <skipped_exps_ok> <ncv_nrad_nvit>"

    logger = setup_logger('MyLogger')
    logger.info(f"Copy script invoked with args: {sys.argv}")

    path = sys.argv[1]
    farsi_inp_dir = sys.argv[2]
    soc_x = int(sys.argv[3])
    soc_y = int(sys.argv[4])
    dag_intarr_time = float(sys.argv[5])
    ndags_exp = int(sys.argv[6])
    nexp = int(sys.argv[7])
    dag_intarr_time_sim = float(sys.argv[8])
    ndags_sim = int(sys.argv[9])
    constrain_topology = int(sys.argv[10])
    skipped_exps_ok = bool(sys.argv[11])
    ncv_nrad_nvit = tuple(sys.argv[12].split(' '))
    ncv, nrad, nvit = int(ncv_nrad_nvit[0]), int(ncv_nrad_nvit[1]), float(ncv_nrad_nvit[2])

    # for all-at-start execution
    if dag_intarr_time != dag_intarr_time_sim:
        dir_ = f"exp-oldDagInterArrTime-{dag_intarr_time_sim}_soc_{soc_x}x{soc_y}_numDags_{ndags_exp}_dagInterArrTime_{dag_intarr_time}_ncv_{ncv}_nrad_{nrad}_nvit_{nvit}"
    else:
        dir_ = f"exp_soc_{soc_x}x{soc_y}_numDags_{ndags_exp}_dagInterArrTime_{dag_intarr_time}_ncv_{ncv}_nrad_{nrad}_nvit_{nvit}"
    assert os.path.exists(f"{path}/{dir_}"), f"Directory {path}/{dir_} not found"
    # TODO adjust for other workload domains
    wrkld = "miniera_" + "_".join(dir_.split("_")[3:])
    wrkld_with_soc_dims = "miniera_" + f"soc_{soc_x}x{soc_y}_" + "_".join(dir_.split("_")[3:])
    _, _, _, exp_id, best_num_dags_met_deadline_all_budget_scales, _, best_power_all_budget_scales, best_area_all_budget_scales, _, _, _, _, _, _ = get_stats_for_workloads(run_root=f"{path}/{dir_}", skipped_exps_ok=skipped_exps_ok)

    best_num_dags_met_deadline = -1
    best_exp_id, best_power, best_area = None, None, None
    for scale_tup, num_dags_met_deadline in best_num_dags_met_deadline_all_budget_scales.items():
        power = best_power_all_budget_scales[scale_tup]
        area = best_area_all_budget_scales[scale_tup]
        if (num_dags_met_deadline > best_num_dags_met_deadline) or \
            (num_dags_met_deadline == best_num_dags_met_deadline and power < best_power) or \
            (num_dags_met_deadline == best_num_dags_met_deadline and area < best_area and power < best_power):
            best_lat_scale, best_pow_scale, best_area_scale = scale_tup
            best_num_dags_met_deadline = num_dags_met_deadline
            best_exp_id, best_power, best_area = exp_id[scale_tup], power, area
    assert best_exp_id != None

    if constrain_topology:
        wrkld = wrkld_with_soc_dims
    if ndags_sim != ndags_exp:
        wrkld = replace_tok(wrkld, "numDags", ndags_sim)
    if dag_intarr_time_sim != dag_intarr_time:
        if ndags_sim == 1:
            dag_intarr_time_sim = 0.0
        wrkld = replace_tok(wrkld, "dagInterArrTime", dag_intarr_time_sim)
    assert best_exp_id != -1, f"Error in calling get_slacks_for_workloads(), wrkld: {wrkld}, dir: {path}/{dir_}, ndags_exp: {ndags_exp}, nexp: {nexp}"
    # copy hw graph and task to hw mapping files corresponding to best exp_id to ../..
    glob_path = f"{path}/{dir_}/final_{best_exp_id}/*/____lat_{best_lat_scale}__pow_{best_pow_scale}__area_{best_area_scale}___workloads*/runs/0/hardware_graph_best.csv"
    logger.info(f"Searching in path: {glob_path}")
    try:
        hw_graph_fname = glob.glob(glob_path)
    except:
        glob_path = f"{path}/{dir_}/final_{best_exp_id}/*/____lat_{float(best_lat_scale)}__pow_{float(best_pow_scale)}__area_{float(best_area_scale)}___workloads*/runs/0/hardware_graph_best.csv"
        logger.info(f"Searching in path: {glob_path}")
    assert len(hw_graph_fname) == 1
    hw_graph_fname = hw_graph_fname[0]
    glob_path = f"{path}/{dir_}/final_{best_exp_id}/*/____lat_{best_lat_scale}__pow_{best_pow_scale}__area_{best_area_scale}___workloads*/runs/0/task_to_hardware_mapping_best.csv"
    logger.info(f"Searching in path: {glob_path}")
    try:
        t_to_hw_map_fname = glob.glob(glob_path)
    except:
        glob_path = f"{path}/{dir_}/final_{best_exp_id}/*/____lat_{float(best_lat_scale)}__pow_{float(best_pow_scale)}__area_{float(best_area_scale)}___workloads*/runs/0/task_to_hardware_mapping_best.csv"
        logger.info(f"Searching in path: {glob_path}")
    assert len(t_to_hw_map_fname) == 1
    t_to_hw_map_fname = t_to_hw_map_fname[0]
    os.makedirs(farsi_inp_dir, exist_ok=True)
    dst = f"{farsi_inp_dir}/{wrkld}_database - Hardware Graph.csv"
    shutil.copyfile(hw_graph_fname, dst)
    logger.info(f"Copied: {hw_graph_fname} -> {dst}")
    dst = f"{farsi_inp_dir}/{wrkld}_database - Task To Hardware Mapping.csv"
    shutil.copyfile(t_to_hw_map_fname, dst)
    logger.info(f"Copied: {t_to_hw_map_fname} -> {dst}")
