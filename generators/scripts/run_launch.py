# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import os
import sys
sys.path.append('../')
from helper import setup_logger

################################################
soc_x_y_dag_iat_all = [
    (3, 3, 0.033), 
    # (3, 3, 0.022), 
    # (3, 3, 0.017)
]

ncv_nrad_nvit_all = [
    (1, 2, 1.0), 
    # (1, 4, 4.0)
]

std_envs = {
    "NDAGS_SIM":                        "50",
    "DROP_TASKS_THAT_PASSED_DEADLINE":  "0",
    "GEN_TRACES":                       "1",
    "DYN_SCHEDULING_MEM_REMAPPING":     "0",
    "BUDGET_SCALES":                    "1. 1. 1.",
    "CUST_SCHED_POLICY_NAME":           "ms_dyn_energy",
    "N_EXP":                            "1",
    "EXPLORE_MODE":                     "all-at-start",
    "CONSTRAIN_TOPOLOGY" :              "0",
    "CUST_SCHED_CONSIDER_DM_TIME":      "1", 
    "EXPLR_TIMEOUT":                    "-1"
}

runs = [
    {"BUDGET_SCALES":           "1. 1. 1.", 
     "CUST_SCHED_POLICY_NAME":  "ms_dyn_energy", 
     "FRAMEWORK":               "ARTEMIS",     
     "N_EXP":                   "10", 
     "EXPLORE_MODE":            "all-at-start", 
     "CONSTRAIN_TOPOLOGY":      "1", 
     "NUM_MEMS":                "2"},
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",  "FRAMEWORK": "FARSI-RR",    "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "1", "NUM_MEMS" : "2"}, # sched pol doesn't affect FARSI-RR
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",  "FRAMEWORK": "FARSI-DYN",   "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "1", "NUM_MEMS" : "2"},

    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_stat_energy", "FRAMEWORK": "ARTEMIS",     "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "1", "NUM_MEMS" : "2"},
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_stat_energy", "FRAMEWORK": "FARSI-DYN",   "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "1", "NUM_MEMS" : "2"},
    
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",  "FRAMEWORK": "ARTEMIS",     "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "0"},
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",  "FRAMEWORK": "FARSI-RR",    "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "0"}, # sched pol doesn't affect FARSI-RR
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",  "FRAMEWORK": "FARSI-DYN",   "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "0"},

    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_stat_energy", "FRAMEWORK": "ARTEMIS",     "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "0"},
    # {"BUDGET_SCALES": "1. 1. 1.", "CUST_SCHED_POLICY_NAME": "ms_stat_energy", "FRAMEWORK": "FARSI-DYN",   "N_EXP": "10", "EXPLORE_MODE": "all-at-start", "CONSTRAIN_TOPOLOGY" : "0"},
]

std_envs.update(os.environ)

################################################

if __name__ == "__main__":
    logger = setup_logger('MyLogger')
    for my_env in runs:
        env = {**my_env, **std_envs}
        for ncv, nrad, nvit in ncv_nrad_nvit_all: 
            for soc_x, soc_y, dag_iat in soc_x_y_dag_iat_all:
                cmd = ["bash", "launch_jobs_ccc.stlt.sh", f"{soc_x} {soc_y} {dag_iat}", f"{ncv} {nrad} {nvit}"]
                constr_top = bool(int(env["CONSTRAIN_TOPOLOGY"]))
                key = (constr_top, (ncv, nrad, nvit), (soc_x, soc_y, dag_iat))
                logger.info(f"Running cmd: {cmd}")
                subprocess.run(cmd, env=env)