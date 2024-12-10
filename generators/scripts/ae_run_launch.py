# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import os
import sys
sys.path.append('../')
from helper import setup_logger

################################################
reduced_exp_set = True # False
soc_x_y_dag_iat_all = [
    [ # set 1
        (3, 3, 0.066667), 
        (3, 3, 0.045454),
        (3, 3, 0.033333), 
        (3, 3, 0.026316),
        (3, 3, 0.022222), 
        (3, 3, 0.019231),
        (3, 3, 0.016667),
        (3, 3, 0.014706) 
    ],
    [ # set 2
        (3, 3, 0.033333), 
        (3, 3, 0.011111), 
    ]
]
if reduced_exp_set:
    soc_x_y_dag_iat_all[0] = soc_x_y_dag_iat_all[0][::2]
ncv_nrad_nvit = [
    (1, 2, 1.0), 
    (1, 4, 4.0)
]

std_envs = {
    "NDAGS_SIM":                        "50",
    "DROP_TASKS_THAT_PASSED_DEADLINE":  "0",
    "GEN_TRACES":                       "1",
    "DYN_SCHEDULING_MEM_REMAPPING":     "0",
    "BUDGET_SCALES":                    "1. 1. 1.",
    "EXPLORE_MODE":                     "all-at-start",
    "CUST_SCHED_CONSIDER_DM_TIME":      "1", 
    "EXPLR_TIMEOUT":                    "-1", 
    "NUM_MEMS":                         "1"
}

# FARSI-RR == FARSI in paper, FARSI-DYN == ARTEMIS/No-DSE in paper
runs_all = [
    [ # set 1
        # experiments for Fig. 8
        {"FRAMEWORK": "FIXED_HET",  "N_EXP":  "1",  "CONSTRAIN_TOPOLOGY": "1", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",}, # for baseline "fixed" SoC
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",}, # for ARTEMIS-generated SoC

        # experiments for Fig. 9 (left)
        {"FRAMEWORK": "FARSI-RR",   "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",}, # sched pol doesn't affect FARSI-RR
        {"FRAMEWORK": "FARSI-DYN",  "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",},
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",},
        
        # experiments for Fig. 9 (right)
        {"FRAMEWORK": "FARSI-RR",   "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "1", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",}, # sched pol doesn't affect FARSI-RR
        {"FRAMEWORK": "FARSI-DYN",  "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "1", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",},
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "1", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",},
    ],
    [ # set 2
        # experiments for Table IV
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "edf",},
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "eft",},
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_stat",},
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_dyn",},
        {"FRAMEWORK": "ARTEMIS",    "N_EXP": "10",  "CONSTRAIN_TOPOLOGY": "0", "CUST_SCHED_POLICY_NAME": "ms_dyn_energy",},
    ]
]

std_envs.update(os.environ)

################################################

if __name__ == "__main__":
    logger = setup_logger('MyLogger')
    for i in [1, 0]: # run set 2 first, then set 1
        soc_x_y_dag_iat = soc_x_y_dag_iat_all[i]
        runs = runs_all[i]
        for my_env in runs:
            env = {**my_env, **std_envs}
            for ncv, nrad, nvit in ncv_nrad_nvit: 
                for soc_x, soc_y, dag_iat in soc_x_y_dag_iat:
                    cmd = ["bash", "launch_jobs_ccc.stlt.sh", f"{soc_x} {soc_y} {dag_iat}", f"{ncv} {nrad} {nvit}"]
                    constr_top = bool(int(env["CONSTRAIN_TOPOLOGY"]))
                    key = (constr_top, (ncv, nrad, nvit), (soc_x, soc_y, dag_iat))
                    logger.info(f"Running cmd: {cmd}")
                    subprocess.run(cmd, env=env)