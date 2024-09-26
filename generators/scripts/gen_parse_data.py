# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append("../")
from datetime import datetime
from math import ceil
from pprint import pprint
from typing import Dict, List, Tuple, Set

import networkx as nx
import numpy as np
import pandas as pd
from DFG import DFG
from filelock import FileLock
from helper import (export_nx_graph, graph_to_csv, parse_args, rotate, setup_logger)

np.random.seed(datetime.now().microsecond)

FARSIRoot = "../../Project_FARSI"
assert os.path.exists(FARSIRoot) and os.listdir(FARSIRoot), f"{FARSIRoot} does not exist or is empty!"

GRAPH_GEN_MODE      = "split" # "as_llp"
ACCEL_SUPPORTS_LLP  = False # True

# Data movement in bytes, profiled by running actual workload.
data_inputs = {
    "miniera": {
        ("souurce", "CV")           : 12_288,
        ("CV", "PlanCtrl")          : 8,

        ("souurce", "VitPre")       : 24_572,
        ("VitPre", "Vit")           : 24_600,

        ("souurce", "Vit")          : 44,
        ("Vit", "VitPost")          : 12_264,

        ("souurce", "VitPost")      : 40,
        ("VitPost", "PlanCtrl")     : 1_604,

        ("souurce", "Radar")        : 8_196,
        ("Radar", "RadarPost")      : 8_192,

        ("souurce", "RadarPost")    : 4,
        ("RadarPost", "PlanCtrl")   : 4,

        ("souurce", "PlanCtrl")     : 16,
        ("PlanCtrl", "siink")       : 12,
    }
}

# Exploration budgets.
def_budgets:Dict[str, Dict[str, float]] = {
    "miniera": {
        "latency"   : 50e-3, # in s
        "power"     : .5,    # in W
        "area"      : 3e-6,  # in m^2
        "cost"      : .1
    }
}

clock_speed = {"miniera": 1200e6}   # in Hz
voltage     = {"miniera": 0.8}      # in V
cpu_ipc     = {"miniera": 1}

# Masks for available PE types.
pe_mask_dict:Dict[str, bool] = {
    "A53"           : True, # Ariane core, actually.
    "IP_CV"         : True,
    "IP_Radar"      : True,
    "IP_Vit"        : True,
}

# From Jia et al., ESSCIRC'22.
power_area_dict = {
    "dynamic_power": {  # in W
        "A53"       : 0.07539267016,
        "IP_CV"     : 0.0802226087,
        "IP_Radar"  : 0.007344761905,
        "IP_Vit"    : 0.03068047059
    }, 
    "static_power": {   # in W
        "A53"       : 0.02513089005,
        "IP_CV"     : 0.02674086957,
        "IP_Radar"  : 0.002448253968,
        "IP_Vit"    : 0.01022682353
    }, 
    "area": {           # in um2
        "A53"       : 339_755.8,
        "IP_CV"     : 577_049.825,
        "IP_Radar"  : 186_951.842,
        "IP_Vit"    : 175_247.674
    }
}
ppa_dict = {
    "perf": {           # in cycles
        "A53": {
            "VitPre"    : lambda x: ceil(x * 1_684_764),
            "Vit"       : lambda x: ceil(x * 2_987_631_454),
            "VitPost"   : lambda x: ceil(x * 1_835_603),
            "Radar"     : lambda x: ceil(x * 3_378_087),
            "RadarPost" : lambda x: ceil(x * 315_491),
            "CV"        : lambda x: ceil(x * 97_890_000),
            "PlanCtrl"  : lambda x: ceil(x * 3_636),
        },
        "IP_CV": {
            "CV"        : lambda x: ceil(x * 41_867_016),
        },
        "IP_Radar": {
            "Radar"     : lambda x: ceil(x * 622_152),
        },
        "IP_Vit": {
            "Vit"       : lambda x: ceil(x * 7_400_008),
        },
    },
    "dynamic_energy": {     # in fJ
        "A53": {
            "VitPre"    : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 1_684_764 / CLK_HZ * 1e15,
            "Vit"       : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 2_987_631_454 / CLK_HZ * 1e15,
            "VitPost"   : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 1_835_603 / CLK_HZ * 1e15,
            "Radar"     : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 3_378_087 / CLK_HZ * 1e15,
            "RadarPost" : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 315_491 / CLK_HZ * 1e15,
            "CV"        : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 97_890_000 / CLK_HZ * 1e15,
            "PlanCtrl"  : lambda x: x * power_area_dict["dynamic_power"]["A53"] * 3_636 / CLK_HZ * 1e15,
        },
        "IP_CV": {
            "CV"        : lambda x: x * power_area_dict["dynamic_power"]["IP_CV"] * 41_867_016 / CLK_HZ * 1e15,
        },
        "IP_Radar": {
            "Radar"     : lambda x: x * power_area_dict["dynamic_power"]["IP_Radar"] * 622_152 / CLK_HZ * 1e15,
        },
        "IP_Vit": {
            "Vit"       : lambda x: power_area_dict["dynamic_power"]["IP_Vit"] * 7_400_008 / CLK_HZ * 1e15,
        },
    },
    "static_power": {       # in W
        "A53": {
            "VitPre"    : lambda x: x * power_area_dict["static_power"]["A53"],
            "Vit"       : lambda x: x * power_area_dict["static_power"]["A53"],
            "VitPost"   : lambda x: x * power_area_dict["static_power"]["A53"],
            "Radar"     : lambda x: x * power_area_dict["static_power"]["A53"],
            "RadarPost" : lambda x: x * power_area_dict["static_power"]["A53"],
            "CV"        : lambda x: x * power_area_dict["static_power"]["A53"],
            "PlanCtrl"  : lambda x: x * power_area_dict["static_power"]["A53"],
        },
        "IP_CV": {
            "CV"        : lambda x: x * power_area_dict["static_power"]["IP_CV"],
        },
        "IP_Radar": {
            "Radar"     : lambda x: x * power_area_dict["static_power"]["IP_Radar"],
        },
        "IP_Vit": {
            "Vit"       : lambda x: power_area_dict["static_power"]["IP_Vit"],
        },
    },
    "area": {
        "A53": {
            "VitPre"    : lambda x: power_area_dict["area"]["A53"],
            "Vit"       : lambda x: power_area_dict["area"]["A53"],
            "VitPost"   : lambda x: power_area_dict["area"]["A53"],
            "Radar"     : lambda x: power_area_dict["area"]["A53"],
            "RadarPost" : lambda x: power_area_dict["area"]["A53"],
            "CV"        : lambda x: power_area_dict["area"]["A53"],
            "PlanCtrl"  : lambda x: power_area_dict["area"]["A53"],
        },
        "IP_CV": {
            "CV"        : lambda x: power_area_dict["area"]["IP_CV"],
        },
        "IP_Radar": {
            "Radar"     : lambda x: power_area_dict["area"]["IP_Radar"],
        },
        "IP_Vit": {
            "Vit"       : lambda x: power_area_dict["area"]["IP_Vit"],
        },
    },
}

def get_bcet(node_type:str, f:int) -> float:
    """Get best case execution time from PPA dictionary"""
    assert ppa_dict
    assert "perf" in ppa_dict
    bcet = np.inf
    for _, task_to_perf_dict in ppa_dict["perf"].items():
        if node_type in task_to_perf_dict:
            bcet = min(bcet, task_to_perf_dict[node_type](f))
    assert bcet != np.inf
    return bcet

class ERAContext():
    def __init__(self, wrkld:str, ppa_dict:dict, pe_mask_dict:dict, num_dags:int, dag_arr_times:List[float], n_cv:int, n_rad:int, n_vit_per_dag:List[int]):
        self.wrkld              = wrkld
        self.dag_arr_times      = dag_arr_times
        self.n_dags             = num_dags
        self.n_cv               = n_cv
        self.n_rad              = n_rad
        self.n_vit_per_dag      = n_vit_per_dag

        self.dfgs = [DFG(ctx=self, wrkld=f"{self.wrkld}_{dag_id}", ppa_dict=ppa_dict, pe_mask_dict=pe_mask_dict, silence=SILENCE, ) for dag_id in range(self.n_dags)]

    def extract_params(self, node_name:str):
        """Extract the llp given a node name, For ERA, the convention is for nodes to be named as {node_name}_x{llp}_{task_id}_{dag_id}."""
        rc = int(node_name.split('_')[-3].split('x')[1])
        return [rc, ]

    def gen_one_dag_s_task_graph(self, dag_id:int, the_dfg:DFG):
        """Populate one DAG"""
        id_suff = f"_{dag_id}"
        souurce_node_name = "souurce"
        n_vit = self.n_vit_per_dag[dag_id]
        # Calculate scaling factor to scale perf/energy with for the plan_ctrl task. Assume linear scaling with respect to number of incoming edges for now.
        plan_ctrl_f = ceil((self.n_cv + n_vit + self.n_rad) / 3)
        if GRAPH_GEN_MODE == 'split':
            # Radar task(s).
            if self.n_rad:
                radar_node_names = [the_dfg.create_node(f"Radar_x1", bcet=get_bcet(node_type="Radar", f=1), llp=1, id=f"{i}{id_suff}") for i in range(self.n_rad)]
                [the_dfg.create_edge(souurce_node_name, radar_node_name, weight=data_inputs["miniera"][("souurce", "Radar")], bcet=get_bcet(node_type="Radar", f=1)) for radar_node_name in radar_node_names] # , 20_480) # Twiddles are pre-loaded.
                radar_post_node_names = [the_dfg.create_node(f"RadarPost_x1", bcet=get_bcet(node_type="RadarPost", f=1), llp=1, id=f"{i}{id_suff}") for i in range(self.n_rad)]
                [the_dfg.create_edge(souurce_node_name, radar_post_node_names[i], weight=data_inputs["miniera"][("souurce", "RadarPost")], bcet=get_bcet(node_type="RadarPost", f=1)) for i in range(self.n_rad)]
                [the_dfg.create_edge(radar_node_names[i], radar_post_node_names[i], weight=data_inputs["miniera"][("Radar", "RadarPost")], bcet=get_bcet(node_type="RadarPost", f=1)) for i in range(self.n_rad)]
            # CV task(s).
            if self.n_cv:
                cv_node_names = [the_dfg.create_node(f"CV_x1", bcet=get_bcet(node_type="CV", f=1), llp=1, id=f"{i}{id_suff}") for i in range(self.n_cv)]
                [the_dfg.create_edge(souurce_node_name, cv_node_name, weight=data_inputs["miniera"][("souurce", "CV")], bcet=get_bcet(node_type="CV", f=1)) for cv_node_name in cv_node_names] # , 1_906_836) # CV accel assumes weights are pre-loaded before.
            # Viterbi decoder task(s).
            if n_vit:
                vit_pre_node_names = [the_dfg.create_node(f"VitPre_x1", bcet=get_bcet(node_type="VitPre", f=1), llp=1, id=f"{i}{id_suff}") for i in range(n_vit)]
                [the_dfg.create_edge(souurce_node_name, vit_pre_node_names[i], weight=data_inputs["miniera"][("souurce", "VitPre")], bcet=get_bcet(node_type="VitPre", f=1)) for i in range(n_vit)]
                vit_node_names = [the_dfg.create_node(f"Vit_x1", bcet=get_bcet(node_type="Vit", f=1), llp=1, id=f"{i}{id_suff}") for i in range(n_vit)]
                [the_dfg.create_edge(souurce_node_name, vit_node_names[i], weight=data_inputs["miniera"][("souurce", "Vit")], bcet=get_bcet(node_type="Vit", f=1)) for i in range(n_vit)]
                [the_dfg.create_edge(vit_pre_node_names[i], vit_node_names[i], weight=data_inputs["miniera"][("VitPre", "Vit")], bcet=get_bcet(node_type="Vit", f=1)) for i in range(n_vit)]
                vit_post_node_names = [the_dfg.create_node(f"VitPost_x1", bcet=get_bcet(node_type="VitPost", f=1), llp=1, id=f"{i}{id_suff}") for i in range(n_vit)]
                [the_dfg.create_edge(souurce_node_name, vit_post_node_names[i], weight=data_inputs["miniera"][("souurce", "VitPost")], bcet=get_bcet(node_type="VitPost", f=1)) for i in range(n_vit)]
                [the_dfg.create_edge(vit_node_names[i], vit_post_node_names[i], weight=data_inputs["miniera"][("Vit", "VitPost")], bcet=get_bcet(node_type="VitPost", f=1)) for i in range(n_vit)]
            # Plan and control task(s).
            plan_ctrl_node_name = the_dfg.create_node(f"PlanCtrl_x{plan_ctrl_f}", bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f), llp=1, id=f"0{id_suff}")
            [the_dfg.create_edge(souurce_node_name, plan_ctrl_node_name, weight=data_inputs["miniera"][("souurce", "PlanCtrl")], bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f))]  # 1 float distance value.
            
            if self.n_rad:
                [the_dfg.create_edge(radar_post_node_names[i], plan_ctrl_node_name, weight=data_inputs["miniera"][("RadarPost", "PlanCtrl")], bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f)) for i in range(self.n_rad)]  # 1 float distance value.
            if n_vit:
                [the_dfg.create_edge(vit_post_node_names[i], plan_ctrl_node_name, weight=data_inputs["miniera"][("VitPost", "PlanCtrl")], bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f)) for i in range(n_vit)]    # sizeof(message_t).
            if self.n_cv:
                [the_dfg.create_edge(cv_node_names[i], plan_ctrl_node_name, weight=data_inputs["miniera"][("CV", "PlanCtrl")], bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f)) for i in range(self.n_cv)] # 1 encoded label value.
        elif GRAPH_GEN_MODE == "as_llp":
            # Radar task(s).
            if self.n_rad:
                llp = self.n_rad if ACCEL_SUPPORTS_LLP else 1
                radar_node_name = the_dfg.create_node(f"Radar_x{self.n_rad}", bcet=get_bcet(node_type="Radar", f=self.n_rad), llp=llp, id=f"0{id_suff}")
                the_dfg.create_edge(souurce_node_name, radar_node_name, weight=data_inputs["miniera"][("souurce", "Radar")]*self.n_rad, bcet=get_bcet(node_type="Radar", f=self.n_rad)) # , 20_480) # Twiddles are pre-loaded.
                radar_post_node_name = the_dfg.create_node(f"RadarPost_x{self.n_rad}", bcet=get_bcet(node_type="RadarPost", f=self.n_rad), llp=llp, id=f"0{id_suff}")
                the_dfg.create_edge(souurce_node_name, radar_post_node_name, weight=data_inputs["miniera"][("souurce", "RadarPost")]*self.n_rad, bcet=get_bcet(node_type="RadarPost", f=self.n_rad))
                the_dfg.create_edge(radar_node_name, radar_post_node_name, weight=data_inputs["miniera"][("Radar", "RadarPost")]*self.n_rad, bcet=get_bcet(node_type="RadarPost", f=self.n_rad))
            # CV task(s).
            if self.n_cv:
                llp = self.n_cv if ACCEL_SUPPORTS_LLP else 1
                cv_node_name = the_dfg.create_node(f"CV_x{self.n_cv}", bcet=get_bcet(node_type="CV", f=self.n_cv), llp=llp, id=f"0{id_suff}")
                the_dfg.create_edge(souurce_node_name, cv_node_name, weight=data_inputs["miniera"][("souurce", "CV")]*self.n_cv, bcet=get_bcet(node_type="CV", f=self.n_cv)) # , 1_906_836) # CV accel assumes weights are pre-loaded before.
            # Viterbi decoder task(s).
            if n_vit:
                llp = n_vit if ACCEL_SUPPORTS_LLP else 1
                vit_pre_node_name = the_dfg.create_node(f"VitPre_x{n_vit}", bcet=get_bcet(node_type="VitPre", f=n_vit), llp=llp, id=f"0{id_suff}")
                the_dfg.create_edge(souurce_node_name, vit_pre_node_name, weight=data_inputs["miniera"][("souurce","VitPre")]*n_vit, bcet=get_bcet(node_type="VitPre", f=n_vit))
                vit_node_name = the_dfg.create_node(f"Vit_x{n_vit}", bcet=get_bcet(node_type="Vit", f=n_vit), llp=llp, id=f"0{id_suff}")
                the_dfg.create_edge(souurce_node_name, vit_node_name, weight=data_inputs["miniera"][("souurce", "Vit")]*n_vit, bcet=get_bcet(node_type="Vit", f=n_vit))
                the_dfg.create_edge(vit_pre_node_name, vit_node_name, weight=data_inputs["miniera"][("VitPre", "Vit")]*n_vit, bcet=get_bcet(node_type="Vit", f=n_vit))
                vit_post_node_name = self.create_node(f"VitPost_x{n_vit}", bcet=get_bcet(node_type="VitPost", f=n_vit), llp=llp, id=f"0{id_suff}")
                the_dfg.create_edge(souurce_node_name, vit_post_node_name, weight=data_inputs["miniera"][("souurce", "VitPost")]*n_vit, bcet=get_bcet(node_type="VitPost", f=n_vit))
                the_dfg.create_edge(vit_node_name, vit_post_node_name, weight=data_inputs["miniera"][("Vit", "VitPost")]*n_vit, bcet=get_bcet(node_type="VitPost", f=n_vit))
            # Plan and control task(s).
            plan_ctrl_node_name = the_dfg.create_node(f"PlanCtrl_x{plan_ctrl_f}", bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f), llp=1, id=f"0{id_suff}")
            the_dfg.create_edge(souurce_node_name, plan_ctrl_node_name, weight=data_inputs["miniera"][("souurce", "PlanCtrl")], bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f))  # 1 float distance value.

            if self.n_rad:
                the_dfg.create_edge(radar_post_node_name, plan_ctrl_node_name, weight=data_inputs["miniera"][("RadarPost", "PlanCtrl")]*self.n_rad, bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f))  # 1 float distance value.
            if n_vit:
                the_dfg.create_edge(vit_post_node_name, plan_ctrl_node_name, weight=data_inputs["miniera"][("VitPost", "PlanCtrl")]*n_vit, bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f))    # sizeof(message_t).
            if self.n_cv:
                the_dfg.create_edge(cv_node_name, plan_ctrl_node_name, weight=data_inputs["miniera"][("CV", "PlanCtrl")]*self.n_cv, bcet=get_bcet(node_type="PlanCtrl", f=plan_ctrl_f)) # 1 encoded label value.
        else:
            raise NotImplementedError
        return plan_ctrl_node_name

    def gen_task_graph(self) -> None:
        """Populate all DAGs in the workload"""
        for dag_id, the_dfg in enumerate(self.dfgs):
            the_dfg.create_source_node()
            plan_ctrl_node_name = self.gen_one_dag_s_task_graph(dag_id=dag_id, the_dfg=the_dfg)
            the_dfg.create_sink_node(dummy_node_name=f"DummyLast_{dag_id}")
            the_dfg.connect_siink(plan_ctrl_node_name, weight=9, dummy_node_name=f"DummyLast_{dag_id}")

    def opt_and_export(self, out_dir:str, output_dot:bool, logger) -> None:
        """Export the workload into a dot file and save the DAG arrival times data"""
        for the_dfg in self.dfgs:
            the_dfg.opt_and_export(out_dir, output_dot, logger)
        gen_dag_arr_times_data(self.wrkld, self.dag_arr_times, out_dir)

def export_task_to_hardware_mapping(block_to_task_dict:Dict[str,Set[str]], out_path:str, out_prefix:str, SILENCE:bool):
    """Save the task to hardware mapping CSV file"""
    # Set -> List.
    for k, v in block_to_task_dict.items():
        block_to_task_dict[k] = list(v)

    # Pad the dictionaries before converting to DataFrame and exporting to CSV.
    max_len = 0
    for _, v in block_to_task_dict.items():
        max_len = max(len(v), max_len)

    for k, v in block_to_task_dict.items():
        len_to_pad = max_len - len(v)
        if len_to_pad:
            v.extend([''] * len_to_pad)

    df = pd.DataFrame.from_dict(block_to_task_dict)
    outfile = f"{out_path}/{out_prefix} - Task To Hardware Mapping.csv"
    df.to_csv(outfile, index=False)
    if not SILENCE:
        logger.info(f"Wrote to file: {outfile}")

def gen_task_to_hw_mapping(hw_graph, dmgs:List[nx.DiGraph], pes:List[str], mems:List[str], ics:List[str], out_path:str, out_prefix:str, method:str) -> None:
    """Generates a file that contains blocks' (PEs, ICs and Mems) names as the CSV header and
    the tasks (or data movements represented as parent_task -> child_task) as entries in the
    corresponding columns. Tasks are allocated using a strategy specified by `method`."""
    
    def get_next_gpp_idx(pes:List[str], curr_pe_idx:int) -> int:
        """Returns the next GPP after `curr_gpp_idx` that is an A53.

        Args:
            curr_gpp_idx (int): last GPP index that was assigned work

        Returns:
            pe_idx: index of GPP to allocate task to
        """
        rot_list = list(range(len(pes)))
        rot_list = rotate(rot_list, -curr_pe_idx-1)
        for pe_idx in rot_list:
            if "A53" in pes[pe_idx]:
                return pe_idx
        assert False
        
    block_to_task_dict = {}

    # Basic checks.
    assert len(pes) >= 1, "there must be at least one PE in the system"
    assert any("A53" in pe for pe in pes), "there must be at least one CPU in the system"
    assert len(ics) >= 1, "there must be at least one memory in the system"
    assert len(mems) >= 1, "there must be at least one memory in the system"

    # Use sets to avoid duplicate assignments.
    for pe in pes:
        block_to_task_dict[pe] = set()
    for ic in ics:
        block_to_task_dict[ic] = set()
    for mem in mems:
        block_to_task_dict[mem] = set()

    # Iterate over each task and assign it to a PE in a greedy fashion.
    curr_pe_idx = get_next_gpp_idx(pes, -1)
    curr_mem_idx = 0

    souurce_done, siink_done = False, False
    for dmg in dmgs:
        tasks:List[str] = list(dmg.nodes())
        np.random.shuffle(tasks)
        for task in tasks:
            # "souurce" and "siink" are assigned just once.
            if task == "souurce":
                if not souurce_done:
                    if method == "greedy_parallel":
                        curr_pe_idx = get_next_gpp_idx(pes, curr_pe_idx)
                    curr_pe = pes[curr_pe_idx]
                    block_to_task_dict[curr_pe].add(task)
                    souurce_pe = curr_pe_idx
                    souurce_done = True
                else:
                    curr_pe_idx = souurce_pe
            elif task == "siink":
                if not siink_done:
                    if method == "greedy_parallel":
                        curr_pe_idx = get_next_gpp_idx(pes, curr_pe_idx)
                    curr_pe = pes[curr_pe_idx]
                    block_to_task_dict[curr_pe].add(task)
                    siink_pe = curr_pe_idx
                    siink_done = True
                else:
                    curr_pe_idx = siink_pe
            else:
                if method == "greedy_parallel":
                    curr_pe_idx = get_next_gpp_idx(pes, curr_pe_idx)
                curr_pe = pes[curr_pe_idx]
                block_to_task_dict[curr_pe].add(task)

            ics_in_path = list(filter(lambda block: "_ic_" in block, nx.shortest_path(hw_graph, source=curr_pe, target=mems[curr_mem_idx])))
            for child_task in dmg.successors(task):
                data_movement = f"{task} -> {child_task}"
                for ic in ics_in_path:
                    block_to_task_dict[ic].add(data_movement)
                block_to_task_dict[mems[curr_mem_idx]].add(data_movement)

                # round robin memory allocation for data writes
                if method == "greedy_parallel": curr_mem_idx = (curr_mem_idx+1)%len(mems)

    export_task_to_hardware_mapping(block_to_task_dict, out_path, out_prefix, SILENCE)

# Stopgap fix for PE/IC/Mem names not including '.' in the clk scaling factor.
def rename_clk_mult(block:str, df, df_dict, col:bool, idx:bool):
    # Rename column if clock multiplier not float.
    split_block = block.split('_')
    changed = False
    # Ic has *MEM_ic_<bus_width>_<clock_freq>; IP has IP_*_<llp>_<clock_freq>.
    if block.startswith("IP_") or block.startswith("LMEM_ic_") or block.startswith("GMEM_ic_"):
        if not '.' in split_block[3]:
            split_block[3] = str(float(split_block[3]))
            changed = True
    # Mem has *MEM_<clock_freq>.
    elif block.startswith("LMEM_") or block.startswith("GMEM_"):
        if not '.' in split_block[2]:
            split_block[2] = str(float(split_block[2]))
            changed = True
    if changed:
        new_block = '_'.join(split_block)
        if col:
            df.rename(columns={block: new_block}, inplace=True)
        if idx:
            df.rename(index={block: new_block}, inplace=True)
        if df_dict is not None:
            df_dict[new_block] = df_dict.pop(block)
        logger.warning(f"Changed block name in header from {block} -> {new_block}")
        block = new_block
    return block

def parse_task_to_hw_mapping(wrkld_dag_ids:Dict[str,List[int]], in_path:str, in_prefix:str, n_dags:int, out_path:str, out_prefix:str, method:str, SILENCE:bool) -> None:
    """Read the task to HW block mapping CSV file"""
    in_filename = f"{in_path}/{in_prefix} - Task To Hardware Mapping.csv"

    logger.info(f"Reading file: {in_filename}")
    df = pd.read_csv(in_filename)
    df_dict = df.to_dict()

    wrkld_names = wrkld_dag_ids.keys()
    # first, find the final dag_id
    final_dag_id = {}
    for w in wrkld_names:
        final_dag_id[w] = -1
    # assuming DummyLast_* is the last task per DAG
    to_del = []
    for block in df.columns:
        tasks_assigned = df[block].to_list()
        for i, task in enumerate(tasks_assigned): # could be task (for PE) or write data movement (for IC and memory)
            if str(task) == "nan":
                to_del.append((block, i))
            elif task.startswith("DummyLast") and '->' not in task:
                w_matched = None
                for w in wrkld_names:
                    if w in task or w == "miniera":
                        w_matched = w
                        break
                assert w_matched is not None
                dag_id = int(task.split('_')[-1])
                if dag_id > final_dag_id[w_matched]: final_dag_id[w_matched] = dag_id

    if not SILENCE:
        logger.info(f"Final DAG IDs are:")
        pprint(final_dag_id)

    for block, i in to_del:
        del df_dict[block][i]
    
    # Collect data on which blocks tasks/DMs belonging to the last were assigned.
    dag_id_task_dm_block_map:Dict[int,Dict[str,List[str]]] = {} # dag ID -> task/dm -> list of blocks
    for block in df.columns:
        block = rename_clk_mult(block, df, df_dict, col=True, idx=False)

        tasks_assigned = df[block].to_list()
        for task in tasks_assigned: # Could be task (for PE) or write data movement (for IC and memory).
            if str(task) == "nan":
                continue
            if '->' in task: # For ICs and memories.
                src_task, dst_task = task.split(" -> ")
                if src_task == "souurce":
                    dst_task_no_dag_id = '_'.join(dst_task.split('_')[:-1])
                    dag_id = int(dst_task.split('_')[-1])
                    task_or_dm = f"souurce -> {dst_task_no_dag_id}_{dag_id}"
                elif dst_task == "siink":
                    src_task_no_dag_id = '_'.join(src_task.split('_')[:-1])
                    dag_id = int(src_task.split('_')[-1])
                    task_or_dm = f"{src_task_no_dag_id}_{dag_id} -> siink"
                else:
                    src_task_no_dag_id = '_'.join(src_task.split('_')[:-1])
                    dst_task_no_dag_id = '_'.join(dst_task.split('_')[:-1])
                    dag_id = int(src_task.split('_')[-1])
                    assert dag_id == int(dst_task.split('_')[-1]), "Cross-DAG dependencies are not supported"
                    task_or_dm = f"{src_task_no_dag_id}_{dag_id} -> {dst_task_no_dag_id}_{dag_id}"
            else: # For PEs.
                if task == "souurce" or task == "siink":
                    continue
                task_no_dag_id = '_'.join(task.split('_')[:-1])
                dag_id = int(task.split('_')[-1])
                task_or_dm = f"{task_no_dag_id}_{dag_id}"
            # Populate the dict.
            if dag_id not in dag_id_task_dm_block_map: dag_id_task_dm_block_map[dag_id] = {}
            if task_or_dm not in dag_id_task_dm_block_map[dag_id]: dag_id_task_dm_block_map[dag_id][task_or_dm] = []
            dag_id_task_dm_block_map[dag_id][task_or_dm].append(block)

    for w, dag_ids in wrkld_dag_ids.items():
        for dag_id in dag_ids: # All the DAG IDs in workload "w".
            if method == "greedy_parallel":
                ref_dag_id = dag_id % (final_dag_id[w]+1)
            elif method == "serial":
                ref_dag_id = final_dag_id[w]
            else:
                raise NotImplementedError
            for task, blocks in dag_id_task_dm_block_map[ref_dag_id].items():
                if '->' in task: # For ICs and memories.
                    src_task, dst_task = task.split(" -> ")
                    if src_task == "souurce":
                        dst_task_no_dag_id = '_'.join(dst_task.split('_')[:-1])
                        task_or_dm = f"souurce -> {dst_task_no_dag_id}_{dag_id}"
                    elif dst_task == "siink":
                        src_task_no_dag_id = '_'.join(src_task.split('_')[:-1])
                        task_or_dm = f"{src_task_no_dag_id}_{dag_id} -> siink"
                    else:
                        src_task_no_dag_id = '_'.join(src_task.split('_')[:-1])
                        dst_task_no_dag_id = '_'.join(dst_task.split('_')[:-1])
                        task_or_dm = f"{src_task_no_dag_id}_{dag_id} -> {dst_task_no_dag_id}_{dag_id}"
                else: # For PEs.
                    if task == "souurce" or task == "siink":
                        continue
                    task_no_dag_id = '_'.join(task.split('_')[:-1])
                    task_or_dm = f"{task_no_dag_id}_{dag_id}"
                for block in blocks:
                    # Replicate the same mappings for the new DAG ID.
                    if task_or_dm not in df_dict[block].values():
                        df_dict[block][len(df_dict[block].keys())] = task_or_dm
        
    df = pd.DataFrame.from_dict(df_dict)
    outfile = f"{out_path}/{out_prefix} - Task To Hardware Mapping.csv"
    df.to_csv(outfile, index=False)
    logger.info(f"Wrote to file: {outfile}")
        
def parse_system(sys_dims:Tuple[int, int], ppa_dict:Dict[str,Dict[str,Dict[str,float]]], in_path:str, in_prefix:str):
    """Parse the HW configuration of the system."""
    pes = []
    mems = []
    ics = []

    in_filename = f"{in_path}/{in_prefix} - Hardware Graph.csv"
    assert os.path.exists(in_filename), f"{in_filename} not found!"
    with open(in_filename) as f:
        header = f.readlines()[0]
        header = header.rstrip('\n').split(',')
        assert header[0] == "Block Name"
        header = header[1:]
        for block_name in header:
            all_blocks = ppa_dict["perf"].keys()
            found = False
            for b in all_blocks:
                if b in block_name:
                    pes.append(block_name)
                    found = True
                    break
            if not found:
                if "_ic_" in block_name:
                    ics.append(block_name)
                else:
                    mems.append(block_name)
    
        assert len(pes) + len(ics) + len(mems) == len(header), f"Error in parsing file: {in_filename}!"
    
    # Read the adjacency matrix.
    logger.info(f"Reading file: {in_filename}")
    adj_mat = np.loadtxt(open(in_filename, "r"), dtype=str, delimiter=",", skiprows=1, usecols=tuple(range(1, len(header)+1)))
    adj_mat = [np.asarray([1 if y == '1' else 0 for y in x.tolist()], dtype=np.int) for x in adj_mat]
    adj_mat = np.stack(adj_mat, axis=0)
    df = pd.DataFrame(adj_mat, index=header, columns=header)
    for block in df.columns:
        rename_clk_mult(block, df, None, col=True, idx=True)
    hw_graph = nx.from_pandas_adjacency(df)

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Block Name"}, inplace=True)
    df.replace(0, np.nan, inplace=True)
    df.to_csv(in_filename, index=False)

    logger.info(f"Wrote to file: {in_filename}")
    return hw_graph, pes, mems, ics

def gen_system(wrkld:str, top:str, sys_dims:Tuple[int, int], out_path:str, out_prefix:str, gen_mode:str, n_cv:int, n_rad:int, n_vit_mean:float, num_io_tiles:int, pe_mask_dict) -> None:
    """Generates a new hardware graph in a `top` topology specified by `sys_dims` from scratch 
    consisting of CPUs, ICs and one memory.

    Args:
        top (str): topology of system
        sys_dims (Tuple[int, int]): x- and y- dimensions of system
        out_path (str): directory where CSV file will be generated
        out_prefix (str): partial path of CSV file without suffix
    """

    assert top is not None
    assert gen_mode in ["cpu_only", "one_ip_per_task", "one_cpu_per_task"]

    def gen_instance_names(num_pes:int, num_mems:int, num_ics:int):
        pes = [f"A53_{pe_id}" for pe_id in range(num_pes)]
        mems = [f"LMEM_0_1.0_{mem_id}" for mem_id in range(num_mems)]
        ics  = [f"LMEM_ic_0_1.0_{ic_id}" for ic_id in range(num_ics)]
        return pes, mems, ics
    
    logger.warning(f"Assuming {args.num_mems} memories in the seed design")
    num_mems = args.num_mems

    if top == "ring":
        assert gen_mode == "cpu_only"
        assert sys_dims[0] >= 2
        hw_graph = nx.empty_graph()
        num_pes = sys_dims[0] - 1
        num_ics = sys_dims[0]
        pes, mems, ics = gen_instance_names(num_pes, num_mems, num_ics)
        [hw_graph.add_node(pes[pe_id]) for pe_id in range(num_pes)]
        [hw_graph.add_node(mems[mem_id]) for mem_id in range(num_mems)]
        ic_node_labs = {}
        for ic_id in range(sys_dims[0]):
            ic_id_next = (ic_id + 1) % sys_dims[0]  # loop over
            ic_node_labs[ic_id] = ics[ic_id]
            if ic_id < num_pes:
                hw_graph.add_edge(ic_id, f"A53_{ic_id}", weight=1)
            elif ic_id - num_pes < num_mems:
                hw_graph.add_edge(ic_id, f"LMEM_0_1.0_{ic_id - num_pes}", weight=1)
            hw_graph.add_edge(ic_id, ic_id_next, weight=1)

    elif top == "mesh":
        assert gen_mode == "cpu_only", f"{gen_mode} not supported for topology \"{top}\""
        hw_graph = nx.grid_2d_graph(sys_dims[0], sys_dims[1])
        num_pes = sys_dims[0] * sys_dims[1] - num_mems
        num_pes -= num_io_tiles # 1 for I/O tile
        num_ics = sys_dims[0] * sys_dims[1]
        pes, mems, ics = gen_instance_names(num_pes, num_mems, num_ics)
        for io_tile_id in range (num_io_tiles):
            pes.append(f"I/O_{io_tile_id}")
            num_pes += 1

        [hw_graph.add_node(pes[pe_id]) for pe_id in range(num_pes)]
        [hw_graph.add_node(mems[mem_id]) for mem_id in range(num_mems)]
        ic_node_labs = {}
        io_tile_id = 0
        for ic_x_id in range(sys_dims[0]):
            for ic_y_id in range(sys_dims[1]):
                ic_id = ic_y_id * sys_dims[0] + ic_x_id
                ic_node_labs[(ic_x_id, ic_y_id)] = ics[ic_id]
                if num_io_tiles > 0:
                    if ic_id < num_io_tiles:
                        hw_graph.add_edge((ic_x_id, ic_y_id), f"I/O_{io_tile_id}", weight=1); io_tile_id += 1
                    elif ic_id < num_pes:
                        hw_graph.add_edge((ic_x_id, ic_y_id), f"A53_{ic_id - num_io_tiles}", weight=1)
                    elif ic_id - num_pes < num_mems:
                        hw_graph.add_edge((ic_x_id, ic_y_id), f"LMEM_0_1.0_{ic_id - num_pes}", weight=1)
                else:
                    if ic_id < num_pes:
                        hw_graph.add_edge((ic_x_id, ic_y_id), f"A53_{ic_id}", weight=1)
                    elif ic_id - num_pes < num_mems:
                        hw_graph.add_edge((ic_x_id, ic_y_id), f"LMEM_0_1.0_{ic_id - num_pes}", weight=1)

    elif top == "bus":
        hw_graph = nx.empty_graph()
        num_ics = 1
        num_pes = 1
        if gen_mode == "one_ip_per_task":
            pes, mems, ics = gen_instance_names(num_pes, num_mems, num_ics)
            for pe_enabled in pe_mask_dict.values():
                assert pe_enabled
            for i in range(n_cv):
                pes.append(f"IP_CV_1_1.0_{i}")
                num_pes += 1
            for i in range(n_rad):
                pes.append(f"IP_Radar_1_1.0_{i}")
                num_pes += 1
            for i in range(int(n_vit_mean)):
                pes.append(f"IP_Vit_1_1.0_{i}")
                num_pes += 1
        else:
            if gen_mode == "one_cpu_per_task":
                num_pes += n_cv + n_rad + int(n_vit_mean)
            elif gen_mode == "cpu_only":
                pass
            else: raise NotImplementedError
            pes, mems, ics = gen_instance_names(num_pes, num_mems, num_ics)
        [hw_graph.add_node(pes[pe_id]) for pe_id in range(num_pes)]
        [hw_graph.add_node(mems[mem_id]) for mem_id in range(num_mems)]
        ic_id = 0
        ic_node_labs = {ic_id: ics[ic_id]}
        for mem_id in range(num_mems):
            hw_graph.add_edge(mems[mem_id], ic_id, weight=1)
        for pe_id in range(num_pes):
            hw_graph.add_edge(pes[pe_id], ic_id, weight=1)

    hw_graph = nx.relabel_nodes(hw_graph, ic_node_labs, copy=False)
    outfile = f"{out_path}/{out_prefix} - Hardware Graph.csv"
    graph_to_csv(hw_graph, outfile, "Block Name")
    if not SILENCE:
        logger.info(f"Wrote to {outfile}.")
    export_nx_graph(graph=hw_graph, outpath=f"{out_path}/dot_outputs/{wrkld}_hardware_graph", ext="dot")
    if not SILENCE:
        logger.info(f"Wrote to {out_path}/dot_outputs/{wrkld}_hardware_graph.dot.")
    return hw_graph, pes, mems, ics

def gen_dag_arr_times_data(wrkld:str, dag_arr_times:List[float], out_dir:str) -> None:
    """Generates a file containing the mapping of a DAG to the time when it
    actually arrives. Only used if DAGS_AS_SEP_WRKLDS == True."""
    # DAG arrival times file.
    dag_arr_times_fname = f"{out_dir}/{wrkld}_database - DAG Arr Times.csv"
    dag_arr_times_dict = {}

    for id, time in enumerate(dag_arr_times):
        dag_arr_times_dict[id] = time
    df = pd.DataFrame(list(dag_arr_times_dict.items()),columns = ['DAG ID','Time (s)'])
    df.to_csv(dag_arr_times_fname, index=False)
    if not SILENCE:
        logger.info(f"Wrote to file {dag_arr_times_fname}.")

def gen_with_params(args, wrkld:str, top:str, constrain_topology:bool, sys_dims:Tuple[int,int], n_dags:int, dag_inter_arrival_time_mean:float, dag_arr_times:Dict[int, List[float]], n_cv:int, n_rad:int, n_vit_mean:float, n_vit_per_dag:List[int], path:str, output_dot:bool, sys:str, task_alloc_method:str, n_traces:int, budget_scale:Dict[str, float], pe_mask_dict:Dict[str,bool]):
    global def_budgets
    if wrkld == "miniera":
        wrkld_dag_ids = {}
        wrkld_dag_ids[wrkld] = list(range(n_dags))
        for trace_id in range(n_traces):
            wrkld_enh = f"{wrkld}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arrival_time_mean}_ncv_{n_cv}_nrad_{n_rad}_nvit_{n_vit_mean}_trace_{trace_id}"
            logger.info(f"Generating trace {trace_id} files for sys_dims={sys_dims} workload={wrkld_enh}, n_dags={n_dags}, dag_inter_arrival_time_mean={dag_inter_arrival_time_mean}, n_cv={n_cv}, n_rad={n_rad}, n_vit={n_vit_mean}, GRAPH_GEN_MODE={GRAPH_GEN_MODE}, ACCEL_SUPPORTS_LLP={ACCEL_SUPPORTS_LLP}.")
            era_context = ERAContext(wrkld_enh, ppa_dict, pe_mask_dict, n_dags, dag_arr_times[trace_id], n_cv, n_rad, n_vit_per_dag)
            era_context.gen_task_graph()

            if constrain_topology:
                wrkld_enh = f"{wrkld}_soc_{sys_dims[0]}x{sys_dims[1]}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arrival_time_mean}_ncv_{n_cv}_nrad_{n_rad}_nvit_{n_vit_mean}_trace_{trace_id}"
                wrkld_enh_no_trace_id = f"{wrkld}_soc_{sys_dims[0]}x{sys_dims[1]}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arrival_time_mean}_ncv_{n_cv}_nrad_{n_rad}_nvit_{n_vit_mean}"
            else:
                wrkld_enh = f"{wrkld}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arrival_time_mean}_ncv_{n_cv}_nrad_{n_rad}_nvit_{n_vit_mean}_trace_{trace_id}"
                wrkld_enh_no_trace_id = f"{wrkld}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arrival_time_mean}_ncv_{n_cv}_nrad_{n_rad}_nvit_{n_vit_mean}"
            # Generate/parse the hardware graph and map tasks onto it.
            if sys == "gen":
                hw_graph, pes, mems, ics = gen_system(wrkld=wrkld_enh, top=top, sys_dims=sys_dims, out_path=path, out_prefix=f"{wrkld_enh_no_trace_id}_database", gen_mode=args.gen_mode, n_cv=n_cv, n_rad=n_rad, n_vit_mean=n_vit_mean, num_io_tiles=args.num_io_tiles, pe_mask_dict=pe_mask_dict)
                gen_task_to_hw_mapping(hw_graph=hw_graph, dmgs=[dfg.dmG for dfg in era_context.dfgs], pes=pes, mems=mems, ics=ics, out_path=path, out_prefix=f"{wrkld_enh_no_trace_id}_database", method=task_alloc_method)
            elif sys == "parse":
                hw_graph, pes, mems, ics = parse_system(sys_dims=sys_dims, ppa_dict=ppa_dict, in_path=path, in_prefix=f"{wrkld_enh_no_trace_id}_database")
                parse_task_to_hw_mapping(wrkld_dag_ids=wrkld_dag_ids, in_path=path, in_prefix=f"{wrkld_enh_no_trace_id}_database", n_dags=n_dags, out_path=path, out_prefix=f"{wrkld_enh_no_trace_id}_database", method=task_alloc_method, SILENCE=SILENCE)
            else:
                assert False
            era_context.opt_and_export(path, output_dot, logger)
            del era_context

        # Update the block characteristics file.
        # TODO should probably move this to DFG.py
        blk_char_fname_src = f"{FARSIRoot}/specs/database_data/parsing/inputs/misc_database - Block Characteristics.{wrkld}.csv"
        blk_char_fname_dst = f"{path}/misc_database - Block Characteristics.{wrkld}.csv"
        if not SILENCE:
            logger.info(f"Copied: {blk_char_fname_src} -> {blk_char_fname_dst}")
        blk_char_lfname = blk_char_fname_dst.replace(".csv", ".lock")
        if not SILENCE:
            logger.info(f"Waiting to get lock on file: {blk_char_lfname}")
        with FileLock(blk_char_lfname):
            try:
                df = pd.read_csv(blk_char_fname_dst, dtype=str, keep_default_na=False)
            except:
                df = pd.read_csv(blk_char_fname_src, dtype=str, keep_default_na=False)
            assert df.at[0, 'Name'] == "A53"
            df.at[0, 'dhrystone_IPC'] = str(CPU_IPC)
            df.at[0, 'Gpp_area'] = str(power_area_dict["area"]["A53"]*1e-6*1e-6) # Convert to mm2.
            df.at[0, 'Inst_per_joul'] = str(CPU_IPC*float(CLK_HZ)/power_area_dict["dynamic_power"]["A53"]) # = inst/(W.s) = (inst/cyc*cyc/s)/W = IPC*clk_freq/W.
            # Update clock frequencies for all blocks to be the same for now.
            for i in range(len(df.index)):
                df.at[i, "Freq"] = str(CLK_HZ)
            df.to_csv(blk_char_fname_dst, index=False)
    else:
        raise NotImplementedError

    for trace_id in range(n_traces):
        if wrkld == "miniera":
            wrkld_enh = f"{wrkld}_numDags_{n_dags}_dagInterArrTime_{dag_inter_arrival_time_mean}_ncv_{n_cv}_nrad_{n_rad}_nvit_{n_vit_mean}_trace_{trace_id}"
            wrklds = [wrkld_enh]
        else:
            raise NotImplementedError
        for wrkld_enh in wrklds:
            # Add workload's last tasks to misc file.
            last_tasks_fname_dst = f"{path}/misc_database - Last Tasks.csv"
            last_tasks_fname_src = f"{FARSIRoot}/specs/database_data/parsing/inputs/misc_database - Last Tasks.csv"
            last_tasks_lfname = last_tasks_fname_dst.replace(".csv", ".lock")
            if not SILENCE:
                logger.info(f"Waiting to get lock on file: {last_tasks_lfname}")
            with FileLock(last_tasks_lfname):
                try:
                    df = pd.read_csv(last_tasks_fname_dst, header=0)
                except:
                    df = pd.read_csv(last_tasks_fname_src, header=0)
                last_tasks_dict = df.set_index('workload').T.to_dict('records')[0]
                for dag_id in range(len(dag_arr_times[trace_id])):
                    last_tasks_dict[f"{wrkld_enh}_{dag_id}"] = f"DummyLast_{dag_id}"
                df = pd.DataFrame.from_dict(last_tasks_dict, orient='index')
                df = df.reset_index().rename(columns={'index': 'workload', 0: 'last_task'})
                df.to_csv(last_tasks_fname_dst, index=False)
                logger.info(f"Wrote to file: {last_tasks_fname_dst}")
            
            # Add workload's budgets to misc file.
            budget_fname_src = f"{FARSIRoot}/specs/database_data/parsing/inputs/misc_database - Budget.csv"
            budget_fname_dst = f"{path}/misc_database - Budget.csv"
            budget_lfname = budget_fname_dst.replace(".csv", ".lock")
            if not SILENCE:
                logger.info(f"Waiting to get lock on file: {budget_lfname}")
            with FileLock(budget_lfname):
                try:
                    df = pd.read_csv(budget_fname_dst, header=0)
                except:
                    df = pd.read_csv(budget_fname_src, header=0)
                budgets_dict = df.set_index('Workload').T.to_dict('index')
                for dag_id in range(len(dag_arr_times[trace_id])):
                    for k, v in def_budgets[wrkld_enh.split('_')[0]].items():
                        v *= budget_scale[k]
                        budgets_dict[k][f"{wrkld_enh}_{dag_id}"] = v # TODO support different budgets for each DAG (from an input file)
                        if k == "power" or k == "area" or k == "cost":
                            budgets_dict[k]["all"] = v
                df = pd.DataFrame.from_dict(budgets_dict, orient='index').T
                df = df.reset_index().rename(columns={'index': 'Workload'})
                df.to_csv(budget_fname_dst, index=False)
                logger.info(f"Wrote to file: {budget_fname_dst}")

def main_det(args, out_path:str, top:str, constrain_topology:bool, sys_dims:Tuple[int,int], wrkld:str, n_dags:int, dag_inter_arrival_time_all:List[float], n_cv_all:List[int], n_rad_all:List[int], n_vit_all:List[float], output_dot:bool, sys:str, task_alloc_method:str, budget_scale:Dict[str, float], pe_mask_dict:Dict[str,bool]):
    """Main function for deterministic DAG arrivals"""
    for dag_inter_arrival_time in dag_inter_arrival_time_all:
        dag_arr_times:Dict[int, List[float]] = {0: [0.]}
        for i in range(n_dags - 1):
            dag_arr_times[0].append(dag_arr_times[0][-1] + dag_inter_arrival_time)
        assert len(dag_arr_times[0]) == n_dags
        if wrkld == "miniera":
            for n_cv in n_cv_all:
                for n_rad in n_rad_all:
                    for n_vit in n_vit_all:
                        n_vit_per_dag = [int(n_vit)] * n_dags
                        assert len(n_vit_per_dag) == n_dags
                        if n_dags == 1:
                            gen_with_params(args, wrkld, top, constrain_topology, sys_dims, n_dags, 0.0, dag_arr_times, n_cv, n_rad, float(n_vit), n_vit_per_dag, out_path, output_dot, sys, task_alloc_method, 1, budget_scale, pe_mask_dict)
                        else:
                            gen_with_params(args, wrkld, top, constrain_topology, sys_dims, n_dags, dag_inter_arrival_time, dag_arr_times, n_cv, n_rad, float(n_vit), n_vit_per_dag, out_path, output_dot, sys, task_alloc_method, 1, budget_scale, pe_mask_dict)
        else:
            if n_dags == 1:
                gen_with_params(args, wrkld, top, constrain_topology, sys_dims, n_dags, 0.0, dag_arr_times, None, None, None, None, out_path, output_dot, sys, task_alloc_method, 1, budget_scale, pe_mask_dict)
            else:
                gen_with_params(args, wrkld, top, constrain_topology, sys_dims, n_dags, dag_inter_arrival_time, dag_arr_times, None, None, None, None, out_path, output_dot, sys, task_alloc_method, 1, budget_scale, pe_mask_dict)

def main_prob(args, out_path:str, top:str, constrain_topology:bool, sys_dims:Tuple[int,int], wrkld:str, n_dags:int, dag_inter_arrival_time_mean_all:List[float], dag_inter_arrival_time_cap:float, n_cv_all:List[int], n_rad_all:List[int], n_vit_mean_all:List[float], n_vit_cap:int, output_dot:bool, sys:str, task_alloc_method:str, budget_scale:Dict[str, float], n_traces:int, pe_mask_dict:Dict[str,bool]):
    """Main function for probabilistic DAG arrivals"""
    dag_arr_times_all:Dict[float, Dict[int, List[float]]] = {}

    for dag_inter_arrival_time_mean in dag_inter_arrival_time_mean_all:
        dag_arr_times_all[dag_inter_arrival_time_mean] = {}
        for trace_id in range(n_traces):
            # Exponential distributed arrival times with dag_inter_arrival_time_mean as the parameter.
            dag_arr_time_deltas = np.random.exponential(scale=dag_inter_arrival_time_mean, size=n_dags-1)
            # Cap off if asked to by the user.
            if dag_inter_arrival_time_cap != None:
                assert dag_inter_arrival_time_cap > 0.
                for i in range(n_dags-1):
                    if dag_arr_time_deltas[i] < dag_inter_arrival_time_cap:
                        dag_arr_time_deltas[i] = dag_inter_arrival_time_cap
            dag_arr_times_all[dag_inter_arrival_time_mean][trace_id] = [0.] + np.cumsum(dag_arr_time_deltas).tolist()
            assert len(dag_arr_times_all[dag_inter_arrival_time_mean][trace_id]) == n_dags

    # Generate number of viterbis per DAG.
    n_vit_per_dag_all:Dict[float, List[int]] = {}
    for n_vit_mean in n_vit_mean_all:
        n_vit_per_dag = np.random.poisson(lam=n_vit_mean, size=n_dags)
        # Cap off if asked to by the user.
        if n_vit_cap != None:
            for dag_id in range(n_dags):
                if n_vit_per_dag[dag_id] > n_vit_cap:
                    n_vit_per_dag[dag_id] = n_vit_cap
        n_vit_per_dag_all[n_vit_mean] = n_vit_per_dag

    for dag_inter_arrival_time_mean in dag_inter_arrival_time_mean_all:
        dag_arr_times = dag_arr_times_all[dag_inter_arrival_time_mean]
        for trace_id in range(n_traces):
            assert len(dag_arr_times[trace_id]) == n_dags
        for n_cv in n_cv_all:
            for n_rad in n_rad_all:
                for n_vit_mean in n_vit_mean_all:
                    n_vit_per_dag = n_vit_per_dag_all[n_vit_mean]
                    assert len(n_vit_per_dag) == n_dags
                    if n_dags == 1:
                        gen_with_params(args, wrkld, top, constrain_topology, sys_dims, n_dags, -1, dag_arr_times, n_cv, n_rad, n_vit_mean, n_vit_per_dag, out_path, output_dot, sys, task_alloc_method, n_traces, budget_scale)
                    else:
                        gen_with_params(args, wrkld, top, constrain_topology, sys_dims, n_dags, dag_inter_arrival_time_mean, dag_arr_times, n_cv, n_rad, n_vit_mean, n_vit_per_dag, out_path, output_dot, sys, task_alloc_method, n_traces, budget_scale)

if __name__ == "__main__":
    global args
    args = parse_args(("miniera", )) # This script is a demonstrator for the ERA workload.
    logger = setup_logger('MyLogger')
    logger.info(f"Generator script called with arguments: {args}")
    os.makedirs(args.out_path, exist_ok=True)
    SILENCE = args.silence

    CLK_HZ  = clock_speed[args.workload]
    CPU_IPC = cpu_ipc[args.workload]

    for io_tile_id in range(args.num_io_tiles):
        ppa_dict["static_power"][f"I/O_{io_tile_id}"] = {}
        ppa_dict["dynamic_energy"][f"I/O_{io_tile_id}"] = {}
        ppa_dict["perf"][f"I/O_{io_tile_id}"] = {}
        ppa_dict["area"][f"I/O_{io_tile_id}"] = {}
        pe_mask_dict[f"I/O_{io_tile_id}"] = True

    if args.budget_scales == None:
        budget_scale = {"latency": 1., "power": 1., "area": 1., "cost": 1.}
    else:
        assert len(args.budget_scales) == 3, "must specify budgets in the order: latency, power and area"
        budget_scale = {"latency": args.budget_scales[0], "power": args.budget_scales[1], "area": args.budget_scales[2], "cost": 1.}

    if args.sys == "gen":
        assert args.gen_mode

    if args.mode == "prob":
        if args.workload == "miniera":
            assert args.dag_inter_arrival_time_mean_all is not None and args.num_viterbi_mean_all is not None and args.dag_inter_arrival_time_all is None and args.num_viterbi_all is None, \
                "You must specify --num-viterbi-mean-all (and --dag-inter-arrival-time-mean-all) instead of --num-viterbi-all (and --dag-inter-arrival-time-all) when --mode is \"prob\""
        main_prob(args, args.out_path, args.top, args.constrain_topology, (args.sys_dim_x, args.sys_dim_y), args.workload, args.num_dags, args.dag_inter_arrival_time_mean_all, args.dag_inter_arrival_time_cap, args.num_cv_all, args.num_radar_all, args.num_viterbi_mean_all, args.num_viterbi_cap, args.output_dot, args.sys, args.task_alloc_method, budget_scale, args.num_traces, pe_mask_dict)
    elif args.mode == "det":
        assert args.num_traces == 1
        if args.workload == "miniera":
            assert args.dag_inter_arrival_time_mean_all is None and args.num_viterbi_mean_all is None and args.dag_inter_arrival_time_all is not None and args.num_viterbi_all is not None, \
                "You must specify --num-viterbi-all (and --dag-inter-arrival-time-all) instead of --num-viterbi-mean-all (and --dag-inter-arrival-time-mean-all) when --mode is \"det\""
        main_det(args, args.out_path, args.top, args.constrain_topology, (args.sys_dim_x, args.sys_dim_y), args.workload, args.num_dags, args.dag_inter_arrival_time_all, args.num_cv_all, args.num_radar_all, args.num_viterbi_all, args.output_dot, args.sys, args.task_alloc_method, budget_scale, pe_mask_dict)
    else:
        raise NotImplementedError
    logger.info(f"All files generated.")
