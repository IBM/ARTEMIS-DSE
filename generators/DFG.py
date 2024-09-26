# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
from typing import List

import networkx as nx
import pandas as pd
import numpy as np
from helper import graph_to_csv, export_nx_graph

class DFG:
    dummy_node_types = ["souurce", "siink", "DummyLast"]
    dot_node_label = True
    dot_edge_label = True

    def __init__(self, ctx, wrkld:str, ppa_dict:dict, pe_mask_dict:dict, silence:bool=False, dag_arr_times:List[float]=None):
        self.ctx = ctx
        self.wrkld = wrkld
        assert ppa_dict
        self.ppa_dict = ppa_dict
        assert pe_mask_dict
        self.pe_mask_dict = pe_mask_dict
        self.dmG = nx.DiGraph(directed=True) # Data movement graph.
        self.dag_arr_times = dag_arr_times
        self.op_cntr_dict = {}
        self.task_itr_count_dict = {}
        self.silence = silence

    @staticmethod
    def is_node_dummy(node_name) -> bool:
        for dummy_node_name in DFG.dummy_node_types:
            if dummy_node_name in node_name:
                return True
        return False

    def create_node(self, node_type:str, bcet:float=0., llp:int=1, id=None):
        """Creates and appends to the parent graph a node of type `node_type` with loop-level parallelism of `llp`.
        An ID can be explicitly specified to append as suffix to the node name."""
        assert bcet != np.inf
        # node_type_no_suff = '_'.join(node_type.split('_')[:-2])
        # assert node_type_no_suff in DFG.node_types or node_type_no_suff in DFG.dummy_node_types, f"Node type {node_type} not found in list"
        if DFG.is_node_dummy(node_type):
            node_name = node_type
        else:
            if node_type not in self.op_cntr_dict:
                self.op_cntr_dict[node_type] = 0
            if id == None:
                id = self.op_cntr_dict[node_type]
            node_name = f"{node_type}_{id}"
            self.op_cntr_dict[node_type] += 1
        assert not self.dmG.has_node(node_name), f"Node {node_name} already exists!"
        if DFG.dot_node_label:
            label = f"{node_name} ({llp})"
            if self.dag_arr_times != None:
                if not DFG.is_node_dummy(node_name):
                    dag_id = int(node_name.split('_')[-1])
                    dag_arr_time = self.dag_arr_times[dag_id]
                    label += f"[{dag_arr_time}]"
            self.dmG.add_node(node_name, label=label, bcet=bcet, sr=1., sdr=1.)
        else:
            self.dmG.add_node(node_name, bcet=bcet, sr=1., sdr=1.)
        self.task_itr_count_dict[node_name] = llp
        return node_name

    def create_edge(self, src_node_name:str, dst_node_name:str, weight:float, bcet:float=0., suff:str=''):
        """Creates and appends to the parent graph an edge between `src_node_name` and `dst_node_name` with weight `weight_src`."""
        assert bcet != None
        assert not self.dmG.has_edge(src_node_name, dst_node_name), f"Node from {src_node_name} to {dst_node_name} already exists!"
        assert src_node_name in self.dmG.nodes, f"Source node {src_node_name} not found in added nodes"
        assert dst_node_name in self.dmG.nodes, f"Destination node {dst_node_name} not found in added nodes"
        if DFG.dot_edge_label:
            self.dmG.add_edge(src_node_name, dst_node_name, weight=weight, label=weight, type=suff, bcet=bcet)
        else:
            self.dmG.add_edge(src_node_name, dst_node_name, weight=weight, bcet=bcet)

    def create_edge_w_souurce_conn(self, src_node_name:str, dst_node_name:str, weight_src:float, weight_souurce:float, souurce_node_name:str="souurce"):
        """Creates and appends to the parent graph an edge between `src_node_name` and `dst_node_name` with weight `weight_src`
            and another one between `src_node_name` and `souurce_node_name` with weight `weight_souurce`.
                weight_src = data transferred from preceding node (e.g. polynomial coefficients).
                weight_souurce = data transferred from local source node (e.g. twiddle factors)."""
        # If src is the same as "souurce", then add the weights up.
        if src_node_name == souurce_node_name:
            self.create_edge(src_node_name, dst_node_name, weight_souurce+weight_src)
        # Otherwise, create two separate edges.
        else:
            self.create_edge(src_node_name, dst_node_name, weight_src)
            self.create_edge(souurce_node_name, dst_node_name, weight_souurce, '[S]')

    def create_source_node(self, souurce_node_name="souurce"):
        self.create_node(souurce_node_name)

    def create_sink_node(self, dummy_node_name="DummyLast", siink_node_name="siink"):
        _ = self.create_node(dummy_node_name)
        _ = self.create_node(siink_node_name)
        self.create_edge(dummy_node_name, siink_node_name, 1)

    """Generates a file containing the mapping of a task to the time when its parent DAG
    actually arrives. Optional if dag_arr_times is not specified during construction."""
    def gen_dag_arr_times_data(self, out_dir:str, logger):
        # DAG arrival times file.
        dag_arr_times_fname = f"{out_dir}/{self.wrkld}_database - Parent Graph Arr Times.csv"
        dag_arr_times_dict = {}
        for node_name in self.dmG.nodes():
            # Add dummy time for dummy nodes.
            # for dummy_node in DFG.dummy_node_types:
            #     if dummy_node in node_name:
            #         continue
            if DFG.is_node_dummy(node_name):
                dag_arr_time = 0.
            else:
                dag_id = int(node_name.split('_')[-1])
                dag_arr_time = self.dag_arr_times[dag_id]
            dag_arr_times_dict[node_name] = dag_arr_time
        df = pd.DataFrame(list(dag_arr_times_dict.items()),columns = ['Task Name','Time (s)'])
        df.to_csv(dag_arr_times_fname, index=False)
        if not self.silence:
            logger.info(f"Wrote to {dag_arr_times_fname}.")

    def gen_dm_itr_cnt_data(self, out_dir, logger):
        # Data movement file.
        dm_fname = f"{out_dir}/{self.wrkld}_database - Task Data Movement.csv"
        graph_to_csv(self.dmG, dm_fname, "Task Name")
        logger.info(f"Wrote to {dm_fname}.")

        # Task iteration count file.
        itr_cnt_fname = f"{out_dir}/{self.wrkld}_database - Task Itr Count.csv"
        # print(self.task_itr_count_dict)
        itr_cnt_df = pd.DataFrame([self.task_itr_count_dict]).T
        itr_cnt_df.columns = ["number of iterations"]
        itr_cnt_df.index.name = "Task Name"
        itr_cnt_df.to_csv(itr_cnt_fname)
        if not self.silence:
            logger.info(f"Wrote to {itr_cnt_fname}.")

    def extract_params(self, node_name:str):
        raise NotImplementedError

    # def max_length(G, node_run='souurce'):
    #     leaf_nodes = [node for node in G.nodes() if G.out_degree(node)==0]
    #     time_dict = nx.get_node_attributes(G, 'wcet')

    #     max_path_length = 0
    #     num_paths = 0

    #     for leaf in leaf_nodes:
    #         for path in nx.all_simple_paths(G, source=node_run, target=leaf):
    #             sum = 0
    #             for key in path:
    #                 sum += int(time_dict[key])

    #             if (sum > max_path_length):
    #                 max_path_length = sum
    #             num_paths += 1
    #     if(num_paths == 0):
    #         min_time = int(time_dict[node])
    #         return min_time

    #     return max_path_length

    #Calculate SDR data
    def gen_task_deadline_ratio(self, out_dir:str, logger):
        sink_nodes = [node for node, outdegree in self.dmG.out_degree(self.dmG.nodes()) if outdegree == 0]
        source_nodes = [node for node, indegree in self.dmG.in_degree(self.dmG.nodes()) if indegree == 0]
        for source, sink in [(source, sink) for sink in sink_nodes for source in source_nodes]:
            for path in nx.all_simple_paths(self.dmG, source=source, target=sink):
                
                # Slack ratio calculation for MS_DYN
                for nid, node in enumerate(path):
                    # print(nid, node, path)
                    path_time = sum([self.dmG.nodes[xnode]["bcet"] for xnode in path[nid:]])
                    if (path_time != 0):
                        bcet = self.dmG.nodes[node]["bcet"]
                        sr = bcet/path_time
                        self.dmG.nodes[node]["sr"] = min(sr, self.dmG.nodes[node]["sr"])
                    else:
                        self.dmG.nodes[node]["sr"] = 0.

                # Sub deadline ratio calculation for MS STAT
                path_time = sum([self.dmG.nodes[node]["bcet"] for node in path])
                for node in path:
                    bcet = self.dmG.nodes[node]["bcet"]
                    sdr = bcet/path_time
                    self.dmG.nodes[node]["sdr"] = min(sdr, self.dmG.nodes[node]["sdr"])
        
        sr_fname = f"{out_dir}/{self.wrkld}_database - Task SR.csv"
        sr_dict = {}
        sdr_fname = f"{out_dir}/{self.wrkld}_database - Task SDR.csv"
        sdr_dict = {}
        for node_name in self.dmG.nodes():
            sr_dict[node_name] = float(self.dmG.nodes[node_name]["sr"])
            sdr_dict[node_name] = float(self.dmG.nodes[node_name]["sdr"])
            
        df_sr = pd.DataFrame(list(sr_dict.items()),columns = ['Task Name','Slack ratio'])
        df_sr.to_csv(sr_fname, index=False)

        df_sdr = pd.DataFrame(list(sdr_dict.items()),columns = ['Task Name','Sub-deadline ratio'])
        df_sdr.to_csv(sdr_fname, index=False)
        if not self.silence:
            logger.info(f"Wrote to {sr_fname}.")
            logger.info(f"Wrote to {sdr_fname}.")

    # Populate performance, power and area data.
    def gen_ppa_data(self, out_dir, logger):
        base_fname = f"{self.wrkld}_database - Task PE "
        fname = {
            "perf"   : base_fname + "Performance",
            "dynamic_energy" : base_fname + "Energy",
            "static_power" : base_fname + "Static Power",
            "area"   : base_fname + "Area"
        }

        # Get the list of PE names to be used for DSE in this system.
        pe_names = list(filter(lambda pe: self.pe_mask_dict[pe], self.ppa_dict["perf"].keys()))
        data_rows = {}
        for metric in fname.keys():
            data_rows[metric] = [','.join(["Task Name"]+pe_names)+'\n']
            # Iterate over all node names.
            for node_name in sorted(self.task_itr_count_dict.keys()):
                if '_' in node_name:
                    node_name_no_params = node_name.split('_')[0]   # e.g. ModSub_N32768l2_0 -> ModSub
                else:
                    node_name_no_params = node_name # e.g. LocSink_1 -> LocSink
                data_row = f"{node_name}"
                for pe in pe_names:
                    val = ''
                    logger.debug(f"Parsing for metric {metric}, node name {node_name}, PE {pe}")
                    if node_name_no_params not in self.ppa_dict[metric][pe]:
                        logger.debug(f"Skipping {node_name_no_params} ({node_name}) for PE {pe}")
                        # Set PPA values for dummy nodes to 0.
                        # print(node_name, node_name_no_params)
                        if (DFG.is_node_dummy(node_name) or node_name in ["LocSource", "LocSink"]) and 'IP' not in pe and (not pe.startswith("I/O")):
                            val = 0
                    else:
                        # TODO virtualize this part of the function
                        eq_fn = self.ppa_dict[metric][pe][node_name_no_params]
                        logger.debug(f"Extracted analytical model {eq_fn} for metric {metric}, pe {pe}, node_name {node_name_no_params}")
                        if eq_fn == 0:
                            val = 0
                        else:
                            params = self.ctx.extract_params(node_name)
                            if params == None:
                                val = eq_fn
                            else:
                                val = eq_fn(*params)
                            logger.debug(f"val = {val} for params = {params}")
                    data_row += f",{val}"
                data_row += f"\n"
                data_rows[metric].append(data_row)

        for metric in ["perf", "dynamic_energy", "static_power", "area"]:
            curr_fname = f"{out_dir}/{fname[metric]}.csv"
            with open(curr_fname, 'w') as result_file:
                result_file.writelines(data_rows[metric])
                if not self.silence:
                    logger.info(f"Wrote to {curr_fname}.")

    def remove_local_sources_sinks(self):
        orig_node_list = deepcopy(self.dmG.nodes())
        edge_weight = nx.get_edge_attributes(self.dmG, 'weight')
        edge_wei_type = nx.get_edge_attributes(self.dmG, 'type')
        for node_name in orig_node_list:
            node_name_no_params = node_name.split('_')[0]
            if node_name_no_params == "LocSource":
                parents = list(self.dmG.predecessors(node_name))
                assert parents

                # Prune local nodes with one parent only.
                if len(parents) == 1 and edge_weight[(parents[0], node_name)] == 1:
                    # Replicate edges from node_name to children to parent[0] to children.
                    out_edges_list = list(self.dmG.out_edges(node_name))
                    for out_edge in out_edges_list:
                        self.create_edge(parents[0], out_edge[1], edge_weight[out_edge], edge_wei_type[out_edge])
                        self.dmG.remove_edge(out_edge[0], out_edge[1])
                    # Remove the current node.
                    self.dmG.remove_node(node_name)

                # Remove corresponding nodes from Task Itr Count dict.
                del self.task_itr_count_dict[node_name]

            elif node_name_no_params == "LocSink":
                children = list(self.dmG.successors(node_name))
                assert children

                # Prune local nodes with one child only.
                if len(children) == 1 and edge_weight[(node_name, children[0])] == 1:
                    # Replicate edges from node_name to children to parent[0] to children.
                    in_edges_list = list(self.dmG.in_edges(node_name))
                    for in_edge in in_edges_list:
                        self.create_edge(in_edge[0], children[0], edge_weight[in_edge], edge_wei_type[in_edge])
                        self.dmG.remove_edge(in_edge[0], in_edge[1])
                    # Remove the current node.
                    self.dmG.remove_node(node_name)

                # Remove corresponding nodes from Task Itr Count dict.
                del self.task_itr_count_dict[node_name]

        del orig_node_list

    def move_static_edges_src_to_souurce(self, souurce_node_name:str="souurce"):
        edge_weight = nx.get_edge_attributes(self.dmG, 'weight')
        edge_wei_type = nx.get_edge_attributes(self.dmG, 'type')
        orig_edge_list = deepcopy(self.dmG.edges())
        # Remove nodes and edges from data movement DFG.
        for edge in orig_edge_list:
            parent_node_name_no_params = edge[0].split('_')[0]
            if edge_wei_type[edge] == '[S]' and parent_node_name_no_params == "LocSource":
                dprint(f"{edge}, {edge_wei_type[edge]}, {edge_weight[edge]}")
                if parent_node_name_no_params == "LocSource":
                    self.create_edge(souurce_node_name, edge[1], edge_weight[edge], edge_wei_type[edge])
                    self.dmG.remove_edge(edge[0], edge[1])
        del orig_edge_list

    def connect_souurce(self, node_conn_to_souurce:str):
        self.create_edge("souurce", node_conn_to_souurce, 1)

    def connect_siink(self, node_conn_to_dummy_last:str, weight:float=1, dummy_node_name="DummyLast"):
        self.create_edge(node_conn_to_dummy_last, dummy_node_name, weight)

    def run_optimizer(self):
        self.move_static_edges_src_to_souurce()
        self.remove_local_sources_sinks()

    def opt_and_export(self, csv_out_dir:str, output_dot:bool, logger):
        # export_nx_graph(graph=self.dmG, wrkld=self.wrkld, suffix="data_movement", ext="dot") # Output DOT file prior to running the graph optimizations.
        self.run_optimizer()
        if output_dot:
            export_nx_graph(graph=self.dmG, outpath=f"{csv_out_dir}/dot_outputs/{self.wrkld}_data_movement", ext="dot")
            if not self.silence:
                logger.info(f"Wrote to {csv_out_dir}/dot_outputs/{self.wrkld}_data_movement.dot.")
        
        self.gen_task_deadline_ratio(csv_out_dir, logger)

        # Export CSV files for FARSI.
        self.gen_dm_itr_cnt_data(csv_out_dir, logger)
        self.gen_ppa_data(csv_out_dir, logger)
        if self.dag_arr_times != None:
            self.gen_dag_arr_times_data(csv_out_dir, logger)
