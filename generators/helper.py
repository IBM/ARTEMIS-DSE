# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import networkx as nx
import pandas as pd

import logging
from colorama import Fore, Style

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Define format that includes the time in HH:MM format
    formatter = logging.Formatter('[%(levelname)s] %(message)s')

    # Customizing the format to include colors
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            log_colors = {
                logging.DEBUG: Fore.CYAN,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.MAGENTA
            }
            levelname_color = log_colors.get(record.levelno, "")
            record.levelname = f"{levelname_color}{record.levelname}{Style.RESET_ALL}"
            return super().format(record)
    
    ch.setFormatter(CustomFormatter('[%(levelname)s] %(message)s'))
    
    # Add handler to the logger
    logger.addHandler(ch)
    return logger

def replace_tok(string:str, tok_tgt:str, val) -> None:
    found = False
    string = string.split("_")
    for i, tok in enumerate(string):
        if tok == tok_tgt:
            string[i+1] = str(val)
            found = True
            break
    assert found
    return "_".join(string)

def get_tok(string:str, tok_tgt:str) -> str:
    string = string.split("_")
    for i, tok in enumerate(string):
        if tok == tok_tgt:
            return string[i+1]
    assert False

def parse_wrkld_specific_args(parser:argparse.ArgumentParser):
    parser.add_argument("--num-cv-all", type=int, nargs='+', default=[1, 1],
            help="number of CV tasks in each DAG (out of <num-dags> DAGs)")
    parser.add_argument("--num-radar-all", type=int, nargs='+', default=[1, 1],
            help="number of radar tasks in each DAG (out of <num-dags> DAGs)")
    parser.add_argument("--sys-dim-x", "-x", type=int, default=2,
            help="x-dimension of the system's mesh topology")
    parser.add_argument("--sys-dim-y", "-y", type=int, default=2,
            help="y-dimension of the system's mesh topology")
    parser.add_argument("--num-dags", "-ndags", type=int, default=2,
            help="number of DAGs")
    parser.add_argument("--out-path", type=str, default="./outputs",
            help="path where generated files will be saved")
    parser.add_argument("--top", "-t", type=str, default="bus", choices=("ring", "bus", "mesh"),
            help="system topology")
    parser.add_argument("--mode", "-m", type=str, default="det", choices=("prob", "det"),
            help="run in probabilistic or deterministic mode")
    parser.add_argument("--constrain-topology", type=int, default=0,
            help="constrain the topology of the SoC to soc_dim[0]xsoc_dim[1]")
    parser.add_argument("--silence", "-s", action="store_true", default=False,
            help="turn off verbose prints")
    parser.add_argument("--num-traces", type=int, default=1,
            help="the number of unique traces to be generated")
    # Deterministic DAG arrivals.
    parser.add_argument("--num-viterbi-all", type=float, nargs='+', default=[2, 2],
            help="number of Viterbi tasks in each DAG (out of <num-dags> DAGs)")
    parser.add_argument("--dag-inter-arrival-time-all", type=float, nargs='+', default=[0, .5],
            help="DAG inter-arrival time for each sweep run")
    # Probabilistic DAG arrivals.
    parser.add_argument("--num-viterbi-mean-all", type=float, nargs='+',
            help="expected number of Viterbi tasks between DAG arrivals")
    parser.add_argument("--num-viterbi-cap", type=int, default=None,
            help="optional cap for max number of Viterbi tasks")
    parser.add_argument("--dag-inter-arrival-time-mean-all", type=float, nargs='+',
            help="mean DAG inter-arrival time for each sweep run")
    parser.add_argument("--dag-inter-arrival-time-cap", type=float, default=None,
            help="optional cap for min inter-arrival time of DAGs")
    parser.add_argument("--num-mems", type=int, default=1,
            help="number of memories in case of constrained topology gen design")
    parser.add_argument("--num-io-tiles", type=int, default=0,
            help="add dummy I/O tile(s) (\"PE\") to the design")

    parser.add_argument("--budget-scales", type=float, nargs='+',
            help="budget scaling factors for latency, power and area in that order")

    parser.add_argument("--gen-mode", type=str, choices=("cpu_only", "one_cpu_per_task", "one_ip_per_task", ''), default='cpu_only',
            help="initial list of PE blocks in the system")

def parse_fhe_args(parser:argparse.ArgumentParser):
    parser.add_argument("-N", type=int, default=2**14,
            help="ciphertext polynomial degree")
    parser.add_argument("-L", type=int, default=35,
            help="multiplicative depth")
    parser.add_argument("-l", type=int, default=35,
            help="number of primes in current level")
    parser.add_argument("-W", type=int, default=8,
            help="machine word size in bytes")
    parser.add_argument("--out-path", type=str, default="./",
            help="path where generated files will be saved")

def parse_args(wrkld_choices:tuple):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", "-w", type=str, default=wrkld_choices[0], choices=wrkld_choices,
            help="name of the workload to generate simulation inputs for")
    parser.add_argument("--verbose", action="store_true", default=False,
            help="machine word size in bytes")
    parser.add_argument("--sys", type=str, default="gen", choices=("gen", "parse"),
            help="generate or parse a hardware graph")
    parser.add_argument("--task-alloc-method", type=str, default="serial", choices=("serial", "greedy_parallel"),
            help="method of task allocation while generating the task to hardware mapping file")
    parser.add_argument("--output-dot", action="store_true", default=False,
            help="output a DOT representation of the final workload DFG")
    parse_wrkld_specific_args(parser)
    args = parser.parse_args()
    os.environ['VERBOSE'] = str(args.verbose)
    return args

def graph_to_csv(graph:nx.Graph, fname:str, idx_name:str):
    dmA = nx.adjacency_matrix(graph, weight="weight").todense()
    header = list(graph.nodes())
    dm_df = pd.DataFrame(dmA, columns=header)
    dm_df.replace(0, '', inplace=True)
    dm_df.insert(0, idx_name, header)
    os.makedirs('/'.join(fname.split('/')[:-1]), exist_ok=True)
    dm_df.to_csv(fname, index=False)

def export_nx_graph(graph:nx.Graph, outpath:str, ext:str):
    dir_path = '/'.join(outpath.split('/')[:-1])
    os.makedirs(dir_path, exist_ok=True)
    nx.drawing.nx_pydot.write_dot(graph, f"{outpath}.{ext}")

def rotate(l, n):
    return l[-n:] + l[:-n]