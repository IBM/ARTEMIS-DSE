# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append("../../Project_FARSI")
from specs.data_base import DataBase
import random
import math

def biased_sample(my_list):
    n = len(my_list)
    weights = [i+1 for i in range(n)]
    return random.choices(my_list, weights=weights, k=1)[0]

class DataBase_ARTEMIS(DataBase):
    def up_sample_down_sample_block_multi_metric_fast(self, blck_to_imprv, sorted_metric_dir, move, selected_kernel, tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks_fast(blck_to_imprv.type, tasks)

        metrics_to_sort_reversed = []
        for metric,dir in sorted_metric_dir.items():
            if metric == "latency":
                metric_to_sort = 'peak_work_rate'
            elif metric == "power":
                #metric_to_sort = 'work_over_energy'
                metric_to_sort = 'one_over_total_power'
            elif metric == "area":
                metric_to_sort = 'one_over_area'
            else:
                assert False, "metric: " + metric + " is not defined"
            metrics_to_sort_reversed.append((metric_to_sort, -1*dir))

        most_important_metric = list(sorted_metric_dir.keys())[-1]
        sampling_dir = sorted_metric_dir[most_important_metric]

        #srtd_comptble_blcks = sorted(all_compatible_blocks, key=attrgetter(metric_to_sort), reverse=reversed)  #

        # sort all compatible blocks (blocks that can run this task) first by the last metric in metrics_to_sort_reversed, then by 2nd last, and so on
        if len(sorted_metric_dir.keys()) == 3:
            srtd_comptble_blcks = sorted(all_compatible_blocks, key=lambda blk: (metrics_to_sort_reversed[2][1]*getattr(blk, metrics_to_sort_reversed[2][0]),
                                                                                 metrics_to_sort_reversed[1][1]*getattr(blk, metrics_to_sort_reversed[1][0]),
                                                                                 metrics_to_sort_reversed[0][1]*getattr(blk, metrics_to_sort_reversed[0][0])))
        elif len(sorted_metric_dir.keys()) == 2:
            srtd_comptble_blcks = sorted(all_compatible_blocks, key=lambda blk: (metrics_to_sort_reversed[1][1]*getattr(blk, metrics_to_sort_reversed[1][0]),
                                                                                 metrics_to_sort_reversed[0][1]*getattr(blk, metrics_to_sort_reversed[0][0])))
        elif len(sorted_metric_dir.keys()) == 1:
            srtd_comptble_blcks = sorted(all_compatible_blocks, key=lambda blk: (metrics_to_sort_reversed[0][1]*getattr(blk, metrics_to_sort_reversed[0][0])))
        else:
            raise ValueError
        idx = 0

        # find the block
        results = []
        """
        # first make sure it can meet across all metrics
        for blck in srtd_comptble_blcks:
            if metrics_to_sort_reversed[2][1]*getattr(blck, metrics_to_sort_reversed[2][0]) > \
                    metrics_to_sort_reversed[2][1]*getattr(blck_to_imprv, metrics_to_sort_reversed[2][0]):
                if metrics_to_sort_reversed[1][1] * getattr(blck, metrics_to_sort_reversed[1][0]) >= \
                        metrics_to_sort_reversed[1][1] * getattr(blck_to_imprv,metrics_to_sort_reversed[1][0]):
                    if metrics_to_sort_reversed[0][1]*getattr(blck, metrics_to_sort_reversed[0][0]) >= \
                            metrics_to_sort_reversed[0][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[0][0]):
                        results.append(blck)

        # meet across two metrics
        if len(results) == 0:
            for blck in srtd_comptble_blcks:
                if metrics_to_sort_reversed[2][1] * getattr(blck, metrics_to_sort_reversed[2][0]) > \
                        metrics_to_sort_reversed[2][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[2][0]):
                    if metrics_to_sort_reversed[1][1] * getattr(blck, metrics_to_sort_reversed[1][0]) >= \
                            metrics_to_sort_reversed[1][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[1][0]):
                        results.append(blck)
        """

        # get latency for selected_kernel task if run on each of the candidate blocks
        # filter in blocks that can make selected_kernel meet its deadline (improve krnl_prob in hill_climbing.py to <1)
        # print(f"srtd_comptble_blcks: {[b_.instance_name for b_ in srtd_comptble_blcks]}")
        if move == "swap" or move == "split_swap":
            # kernel_probs_for_srtd_comptble_blcks = []
            for blck in srtd_comptble_blcks:
                blck_s_lat_this_krnl = selected_kernel.get_comp_latency_if_krnel_run_in_isolation(blck)
                kernel_s_wait_time = selected_kernel.completion_time - selected_kernel.arrival_time # Time that the kernel has waited
                kernel_lat = selected_kernel.stats.get_metric("latency")
                kernel_prob_raw = -round(blck_s_lat_this_krnl*(selected_kernel.deadline - (kernel_s_wait_time + blck_s_lat_this_krnl)),9)
                try:
                    kernel_prob = math.exp(kernel_prob_raw)
                except:
                    kernel_prob = math.inf
                # kernel_probs_for_srtd_comptble_blcks.append(kernel_prob)
                if kernel_prob <= 1: # task meets deadline on this block
                    results.append(blck)
            # print(f"Move = {move}, task to opt: {selected_kernel.task_name}, blocks that can meet deadline: {[b_.instance_name for b_ in results]}")

            # if there is no block on which the task can run and meet its subdeadline, then pick the block that is closest to making the task meet
            # adding random sampling here to avoid overshooting with huge area PEs
            if not results: # and kernel_probs_for_srtd_comptble_blcks:
                # blk_idx = kernel_probs_for_srtd_comptble_blcks.index(min(kernel_probs_for_srtd_comptble_blcks, key=lambda x:abs(x-1.)))
                # results = [srtd_comptble_blcks[blk_idx]]
                results = [biased_sample(srtd_comptble_blcks)]
                # print(f"Giving up... Blocks most likely to make meet deadline: {[b_.instance_name for b_ in results]}")
                return results

        # if nothing can make it meet deadline, improve across at least one metric (most important one - last one in metrics_to_sort_reversed)
        # meet across at least one metric
        if len(results) == 0:
            for blck in srtd_comptble_blcks:
                metric_idx = len(sorted_metric_dir.keys())-1
                if metrics_to_sort_reversed[metric_idx][1] * getattr(blck, metrics_to_sort_reversed[metric_idx][0]) > \
                        metrics_to_sort_reversed[metric_idx][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[metric_idx][0]):
                    results.append(blck)

       # we need pareto front calculation here, but we are doing something simple at the moment instead
        if len(results) > 1:
            results = [random.choice(results)]

        # giving up - revert back to old block ; blck_to_imprv
        if len(results) == 0:
            for el in srtd_comptble_blcks:
                if el.get_generic_instance_name() == blck_to_imprv.get_generic_instance_name():
                    results = [el]
                    break

        return results