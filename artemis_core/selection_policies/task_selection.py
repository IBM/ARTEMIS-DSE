# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append("../../Project_FARSI")
from DSE_utils.hill_climbing import HillClimbing
from settings import config
import math
    
class HillClimbing_ARTEMIS(HillClimbing):
    # get each kernels_contribution to the metric of interest
    def get_kernels_s_contribution(self, selected_metric, sim_dp):
        krnl_prob_dict = {}  # (kernel, metric_value)

        #krnls = sim_dp.get_dp_stats().get_kernels()
        # filter it kernels whose workload meet the budget
        krnls = self.filter_in_kernels_meeting_budget(selected_metric, sim_dp)
        if krnls == []: # the design meets the budget, hence all kernels can be improved for cost improvement
            krnls = sim_dp.get_dp_stats().get_kernels()
            # remove dummy tasks
            new_krnls = []
            for krnl in krnls:
                if not krnl.get_task().is_task_dummy():
                    new_krnls.append(krnl)
            krnls = new_krnls

        metric_total = sum([krnl.stats.get_metric(selected_metric) for krnl in krnls])
        if selected_metric == "latency":
            # sort kernels based on their contribution to the metric of interest
            for krnl in krnls:
                # for each kernel, compute lat*slack = lat*(service_time)
                krnl_lat = krnl.stats.get_metric("latency")
                krnl_prob_raw = -round(krnl_lat*(krnl.deadline - (krnl.completion_time - krnl.arrival_time)),9)
                try:
                    krnl_prob_dict[krnl] = math.exp(krnl_prob_raw)
                except:
                    krnl_prob_dict[krnl] = math.inf
        else:
            # sort kernels based on their contribution to the metric of interest
            for krnl in krnls:
                krnl_prob_dict[krnl] = round(krnl.stats.get_metric(selected_metric)/metric_total,9)

        if not "bottleneck" in self.move_s_krnel_selection:
            for krnl in krnls:
                krnl_prob_dict[krnl] = 1
        return krnl_prob_dict

    def select_kernel(self, ex_dp, sim_dp, selected_metric, move_sorted_metric_dir):
        # get each kernel's contributions
        krnl_contribution_dict = self.get_kernels_s_contribution(selected_metric, sim_dp)
        # get each kernel's improvement cost
        krnl_improvement_ease = self.get_kernels_s_improvement_ease(ex_dp, sim_dp, selected_metric, move_sorted_metric_dir)

        # combine the selections methods
        # multiply the probabilities for a more complex metric
        krnl_prob_dict = {}
        for krnl in krnl_contribution_dict.keys():
            krnl_prob_dict[krnl] = krnl_contribution_dict[krnl] * krnl_improvement_ease[krnl]

        # sanity check
        for k in krnl_prob_dict.keys():
            assert k.task_name != "souurce"
            assert k.task_name != "siink"
            assert "DummyLast" not in k.task_name

        # give zero probablity to the krnls that you filtered out
        for krnl in sim_dp.get_dp_stats().get_kernels():
            if krnl not in krnl_prob_dict.keys():
                krnl_prob_dict[krnl] = 0.
        # sort
        #krnl_prob_dict_sorted = {k: v for k, v in sorted(krnl_prob_dict.items(), key=lambda item: item[1])}
        if selected_metric == "latency":
            krnl_prob_dict_sorted = sorted(krnl_prob_dict.items(), key=lambda item: -item[1])
        else:
            krnl_prob_dict_sorted = sorted(krnl_prob_dict.items(), key=lambda item: (-item[1], item[0].parent_graph_arr_time))

        # get the worse kernel
        if config.move_krnel_ranking_mode == "exact":  # for area to allow us pick scenarios that are not necessarily the worst
            #selected_krnl = list(krnl_prob_dict_sorted.keys())[
            #    len(krnl_prob_dict_sorted.keys()) - 1 - self.krnel_rnk_to_consider]
            for krnl, prob in krnl_prob_dict_sorted:
                if krnl.get_task_name() in self.krnels_not_to_consider:
                    continue
                selected_krnl = krnl
                break
        else:
            selected_krnl = self.pick_from_prob_dict(krnl_prob_dict_sorted)

        if config.transformation_selection_mode == "random":
            krnls = sim_dp.get_dp_stats().get_kernels()
            if (config.DEBUG_FIX):
                random.seed(0)
            else:
                random.seed(datetime.now().microsecond)
            selected_krnl = random.choice(krnls)

        return selected_krnl, krnl_prob_dict, krnl_prob_dict_sorted