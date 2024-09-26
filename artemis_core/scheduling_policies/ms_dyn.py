# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# SCHEDULING POLICY DESCRIPTION:
#  This scheduling policy tries to schedule tasks according to the HetSched paper published in CAL 2021
#

import logging
import random
from datetime import datetime, timedelta

import numpy
from settings import config
from design_utils.components.scheduling import BaseSchedulingPolicy

class SchedulingPolicy(BaseSchedulingPolicy):

    def init(self, farsi_mode, perf_sim, hardware_graph, task_to_mappable_pe_map, abs_deadlines_per_workload = None):
        self.perf_sim                   = perf_sim
        self.hardware_graph             = hardware_graph
        self.blocks                     = hardware_graph.get_blocks_by_type("pe")
        self.task_to_mappable_pe_map    = task_to_mappable_pe_map
        self.abs_deadlines_per_workload = abs_deadlines_per_workload
        self.bin_size                   = 1
        self.num_bins                   = 12
        self.stats                      = {}
        self.stats['Kernel Issue Posn'] = numpy.zeros(self.num_bins, dtype=int)  # N-bin histogram
        self.farsi_mode = farsi_mode
        if farsi_mode == "sim":
            self.max_task_depth_to_check = 100
        elif farsi_mode == "exp":
            self.max_task_depth_to_check = 100
        else:
            raise NotImplementedError
        # self.ta_time                    = timedelta(microseconds=0)
        # self.to_time                    = timedelta(microseconds=0)

    def assign_rank_and_type(self, t, wcet_slack, bcet_slack, actual_task_slack, actual_dag_slack):
        #NEW1
        # if t.get_task().is_task_dummy():
        #     t.rank = 0
        #     t.rank_type = 1
        # else:
        #     if wcet_slack >= 0 and wcet_slack != bcet_slack:
        #         slack = wcet_slack
        #         t.rank = slack   #t.parent_graph_arr_time
        #         t.rank_type = 3 
        #     elif bcet_slack >= 0:
        #         slack = bcet_slack
        #         t.rank = bcet_slack  #t.parent_graph_arr_time
        #         t.rank_type = 2
        #     elif actual_task_slack > 0:
        #         slack = actual_task_slack
        #         t.rank = actual_task_slack  #t.parent_graph_arr_time
        #         t.rank_type = 4
        #     elif actual_dag_slack > 0:
        #         slack = actual_dag_slack
        #         t.rank = t.parent_graph_arr_time
        #         t.rank_type = 5              
        #     else:
        #         dtime = t.arrival_time + t.deadline
        #         t.rank = t.parent_graph_arr_time # priortize the one with latest deadline first
        #         t.rank_type = 6
        #NEW2
        if t.get_task().is_task_dummy():
            t.rank = 0
            t.rank_type = 1
        else:
            if wcet_slack >= 0 and wcet_slack != bcet_slack:
                slack = wcet_slack
                t.rank = slack   #t.parent_graph_arr_time
                t.rank_type = 4 
            elif bcet_slack >= 0:
                slack = bcet_slack
                t.rank = slack  #t.parent_graph_arr_time
                t.rank_type = 3
            elif actual_task_slack > 0:
                slack = bcet_slack
                t.rank = slack  #t.parent_graph_arr_time
                t.rank_type = 2
            elif actual_dag_slack > 0:
                # slack = actual_dag_slack
                t.rank = t.parent_graph_arr_time
                t.rank_type = 5              
            else:
                # dtime = t.arrival_time + t.deadline
                t.rank = t.parent_graph_arr_time # priortize the one with latest deadline first
                t.rank_type = 6

    def calculate_slacks(self, kernel, clock_time):
        # Done now at META init.
        #
        # max_time = 0
        # min_time = 100000
        # for stype in kernel.per_server_service_dict:
        #     service_time   = kernel.per_server_service_dict[stype]
        #     if(max_time < float(service_time)):
        #         max_time = float(service_time)
        #     if(min_time > float(service_time)):
        #         min_time = float(service_time)
        # assert max_time == kernel.max_time
        # assert min_time == kernel.min_time
        kernel.deadline = (kernel.dag_dtime - clock_time) * kernel.get_task().sr
        wcet_slack = kernel.deadline - (clock_time - kernel.arrival_time) - kernel.max_time
        bcet_slack = kernel.deadline - (clock_time - kernel.arrival_time) - kernel.min_time
        actual_task_slack = kernel.deadline -(clock_time-kernel.arrival_time)
        actual_dag_slack = kernel.dag_dtime - clock_time
        return wcet_slack, bcet_slack, actual_task_slack, actual_dag_slack

    def rank_kernels(self, clock_time, kernels):
        for kernel in kernels:
            # Get the min and max service_time of this task across all servers.
            wcet_slack, bcet_slack, actual_task_slack, actual_dag_slack = self.calculate_slacks(kernel, clock_time)
            self.assign_rank_and_type(kernel, wcet_slack, bcet_slack, actual_task_slack, actual_dag_slack)

    # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def assign_kernel(self, clock_time, kernel, window, scheduled_kernels, yet_to_schedule_kernels):
        tidx = 0
        if config.CACHE_TASK_LAT_ESTIMATES:
            task_on_block_est_lat_cache = {}
        # t0, t1 = 0., 0.
        for kernel in window:
            # s0 = time.time()
            block_to_assign = None
            if not kernel.get_task().is_task_dummy():
                # Compute execution times for each target block, factoring in
                # the remaining execution time of kernels_to_schedule already running.
                target_block_time = []
                for block in self.blocks:
                    # Get ETF for each block the task can run on
                    if block.instance_type in self.task_to_mappable_pe_map[kernel.task_name]:
                        if block.busy:
                            # this is to account for the time left until finish of the kernel that's currently running on block
                            if config.CACHE_TASK_LAT_ESTIMATES:
                                key = (block.kernel_running.task_name, block)
                                if key in task_on_block_est_lat_cache:
                                    task_on_block_est_lat = task_on_block_est_lat_cache[key]
                                else:
                                    task_on_block_est_lat = self.perf_sim.get_estimated_latency(block.kernel_running, block)
                                    task_on_block_est_lat_cache[key] = task_on_block_est_lat
                            else:
                                task_on_block_est_lat = self.perf_sim.get_estimated_latency(block.kernel_running, block)
                            remaining_time = block.kernel_running.starting_time + task_on_block_est_lat - clock_time

                            # this is to account for all tasks in ready queue with a higher priority that are waiting for a busy PE to complete execution.
                            for stask in window:
                                # Break condition to only acount for tasks in the ready queue with a higher priority
                                if stask == kernel:
                                    break
                                #Consider for remaining time if the block the higher priority task is waiting for is same as the potential block we are computing remaining time for
                                if block.instance_name == stask.possible_block_instance_name:
                                    if config.CACHE_TASK_LAT_ESTIMATES:
                                        key = (stask.task_name, block)
                                        if key in task_on_block_est_lat_cache:
                                            stask_est_lat = task_on_block_est_lat_cache[key]
                                        else:
                                            stask_est_lat = self.perf_sim.get_estimated_latency(stask, block)
                                            task_on_block_est_lat_cache[key] = stask_est_lat
                                    else:
                                        stask_est_lat = self.perf_sim.get_estimated_latency(stask, block)
                                    remaining_time += stask_est_lat
                        else:
                            remaining_time = 0

                        # Get execution time if kernel is run on this potential block
                        if config.CACHE_TASK_LAT_ESTIMATES:
                            key = (kernel.task_name, block)
                            if key in task_on_block_est_lat_cache:
                                cur_task_est_lat = task_on_block_est_lat_cache[key]
                            else:
                                cur_task_est_lat = self.perf_sim.get_estimated_latency(kernel, block)
                                task_on_block_est_lat_cache[key] = cur_task_est_lat
                        else:
                            cur_task_est_lat = self.perf_sim.get_estimated_latency(kernel, block)
                        mean_service_time = cur_task_est_lat

                        # Add in the wait/remaining time to run on this block
                        actual_service_time = mean_service_time + remaining_time

                        # Append, in the order of the block idx, the potential service time into a list to compute minimum later
                        target_block_time.append(actual_service_time)
                    else: # for incompatible blocks
                        target_block_time.append(float("inf"))
                    
                    # Look for the block with smaller actual_service_time
                    # and check if it's available
                    # if(min(target_block_time) == float("inf")):
                    #     print("NEED TO TERMINATE.. CAN'T RUN TASK ON ANYTHING AND MEET DEADLINE")
                    #     continue

                #Find the block idx, block with the fastest finish time
                block_idx = target_block_time.index(min(target_block_time))
                block_to_assign = self.blocks[block_idx]
            else:            
                # Handle block assignment for dummy tasks         
                gpp_blks = [b for b in self.hardware_graph.get_blocks_by_type("pe") if b.subtype == "gpp"]
                for blk in gpp_blks:
                    if not blk.busy:
                        block_to_assign = blk
                        break
                
                if block_to_assign == None:
                    block_to_assign = gpp_blks[0]
            # t0 += (time.time()-s0)

            # s1 = time.time()
            if not block_to_assign.busy:           # Server is not busy.
                #Reschedule task to new block chosen
                # print(f"Scheduling @{clock_time}, {kernel.task_name},{kernel.dag_id}, {block_to_assign.instance_name}", flush=True)
                self.reschedule_task_to_block(clock_time, self.hardware_graph, block_to_assign, kernel)    
                # Launch the kernel
                self.perf_sim.launch_kernel(kernel)

                bin = int(tidx / self.bin_size)
                if (bin >= len(self.stats['Kernel Issue Posn'])):
                    bin = len(self.stats['Kernel Issue Posn']) - 1
                # logging.debug('[          ] Set BIN from %d / %d to %d vs %d = %d' % (tidx, self.bin_size, int(tidx / self.bin_size), len(self.stats['Kernel Issue Posn']), bin))
                self.stats['Kernel Issue Posn'][bin] += 1
            else:
                kernel.possible_block_instance_name = block_to_assign.instance_name
            # t1 += (time.time()-s1)
            tidx += 1  # Increment task idx
        # print(f"!! blk sel, assignment time (s): {t0},{t1}")
        # print(clock_time, [k.task_name for k in scheduled_kernels], [k.task_name for k in window])
        if config.CACHE_TASK_LAT_ESTIMATES:
            del task_on_block_est_lat_cache
        return None

    def assign_kernel_to_block(self, clock_time, kernels_to_schedule, scheduled_kernels, yet_to_schedule_kernels):

        if (len(kernels_to_schedule) == 0):
            # There aren't tasks to serve
            return None

        # start = datetime.now()
        # s = time.time()
        self.rank_kernels(clock_time, kernels_to_schedule)
        kernels_to_schedule.sort(key=lambda kernel: (kernel.rank_type,kernel.rank), reverse=False)

        # print(clock_time, [(t.task_name, t.rank_type, t.rank, self.calculate_slacks(t, clock_time)) for t in kernels_to_schedule])
        # e = time.time()
        # print(f"rank_kernels and sort (s): {e-s}")
        # end = datetime.now()
        # self.to_time += end - start

        #Non-blocking in a window
        window_len = min(self.max_task_depth_to_check, len(kernels_to_schedule))
        if self.farsi_mode == "sim":
            window_len = len(kernels_to_schedule)

        # print("Window len is:", window_len, "max queue len is:", len(kernels_to_schedule))
        window = kernels_to_schedule[:window_len]

        # start = datetime.now()
        # s = time.time()
        block_assigned = self.assign_kernel(clock_time, kernels_to_schedule, window, scheduled_kernels, yet_to_schedule_kernels)
        # e = time.time()
        # print(f"assign_kernel (s): {e-s}")
        # end = datetime.now()
        # self.ta_time += end - start
        return block_assigned

    def remove_kernel_from_block(self, clock_time, server):
        pass


    def output_final_stats(self, clock_time):
        logging.info('   Kernel Issue Position: %s' % (', '.join(map(str,self.stats['Kernel Issue Posn']))))
        idx = 0;
        bin = 0;
        logging.info('         %4s  %10s' % ("Bin", "Issues"))
        c_time = 0
        c_pct_time = 0
        for count in self.stats['Kernel Issue Posn']:
            sbin = str(bin)
            logging.info('         %4s  %10d' % (sbin, count))
            idx += 1
            if (idx < (self.num_bins - 1)):
                bin += self.bin_size
            else:
                bin = ">" + str(bin)
        logging.info('')
