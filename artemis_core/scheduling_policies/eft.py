# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# SCHEDULING POLICY DESCRIPTION:
#  This scheduling policy tries to schedule the task at the head of the
#  queue into the queue that will result in the earliest estimated
#  completion time for this task (factoring in the given start time
#  of the task taking into account the current busy status of a server).
#  IF that task does not immeidately "issue" to the selected server
#   (i.e. that server is "busy") then it considers the next task on the task list,
#   while factoring the remaining time of the preceding tasks in the list,
#   and continues to do so until it has checked a number of tasks equal to
#   the max_task_depth_to_check parm (defined below).
#  This policy effectively attempts to provide the least utilization time
#  (overall) for all the servers during the run.  For highly skewed
#  mean service times, this policy may delay the start time of a task until
#  a fast server is available.
# This is the first example that includes "issue" of tasks other than the
#  one at the head of the queue...
#

import logging
import random
from datetime import datetime, timedelta

import numpy
import copy
from settings import config
from design_utils.components.scheduling import BaseSchedulingPolicy

class SchedulingPolicy(BaseSchedulingPolicy):

    def init(self, farsi_mode, perf_sim, hardware_graph, task_to_mappable_pe_map, abs_deadlines_per_workload = None):
        self.perf_sim                   = perf_sim
        self.hardware_graph             = hardware_graph
        self.blocks                     = hardware_graph.get_blocks_by_type("pe")
        self.task_to_mappable_pe_map    = task_to_mappable_pe_map
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


    def assign_kernel_to_block(self, clock_time, kernels_to_schedule, scheduled_kernels, yet_to_schedule_kernels):
        if len(kernels_to_schedule) == 0:
            # There aren't tasks to serve
            return None

        #Non-blocking in a window
        if len(kernels_to_schedule) > self.max_task_depth_to_check:
            window_len = self.max_task_depth_to_check
        else:
            window_len = len(kernels_to_schedule)

        window = kernels_to_schedule[:window_len]

        tidx = 0
        if config.CACHE_TASK_LAT_ESTIMATES:
            task_on_block_est_lat_cache = {}
        for kernel in window:
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
                block_to_assign = kernel.get_ref_block()

            if not block_to_assign.busy:           # Server is not busy.
                #Reschedule task to new block chosen
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
            tidx += 1  # Increment task idx
        if config.CACHE_TASK_LAT_ESTIMATES:
            del task_on_block_est_lat_cache
        return None

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
