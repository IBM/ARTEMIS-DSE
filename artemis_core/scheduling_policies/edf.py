# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# SCHEDULING POLICY DESCRIPTION:
#  This scheduling policy tries to schedule the task at the head of the
#  queue in its best scheduling option (i.e. fastest server). If the best
#  scheduling option isn't available, the policy will try to schedule the
#  task in less-optimal server platforms.  This policy tries to clear the
#  first task by allowing any (eligible) server to execute it even if it is
#  not the most optimal execution platform.
#  This is effectively a sorted earliest-out-of-queue approach, where the
#  task checks the fastest servers, then the next-fastest servers, etc. until
#  it finds one that is not busy.

from design_utils.components.scheduling import BaseSchedulingPolicy 
import numpy
import logging
from datetime import datetime, timedelta

class SchedulingPolicy(BaseSchedulingPolicy):

    def init(self, farsi_mode, perf_sim, hardware_graph, task_to_mappable_pe_map, abs_deadlines_per_workload):
        self.perf_sim                   = perf_sim
        self.hardware_graph             = hardware_graph
        self.blocks                     = hardware_graph.get_blocks_by_type("pe")
        self.task_to_mappable_pe_map    = task_to_mappable_pe_map
        self.abs_deadlines_per_workload = abs_deadlines_per_workload
        self.bin_size                   = 1
        self.num_bins                   = 12
        self.stats                      = {}
        self.stats['Kernel Issue Posn'] = numpy.zeros(self.num_bins, dtype=int)  # N-bin histogram

    def assign_kernel_to_block(self, clock_time, kernels_to_schedule, scheduled_kernels, yet_to_schedule_kernels):
        if len(kernels_to_schedule) == 0:
            # There aren't tasks to serve
            return None

        kernels_to_schedule.sort(key=lambda kernel: self.abs_deadlines_per_workload[kernel.dag_id], reverse=False)
        # Look for an available server to process the task
        #Assumption task_to_mappable_pe_map has block names in the order of their execution (fastest to slowest)
        tidx = 0;
        start = datetime.now()
        for kernel in kernels_to_schedule:
            block_to_assign = None
            if not kernel.get_task().is_task_dummy():
                for block_name in self.task_to_mappable_pe_map[kernel.task_name]:
                    assignable_blocks = [b for b in self.blocks if b.instance_type == block_name and not b.busy]
                    # print("Assignable free blocks", block_name, [x.instance_name for x in assignable_blocks])
                    if not assignable_blocks:
                        #Block of this type not found in SoC or all are busy, continue
                        continue
                    block_to_assign = assignable_blocks[0]
                    # print("Block assigned: ")
                    # print(block_name, block_to_assign.instance_type, block_to_assign.instance_name, block_to_assign.busy)
                    break
            else:
                block_to_assign = kernel.get_ref_block()
            
            if block_to_assign is None:
                #No free PE found moving on in a non-blocking manner
                #TODO: If this should be blocking 
                continue
            
            if not block_to_assign.busy:

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
                break
            tidx += 1  # Increment task idx
            blocks_busy_status = [b.busy for b in self.blocks]
            if any(blocks_busy_status) == True:
                # All blocks are busy
                break
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
    
