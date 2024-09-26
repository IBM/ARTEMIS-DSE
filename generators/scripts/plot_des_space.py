import math
from copy import *
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import cm
from sympy import factorial
from sympy.functions.combinatorial import numbers

plt.style.use('seaborn-whitegrid')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# mpl.use('agg')
plt.style.use('seaborn-whitegrid')
# plt.style.use("seaborn-deep")
# plt.rcParams['lines.linewidth'] = 1.5
# plt.rcParams['lines.markersize'] = 5
# # plt.rcParams['axes.grid'] = True
# # plt.rcParams['axes.grid.axis'] = 'x'
# plt.rcParams['grid.linestyle'] = '--'
# plt.rcParams['grid.alpha'] = 0.5
# plt.rcParams['text.usetex'] = False
# plt.rcParams['axes.edgecolor'] = "lightgray"

font_path = 'fonts/JournalSans.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = 12

# ------------------------------
# Functionality:
#       calculate stirling values, ie., the number of ways to partition a set. For mathematical understanding refer to
#       https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
# Variables:
#       n, k are both  stirling inputs. refer to:
#       https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
# ------------------------------
# n: balls, k: bins
def calc_stirling(n, k):
    # multiply by k! if you want the boxes to at least contain 1 value
    return numbers.stirling(n, k, d=None, kind=2, signed=False)
    
# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation

def plot_design_space_size(pe_range:np.array, knob_count:int, task_cnt_range:np.array):
    pe_cnts, task_cnts = np.meshgrid(pe_range, task_cnt_range)
    design_space_size = {}
    design_space_size = {"all": [], "all_but_mapping": [], "map_savings": []}
    label = {"all": "Full Design Space", "all_but_mapping": "Design Space\nwithout Mapping", "map_savings": "$\times$ Savings with no Mapping"}
    color = {"all": "#00429d", "all_but_mapping": "#b04351", "map_savings": "green"}
    cmap = {"all": "cool", "all_but_mapping": "autumn", "map_savings": "summer"}
    for k in design_space_size.keys():
        for task_cnt in task_cnt_range:
            design_space_size[k].append([])
            for pe_cnt in pe_range:
                if k == "map_savings":
                    d = count_system_variation(pe_cnt, task_cnt, knob_count, "all_but_mapping")
                    if d == np.nan or d == 0.:
                        design_space_size[k][-1].append(0.)
                    else:
                        # design_space_size[k][-1].append(math.log10(count_system_variation(pe_cnt, task_cnt, knob_count, "all") - \
                        #                                            count_system_variation(pe_cnt, task_cnt, knob_count, "all_but_mapping")))
                        # design_space_size[k][-1].append(float(count_system_variation(pe_cnt, task_cnt, knob_count, "all") / d))
                        design_space_size[k][-1].append(math.log10(float(count_system_variation(pe_cnt, task_cnt, knob_count, "all") / d)))
                else:
                    n = count_system_variation(pe_cnt, task_cnt, knob_count, k)
                    if n == np.nan or n == 0.:
                        design_space_size[k][-1].append(0.)
                    else:
                        # design_space_size[k][-1].append(n)
                        design_space_size[k][-1].append(math.log10(n))
        design_space_size[k] = np.array(design_space_size[k])

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ax.plot_surface(pe_cnts, task_cnts, design_space_size["all"], rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # ax.plot_wireframe(pe_cnts, task_cnts, design_space_size["all_but_mapping"], rstride=5, cstride=5)

    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, k in enumerate(["all", "all_but_mapping"]):
        # ax.plot_surface(pe_cnts, task_cnts, design_space_size[k], rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.plot_wireframe(pe_cnts, task_cnts, design_space_size[k], rstride=2, cstride=2, label=label[k], color=color[i])
    # print(design_space_size["map_savings"])
    # ax.plot_surface(pe_cnts, task_cnts, design_space_size["map_savings"], rstride=1, cstride=1, cmap=cmap["map_savings"], linewidth=0, antialiased=False)
    # ax.plot_wireframe(pe_cnts, task_cnts, design_space_size["map_savings"], rstride=1, cstride=1, color=color["map_savings"])
    ax.set_xlabel('No. of PEs')
    ax.set_ylabel('No. of Tasks')
    ax.set_zlabel('Design Space Size')
    ax.set_box_aspect(aspect = (1.5,1.2,1))
    ax.zaxis.set_ticks(np.arange(0, 60, 20))
    plt.legend(loc=(0.05,0.75))
    # plt.savefig("exploration_time.pdf")
    fig.tight_layout()
    plt.show()
    print("ok")

def count_system_variation(pe_cnt:int, task_cnt:int, knob_count:int, mode="all"):
    # assuming that we have 5 different tasks and hence (can have up to 5 different blocks).
    # we'd like to know how many different migration/allocation combinations are out there.
    # Assumptions:
    #           PE's are identical.
    #           Buses are identical
    #           memory is ignored for now
    if pe_cnt > task_cnt:
        return np.nan

    MAX_PE_CNT = pe_cnt
    system_variations = 0

    topology_dict = [1,
                     1,
                     4,
                     38,
                     728,
                     26704,
                     1866256,
                     251548592,
                     66296291072,
                     34496488594816,
                     35641657548953344,
                     73354596206766622208,
                     301272202649664088951808,
                     2471648811030443735290891264,
                     40527680937730480234609755344896,
                     1328578958335783201008338986845427712,
                     87089689052447182841791388989051400978432,
                     11416413520434522308788674285713247919244640256,
                     2992938411601818037370034280152893935458466172698624,
                     1569215570739406346256547210377768575765884983264804405248,
                     1645471602537064877722485517800176164374001516327306287561310208]

    for pe_cnt in range(1, MAX_PE_CNT):
        try:
            topology = topology_dict[pe_cnt-1]
        except:
            print(f"Error for {pe_cnt-1}")
            exit(1)
        # then calculate mapping (migrate)
        mapping = calc_stirling(task_cnt, pe_cnt) # *factorial(pe_cnt)
        # then calculate customization (swap)
        swap = int(math.pow(knob_count, (pe_cnt)))

        # print(f"topology[{pe_cnt}][{task_cnt}] = {topology}")
        # print(f"mapping[{pe_cnt}][{task_cnt}] = {mapping}")
        # print(f"swap[{pe_cnt}][{task_cnt}] = {swap}")

        if mode == "all":
            r = topology*mapping*swap
            system_variations += r
        elif mode == "all_but_mapping":
            r = topology*swap
            system_variations += r
        elif mode == "customization":
            system_variations += swap
        elif mode == "mapping":
            system_variations += mapping
        elif mode == "topology":
            system_variations += topology

        # print(f"system_variations[{pe_cnt}][{task_cnt}] = {system_variations}")
    # if system_variations == 0:
    #     system_variations = 1
    return system_variations
    #print("{:e}".format(system_variations))

# pe_cnt = 7
# task_cnt = 7*5
# knob_count = 4
# ds_size = {}
# ds_size_digits = {}
# for design_stage in ["topology", "mapping", "customization", "all", "all_but_mapping"]:
#     ds_size[design_stage] = count_system_variation(pe_cnt, task_cnt, knob_count, design_stage)
#     ds_size_digits[design_stage] = math.log10(count_system_variation(pe_cnt, task_cnt, knob_count, design_stage))

# pprint(ds_size_digits)
# pprint(ds_size)

tasks_per_dag = 3
n_dags = 10
plot_design_space_size(pe_range=np.arange(start=1, stop=14, step=1, dtype=np.int32), knob_count=4, task_cnt_range=np.arange(start=1, stop=tasks_per_dag*n_dags, step=1, dtype=np.int32))
    # pe_range=[2*2-2, 2*3-2, 3*3-2, 3*4-2, 4*4-2, 4*5-2], knob_count=4, task_cnt_range=[3*1, 3*2, 3*3, 3*4, 3*5])
