# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from gen_parse_data import def_budgets
import os
import numpy as np
from matplotlib import cm
import matplotlib.patches as patches
import sys
sys.path.append("../")
from helper import setup_logger

results_root = "./results"
FONTSIZE=13
SKIP_OK=False
FILTER_IF_NOT_MET_BUDGET=False
COLOR_CODE_IF_NOT_MET_BUDGET=False
DRAW_BUDGET_LINE=True
PLOT_NORMALIZED_METRICS=1
TITLEPAD=0
LABELPAD=5
width=.5*.75# *2
# mpl.use('agg')
# plt.style.use("fivethirtyeight")
plt.style.use('seaborn-whitegrid')
plt.rcParams['lines.linewidth'] = .5
# plt.rcParams['lines.markersize'] = 5
# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.axis'] = 'x'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
# plt.rcParams['text.usetex'] = False
plt.rcParams['axes.edgecolor'] = "lightgray"
plt.rcParams['axes.linewidth'] = .5

font_path = 'fonts/JournalSans.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['figure.constrained_layout.use'] = True
# color = ['#00429d', '#b04351', '#73a2c6', '#fe908f', ] # 
color = ['#d1e5f0', 'yellow', 'gray', '#67001f', '#2166ac', ] # 
color = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#053061','#d1e5f0','#92c5de','#4393c3','#2166ac']
color = ['#67001f','#b2182b','#053061','#2166ac']
color = ['#750014','#ca0020','#2a5173','#437fb5']
# color = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

path = {
    "better-case": {
        "FARSI":    f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP_ms_dyn_energy/FARSI-RR/exp-oldDagInterArrTime-0.022_soc_3x3_numDags_3_dagInterArrTime_0.0_ncv_1_nrad_4_nvit_4.0/final_6/run.log",
        "ARTEMIS":  f"{results_root}/miniera/real_EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP_ms_dyn_energy/ARTEMIS/exp-oldDagInterArrTime-0.022_soc_3x3_numDags_3_dagInterArrTime_0.0_ncv_1_nrad_4_nvit_4.0/final_10/run.log",
    }
}

bar_colors = {
    "arch": color[0],
    "mapping": color[1],
    "arch_mapping": color[2],
}

def arrowplot(axes, x, y, nArrs=30, mutateSize=10, color='gray', markerStyle='o', label=''): 
    '''arrowplot : plots arrows along a path on a set of axes
        axes   :  the axes the path will be plotted on
        x      :  list of x coordinates of points defining path
        y      :  list of y coordinates of points defining path
        nArrs  :  Number of arrows that will be drawn along the path
        mutateSize :  Size parameter for arrows
        color  :  color of the edge and face of the arrow head
        markerStyle : Symbol
    
        Bugs: If a path is straight vertical, the matplotlab FanceArrowPatch bombs out.
          My kludge is to test for a vertical path, and perturb the second x value
          by 0.1 pixel. The original x & y arrays are not changed
    
        MHuster 2016, based on code by 
    '''
    # recast the data into numpy arrays
    x = np.array(x, dtype='f')
    y = np.array(y, dtype='f')
    nPts = len(x)

    # Plot the points first to set up the display coordinates
    axes.plot(x,y, markerStyle, ms=1, color=color, label=label)

    # get inverse coord transform
    inv = axes.transData.inverted()

    # transform x & y into display coordinates
    # Variable with a 'D' at the end are in display coordinates
    xyDisp = np.array(axes.transData.transform(list(zip(x,y))))
    xD = xyDisp[:,0]
    yD = xyDisp[:,1]

    # drD is the distance spanned between pairs of points
    # in display coordinates
    dxD = xD[1:] - xD[:-1]
    dyD = yD[1:] - yD[:-1]
    drD = np.sqrt(dxD**2 + dyD**2)

    # Compensating for matplotlib bug
    dxD[np.where(dxD==0.0)] = 0.1


    # rtotS is the total path length
    rtotD = np.sum(drD)

    # based on nArrs, set the nominal arrow spacing
    arrSpaceD = rtotD / nArrs

    # Loop over the path segments
    iSeg = 0
    while iSeg < nPts - 1:
        # Figure out how many arrows in this segment.
        # Plot at least one.
        nArrSeg = max(1, int(drD[iSeg] / arrSpaceD + 0.5))
        xArr = (dxD[iSeg]) / nArrSeg # x size of each arrow
        segSlope = dyD[iSeg] / dxD[iSeg]
        # Get display coordinates of first arrow in segment
        xBeg = xD[iSeg]
        xEnd = xBeg + xArr
        yBeg = yD[iSeg]
        yEnd = yBeg + segSlope * xArr
        # Now loop over the arrows in this segment
        for iArr in range(nArrSeg):
            # Transform the oints back to data coordinates
            xyData = inv.transform(((xBeg, yBeg),(xEnd,yEnd)))
            # Use a patch to draw the arrow
            # I draw the arrows with an alpha of 0.5
            p = patches.FancyArrowPatch( 
                xyData[0], xyData[1], 
                arrowstyle='->',
                mutation_scale=mutateSize,
                color=color, alpha=0.5)
            axes.add_patch(p)
            # Increment to the next arrow
            xBeg = xEnd
            xEnd += xArr
            yBeg = yEnd
            yEnd += segSlope * xArr
        # Increment segment number
        iSeg += 1

if __name__ == "__main__":
    logger = setup_logger('MyLogger')
    pap, avg_slacks, area, perc_met, power, all_bar_colors = {}, {}, {}, {}, {}, {}
    frame_all = ["FARSI", "ARTEMIS"]
    # frame_all = ["ARTEMIS"]
    exp_id_all = 1

    rounds = [1, 2]
    strides = {"FARSI": 20, "ARTEMIS": 10}
    for metric in ["pow", "area"]:
        for round in rounds:
            fig = plt.figure(figsize=(2.6,2.4))
            for i, case in enumerate(reversed(sorted(path.keys()))):
                print(f"case: {case}")
                ax1 = fig.add_subplot(1,len(path.keys()), i+1)
                c = 0
                for j, frame in enumerate(frame_all):
                    print(f"\tpath: {path[case][frame]}")
                    for exp_id in range(exp_id_all):
                        idx = f"frame_{exp_id+1}"
                        perc_met[idx] = []
                        avg_slacks[idx] = []
                        power[idx] = []
                        area[idx] = []
                        pap[idx] = []
                        all_bar_colors[idx] = []
                        skip = False
                        itr_count = 0
                        path_ = path[case][frame]
                        if not os.path.exists(path_): continue
                        # print(f"Reading: {path_}")
                        init_des_done = False
                        with open(path_, 'r') as data_log:
                            lines = data_log.readlines()
                            for line_num, line in enumerate(lines):
                                line = line.rstrip('\n')
                                if line.startswith("info: tp::") and lines[line_num+1].startswith("energy"):
                                    itr_count += 1
                                elif (line.startswith("design's latency") and not skip) or (line.startswith("Init design's latency") and not init_des_done):
                                    num_met = 0
                                    if (line.startswith("Init design's latency") and not init_des_done):
                                        line = line.split(' ')[3:][1::2]
                                    else:
                                        line = line.split(' ')[2:][1::2]
                                    slack = []
                                    for k in range(len(line)): # over all DAGs
                                        if k == len(line)-1:
                                            line[k] = float(line[k].rstrip('}'))
                                        else:
                                            line[k] = float(line[k].rstrip(','))
                                        if line[k] <= def_budgets["miniera"]["latency"]:
                                            num_met += 1
                                        slack.append(-def_budgets["miniera"]["latency"]+line[k])
                                    avg_slack = sum(slack)/len(slack)
                                    perc_met[idx].append(100.*num_met/len(line))
                                    avg_slacks[idx].append(avg_slack)
                                    # print(perc_met[idx][-1], num_met, line)
                                elif line.startswith("design's power") or (line.startswith("Init design's power") and not init_des_done):
                                    if not skip:
                                        if (line.startswith("Init design's power") and not init_des_done):
                                            power[idx].append(float(line.split(' ')[3])*1e3)
                                        else:
                                            power[idx].append(float(line.split(' ')[2])*1e3)
                                    else:
                                        skip = False
                                elif line.startswith("design's area") or (line.startswith("Init design's area") and not init_des_done):
                                    if not skip:
                                        if (line.startswith("Init design's area") and not init_des_done):
                                            area[idx].append(float(line.split(' ')[3])*1e6)
                                            init_des_done = True
                                            itr_count += 1
                                        else:
                                            area[idx].append(float(line.split(' ')[2])*1e6)
                                    else:
                                        skip = False
                                    # perc_met_per_mW = perc_met / (power*1e3)
                        assert len(power[idx]) == itr_count, f"{len(power[idx])}, {itr_count}"
                        assert len(perc_met[idx]) == itr_count, f"{len(perc_met[idx])}, {itr_count}"
                        assert len(power[idx]) == len(avg_slacks[idx])
                        assert len(power[idx]) == len(area[idx]), f"{len(area[idx])}, {len(power[idx])}"
                        assert avg_slacks[idx]
                        stride = strides[frame]
                        for v in zip(power[idx], area[idx]):
                            pap[idx].append(v[0]*v[1])
                        if metric == "pow":
                            ax1.set_xlabel("Power ($mW$)")
                            metric_val = power
                        elif metric == "area":
                            ax1.set_xlabel("Area ($mm^2$)")
                            metric_val = area
                        else: raise NotImplementedError
                        arrowplot(ax1, metric_val[idx][::stride], avg_slacks[idx][::stride], nArrs=1*(len(metric_val[idx][::stride])-1), mutateSize=10, color=color[2*c], label=frame)
                        ax1.plot(metric_val[idx][::stride][0], avg_slacks[idx][::stride][0], marker='s', ms=5, color=color[2*c+1])
                        ax1.plot(metric_val[idx][::stride][-1], avg_slacks[idx][::stride][-1], marker='*', ms=10, color=color[2*c+1])
                        ax1.set_ylabel("Avg. Negative Slack (s)")
                        if round == 2:
                            if case == "worst-case":
                                ax1.set_ylim((-.02, .15))
                            else:
                                ax1.set_ylim((-.02, .05))
                        c += 1
            os.makedirs("outputs", exist_ok=True)
            outpath = f"outputs/trail.{round}.power.pdf"
            plt.savefig(outpath)
            logger.info(f"Wrote to file: {outpath}")
            plt.close()
