# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd

frame_all = ("EDF", "EFT", "MS_STAT", "MS_DYN", "ART_DSE")
inter_arr_time_all = (0.033333, 0.011111)
ncv_nrad_nvit_all = ((1, 4, 4.0), )

results_root = "./results/miniera/EXPLORE_MODE_all-at-start_USE_DYN_NDAGS_1_f1_CONSTRAIN_TOPOLOGY_0_BUDGET_SCALES_1._1._1._DEADLINE_0.05_LAT_AMP_NO_REMAP"
paths_dict = {
    "EDF":      f"{results_root}_edf/ARTEMIS",
    "EFT":      f"{results_root}_eft/ARTEMIS",
    "MS_STAT":  f"{results_root}_ms_stat/ARTEMIS",
    "MS_DYN":   f"{results_root}_ms_dyn/ARTEMIS",
    "ART_DSE":  f"{results_root}_ms_dyn_energy/ARTEMIS",
}

for ncv, nrad, nvit in ncv_nrad_nvit_all:
    print(f"N_CV, N_rad, N_vit: ({ncv},{nrad},{nvit})")
    for inter_arr_time in inter_arr_time_all:
        power_mW_dict, time_s_dict, deadlines_met_dict = {}, {}, {}
        for frame in frame_all:
            path = paths_dict[frame]
            df = pd.read_csv(f"{path}/{inter_arr_time}_{ncv}_{nrad}_{nvit}.results.csv")
            power_mW_dict[frame] = df[df["Prefix"] == "sim_det"]["Best Power (mW)"].item()
            time_s_dict[frame] = df[df["Prefix"] == "exp"]["Best Exp Elapsed Time (s)"].item()
            deadlines_met_dict[frame] = df[df["Prefix"] == "sim_det"]["Best DAG Deadline Meet %"].item() == 100.
        print(f"\tr_DAG: {1/inter_arr_time:.0f} Hz")
        print(f"\t\tExploration Time (s):")
        print("\t\t\t", end='')
        for frame in frame_all:
            print(f"{frame}: {time_s_dict[frame]:.0f}, ", end='')
        print('')
        print(f"\t\tPower (mW):")
        print("\t\t\t", end='')
        for frame in frame_all:
            print(f"{frame}: {power_mW_dict[frame]:.0f}, ", end='')
        print('')
        print(f"\t\tDeadlines Met?")
        print("\t\t\t", end='')
        for frame in frame_all:
            if deadlines_met_dict[frame] == 0:
                print(f"{frame}: No, ", end='')
            else:
                print(f"{frame}: Yes, ", end='')
        print('')
        