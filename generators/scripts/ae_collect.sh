# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p paper_results
cp outputs/real_deadline_50ms_fixed_het_ut.lat_area.pdf  paper_results/fig_8.pdf
cp outputs/real_deadline_50ms_farsi_ut.lat_area.pdf      paper_results/fig_9_top_left.pdf
cp outputs/real_deadline_50ms_farsi_ut.lat_wall_time.pdf paper_results/fig_9_bot_left.pdf
cp outputs/real_deadline_50ms_farsi_ct.lat_pow.pdf       paper_results/fig_9_top_right.pdf
cp outputs/real_deadline_50ms_farsi_ct.lat_wall_time.pdf paper_results/fig_9_bot_right.pdf
echo "Copied all files."