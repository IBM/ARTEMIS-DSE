# Copyright (c) IBM.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python3 gen_parse_data.py \
    --workload=miniera \
    --num-cv-all 1 \
    --num-radar-all=4 \
    --num-viterbi-all=4 \
    --output-dot \
    --num-dags=2 \
    --dag-inter-arrival-time-all 0 \
    -x 2 \
    -y 2 \
    --out-path=./outputs \
    --mode=det \
    --sys=gen \
    --gen-mode cpu_only \
    --task-alloc-method serial \
    --top bus \
    --constrain-topology 1 \
    --num-mems 2
