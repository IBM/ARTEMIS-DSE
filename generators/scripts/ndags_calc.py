# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import sys

assert len(sys.argv) == 4
mode = sys.argv[1]
assert mode == "lcm" or "simple"
dline = float(sys.argv[2])
iat = float(sys.argv[3])

# we want ndags_exp to be < 10 for tractable simtime
# number with most factors; this is 8 for natural numbers < 10
prec = 3 # math.log(4/dline, 10)
# for iat in np.random.uniform(low=0.01, high=0.07, size=(60,)): #  np.arange(0.01, 0.07, 0.01):
dline_scaled = int(math.ceil(dline*(10**prec)))
iat_scaled = int(math.ceil(iat*(10**prec)))
# print(iat, dline, iat_scaled, dline_scaled)
if mode == "lcm":
    if dline < iat:
        ndags = 1
    else:
        ndags = abs(dline_scaled * iat_scaled) // math.gcd(dline_scaled, iat_scaled) // iat_scaled
elif mode == "simple":
    ndags = math.ceil(dline / iat)
# print(f"iat {iat}, iat_scaled {iat_scaled}, dline {dline}, dline_scaled {dline_scaled}, ndags_simple {ndags_simple}, ndags_lcm {ndags_lcm}")
print(ndags, end="")