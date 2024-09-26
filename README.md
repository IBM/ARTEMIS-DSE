# ARTEMIS: A Design Space Exploration Tool for Real-Time, Heterogeneous Systems-on-Chips
ARTEMIS is a framework for the exploration of SoC configurations for real-time, heterogeneous systems. It is built upon a prior work called [FARSI](https://github.com/facebookresearch/Project_FARSI). ARTEMIS builds upon FARSI to perform task-to-hardware-block mapping using a set of real-time, heterogeneity-aware scheduling policies.

## Infrastructure
This repo is structured as follows. We clone `Project_FARSI` as a submodule and then apply several diff files that were developed as part of ARTEMIS into the cloned submodule directory.
On the other hand, completely independent code, such as generators, scheduling policies, and block and task selection policies, are directly available in `artemis_core/`.

### Quick Setup

This needs to be done only once.
```bash
git clone --recurse-submodules https://github.com/IBM/ARTEMIS-DSE.git && cd ARTEMIS-DSE
sh bootstrap.sh
```

### Contributing

This needs to be done if you modify files inside `Project_FARSI/`.
```bash
sh create_diffs.sh
```
Now, you can commit and push the files in `diffs/`.

## ERA Reference Workload
As a starting point, we demonstrate the use of ARTEMIS using the [ERA workload](https://github.com/IBM/mini-era). We provide a set of scripts in `generators/scripts`. These scripts are responsible for preparing the inputs to ARTEMIS. This demo considers a mesh configuration for the SoC and therefore uses a fixed-template mode.

The main wrapper script that the user can directly invoke for this demo is called `run_launch.py`. The following blurb contains descriptions of some parameters settings specified within the script.

First, these are the regression parameters to be able to run multiple simulations.
* `soc_x_y_dag_iat_all`: Tuple of _x_ and _y_ dimensions of the mesh, and the DAG inter-arrival time in seconds.
* `ncv_nrad_nvit_all`: Tuple of the number of CV, radar, and Viterbi decoding tasks, per DAG.

Next, we have the environment variables that are defined and used in ARTEMIS.
* `NDAGS_SIM`: Number of DAGs to consider for the simulated deplyoment of the SoC.
* `N_EXP`: Number of DAGs to consider for exploration of the SoC.
* `DROP_TASKS_THAT_PASSED_DEADLINE`: Whether to drop (1) or to complete (0) DAGs that have passed their deadline during simulation.
* `BUDGET_SCALES`: Scaling factors to apply on top of the default budgets for latency, power, and area, respectively.
* `CUST_SCHED_POLICY_NAME`: Name of the custom scheduling policy defined in `artemis_core/scheduling_policies`, e.g., "ms_dyn_energy"
* `CONSTRAIN_TOPOLOGY`: Whether to run in fixed-template mode (1), or not (0).

The remainder of the parameters can be left as-is for the demo.

## Compatibility
This project has been tested extensively on Red Hat Enterprise Linux 8 (RHEL8) and is expected to work on other Linux distributions as well. While RHEL8 has been the primary testing environment, the project adheres to common Linux standards, making it compatible with most modern Linux systems. If you encounter any distribution-specific issues, please report them, and we will work to resolve them.
