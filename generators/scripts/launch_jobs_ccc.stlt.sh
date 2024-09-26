# Copyright (c) 2024 IBM Corp.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e
source helper.sh

function count_num_vals() {
    str=$1
    res="${str//[^ ]}"
    res="${#res}"
    res=$((${res}+1))
    echo $res
}
if [ $# -ne 2 ]; then
    log "ERROR" "Exactly 2 arguments required"
    exit 1
fi
SOC_X_SOC_Y_DAG_INTARR_TIME=$1
NC_NR_NV=$2
log "INFO" "SOC_X_SOC_Y_DAG_INTARR_TIME: ${SOC_X_SOC_Y_DAG_INTARR_TIME}"
log "INFO" "NC_NR_NV: ${NC_NR_NV}"
GEN_TRACES=$GEN_TRACES
DEADLINE=0.05 # also change in gen_parse_data.py
SKIPPED_EXPS_OK=0 # is it ok to skip failed exp runs?

# FARSI-RR extrapolates mappings for DAG IDs NDAGS_EXP...NDAGS_SIM-1 based on a round-robin method, i.e., DAG[NDAGS_EXP] == DAG[0], DAG[NDAGS_EXP+1] == DAG[1],...
# FARSI-LAST extrapolates mappings for DAG IDs NDAGS_EXP...NDAGS_SIM-1 solely based on a DAG[NDAGS_EXP-1]; most likely it doesn't make sense to use this
# FARSI-DYN uses scheduler to do the mapping
# the exploration for all of these three are common
# for the paper, we are using: "FARSI-RR" "ARTEMIS" "FARSI-DYN" "FIXED_HET"
FRAMEWORK=$FRAMEWORK # "ARTEMIS" "FARSI-LAST" "FARSI-RR" "FARSI-DYN" "FIXED_HET" "FIXED_HOM" "CUSTOM"
CONSTRAIN_TOPOLOGY=$CONSTRAIN_TOPOLOGY
CUST_SCHED_POLICY_NAME=$CUST_SCHED_POLICY_NAME
CUST_SCHED_CONSIDER_DM_TIME=$CUST_SCHED_CONSIDER_DM_TIME
DYN_SCHEDULING_MEM_REMAPPING=$DYN_SCHEDULING_MEM_REMAPPING
NUM_MEMS=$NUM_MEMS # TODO

if [[ $CONSTRAIN_TOPOLOGY -eq 0 ]]; then
    NUM_MEMS=1 # doesn't matter since we generate HW graph from scratch
fi

if [[ $FRAMEWORK == "FARSI-LAST" ]]; then
    TASK_ALLOC_METHOD_PARSE="serial"
else
    TASK_ALLOC_METHOD_PARSE="greedy_parallel"
fi
if [[ $FRAMEWORK == "ARTEMIS" ]] || [[ $FRAMEWORK == "CUSTOM" ]] || [[ $FRAMEWORK == "FARSI-DYN" ]] || [[ $FRAMEWORK == "FARSI-RR" ]] || [[ $FRAMEWORK == "FARSI-LAST" ]]; then
    TOPOLOGY="mesh" # "ring"
    HW_GEN_MODE="cpu_only"
else
    if [[ $CONSTRAIN_TOPOLOGY -ne 1 ]]; then
        log "ERROR" "fixed SoC modes must be run with constrained topology"
        exit 1
    fi
    TOPOLOGY="bus"
    if [[ $FRAMEWORK == "FIXED_HOM" ]]; then
        HW_GEN_MODE="one_cpu_per_task"
    elif [[ $FRAMEWORK == "FIXED_HET" ]]; then
        HW_GEN_MODE="one_ip_per_task"
    else
        log "ERROR" ""
        exit 1
    fi
fi
if [[ $CONSTRAIN_TOPOLOGY -eq 0 ]]; then
    TOPOLOGY="bus" # don't care
fi

EXPLORE_MODE=${EXPLORE_MODE} # "staggered" # "all-at-start"
USE_DYN_NDAGS=1
if [[ $EXPLORE_MODE == "all-at-start" ]]; then
    NDAGS_EXP_SCALE_F=1 # only used for USE_DYN_NDAGS = 1
    NDAGS_CALC_METHOD="simple" # don't change this
elif [[ $EXPLORE_MODE == "staggered" ]]; then
    NDAGS_EXP_SCALE_F=2 # only used for USE_DYN_NDAGS = 1
    NDAGS_CALC_METHOD="lcm" # "simple", only for EXPLORE_MODE == "staggered"
else
    log "ERROR" "Unsupported EXPLORE_MODE: $EXPLORE_MODE"
    exit 1
fi
DROP_TASKS_THAT_PASSED_DEADLINE=$DROP_TASKS_THAT_PASSED_DEADLINE # only for simulation, not exploration # TODO: broken
BUDGET_SCALES=${BUDGET_SCALES}
BUDGET_SCALES_STR=`echo "${BUDGET_SCALES}" | tr ' ' '_'`
log "INFO" "BUDGET_SCALES_STR: $BUDGET_SCALES_STR"

if [[ $CONSTRAIN_TOPOLOGY -eq 0 ]]; then
    LOG_DIR="results/miniera/EXPLORE_MODE_${EXPLORE_MODE}_USE_DYN_NDAGS_${USE_DYN_NDAGS}_f${NDAGS_EXP_SCALE_F}_CONSTRAIN_TOPOLOGY_${CONSTRAIN_TOPOLOGY}_BUDGET_SCALES_${BUDGET_SCALES_STR}_DEADLINE_${DEADLINE}_LAT_AMP_NO_REMAP_${CUST_SCHED_POLICY_NAME}"
else
    LOG_DIR="results/miniera/EXPLORE_MODE_${EXPLORE_MODE}_USE_DYN_NDAGS_${USE_DYN_NDAGS}_f${NDAGS_EXP_SCALE_F}_CONSTRAIN_TOPOLOGY_${CONSTRAIN_TOPOLOGY}_${TOPOLOGY}_BUDGET_SCALES_${BUDGET_SCALES_STR}_DEADLINE_${DEADLINE}_LAT_AMP_NO_REMAP_${CUST_SCHED_POLICY_NAME}"
fi

NC_ALL="\"`echo ${NC_NR_NV} | cut -d ' ' -f1`\""
NR_ALL="\"`echo ${NC_NR_NV} | cut -d ' ' -f2`\""
NV_ALL="\"`echo ${NC_NR_NV} | cut -d ' ' -f3`\""
LOG_ROOT="${PWD}/${LOG_DIR}"

NDAGS_SIM=$NDAGS_SIM
N_EXP=$N_EXP
if [[ $FRAMEWORK == "ARTEMIS" ]]; then
    if [[ $USE_DYN_NDAGS -eq 0 ]]; then
        NDAGS_EXP=6
    fi
    # N_EXP=10 # need this only if there is possibility of random exploration, e.g. if we have multiple PE choices with clock speeds and LLP
    MEM=32g
elif [[ $FRAMEWORK == "FARSI-DYN" ]] || [[ $FRAMEWORK == "FARSI-RR" ]] || [[ $FRAMEWORK == "FARSI-LAST" ]] || [[ $FRAMEWORK == "CUSTOM" ]]; then
    if [[ $USE_DYN_NDAGS -eq 0 ]]; then
        NDAGS_EXP=6
    fi
    # N_EXP=5
    MEM=32g
elif [[ $FRAMEWORK == "FIXED_HOM" ]] || [[ $FRAMEWORK == "FIXED_HET" ]]; then
    if [[ $USE_DYN_NDAGS -eq 0 ]]; then
        NDAGS_EXP=6
    fi
    # N_EXP=1
    MEM=8g
else
    exit 1
fi

FARSI_INT_PARALLELISM=1
if [[ $FRAMEWORK == "FARSI-DYN" ]]; then
    heuristic_type=FARSI
    USE_CUST_SCHED_POLICIES_exp=0
    USE_CUST_SCHED_POLICIES_sim=1
    RT_AWARE_BLK_SEL=0
    RT_AWARE_TASK_SEL=0
    CLUSTER_KRNLS_NOT_TO_CONSIDER=0
    DYN_SCHEDULING_INSTEAD_OF_MAPPING=0
    SINGLE_RUN_EXP=0
elif [[ $FRAMEWORK == "FARSI-RR" ]] || [[ $FRAMEWORK == "FARSI-LAST" ]]; then
    heuristic_type=FARSI
    USE_CUST_SCHED_POLICIES_exp=0
    USE_CUST_SCHED_POLICIES_sim=0
    RT_AWARE_BLK_SEL=0
    RT_AWARE_TASK_SEL=0
    CLUSTER_KRNLS_NOT_TO_CONSIDER=0
    DYN_SCHEDULING_INSTEAD_OF_MAPPING=0
    SINGLE_RUN_EXP=0
elif [[ $FRAMEWORK == "ARTEMIS" ]]; then
    heuristic_type=FARSI
    USE_CUST_SCHED_POLICIES_exp=1
    USE_CUST_SCHED_POLICIES_sim=1
    RT_AWARE_BLK_SEL=1
    RT_AWARE_TASK_SEL=1
    CLUSTER_KRNLS_NOT_TO_CONSIDER=1
    DYN_SCHEDULING_INSTEAD_OF_MAPPING=1
    SINGLE_RUN_EXP=0
elif [[ $FRAMEWORK == "FIXED_HET" ]] || [[ $FRAMEWORK == "FIXED_HOM" ]]; then
    heuristic_type=FARSI    # don't care
    USE_CUST_SCHED_POLICIES_exp=1   # don't care
    USE_CUST_SCHED_POLICIES_sim=1
    RT_AWARE_BLK_SEL=1  # don't care
    RT_AWARE_TASK_SEL=1 # don't care
    CLUSTER_KRNLS_NOT_TO_CONSIDER=1 # don't care
    DYN_SCHEDULING_INSTEAD_OF_MAPPING=1 # don't care
    SINGLE_RUN_EXP=1 # we don't want to run exploration for FIXED_HOM and FIXED_HET SoCs, so set this override flag
elif [[ $FRAMEWORK == "CUSTOM" ]]; then
    if [[ -z ${heuristic_type} ]] || [[ -z ${USE_CUST_SCHED_POLICIES_exp} ]] || [[ -z ${USE_CUST_SCHED_POLICIES_sim} ]] || [[ -z ${RT_AWARE_BLK_SEL} ]] || [[ -z ${RT_AWARE_TASK_SEL} ]] || [[ -z ${CLUSTER_KRNLS_NOT_TO_CONSIDER} ]] || [[ -z ${DYN_SCHEDULING_INSTEAD_OF_MAPPING} ]]; then
        log "ERROR" "Env variables heuristic_type, USE_CUST_SCHED_POLICIES_exp, USE_CUST_SCHED_POLICIES_sim, RT_AWARE_BLK_SEL, RT_AWARE_TASK_SEL, CLUSTER_KRNLS_NOT_TO_CONSIDER, DYN_SCHEDULING_INSTEAD_OF_MAPPING must be defined"
        exit 1
    fi
    SINGLE_RUN_EXP=0
else
    log "ERROR" "Unsupported FRAMEWORK: ${FRAMEWORK}"
    exit 1
fi

mkdir -p ${LOG_ROOT}/${FRAMEWORK}
DAG_REPO_ROOT="${LOG_ROOT}/${FRAMEWORK}/inputs"
# rm -rf ccc_${SUFF}_*.log ${LOG_ROOT}/*
# echo "Deleting CSV files in ${DAG_REPO_ROOT}"
# rm -rf ${DAG_REPO_ROOT}/miniera_soc_*.csv

# # replace_ndags <DAG_INTARR_TIME> <DEADLINE>
# function update_ndags() {
#     DAG_INTARR_TIME="$1"
#     DEADLINE="$2"
#     result=`echo "scale=2 ; $DEADLINE / $DAG_INTARR_TIME" | bc`
#     NDAGS_EXP_UPDATED=`echo "define ceil (x) {if (x<0) {return x/1} \
#         else {if (scale(x)==0) {return x} \
#         else {return x/1 + 1 }}} ; ceil($result)" | bc`
#     # echo $DAG_INTARR_TIME $DEADLINE $NDAGS_EXP_UPDATED
#     # NDAGS_SIM=$NDAGS_EXP_UPDATED
#     # NDAGS_SIM=`echo $((NDAGS_EXP_UPDATED>NDAGS_SIM ? NDAGS_EXP_UPDATED : NDAGS_SIM))`
# }

function update_ndags() {
    DAG_INTARR_TIME="$1"
    DEADLINE="$2"
    NDAGS_EXP_UPDATED=`python3 ndags_calc.py $NDAGS_CALC_METHOD $DEADLINE $DAG_INTARR_TIME`
    NDAGS_EXP_UPDATED="$((NDAGS_EXP_UPDATED*NDAGS_EXP_SCALE_F))"
    # echo $NDAGS_EXP_UPDATED
}

# enable for det trace generation
if [[ $GEN_TRACES -eq 1 ]]; then
    SOC_X=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | cut -d ' ' -f1`
    SOC_Y=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | cut -d ' ' -f2`
    DAG_INTARR_TIME=`echo $SOC_X_SOC_Y_DAG_INTARR_TIME | cut -d ' ' -f3`
    if [[ ${EXPLORE_MODE} == "all-at-start" ]]; then
        DAG_INTARR_TIME_UPDATED="0.0"
    else
        DAG_INTARR_TIME_UPDATED=$DAG_INTARR_TIME
    fi
    if [[ $USE_DYN_NDAGS -eq 1 ]]; then
        update_ndags $DAG_INTARR_TIME $DEADLINE
        if [[ $NDAGS_SIM -lt $NDAGS_EXP_UPDATED ]]; then
            log "ERROR" "NDAGS_SIM ($NDAGS_SIM) must be >= NDAGS_EXP ($NDAGS_EXP_UPDATED) ($SOC_X_SOC_Y_DAG_INTARR_TIME)"
            exit 1
        fi
        if [[ $NDAGS_EXP_UPDATED -eq 1 ]]; then
            DAG_INTARR_TIME_UPDATED=0.0
        fi
    else
        NDAGS_EXP_UPDATED=$NDAGS_EXP
    fi
    SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED="${SOC_X} ${SOC_Y} ${DAG_INTARR_TIME_UPDATED}"

    if [[ $CONSTRAIN_TOPOLOGY -eq 1 ]]; then
        inner_suffix=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | tr " " _`
    else # ignore SoC dims as these are not used
        inner_suffix=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | cut -d' ' -f3`
    fi
    STATUS_LOGFILE=${LOG_ROOT}/${FRAMEWORK}/${inner_suffix}.status.log
    log "INFO" generate_det_1 >> ${STATUS_LOGFILE}
    DET_DAGS=1 \
        BUDGET_SCALES=${BUDGET_SCALES} \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        EXPS_RUN_DIFF_TRACES=0 \
        N_EXP=1 \
        TOPOLOGY=$TOPOLOGY \
        SYS_GEN_MODE=gen \
        HW_GEN_MODE=${HW_GEN_MODE} \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        TASK_ALLOC_METHOD=greedy_parallel \
        NUM_MEMS=${NUM_MEMS} \
        FRAMEWORK=${FRAMEWORK} \
        bash run_all.sh gen ${SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED} ${NDAGS_EXP_UPDATED} "${NC_ALL}" "${NR_ALL}" "${NV_ALL}"
fi
    # enable for prob trace generation
    # if [[ ${EXPLORE_MODE} == "all-at-start" ]]; then
    #     if [[ ${FRAMEWORK} != "FARSI" ]]; then
    #         echo "$EXPLORE_MODE exploration mode only valid for FARSI, not ARTEMIS"
    #         exit 1
    #     fi
    #     replace_dag_iat_ndags "${SOC_X_SOC_Y_DAG_INTARR_TIME}" ${DEADLINE}
    # fi
    # else
    #     SOC_X_SOC_Y_DAG_INTARR_TIME_NDAGS_EXP=`echo "$SOC_X_SOC_Y_DAG_INTARR_TIME $NDAGS_EXP"`
    # fi
    # inner_suffix=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | tr " " _`
    # STATUS_LOGFILE=${LOG_ROOT}/${FRAMEWORK}/${inner_suffix}.status.log
    # echo generate_prob_1 >> {STATUS_LOGFILE}
    # DET_DAGS=0 \
    #     BUDGET_SCALES=${BUDGET_SCALES} \
    #    CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
    #     EXPS_RUN_DIFF_TRACES=1 \
    #     N_EXP=$N_EXP \
    #     TOPOLOGY=$TOPOLOGY \
    #     SYS_GEN_MODE=gen \
    #     FARSI_INP_DIR=${DAG_REPO_ROOT} \
    #     TASK_ALLOC_METHOD=greedy_parallel \
    #     NUM_MEMS=${NUM_MEMS} \
    #     bash run_all.sh gen ${SOC_X_SOC_Y_DAG_INTARR_TIME_NDAGS_EXP} "${NC_ALL}" "${NR_ALL}" "${NV_ALL}"
# fi

# 1. generate inputs, 2. run exploration with FARSI mapper, 3. copy explored SoC config for next use, 4. run simulation with scheduling policy in config.py
# 6. generate inputs, 7. run exploration with scheduling policy, 8. copy explored SoC config for next use, 9. run simulation with scheduling policy in config.py, 10. collect results
log "INFO" "SOC $FRAMEWORK, CONSTRAIN_TOPOLOGY $CONSTRAIN_TOPOLOGY, DROP_TASKS_THAT_PASSED_DEADLINE $DROP_TASKS_THAT_PASSED_DEADLINE, USE_CUST_SCHED_POLICIES: $USE_CUST_SCHED_POLICIES_exp, $USE_CUST_SCHED_POLICIES_sim, DYN_SCHEDULING_INSTEAD_OF_MAPPING: $DYN_SCHEDULING_INSTEAD_OF_MAPPING"
mkdir -p ${LOG_ROOT}/${FRAMEWORK}
SOC_X=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | cut -d ' ' -f1`
SOC_Y=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | cut -d ' ' -f2`
DAG_INTARR_TIME=`echo $SOC_X_SOC_Y_DAG_INTARR_TIME | cut -d ' ' -f3`
NEW_EXP_PREFIX_SUFFIX=""
EXP_PREFIX="exp"
if [[ ${EXPLORE_MODE} == "all-at-start" ]]; then
    NEW_EXP_PREFIX_SUFFIX="-oldDagInterArrTime-${DAG_INTARR_TIME}"
    EXP_PREFIX="exp-oldDagInterArrTime-${DAG_INTARR_TIME}"
    DAG_INTARR_TIME_UPDATED="0.0"
else
    DAG_INTARR_TIME_UPDATED=$DAG_INTARR_TIME
fi
if [[ $USE_DYN_NDAGS -eq 1 ]]; then
    update_ndags $DAG_INTARR_TIME $DEADLINE
    if [[ $NDAGS_SIM -lt $NDAGS_EXP_UPDATED ]]; then
        log "ERROR" "NDAGS_SIM ($NDAGS_SIM) must be >= NDAGS_EXP ($NDAGS_EXP_UPDATED) ($SOC_X_SOC_Y_DAG_INTARR_TIME)"
        exit 1
    fi
    if [[ $NDAGS_EXP_UPDATED -eq 1 ]]; then
        NEW_EXP_PREFIX_SUFFIX="-oldDagInterArrTime-${DAG_INTARR_TIME}"
        EXP_PREFIX="exp-oldDagInterArrTime-${DAG_INTARR_TIME}"
        DAG_INTARR_TIME_UPDATED=0.0
    fi
else
    NDAGS_EXP_UPDATED=$NDAGS_EXP
fi
SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED="${SOC_X} ${SOC_Y} ${DAG_INTARR_TIME_UPDATED}"

NC_NO_QUOTES=`echo $NC_ALL | tr -d '\"'`
NR_NO_QUOTES=`echo $NR_ALL | tr -d '\"'`
NV_NO_QUOTES=`echo $NV_ALL | tr -d '\"'`
if [[ $CONSTRAIN_TOPOLOGY -eq 1 ]]; then
    inner_suffix=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | tr " " _`_${NC_NO_QUOTES}_${NR_NO_QUOTES}_${NV_NO_QUOTES}
else # ignore SoC dims as these are not used
    inner_suffix=`echo ${SOC_X_SOC_Y_DAG_INTARR_TIME} | cut -d' ' -f3`_${NC_NO_QUOTES}_${NR_NO_QUOTES}_${NV_NO_QUOTES}
fi

CCC_LOGFILE=${LOG_ROOT}/${FRAMEWORK}/${inner_suffix}.ccc.log
STATUS_LOGFILE=${LOG_ROOT}/${FRAMEWORK}/${inner_suffix}.status.log
# if NOT using custom scheduling policies for sim
if [[ $USE_CUST_SCHED_POLICIES_sim -eq 0 ]]; then # perform explore, (re-)generate (gen), copy, simulate, collect
    cmd="\
    log INFO explore >> ${STATUS_LOGFILE} && \
        N_EXP=${N_EXP} \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        DROP_TASKS_THAT_PASSED_DEADLINE=0 \
        USE_CUST_SCHED_POLICIES=${USE_CUST_SCHED_POLICIES_exp} \
        SINGLE_RUN=${SINGLE_RUN_EXP} \
        RUN_SUFFIX=exp${NEW_EXP_PREFIX_SUFFIX} \
        OVERWRITE_INIT_SOC=0 \
        DYN_SCHEDULING_INSTEAD_OF_MAPPING=${DYN_SCHEDULING_INSTEAD_OF_MAPPING} \
        heuristic_type=${heuristic_type} \
        RT_AWARE_BLK_SEL=${RT_AWARE_BLK_SEL} \
        RT_AWARE_TASK_SEL=${RT_AWARE_TASK_SEL} \
        CLUSTER_KRNLS_NOT_TO_CONSIDER=${CLUSTER_KRNLS_NOT_TO_CONSIDER} \
        LOG_ROOT=${LOG_ROOT}/${FRAMEWORK} \
        EXPS_RUN_DIFF_TRACES=0 \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        NUM_MEMS=${NUM_MEMS} \
        FRAMEWORK=${FRAMEWORK} \
        CUST_SCHED_POLICY_NAME=${CUST_SCHED_POLICY_NAME} \
        CUST_SCHED_CONSIDER_DM_TIME=${CUST_SCHED_CONSIDER_DM_TIME} \
        DYN_SCHEDULING_MEM_REMAPPING=${DYN_SCHEDULING_MEM_REMAPPING} \
        FARSI_INT_PARALLELISM=${FARSI_INT_PARALLELISM} \
    bash run_all.sh run ${SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED} ${NDAGS_EXP_UPDATED} ${NC_ALL} ${NR_ALL} ${NV_ALL} && sleep 10 && \
    log INFO copy_best >> ${STATUS_LOGFILE} && \
    python3 copy_best.py ${LOG_ROOT}/${FRAMEWORK} ${DAG_REPO_ROOT}_dagInterArrTime_${DAG_INTARR_TIME} ${SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED} ${NDAGS_EXP_UPDATED} ${N_EXP} ${DAG_INTARR_TIME} ${NDAGS_SIM} ${CONSTRAIN_TOPOLOGY} ${SKIPPED_EXPS_OK} \"${NC_NR_NV}\" &&\
    log INFO generate_det_2 >> ${STATUS_LOGFILE} && \
        DET_DAGS=1 \
        BUDGET_SCALES=\"${BUDGET_SCALES}\" \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        EXPS_RUN_DIFF_TRACES=0 \
        N_EXP=1 \
        TOPOLOGY=$TOPOLOGY \
        SYS_GEN_MODE=parse \
        TASK_ALLOC_METHOD=${TASK_ALLOC_METHOD_PARSE} \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        NUM_MEMS=${NUM_MEMS} \
        FRAMEWORK=${FRAMEWORK} \
    bash run_all.sh gen ${SOC_X_SOC_Y_DAG_INTARR_TIME} ${NDAGS_SIM} ${NC_ALL} ${NR_ALL} ${NV_ALL} && sleep 10 && \
    log INFO run_det >> ${STATUS_LOGFILE} && \
        N_EXP=1 \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        DROP_TASKS_THAT_PASSED_DEADLINE=${DROP_TASKS_THAT_PASSED_DEADLINE} \
        USE_CUST_SCHED_POLICIES=${USE_CUST_SCHED_POLICIES_sim} \
        SINGLE_RUN=1 \
        RUN_SUFFIX=sim_det \
        OVERWRITE_INIT_SOC=0 \
        DYN_SCHEDULING_INSTEAD_OF_MAPPING=${DYN_SCHEDULING_INSTEAD_OF_MAPPING} \
        heuristic_type=${heuristic_type} \
        RT_AWARE_BLK_SEL=${RT_AWARE_BLK_SEL} \
        RT_AWARE_TASK_SEL=${RT_AWARE_TASK_SEL} \
        CLUSTER_KRNLS_NOT_TO_CONSIDER=${CLUSTER_KRNLS_NOT_TO_CONSIDER} \
        LOG_ROOT=${LOG_ROOT}/${FRAMEWORK} \
        RUN_PARSED_SYSTEM=1 \
        EXPS_RUN_DIFF_TRACES=0 \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        NUM_MEMS=${NUM_MEMS} \
        FRAMEWORK=${FRAMEWORK} \
        CUST_SCHED_POLICY_NAME=${CUST_SCHED_POLICY_NAME} \
        CUST_SCHED_CONSIDER_DM_TIME=${CUST_SCHED_CONSIDER_DM_TIME} \
        DYN_SCHEDULING_MEM_REMAPPING=${DYN_SCHEDULING_MEM_REMAPPING} \
        FARSI_INT_PARALLELISM=${FARSI_INT_PARALLELISM} \
    bash run_all.sh run ${SOC_X_SOC_Y_DAG_INTARR_TIME} ${NDAGS_SIM} ${NC_ALL} ${NR_ALL} ${NV_ALL} && sleep 10 && \
    log INFO collect_all_2 >> ${STATUS_LOGFILE} && \
    python3 collect_all.py ${LOG_ROOT}/${FRAMEWORK} \"${EXP_PREFIX} sim_det\" ${SOC_X_SOC_Y_DAG_INTARR_TIME} \"${NDAGS_EXP_UPDATED} $NDAGS_SIM\" ${NC_ALL} ${NR_ALL} ${NV_ALL} \"${N_EXP} 1\" ${SKIPPED_EXPS_OK} 2>&1 | tee ${LOG_ROOT}/${FRAMEWORK}/${inner_suffix}.results.csv && \
    log INFO done >> ${STATUS_LOGFILE}\
"
# if using custom scheduling policies for sim
else # perform explore, copy, collect
    cmd="\
log INFO explore >> ${STATUS_LOGFILE} && \
        N_EXP=${N_EXP} \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        DROP_TASKS_THAT_PASSED_DEADLINE=0 \
        USE_CUST_SCHED_POLICIES=${USE_CUST_SCHED_POLICIES_exp} \
        SINGLE_RUN=${SINGLE_RUN_EXP} \
        RUN_SUFFIX=exp${NEW_EXP_PREFIX_SUFFIX} \
        OVERWRITE_INIT_SOC=0 \
        DYN_SCHEDULING_INSTEAD_OF_MAPPING=${DYN_SCHEDULING_INSTEAD_OF_MAPPING} \
        heuristic_type=${heuristic_type} \
        RT_AWARE_BLK_SEL=${RT_AWARE_BLK_SEL} \
        RT_AWARE_TASK_SEL=${RT_AWARE_TASK_SEL} \
        CLUSTER_KRNLS_NOT_TO_CONSIDER=${CLUSTER_KRNLS_NOT_TO_CONSIDER} \
        LOG_ROOT=${LOG_ROOT}/${FRAMEWORK} \
        EXPS_RUN_DIFF_TRACES=0 \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        NUM_MEMS=${NUM_MEMS} \
        FRAMEWORK=${FRAMEWORK} \
        CUST_SCHED_POLICY_NAME=${CUST_SCHED_POLICY_NAME} \
        CUST_SCHED_CONSIDER_DM_TIME=${CUST_SCHED_CONSIDER_DM_TIME} \
        DYN_SCHEDULING_MEM_REMAPPING=${DYN_SCHEDULING_MEM_REMAPPING} \
        FARSI_INT_PARALLELISM=${FARSI_INT_PARALLELISM} \
    bash run_all.sh run ${SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED} ${NDAGS_EXP_UPDATED} ${NC_ALL} ${NR_ALL} ${NV_ALL} && sleep 10 && \
    log INFO copy_best >> ${STATUS_LOGFILE} && \
    python3 copy_best.py ${LOG_ROOT}/${FRAMEWORK} ${DAG_REPO_ROOT}_dagInterArrTime_${DAG_INTARR_TIME} ${SOC_X_SOC_Y_DAG_INTARR_TIME_UPDATED} ${NDAGS_EXP_UPDATED} ${N_EXP} ${DAG_INTARR_TIME} ${NDAGS_SIM} ${CONSTRAIN_TOPOLOGY} ${SKIPPED_EXPS_OK} \"${NC_NR_NV}\" &&\
    log INFO generate_det_2 >> ${STATUS_LOGFILE} && \
        DET_DAGS=1 \
        BUDGET_SCALES=\"${BUDGET_SCALES}\" \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        EXPS_RUN_DIFF_TRACES=0 \
        N_EXP=1 \
        TOPOLOGY=$TOPOLOGY \
        SYS_GEN_MODE=parse \
        TASK_ALLOC_METHOD=${TASK_ALLOC_METHOD_PARSE} \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        NUM_MEMS=${NUM_MEMS} \
        FRAMEWORK=${FRAMEWORK} \
    bash run_all.sh gen ${SOC_X_SOC_Y_DAG_INTARR_TIME} ${NDAGS_SIM} ${NC_ALL} ${NR_ALL} ${NV_ALL} && sleep 10 && \
    log INFO run_det >> ${STATUS_LOGFILE} && \
        N_EXP=1 \
        CONSTRAIN_TOPOLOGY=${CONSTRAIN_TOPOLOGY} \
        DROP_TASKS_THAT_PASSED_DEADLINE=${DROP_TASKS_THAT_PASSED_DEADLINE} \
        USE_CUST_SCHED_POLICIES=${USE_CUST_SCHED_POLICIES_sim} \
        SINGLE_RUN=1 \
        RUN_SUFFIX=sim_det \
        OVERWRITE_INIT_SOC=0 \
        DYN_SCHEDULING_INSTEAD_OF_MAPPING=${DYN_SCHEDULING_INSTEAD_OF_MAPPING} \
        heuristic_type=${heuristic_type} \
        RT_AWARE_BLK_SEL=${RT_AWARE_BLK_SEL} \
        RT_AWARE_TASK_SEL=${RT_AWARE_TASK_SEL} \
        CLUSTER_KRNLS_NOT_TO_CONSIDER=${CLUSTER_KRNLS_NOT_TO_CONSIDER} \
        LOG_ROOT=${LOG_ROOT}/${FRAMEWORK} \
        RUN_PARSED_SYSTEM=1 \
        EXPS_RUN_DIFF_TRACES=0 \
        FARSI_INP_DIR=${DAG_REPO_ROOT} \
        FRAMEWORK=${FRAMEWORK} \
        CUST_SCHED_POLICY_NAME=${CUST_SCHED_POLICY_NAME} \
        CUST_SCHED_CONSIDER_DM_TIME=${CUST_SCHED_CONSIDER_DM_TIME} \
        DYN_SCHEDULING_MEM_REMAPPING=${DYN_SCHEDULING_MEM_REMAPPING} \
        FARSI_INT_PARALLELISM=${FARSI_INT_PARALLELISM} \
    bash run_all.sh run ${SOC_X_SOC_Y_DAG_INTARR_TIME} ${NDAGS_SIM} ${NC_ALL} ${NR_ALL} ${NV_ALL} && sleep 10 && \
    log info collect_all_2 >> ${STATUS_LOGFILE} && \
    python3 collect_all.py ${LOG_ROOT}/${FRAMEWORK} \"${EXP_PREFIX} sim_det\" ${SOC_X_SOC_Y_DAG_INTARR_TIME} \"${NDAGS_EXP_UPDATED} $NDAGS_SIM\" ${NC_ALL} ${NR_ALL} ${NV_ALL} \"${N_EXP} 1\" ${SKIPPED_EXPS_OK} 2>&1 | tee ${LOG_ROOT}/${FRAMEWORK}/${inner_suffix}.results.csv && \
    log info done >> ${STATUS_LOGFILE}\
"
fi
log "INFO" "Launching command: ${cmd} and logging to ${CCC_LOGFILE}"
eval ${cmd} 2>&1 | tee ${CCC_LOGFILE}
log "INFO" "./launch_jobs_ccc.stlt.sh finished."