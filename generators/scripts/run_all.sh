# Copyright (c) IBM.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e
source helper.sh

SCRIPT_PWD=$PWD
FARSI_ROOT=$PWD/../../Project_FARSI

NUM_IO_TILES=1

if [[ $1 == "gen" ]]; then
    GEN_PARSE_DATA=1
    RUN_FARSI=0
elif [[ $1 == "run" ]]; then
    GEN_PARSE_DATA=0
    RUN_FARSI=1
fi

function gen_parse_data_loc() {
    if [[ $DET_DAGS -eq 1 ]]; then
        cmd="python3 gen_parse_data.py \
--workload=miniera \
--mode=det \
--num-cv-all ${nc} \
--num-radar-all ${nr} \
--num-viterbi-all ${nv} \
--output-dot \
--out-path=${FARSI_INP_DIR}_dagInterArrTime_${dag_arr_time} \
--num-dags=${num_dags} \
--dag-inter-arrival-time-all ${dag_arr_time} \
-x ${soc_x_dim} \
-y ${soc_y_dim} \
--sys=${sys_gen_mode} \
--task-alloc-method=${TASK_ALLOC_METHOD} \
--silence \
--constrain-topology ${CONSTRAIN_TOPOLOGY} \
--budget-scales ${BUDGET_SCALES} \
--top=$TOPOLOGY \
--num-mems=${NUM_MEMS}"
    elif [[ $DET_DAGS -eq 0 ]]; then
        if [ -z "${DAG_IAT_CAP}" ] && [ -z "${NVIT_CAP}" ]; then
            cmd="python3 gen_parse_data.py \
--workload=miniera \
--mode=prob \
--num-cv-all ${nc} \
--num-radar-all ${nr} \
--num-viterbi-mean-all ${nv} \
--output-dot \
--out-path=${FARSI_INP_DIR}_dagInterArrTime_${dag_arr_time} \
--num-dags=${num_dags} \
--dag-inter-arrival-time-mean-all ${dag_arr_time} \
-x ${soc_x_dim} \
-y ${soc_y_dim} \
--sys=${sys_gen_mode} \
--task-alloc-method=${TASK_ALLOC_METHOD} \
--num-traces ${N_EXP} \
--silence \
--constrain-topology ${CONSTRAIN_TOPOLOGY} \
--budget-scales ${BUDGET_SCALES} \
--top=$TOPOLOGY \
--num-mems=${NUM_MEMS}"
        else
            cmd="python3 gen_parse_data.py \
--workload=miniera \
--mode=prob \
--num-cv-all ${nc} \
--num-radar-all ${nr} \
--num-viterbi-mean-all ${nv} \
--num-viterbi-cap $NVIT_CAP \
--output-dot \
--out-path=${FARSI_INP_DIR}_dagInterArrTime_${dag_arr_time} \
--num-dags=${num_dags} \
--dag-inter-arrival-time-mean-all ${dag_arr_time} \
--dag-inter-arrival-time-cap $DAG_IAT_CAP \
-x ${soc_x_dim} \
-y ${soc_y_dim} \
--sys=${sys_gen_mode} \
--task-alloc-method=${TASK_ALLOC_METHOD} \
--num-traces ${N_EXP} \
--silence \
--constrain-topology ${CONSTRAIN_TOPOLOGY} \
--budget-scales ${BUDGET_SCALES} \
--top=$TOPOLOGY \
--num-mems=${NUM_MEMS}"
        fi
    else
        log "ERROR" "Error in script!"
    fi
}

function run() {
    if [ "$#" -ne 7 ]; then
        log "ERROR" "Illegal number of parameters to run(), expected 7, got $#"
        exit 1
    fi
    soc_x_dim=$1
    soc_y_dim=$2
    dag_arr_time=$3
    num_dags=$4
    nc=$5
    nr=$6
    nv=$7
    env_vars=("EXPS_RUN_DIFF_TRACES" "N_EXP" "CONSTRAIN_TOPOLOGY" "FARSI_INP_DIR" "FRAMEWORK")
    for var in "${env_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log "ERROR" "Env variable $var must be defined"
            exit 1
        fi
    done

    if [[ $GEN_PARSE_DATA -eq 1 ]]; then
        env_vars=("TASK_ALLOC_METHOD" "BUDGET_SCALES" "DET_DAGS" "TOPOLOGY" "SYS_GEN_MODE" "NUM_MEMS")
        for var in "${env_vars[@]}"; do
            if [[ -z "${!var}" ]]; then
                log "ERROR" "Env variable $var must be defined"
                exit 1
            fi
        done
        if [[ $EXPS_RUN_DIFF_TRACES -eq 0 ]]; then
            N_EXP=1
        fi
        if [[ ${SYS_GEN_MODE} == "gen" ]]; then
            if [[ -z "${HW_GEN_MODE}" ]]; then
                log "ERROR" "Env variable HW_GEN_MODE must be defined"
                exit 1
            fi
        fi
        sys_gen_mode=$SYS_GEN_MODE
        mkdir -p ${FARSI_INP_DIR}_dagInterArrTime_${dag_arr_time}
        cd ${SCRIPT_PWD}
        nc=`echo "${nc//\"}"`
        nr=`echo "${nr//\"}"`
        nv=`echo "${nv//\"}"`
        log "INFO" "Generating parse data (CONSTRAIN_TOPOLOGY=$CONSTRAIN_TOPOLOGY) for ${soc_x_dim}x${soc_y_dim}, IAT=${dag_arr_time}, NC=$nc, NR=$nr, NVIT=$nv with DET=$DET_DAGS."
        gen_parse_data_loc

        if [[ ${SYS_GEN_MODE} == "gen" ]]; then
            cmd="$cmd --gen-mode=$HW_GEN_MODE"
        fi
        # add a fake I/O tile in case of constrained topology mode
        if [[ ${CONSTRAIN_TOPOLOGY} -eq 1 ]]; then
            cmd="$cmd --num-io-tiles=${NUM_IO_TILES}"
        fi
        log "INFO" "Running command: $cmd"
        # echo "cd $PWD && $cmd" >> cmd.sh
        eval $cmd
    elif [[ $RUN_FARSI -eq 1 ]]; then
        env_vars=("DROP_TASKS_THAT_PASSED_DEADLINE" "USE_CUST_SCHED_POLICIES" 
                  "SINGLE_RUN" "RUN_SUFFIX" "OVERWRITE_INIT_SOC" 
                  "DYN_SCHEDULING_INSTEAD_OF_MAPPING" "RT_AWARE_BLK_SEL" 
                  "RT_AWARE_TASK_SEL" "CLUSTER_KRNLS_NOT_TO_CONSIDER" 
                  "heuristic_type" "LOG_ROOT" "CUST_SCHED_POLICY_NAME"
                  "CUST_SCHED_CONSIDER_DM_TIME" "DYN_SCHEDULING_MEM_REMAPPING"
                  "EXPLR_TIMEOUT" "FARSI_INT_PARALLELISM")
        for var in "${env_vars[@]}"; do
            if [[ -z "${!var}" ]]; then
                log "ERROR" "Env variable $var must be defined"
                exit 1
            fi
        done
        cd ${FARSI_ROOT}/data_collection/collection_utils/what_ifs
        for ncv in `echo "${nc//\"}"`; do
            for nrad in `echo "${nr//\"}"`; do
                for nvit in `echo "${nv//\"}"`; do
                    SOC_DIM="${soc_x_dim}x${soc_y_dim}"
                    suffix="numDags_${num_dags}_dagInterArrTime_${dag_arr_time}_ncv_${ncv}_nrad_${nrad}_nvit_${nvit}"
                    LOG_PATH="$LOG_ROOT/${RUN_SUFFIX}_soc_${SOC_DIM}_${suffix}"
                    mkdir -p ${LOG_PATH}

                    for ((exp_id=1;exp_id<=N_EXP;exp_id++)); do
                        if [[ $EXPS_RUN_DIFF_TRACES -eq 1 ]]; then
                            suffix_app=${suffix}_trace_$((${exp_id}-1))
                        else
                            suffix_app=${suffix}_trace_0
                        fi
                        # if path already exists, skip...
                        if [[ -d "${LOG_PATH}" ]]; then
                            if [[ -d "${LOG_PATH}/final_${exp_id}" ]]; then
                                log "WARNING" "Skipping run that already exists: ${LOG_PATH}/final_${exp_id}"
                                continue
                            else
                                rm -rf ${LOG_PATH}/${exp_id}
                            fi
                        fi
                        log "INFO" "Started experiment in dir: ${LOG_PATH}/${exp_id}"
                        mkdir ${LOG_PATH}/${exp_id}
                        # python -m cProfile -s cumtime FARSI_what_ifs_with_params.py \
                        if [[ -z "${RUN_PARSED_SYSTEM}" ]]; then
                            cmd="REGRESSION_RUNS=1 python FARSI_what_ifs_with_params.py \
${FARSI_INP_DIR}_dagInterArrTime_${dag_arr_time} \
miniera_${suffix_app} \
${num_dags} \
${LOG_PATH}/${exp_id} \
$SOC_DIM \
${RT_AWARE_BLK_SEL} \
${RT_AWARE_TASK_SEL} \
${CLUSTER_KRNLS_NOT_TO_CONSIDER} \
${heuristic_type} \
$CONSTRAIN_TOPOLOGY \
$DROP_TASKS_THAT_PASSED_DEADLINE \
$USE_CUST_SCHED_POLICIES \
$SINGLE_RUN \
$OVERWRITE_INIT_SOC \
$DYN_SCHEDULING_INSTEAD_OF_MAPPING \
$CUST_SCHED_POLICY_NAME \
$CUST_SCHED_CONSIDER_DM_TIME \
$DYN_SCHEDULING_MEM_REMAPPING \
$EXPLR_TIMEOUT \
$FARSI_INT_PARALLELISM \
&> ${LOG_PATH}/${exp_id}/run.log &"
                        else
                            cmd="REGRESSION_RUNS=1 python FARSI_what_ifs_with_params.py \
${FARSI_INP_DIR}_dagInterArrTime_${dag_arr_time} \
miniera_${suffix_app} \
${num_dags} \
${LOG_PATH}/${exp_id} \
$SOC_DIM \
${RT_AWARE_BLK_SEL} \
${RT_AWARE_TASK_SEL} \
${CLUSTER_KRNLS_NOT_TO_CONSIDER} \
${heuristic_type} \
$CONSTRAIN_TOPOLOGY \
$DROP_TASKS_THAT_PASSED_DEADLINE \
$USE_CUST_SCHED_POLICIES \
$SINGLE_RUN \
$OVERWRITE_INIT_SOC \
$DYN_SCHEDULING_INSTEAD_OF_MAPPING \
$CUST_SCHED_POLICY_NAME \
$CUST_SCHED_CONSIDER_DM_TIME \
$DYN_SCHEDULING_MEM_REMAPPING \
$EXPLR_TIMEOUT \
$FARSI_INT_PARALLELISM \
$RUN_PARSED_SYSTEM \
 &> ${LOG_PATH}/${exp_id}/run.log &"
                        fi
                        log "INFO" "Running command: ${cmd}"
                        echo "cd ${PWD} && ${cmd}" > ${LOG_PATH}/${exp_id}/cmd.sh
                        eval $cmd
                    done
                done
            done
        done
    fi
    log "INFO" "Waiting until previous commands finish..."
    wait
}

soc_x_dim=$2
soc_y_dim=$3
dag_arr_time=$4
ndags=$5
nc="$6"
nr="$7"
nv="$8"

run $soc_x_dim $soc_y_dim $dag_arr_time $ndags "$nc" "$nr" "$nv"

log "INFO" "./run_all.sh finished."
