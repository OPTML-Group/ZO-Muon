MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
EPS=${EPS:-1e-3}
RANK=${RANK:-2}
TRUNCATION_RANK=${TRUNCATION_RANK:-2}
ENERGY_THRESHOLD=${ENERGY_THRESHOLD:0.9}
STEP_INTERVAL=${STEP_INTERVAL:-100}
TASK=${TASK:-'SST2'}
# Trainer=${Trainer:-'LowDim'}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
LR=${LR:-1e-5}
STEPS=${STEPS:-10000}
EVAL_STEPS=${EVAL_STEPS:-10000}
OPT=${OPT:-'sgd'}
MODE=${MODE:-ft}
REDUCE_RANK=${REDUCE_RANK:-False}
MULTIPLE_SAMPLE=${MULTIPLE_SAMPLE:-False}
SWITCH_OPTIM=${SWITCH_OPTIM:-False}
COMPUTE_VAR=${COMPUTE_VAR:-False}
MAX_TIME=${MAX_TIME:-0}
LOAD_MODE=${LOAD_MODE:-float16} 

THRESHOLD=${THRESHOLD:-15000}
NUM_SAMPLES=${NUM_SAMPLES:-4}

EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
SEED=0
Tainer=subzero


LOAD_FLAG=""
if [ "$LOAD_MODE" == "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" == "float16" ]; then
    LOAD_FLAG="--load_float16"
else
    echo "Warning: No specific load mode selected, defaulting to standard float32 or model default."
fi

case $TASK in
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

TAG=$Tainer-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL-$RANK-$MAX_TIME


echo $TAG
echo "Task: $TASK"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "RANK: $RANK"
echo "STEP INTERVAL: $STEP_INTERVAL"

python ../run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $Tainer \
    $LOAD_FLAG \
    --zo_optimizer $OPT \
    --reduce_rank $REDUCE_RANK \
    --multiple_sample $MULTIPLE_SAMPLE \
    --adam_eps 1e-6 \
    --max_time $MAX_TIME \
    --truncation_rank $TRUNCATION_RANK \
    --switch_optim $SWITCH_OPTIM \
    --num_samples $NUM_SAMPLES \
    --threshold $THRESHOLD \
    --compute_variance $COMPUTE_VAR \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --step_interval $STEP_INTERVAL \
    --rank_r $RANK \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 





