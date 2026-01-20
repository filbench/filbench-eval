#!/bin/bash

REVISION=$1
HF_ORG="ljvmiranda921"

cd lighteval
# For models in HuggingFace and accessible via vLLM
cat examples/tasks/all_filbench_tasks.txt | xargs -I {} \
    python3 -m lighteval vllm "model_name=ljvmiranda921/msde-sft-dev,revision=$REVISION,tensor_parallel_size=1,max_model_length=4096" {} \
    --push-to-hub \
    --results-org $HF_ORG \
    --custom-tasks community_tasks/filbench_evals.py \
    --max-samples 1000