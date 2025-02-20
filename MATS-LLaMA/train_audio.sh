export TOKENIZERS_PARALLELISM=false
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_port 1236 \
    train.py --cfg-path configs/audio_config_open.yaml \
    --options gpu=0,1