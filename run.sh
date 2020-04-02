#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=3 --node_rank=0 --master_addr="128.200.93.82" --master_port=1234 dist_main.py