torchrun --nnodes 1 --nproc_per_node=3 --group_rank $1 --master_addr="127.0.0.1"  --master_port="29700" tests/test_launch.py
