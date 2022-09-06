torchrun --nnodes $1 --nproc_per_node=$2 --node_rank $3 --master_addr="0.0.0.0"  --master_port="29700" tests/test_launch.py
