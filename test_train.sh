#!/bin/zsh
for i in {0..2}
do
  nohup python tests/test_train.py $i 3 > logs/nohups/nohup.out.$i &
  sleep 1
done
