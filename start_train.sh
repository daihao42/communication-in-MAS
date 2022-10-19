#!/usr/bin/zsh

nohup python test_learner_inc.py --rank 0 --world-size 4 > logs/nohups/learner-0.out &

for i in {1..3}
do
  nohup python test_actor_inc.py --rank $i --world-size 4 --max-epochs 1000000 > logs/nohups/actor-$i.out &
done
