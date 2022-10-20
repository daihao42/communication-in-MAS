#!/usr/bin/zsh
#
learner_num=$1
actor_per_learner=$2
((world_size=$learner_num*$actor_per_learner+$learner_num))

for i in {$learner_num..$(($world_size-1))}
do
  nohup python test_actor_inc.py --rank $i --world-size $world_size --parallelism 3 --learner-rank $((($i-$learner_num)/$actor_per_learner)) --max-epochs 10000 > logs/nohups/actor-$i.out &
  sleep 1
done

log_time=`date +%s`

for i in {0..$(($learner_num-1))}
do
  sleep 1
  nohup python test_learner_inc.py --rank $i --world-size $world_size --log-time $log_time > logs/nohups/learner-$i.out &
done

