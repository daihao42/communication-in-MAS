#!/usr/bin/zsh

learner_num=$1
actor_per_learner=$2
((world_size=$learner_num*$actor_per_learner+$learner_num))

for i in {0..$(($learner_num-1))}
do
  echo "rank_"$i 
  echo "world_size_"$world_size
done

for i in {$learner_num..$(($world_size-1))}
do
  echo "rank_"$i 
  echo "learner_rank_"$((($i-$learner_num)/$actor_per_learner))
done
