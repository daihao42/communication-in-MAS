#!/usr/bin/zsh

ps aux|grep test_learner|awk '{print $2}'|xargs kill -9
ps aux|grep test_actor|awk '{print $2}'|xargs kill -9
