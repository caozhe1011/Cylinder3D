#!/bin/sh

NAME=/home/yy/.conda/envs/condayy/bin/python
echo $NAME
ID=`ps -ef | grep "$NAME" | grep -v "grep" | awk '{print $2}'`
#ID=`ps -ef | grep "$NAME" | grep -v "grep" | awk '{print $2}' | awk '{print$8}'`
echo $ID
echo "---------------"
for id in $ID
do
	kill -9 $id
	echo "killed $id"
done
echo "---------------"

