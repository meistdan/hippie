#!/bin/bash

cd  ../../bin
CUDART=cudart.exe
ENV_FILES=../test/InsertBvh/env/*.env;
NUMBER_OF_ENV_FILES=$(ls -l $ENV_FILES | wc -l);
i=1;

date;
for ENV in $ENV_FILES; do
	echo $ENV;
	./$CUDART $ENV
	rm $ENV;
	echo $i/$NUMBER_OF_ENV_FILES
	i=$((i+1));
done
date;