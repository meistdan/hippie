#!/bin/bash

cd  ../../bin
CUDART=cudart.exe
ENV_FILES=../test/RaySorting/env/*.env;
NUMBER_OF_ENV_FILES=$(ls -l $ENV_FILES | wc -l);
i=1;

START=$(date)
for ENV in $ENV_FILES; do
	echo $ENV;
	./$CUDART $ENV
	rm $ENV;
	echo $i/$NUMBER_OF_ENV_FILES
	i=$((i+1));
done
END=$(date)

echo $START
echo $END
