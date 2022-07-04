#!/bin/bash

cd  ../../bin
HIPPIE=hippie.exe
ENV_FILES=../test/Comparison/env/*.env;
NUMBER_OF_ENV_FILES=$(ls -l $ENV_FILES | wc -l);
i=1;

START=$(date)
for ENV in $ENV_FILES; do
	echo $ENV;
	./$HIPPIE $ENV
	rm $ENV;
	echo $i/$NUMBER_OF_ENV_FILES
	i=$((i+1));
done
END=$(date)

echo $START
echo $END
