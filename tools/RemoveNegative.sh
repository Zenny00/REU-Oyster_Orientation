#!/bin/bash

for file in *.txt
do
	filename="${file%%.txt}"
	filename="${filename##*/}"
	
	awk -vOFS=' ' '{for(i=1;i<=NF;i++)if($i<0||$i=="-nan")$i=0}1' $filename.txt > $filename.tmp && mv $filename.tmp $filename.txt
done

exit 0

