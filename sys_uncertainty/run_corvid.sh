#!/bin/bash 
 
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value


procIdx=${SLURM_PROCID}
configFile=$1
# set up directory for this run
outPath=out_$procIdx
mkdir $outPath
cp ./corvid $configFile ./*.dat $outPath
cd $outPath
ls

# adjust random seed
seed=`od -A n -t d -N 3 /dev/random | tr -d ' '`
sed -i "s/randomnumberseed 1/randomnumberseed $seed/" $configFile
cat $configFile

# and run
./corvid $configFile


