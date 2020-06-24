#!/bin/bash 
 
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value



myRandomF () {
    rndI=`od -A n -t d -N 3 /dev/random | tr -d ' '`
    # This is 3-byte truly random generator, max value: 256^3=16777216-1
    rndF=` echo $rndI | awk  '{printf "%.7f", $1/ 16777216 *2 -1. }'`
}


procIdx=${SLURM_PROCID}
configFile=$1
# set up directory for this run
outPath=out_$procIdx
mkdir $outPath
cp ./corvid $configFile ./*.dat $outPath
cd $outPath
ls

# adjust random seed
myRandomF
seed=$rndI
sed -i "s/randomnumberseed 1/randomnumberseed $seed/" $configFile

# adjust params
myRandomF
uresponseday=$rndF
myRandomF
uwfhcomp=$rndF
myRandomF
ullcomp=$rndF


responseday=` echo $uresponseday | awk '{printf "%d", 7*(8+int(2*$1))}'`
wfhcomp=` echo $uwfhcomp | awk '{printf "%f", int(5+5*$1)/10.}'`
llcomp=` echo $ullcomp | awk '{printf "%f", int(5+5*$1)/10.}'`

sed -i "s/workfromhome .*/workfromhome $wfhcomp/" $configFile
sed -i "s/liberalleave .*/liberalleave $llcomp/" $configFile
sed -i "s/responseday .*/responseday $responseday/" $configFile
cat $configFile

# and run
./corvid $configFile


