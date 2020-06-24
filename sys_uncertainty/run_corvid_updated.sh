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

# adjust R0 and WFH params
myRandomF
uR0=$rndF
myRandomF
uwfhdays=$rndF
myRandomF
uwfhcomp=$rndF

R0=` echo $uR0 | awk '{printf "%f", (2.5+0.5*$1)}'`
wfhdays=` echo $uwfhdays | awk '{printf "%d", int(67+23*$1)}'`
wfhcomp=` echo $uwfhcomp | awk '{printf "%f", (0.45+0.45*$1)}'`
sed -i "s/R0 .*/R0 $R0/" $configFile
sed -i "s/workfromhome .*/workfromhome $wfhcomp/" $configFile
sed -i "s/workfromhomedays .*/workfromhomedays $wfhdays/" $configFile
cat $configFile

# and run
./corvid $configFile


