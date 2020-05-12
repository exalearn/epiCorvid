#!/usr/bin/env python3
""" 
test validity of output files 
on Cori:
 module load tensorflow/gpu-2.0.0-py37

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from Util_Func import read_one_csv
from pprint import pprint
import numpy as np

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--dataPath",
                        default='./',
                        help="path to input")
    parser.add_argument("--outPath",
                        default='out',help="output path for plots and tables")

    parser.add_argument("-k", "--depth", type=int, default=2,
                        help="depth of printed  tree ")

    parser.add_argument("-N", "--name",
                        default='Individuals-seattle26b.txt',
                        help="core name")
    
    args = parser.parse_args()

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================
args=get_parser()

print('\n ---- check meta file ----')
inpF=args.dataPath+args.name
dataL,keyL=read_one_csv(inpF)
print('M:',keyL)

useK=['infectedtime', 'incubationdays', 'dayneighborhood']
print('\n ---- select some keys ----',useK)
dataL2=[]
for rec in dataL:
    rowL=[ int(rec[x]) for x in useK ]
    dataL2.append(rowL)
    #break

dataA=np.array(dataL2)
print('found ',dataA.shape,dataA)
for i,k in enumerate(useK):
    print(i,k, np.amin(dataA[:,i]), np.amax(dataA[:,i]))

