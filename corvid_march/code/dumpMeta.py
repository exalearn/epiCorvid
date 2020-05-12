#!/usr/bin/env python3
""" 
test validity of YAML output files
on Cori:
 module load tensorflow/gpu-2.0.0-py37
 
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from Util_Func import read_yaml
from pprint import pprint

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

    parser.add_argument("-n", "--events", type=int, default=100,
                        help="events for training, use 0 for all")

    parser.add_argument("-N", "--name",
                        default='Summary-seattle26b',
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

if '.meta.yaml' != args.name[-10:] : args.name+='.meta.yaml'

print('\n ---- check meta file ----')
metaF=args.dataPath+args.name
blob=read_yaml(metaF)
pprint(blob)

