#!/usr/bin/env python3
""" 
test validity of HD5 output files 
on Cori:
 module load tensorflow/gpu-2.0.0-py37

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from Util_Func import read_data_hdf5
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

    parser.add_argument("-k", "--depth", type=int, default=2,
                        help="depth of printed  tree ")

    parser.add_argument("-N", "--name",
                        default='h5tutr_dset',
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
if '.h5' != args.name[-3:] : args.name+='.h5'
print('\n ---- check HD5 file ----')
inpF=args.dataPath+args.name
blob=read_data_hdf5(inpF)
    
print('\ncheck parBio:')
ds=blob['parBio']
print(ds)
    
print('\ncheck uniBio:')
ds=blob['uniBio']
print(ds)
    
print('\ncheck symptomatic3D: iTr, tag, [..days..]')
ds=blob['symptomatic3D']
 
for itr in range(ds.shape[0]):
    for tag in range(ds.shape[1]):
        vec=ds[itr,tag,:]
        if sum(vec)<=0: continue
        print( itr,tag,vec)
