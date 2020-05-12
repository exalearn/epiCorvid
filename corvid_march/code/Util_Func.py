__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# module load tensorflow/gpu-2.0.0-py37
import numpy as np
import time, os
import ruamel.yaml  as yaml
from pprint import pprint

import csv
import h5py

#...!...!..................
def read_yaml(ymlFn):
        print('  read  yaml:',ymlFn,end='')
        start = time.time()
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
        ymlFd.close()
        print(' done, size=%d'%len(bulk),'  elaT=%.1f sec'%(time.time() - start))
        return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):
        start = time.time()
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1048576
        if verb:
                print('  closed  yaml:',ymlFn,' size=%.2f MB'%xx,'  elaT=%.1f sec'%(time.time() - start))

#...!...!..................
def write_data_hdf5(dataD,outF,verb=1):
    h5f = h5py.File(outF, 'w')
    if verb>0: print('saving data as hdf5:',outF)
    for item in dataD:
        rec=dataD[item]
        h5f.create_dataset(item, data=rec)
        if verb>0:print('h5-write :',item, rec.shape)
    h5f.close()
    xx=os.path.getsize(outF)/1048576
    print('closed  hdf5:',outF,' size=%.2f MB'%(xx))

    
#...!...!..................
def read_data_hdf5(inpF):
        print('read data from hdf5:',inpF)
        h5f = h5py.File(inpF, 'r')
        objD={}
        for x in h5f.keys():
            obj=h5f[x][:]
            print('read ',x,obj.shape,obj.dtype)
            objD[x]=obj

        h5f.close()
        return objD

    
#............................
def read_one_csv(fname,delim=','):
    print('read_one_csv:',fname)
    tabL=[]
    with open(fname) as csvfile:
        drd = csv.DictReader(csvfile, delimiter=delim)
        print('see %d columns'%len(drd.fieldnames),drd.fieldnames)
        for row in drd:
            tabL.append(row)

    print('got %d rows \n'%(len(tabL)))
    #print('LAST:',row)
    return tabL,drd.fieldnames

#............................
def write_one_csv(fname,rowL,colNameL):
    print('write_one_csv:',fname)
    print('export %d columns'%len(colNameL), colNameL)
    with open(fname,'w') as fou:
        dw = csv.DictWriter(fou, fieldnames=colNameL)#, delimiter='\t'
        dw.writeheader()
        for row in rowL:
            dw.writerow(row)    
