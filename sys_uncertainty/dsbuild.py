import numpy as np
from utils import DatasetBuilder

dataDir = '/global/cscratch1/sd/pharring/corvid_demo/'
Cdir = '/global/cfs/cdirs/covid19/datasets/corvid2parB_10k/'
dataTag = 'MLruns'
Narr = 157
Nperjob = 64

builder = DatasetBuilder(dataDir, dataTag, Narr, Nperjob)

builder.copy_HD5s_to_cfs(Cdir)
#builder.copy_summaries_to_cfs(Cdir)

