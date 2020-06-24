import numpy as np
import glob
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.MantissaNoDotYAML1_1Warning)
import h5py
import time
import sys
import matplotlib.pyplot as plt
import subprocess


def read_yaml_silent(ymlFn):
        """Non-verbose version of exalearnEpi.corvid_march.code.Util_Func.read_yaml"""
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
        ymlFd.close()
        return bulk



class DatasetBuilder():

    def __init__(self, dataDir, dataTag, Narr, Nperjob):
        self.dataDir = dataDir # location of data files (outputs of batchRun job array)
        self.dataTag = dataTag # name tag of all batchRun outputs
        self.Narr = Narr # Number of jobs in job array
        self.Nperjob = Nperjob # number of runs per job (equals srun --ntasks in batch script)
        self.total = Narr*Nperjob
        print('Expecting  %d total runs in %s'%(self.total, self.dataDir))


    def copy_HD5s_to_cfs(self, cDir):
        print('Copying HDF5s to %s'%cDir)
        run=0
        parBios = []
        uniBios = []
        symptomatic3Ds = []

        start=time.time()
        for arrIdx in range(self.Narr):
            jobDir=self.dataDir+self.dataTag+'-'+str(arrIdx)+'/'
            print('Parsing job directory %s'%jobDir)
            for procIdx in range(self.Nperjob):
                metaFile=glob.glob(jobDir+'out_'+str(procIdx)+'/*meta.yaml')[0]
                meta = read_yaml_silent(metaFile)
                metah5 = meta['hd5meta']
                paramnames = metah5['parBio']['axis_bins']
                arraynames = metah5['symptomatic3D']['axis_name']
                age_bins = metah5['symptomatic3D']['age_group_bins']

                h5File = glob.glob(jobDir+'out_'+str(procIdx)+'/*.h5')[0]
                with h5py.File(h5File, 'r') as f:
                    parBios.append(f['parBio'][...])
                    uniBios.append(f['uniBio'][...])
                    symptomatic3Ds.append(f['symptomatic3D'][...])
                run += 1
        parBios = np.stack(parBios, axis=0)
        uniBios = np.stack(uniBios, axis=0)
        symptomatic3Ds = np.stack(symptomatic3Ds, axis=0).astype(np.ushort)
        with h5py.File(cDir+'data.h5', 'w') as f:
            f.create_dataset('parBio', data=parBios)
            print('Wrote field %s, shape %s'%('parBio', str(parBios.shape)))
            f.create_dataset('uniBio', data=uniBios)
            print('Wrote field %s, shape %s'%('uniBio', str(uniBios.shape)))
            f.create_dataset('symptomatic3D', data=symptomatic3Ds)
            print('Wrote field %s, shape %s'%('symptomatic3D', str(symptomatic3Ds.shape)))
        print('DONE')


    def copy_summaries_to_cfs(self, cDir):
        print('Copying summaries to %s'%cDir)
        parser = RunParser(self.dataDir+self.dataTag, self.Narr, self.Nperjob)
        parser.parse_runs()
        parser.save_h5(cDir+'summary.h5')

    def inspect(self):
        run=0
        parBios = []
        uniBios = []
        symptomatic3Ds = []
        
        start=time.time()
        for arrIdx in range(self.Narr):
            jobDir=self.dataDir+self.dataTag+'-'+str(arrIdx)+'/'
            print('Parsing job directory %s'%jobDir)
            for procIdx in range(self.Nperjob):
                metaFile=glob.glob(jobDir+'out_'+str(procIdx)+'/*meta.yaml')[0]
                meta = read_yaml_silent(metaFile)
                metah5 = meta['hd5meta']
                paramnames = metah5['parBio']['axis_bins']
                arraynames = metah5['symptomatic3D']['axis_name']
                age_bins = metah5['symptomatic3D']['age_group_bins']

                h5File = glob.glob(jobDir+'out_'+str(procIdx)+'/*.h5')[0]
                with h5py.File(h5File, 'r') as f:
                    parBios.append(f['parBio'][...])
                    uniBios.append(f['uniBio'][...]) 
                    symptomatic3Ds.append(f['symptomatic3D'][...])
                run += 1
        parBios = np.stack(parBios, axis=0)
        uniBios = np.stack(uniBios, axis=0)
        symptomatic3Ds = np.stack(symptomatic3Ds, axis=0)
        self.plot_params(parBios, uniBios, paramnames)
        self.plot_arrays(symptomatic3Ds, arraynames)
        print('Parsing %d runs took %f s'%(run, time.time() - start)) 


    def plot_arrays(self, arr, arrnames):
        hbins = np.arange(-0.5,134.5,1)
        centers = np.arange(134)

        hist, _ = np.histogram(arr, bins=hbins)

        plt.figure()
        plt.plot(centers, hist, 'o')
        plt.yscale('log')
        plt.xlabel('Num. new infected per age/day/tract')
        plt.ylabel('Counts')
        plt.title('symptomatic3D')
        plt.show()


    def plot_params(self, par, uni, names, nbins=20):
        numpars = par.shape[1]
        numruns = par.shape[0]
        plt.figure(figsize=(10,10))
        for idx in range(16):
           plt.subplot(4,4,idx+1)
           if idx%2 == 0:
                plt.hist(par[:,idx//2], bins=nbins)
                plt.title(names[idx//2])
           else:
                plt.hist(uni[:,idx//2], bins=nbins)
                plt.title(names[idx//2]+' (normed)')
        plt.tight_layout()
        plt.show()




class RunParser():
    """Class to parse yaml summary outputs of a set of runs into h5"""

    def __init__(self, dataDir=None, Narr=None, Nperjob=None, load_from_h5=False):
        # Usage check
        if not load_from_h5 and dataDir is None:
            print("Error: must specify either dataDir to parse a set of runs or load_from_h5 to load from an HDF5 file")
            sys.exit()
        if dataDir is not None and (Narr is None or Nperjob is None):
            print("Error: must specify Narr and Nperjob if parsing from dataDir '%s'"%dataDir)
            sys.exit()

        # fields to track
        self.fields = ['Number symptomatic (daily)',
                       'Cumulative symptomatic (daily)',
                       'Total infection attack rates by age', 
                       'Total infection attack rate', 
                       'Total infected individuals by age', 
                       'Total symptomatic attack rates by age', 
                       'Total symptomatic attack rate', 
                       'Total symptomatic individuals by age', 
                       'Total symptomatic unvaccinated individuals by age', 
                       'Total unvaccinated individuals by age', 
                       'Total individuals by age'] 

        # fields already listed by corvid as daily output
        self.fields_daily = ['Number symptomatic (daily)',
                             'Cumulative symptomatic (daily)'] 

        # Age ranges (first bin empty for mplt xtick labels)
        self.age_bins = ['', '0-4', '5-18', '19-29', '30-64', '65+'] 

        if load_from_h5:
            self.load_h5(load_from_h5)
        else: 
            self.dataDir = dataDir # location of data files (outputs of batchRun job array)
            self.Narr = Narr # Number of jobs in job array
            self.Nperjob = Nperjob # number of runs per job (equals srun --ntasks in batch script)
            self.total = Narr*Nperjob
            print('Expecting  %d total runs in %s'%(self.total, self.dataDir))
            
            metaFile = glob.glob(dataDir+'-0/out_0/*meta.yaml')[0]
            meta = read_yaml_silent(metaFile)['config']
            self.summfname = meta['summaryfilename']
            self.Ndays = meta['runlength']
            print('Summary names=%s,'%self.summfname, 'Ndays=%d'%self.Ndays)

            # Allocate arrays in the data aggregation dict
            self.allocate_arrays()
            print('Allocated arrays:')
            for k,v in self.agg.items():
                print(k, v.shape)


    def allocate_arrays(self):
        # Allocate arrays
        yamfile = self.dataDir+'-0/out_0/'+self.summfname
        dat = read_yaml_silent(yamfile)
        self.agg = {}
        for k,v in dat['DailySummary'][-1].items():
            if k in self.fields:
                if type(v) != list:
                    # Scalar field
                    self.agg[k] = np.zeros(shape=(self.total,self.Ndays,1), dtype=np.float)
                elif len(v) == self.Ndays:
                    # Fields automatically gathered as daily counts
                    self.agg[k] = np.zeros(shape=(self.total,self.Ndays,1), dtype=np.float)
                else:
                    # Others (age binned data)
                    axdim = np.array(v).shape[0]
                    self.agg[k] = np.zeros(shape=(self.total,self.Ndays,axdim), dtype=np.float)



    def parse_runs(self):
        run=0
        start=time.time()
        for arrIdx in range(self.Narr):
            jobDir=self.dataDir+'-'+str(arrIdx)+'/'
            print('Parsing job directory %s'%jobDir)
            for procIdx in range(self.Nperjob):
                yamFl=jobDir+'out_'+str(procIdx)+'/'+self.summfname
                
                days = read_yaml_silent(yamFl)['DailySummary']
                for idx, day in enumerate(days):
                    for k,v in day.items():
                        if k in self.fields:
                            if k in self.fields_daily:
                                # Don't read pre-formatted daily data until last day
                                if len(v)==self.Ndays:
                                    self.agg[k][run,:,0] = np.array(v)
                            else:
                                self.agg[k][run,idx,:] = np.array(v)
                run += 1
        print('Parsing %d runs took %f s'%(run, time.time() - start))


    def save_h5(self, fname):
            """Saves datadict fields into h5 file"""
            with h5py.File(fname, 'w') as f:
                for k,v in self.agg.items():
                    f.create_dataset(k, data=v)
            print('Saved h5 to %s'%fname)

    def load_h5(self, fname):
            """Loads h5 fields into datadict"""
            self.agg = {}
            with h5py.File(fname, 'r') as f:
                fields = list(f.keys())
                for field in fields:
                    self.agg[field] = f[field][...]
            print('From %s loaded following fields:'%fname)
            for k,v in self.agg.items():
                print(k, v.shape)

