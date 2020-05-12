/* unit test  of EpiData_to_hd5
accumulate 3D array in memory, write at the end

Compile:
 module load cray-hdf5
 CC test_EpiData2hd5.cpp EpiData2hd5.cpp

Execute:  ./a.out
Print HD5 output:  /dumpHD5.py -N epiFab2

 module load tensorflow/gpu-2.0.0-py37
*/

#include "EpiData2hd5.h"

THIS TEST IS BROKEN

//=======================
//=======================
//=======================

int main (void) {

  int nDay=5, nx=4, ny=3;
  EpiData2hd5 epi5(nx,ny,nDay);

  // populate data-buff w/ values
  for (int k=0; k<nDay; k++){
    printf("new day=%d\n",k);
    for (int j=0 ; j<ny;j++) {
      for (int i=0 ; i<nx;i++)	{
	int val=1+i+2*j*j+k;
	epi5.addTr(i,j,k,val);
      }
    }
  }
  epi5.doDiff();// convert integrals to day-differences
  epi5.writeH5("setTr","epiFab2.h5");
  
}
