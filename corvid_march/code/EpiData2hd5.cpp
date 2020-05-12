#include <cstring>
#include "EpiData2hd5.h"
// needs: module load cray-hdf5

EpiData2hd5::EpiData2hd5(int nx1, int ny1, int nz1){
  //Create the data space with fixed dimensions.
  nx=nx1; ny=ny1; nz=nz1;
  int size=nx*ny*nz*sizeof(int);
  dataTract=(int*)malloc(size);
  memset(dataTract, 0, size);
  nzy=nz*ny;
 
 }


void EpiData2hd5::addLabel(string name,float value, float lb, float ub){
  assert (numLab<maxLab);
  labelName[numLab]=name;
  labelValue[numLab]=value;
  labelBound[numLab][0]=lb;
  labelBound[numLab][1]=ub;
  assert (lb <=value);
  assert (value <=ub);
  assert (lb <ub);
  float avr=(ub+lb)/2;
  float del=(ub-lb)/2;
  labelNorm[numLab]= (value-avr)/del;
  
  printf("addLab: %d %s %f\n",numLab,labelName[numLab].c_str(),labelValue[numLab]);
  numLab++;

}

void EpiData2hd5::addTractItem(int i, int j, int k, int val){
  if( val<=0 ) return;
  //if (i==120 && j==2)
  // printf("i=%d, j=%d, k=%d , idx=%d  val=%d\n",i,j,k,i*nzy + j*nz + k,val);
  dataTract[i*nzy + j*nz + k]+=val;
}  


void EpiData2hd5::doDayDiff(){
  for (int i=0; i<nx; i++)
    for(int j=0; j<ny; j++)
      for (int k=1; k<nz; k++){
	int i1=nz-k +i*nzy + j*nz;
	int i2=i1-1;
	//printf("\ni=%d, j=%d, k=%d , i1=%d, i2=%d\n",i,j,k,i1,i2);
	dataTract[i1]=dataTract[i1]-dataTract[i2];
      }
  printf("doDayDiff done");
  
}


void EpiData2hd5::dumpNonZeroTract(){
  printf("\n\n dump Tract start\n");
  for (int i=0; i<nx; i++)
    for(int j=0; j<ny; j++) {
      int cnt=0;
      for (int k=0; k<nz; k++)
	if (dataTract[i*nzy + j*nz + k] !=0) cnt++;
      if (cnt==0) continue;
      printf("\ni=%d, j=%d,",i,j);
      for (int k=0; k<nz; k++)
	printf("%2d, ",dataTract[i*nzy + j*nz + k]);
    }
  printf("\n dump Tract done\n");
}


void EpiData2hd5::dumpLabels(){
  printf("\n dump %d labels\n",maxLab);
  for (int i=0; i<maxLab; i++){
    printf("%d %s %f\n",i,labelName[i].c_str(),labelValue[i]);
  }
}


int EpiData2hd5::writeH5(string coreName){

  h5name=coreName+".h5";
  H5std_string  setName=H5std_string("symptomatic3D");
  H5std_string  setName2=H5std_string("parBio");
  H5std_string  setName3=H5std_string("uniBio");
  H5std_string  h5Name=H5std_string(h5name);

  // repack 3D array to ushort
  int size=nx*ny*nz;
  ushort*data16=(ushort*)malloc(size*sizeof(ushort));
  int nNeg=0, nHuge=0;
  for (int i=0 ; i<size; i++){
    int x=dataTract[i];
    if (x<0) { nNeg++; x=0; }
    if ( x>65535 ) {nHuge++; x=65535;}
    data16[i]=x;
  }
  if (nNeg>0)  printf("\nWarn, %d negative 3D values clipped to 0\n",nNeg);
  if (nHuge>0)  printf("\nWarn, %d huge 3D values clipped to 64k\n",nHuge);

  
  printf("EpiData2hd5::writeH5 -->%s\n",coreName.c_str());
  // Create a new file using the default property lists. 
  H5File file(h5Name, H5F_ACC_TRUNC);

  // Create the data space for the dataset and the dataset
  hsize_t dimsTr[rankTr]={nx,ny,nz};
  DataSpace dataspace(rankTr, dimsTr);
  DataSet data = file.createDataSet(setName, PredType::NATIVE_USHORT, dataspace);

  // write 3D array
  data.write(data16, PredType::NATIVE_USHORT);

  // the same for bioPar
  hsize_t dimsLab[1]={maxLab};
  DataSpace dataspace2(1,dimsLab);
  data = file.createDataSet(setName2, PredType::NATIVE_FLOAT, dataspace2);
  data.write(labelValue, PredType::NATIVE_FLOAT);
 
  // the same for uniPar
  data = file.createDataSet(setName3, PredType::NATIVE_FLOAT, dataspace2);
  data.write(labelNorm, PredType::NATIVE_FLOAT);
 
  
  return 0;
}

