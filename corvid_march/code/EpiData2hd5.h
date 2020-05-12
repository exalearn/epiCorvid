#include <string>
#include <assert.h>
using namespace std;

#include "H5Cpp.h"
using namespace H5;
class EpiPara2yaml;

class EpiData2hd5 {
private:
  friend class EpiPara2yaml; 
  enum  {  rankTr = 3, maxLab=9};
  int *dataTract;
  float labelValue[maxLab];
  float labelNorm[maxLab];
  float labelBound[maxLab][2];
  string labelName[maxLab];
  int numLab=0;
  int nx,ny,nz,nzy;  // dataset fixed dimensions
  string h5name="";
public:
  EpiData2hd5( int nx, int ny, int nz);
  ~EpiData2hd5() { }
  void addTractItem(int i, int j, int k, int val);
  void doDayDiff();
  int  writeH5(string coreName);
  void dumpNonZeroTract();
  void dumpLabels();
  void addLabel(string a,float b) {addLabel(a,b,0.,1.);}
  void addLabel(string,float,float, float);
  const EpiData2hd5 *getEpiDataInfo() { return  this; }
};
