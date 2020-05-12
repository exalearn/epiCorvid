/* main program for Corvid
 *
 * Dennis Chao
 * March 2020

Added HD5 + YAML outputs by LBNL ExaLearn team: April 2020
To compile on Cori:

module load cray-hdf5
cd corvid_march/code
make

 */

#include "epimodel.h"
#include "epimodelparameters.h"
#include "EpiParam2yaml.h"

int main(int argc, char *argv[]) {
  char *configname=NULL;
  if (argc==2)
    configname = argv[1];
  else {
    cerr << "Usage: " << argv[0] << " configfilename" << endl;
    exit(-1);
  }

  EpiModelParameters parms(configname);
  EpiModel model(parms);
  //Jan: keep this order: information flows between all 3 classes - it is complicated
  string coreName=configname;
  coreName.replace(0,7,"");
  
  EpiPara2yaml meta(coreName );
  meta.save_config(configname);
  model.prepH5(parms);
  model.run();
  model.saveH5(coreName);
  meta.save_hd5meta(model.getEpiDataInfo());

  return 0;
}
