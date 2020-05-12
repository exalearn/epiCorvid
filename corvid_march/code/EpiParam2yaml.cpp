/* class EpiPara2yaml
 */

#include <assert.h>
#include <stdio.h>
#include "EpiData2hd5.h"
#include "EpiParam2yaml.h"

EpiPara2yaml::EpiPara2yaml( string name){
  printf("EpiPara2yaml sumFname=%s\n",name.c_str());
 
  string metaFname=name+".meta.yaml";
  printf("EpiPara2yaml metaFname=%s\n",metaFname.c_str());
  fdm = fopen(metaFname.c_str(), "w");
  assert (fdm != NULL);
}



void EpiPara2yaml::save_hd5meta(const EpiData2hd5 * h5Info) {
  assert (fdm != NULL);
  //printf("AAA=%s %d",h5Info->labelName[0].c_str(),h5Info->numLab);
  fprintf(fdm,"hd5meta:\n");
  fprintf(fdm,"  fileName: %s\n",h5Info->h5name.c_str());
  fprintf(fdm,"  symptomatic3D:\n");
  fprintf(fdm,"    axis_name: [ census_tract, age_group, simu_day]\n");
  fprintf(fdm,"    age_group_bins: [0-4, 5-18, 19-29, 30-64, 65-plus]\n");
  fprintf(fdm,"    axis_shape: [ %d, %d, %d ]\n",h5Info->nx,h5Info->ny,h5Info->nz);

  fprintf(fdm,"  parBio:\n");
  fprintf(fdm,"    axis_bins:\n");
  for (int i=0; i< h5Info->numLab; i++)
    fprintf(fdm,"      - %s\n",h5Info->labelName[i].c_str());

  fprintf(fdm,"  bioRange:\n");
  for (int i=0; i< h5Info->numLab; i++)
    fprintf(fdm,"   %s: [%.1f, %.1f]\n",h5Info->labelName[i].c_str(),h5Info->labelBound[i][0],h5Info->labelBound[i][1]);

  
}



void EpiPara2yaml::save_config(const char *configname) {
  assert (fdm != NULL);
  fprintf(fdm,"config: \n");
      
  const int mxbuf = 255;
  char line[mxbuf+1];

  FILE* fd = fopen(configname, "r");
  //check if file exists
  if (fd == NULL){
    printf("file does not exists %s", configname);
    return ;
  }
  
  const int mxwrd=100;
  char *words[mxwrd];
  //read line by line
  while (fgets(line, mxbuf, fd) != NULL)  {
    printf(line);
    if ( line[0]== '#' ) continue;
    if (strlen(line)<2 ) continue; // skip empty lines
    int nw=get_words(line,words,mxwrd); // is destructive for line
    //printf("nw=%d, key==%s linelen=%ld\n",nw,words[0],strlen(line));
    assert ( nw >1 ); // needs at least key,value
    if (nw==2) {
      fprintf(fdm,"  %s: %s",words[0],words[1]);
    } else { // it is a list
      fprintf(fdm,"  %s: [ ",words[0]);
      for ( int i=1; i<nw-1; i++) fprintf(fdm," %s, ",words[i]);
      fprintf(fdm," %s ] ",words[nw-1]);
    }    
  }
  fclose(fd);
}

int EpiPara2yaml::get_words(char *buffer, char **array, int mxwrd){
  int i=0;
  array[i] = strtok(buffer," ");
  while(array[i]!=NULL) {
    array[++i] = strtok(NULL," ");
    assert (i <mxwrd);
  }
  return i;
}
