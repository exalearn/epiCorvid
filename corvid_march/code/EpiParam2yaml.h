/* class EpiPara2yaml
 *
 * Jan Balewski
 * 03/2020

Primitive parser of Epi-config file.
Processes 1 line at a time and writes instantly to [confname.meta].yaml
*first word is name of the dictionary
* count number of subsequent words (space separated)
* if 1 --> it is a value
* if more than one, it is a list of values
*/

#include <climits>
#include <cstring>
#include <string>
using namespace std;
class EpiData2hd5;

class EpiPara2yaml {
 public:
  EpiPara2yaml( string sumFname);
  ~EpiPara2yaml() { fclose(fdm); }
  void save_config(const char *configname);
  void save_hd5meta(const EpiData2hd5 * );
 private:
  FILE* fdm;
  int get_words(char *buffer, char **words, int mxwrd);
  
};

