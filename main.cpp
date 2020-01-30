/*
 * This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/BTCCollider).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Timer.h"
#include "BTCCollider.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include "hash/sha512.h"
#include "hash/sha256.h"

#define RELEASE "1.1"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("BTCCollider [-check] [-v] [-gpu]\n");
  printf("            [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
  printf("            [-o outputfile] [-s collisionSize] [-t nbThread] [-d dpBit]\n");
  printf("            [-w workfile] [-i inputWorkFile] [-wi workInterval]\n");
  printf("            [-e] [-check]\n\n");
  printf(" -v: Print version\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -o outputfile: Output results to the specified file\n");
  printf(" -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
  printf(" -s: Specify size of the collision in bit (minimum 16,default is 40)\n");
  printf(" -d: Specify number of leading zeros for the DP method (default is auto)\n");
  printf(" -e: Enable extra points (symetry and endomorphisms), reduce needed step by sqrt(2)\n");
  printf(" -t threadNumber: Specify number of CPU thread, default is number of core\n");
  printf(" -w workfile: Specify file to save work into\n");
  printf(" -i workfile: Specify file to load work from\n");
  printf(" -wi workInterval: Periodic interval (in seconds) for saving work\n");
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check CPU and GPU kernel vs CPU\n");
  exit(0);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens, const string &text, char sep) {

  size_t start = 0, end = 0;
  tokens.clear();
  int item;

  try {

    while ((end = text.find(sep, start)) != string::npos) {
      item = std::stoi(text.substr(start, end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  } catch(std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

}

// ------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  // Global Init
  Timer::Init();
  rseed((unsigned long)time(NULL));

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  int a = 1;
  int dp = -1;
  bool gpuEnable = false;
  bool stop = false;
  vector<int> gpuId = {0};
  vector<int> gridSize;
  string seed = "";
  vector<string> prefix;
  string outputFile = "";
  int nbCPUThread = Timer::getCoreNumber();
  bool tSpecified = false;
  bool extraPts = false;
  bool checkFlag = false;
  uint32_t cSize = 40;
  uint64_t rekey = 0;
  Point startPuKey;
  startPuKey.Clear();
  string workFile = "";
  string iWorkFile = "";
  uint32_t savePeriod = 60;

  while (a < argc) {

    if (strcmp(argv[a], "-gpu")==0) {
      gpuEnable = true;
      a++;
    } else if (strcmp(argv[a], "-gpuId")==0) {
      a++;
      getInts("gpuId",gpuId,string(argv[a]),',');
      a++;
    } else if (strcmp(argv[a], "-stop") == 0) {
      stop = true;
      a++;
    } else if (strcmp(argv[a], "-v") == 0) {
      printf("%s\n",RELEASE);
      exit(0);
    } else if (strcmp(argv[a], "-check") == 0) {
      checkFlag = true;
      a++;
    } else if (strcmp(argv[a], "-l") == 0) {

#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if (strcmp(argv[a], "-e") == 0) {
      extraPts = true;
      a++;
    } else if (strcmp(argv[a], "-g") == 0) {
      a++;
      getInts("gridSize",gridSize,string(argv[a]),',');
      a++;
    } else if (strcmp(argv[a], "-o") == 0) {
      a++;
      outputFile = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-w") == 0) {
      a++;
      workFile = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-i") == 0) {
      a++;
      iWorkFile = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-t") == 0) {
      a++;
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      a++;
      tSpecified = true;
    } else if (strcmp(argv[a], "-wi") == 0) {
      a++;
      savePeriod = getInt("savePeriod", argv[a]);
      a++;
    } else if (strcmp(argv[a], "-s") == 0) {
      a++;
      cSize = getInt("collisionSize", argv[a]);
      if (cSize < 16 || cSize>160) {
        printf("Unexpected collision size [16,160] expected\n");
        exit(-1);
      }
      a++;
    } else if (strcmp(argv[a], "-d") == 0) {
      a++;
      dp = getInt("dpSize", argv[a]);
      a++;
    } else if (strcmp(argv[a], "-h") == 0) {
      printUsage();
    } else if (a == argc - 1) {
      prefix.push_back(string(argv[a]));
      a++;
    } else {
      printf("Unexpected %s argument\n",argv[a]);
      exit(-1);
    }

  }

  printf("BTCCollider v" RELEASE "\n");

  if(gridSize.size()==0) {
    for (int i = 0; i < gpuId.size(); i++) {
      gridSize.push_back(0);
      gridSize.push_back(0);
    }
  } else if(gridSize.size() != gpuId.size()*2) {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }

  // Let one CPU core free per gpu is gpu is enabled
  // It will avoid to hang the system
  if( !tSpecified && nbCPUThread>1 && gpuEnable)
    nbCPUThread-=(int)gpuId.size();
  if(nbCPUThread<0)
    nbCPUThread = 0;

  BTCCollider *v = new BTCCollider(secp, gpuEnable, stop, outputFile, workFile, iWorkFile, savePeriod, cSize, dp, extraPts);
  if(checkFlag)
    v->Check(gpuId, gridSize);
  else
    v->Search(nbCPUThread,gpuId,gridSize);

  return 0;
}
