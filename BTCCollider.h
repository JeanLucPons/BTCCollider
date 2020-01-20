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

#ifndef BTCCOLLIDERH
#define BTCCOLLIDERH

#include <string>
#include <vector>
#include "SECP256k1.h"
#include "HashTable.h"
#include "IntGroup.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#endif

class BTCCollider;

typedef struct {

  BTCCollider *obj;
  int  threadId;
  bool isRunning;
  bool hasStarted;
  int  gridSizeX;
  int  gridSizeY;
  int  gpuId;
  Int  localSeed;
  hash160_t start;
  hash160_t end;

} TH_PARAM;

#ifdef WIN64
typedef HANDLE THREAD_HANDLE;
#else
typedef pthread_t THREAD_HANDLE;
#endif

#define NUM_PARTS(n) (((n) - 1) / 16 + 1)
#define CPU_AFFINE

class BTCCollider {

public:

  BTCCollider(Secp256K1 *secp, bool useGpu, bool stop, std::string outputFile, bool useSSE,uint32_t n,int dp);
  void Search(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize);
  void Check(std::vector<int> gpuId, std::vector<int> gridSize);
  void FindCollisionCPU(TH_PARAM *p);
  void FindCollisionGPU(TH_PARAM *p);
  void UndistinguishCPU(TH_PARAM *p);
  void InitKey(TH_PARAM *p);

private:

  Int GetPrivKey(hash160_t x);
  hash160_t F(hash160_t x);
  bool IsDP(hash160_t x);
  bool IsEqual(hash160_t x1, hash160_t x2);
  void SetDP(int size);
  void FGroup(IntGroup *grp, Point *pts, Int *di, hash160_t *x);
  void AddGroup(IntGroup *grp, hash160_t *x, Point *p1, Int *dx, int i, uint16_t colMask);
  void Lock();
  void Unlock();
  std::string GetTimeStr(double s);

#ifdef WIN64
  THREAD_HANDLE LaunchThread(LPTHREAD_START_ROUTINE func,TH_PARAM *p);
#else
  THREAD_HANDLE LaunchThread(void *(*func) (void *), TH_PARAM *p);
#endif
  void JoinThreads(THREAD_HANDLE *handles, int nbThread);
  void FreeHandles(THREAD_HANDLE *handles, int nbThread);

  std::string GetHex(hash160_t x);
  void Rand(Int *seed,Int *i);
  void Rand(Int *seed,hash160_t *i);
  bool isAlive(TH_PARAM *p);
  uint64_t getGPUCount();
  uint64_t getCPUCount();
  bool hasStarted(TH_PARAM *p);

  Int seed;
  Secp256K1 *secp;
  HashTable hashTable;
  uint64_t counters[256];
  uint64_t dMask;
  int dpSize;
  int initDPSize;
  int colSize;
  int nbFull;
  uint16_t colMask;
  int  nbCPUThread;
  int  nbGPUThread;
  std::string outputFile;
  bool useSSE;
  bool useGpu;
  bool endOfSearch;
  double startTime;
  int CPU_GRP_SIZE;

  // Hash160 to key mapping
  Point pub[10][65536];
  Int   priv[10][65536];

#ifdef WIN64
  HANDLE ghMutex;
#else
  pthread_mutex_t  ghMutex;
#endif

};

#endif // BTCCOLLIDERH
