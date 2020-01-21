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

#include "BTCCollider.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

#ifdef WIN64
DWORD WINAPI _InitKey(LPVOID lpParam) {
#else
void *_InitKey(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->InitKey(p);
  return 0;
}

// ----------------------------------------------------------------------------

BTCCollider::BTCCollider(Secp256K1 *secp, bool useGpu, bool stop, std::string outputFile, bool useSSE, uint32_t n, int dp) {

  this->secp = secp;
  this->useGpu = useGpu;
  this->outputFile = outputFile;
  this->useSSE = useSSE;
  this->nbGPUThread = 0;
  this->colSize = n;
  this->CPU_GRP_SIZE = 128;
  this->initDPSize = dp;

  // Seed
  string seedStr = Timer::getSeed(32);
  seed.SetInt32(0);
  sha256((uint8_t *)seedStr.c_str(), (int)seedStr.length(), (uint8_t *)seed.bits64);

  printf("Collision: %d bits\n", n);
  printf("Seed: %s\n", seed.GetBase16().c_str());

  // Derived from pairgen (https://github.com/basil00/pairgen.git)
  // Given a hash160 H comprised on {h0, h1, .., h9} 16-bit parts, then H is
  // mapped to a public key P as follows:
  //     P = pub[0][h0] + pub[1][h1] + .. + pub[9][h9]
  // The calculation is truncated according to the length n.  The corresponding
  // private key P' is:
  //     P' = priv[0]+h0 + priv[1]+h1*2^16 + .. + priv[9]+h9*2^144
  // Each base private key is chosen randomly and computed in advanced. 

  TH_PARAM params[10];
  memset(params, 0, sizeof(params));

  printf("Initializing:");
#ifndef WIN64
  fflush(stdout);
#endif

  Point GP = secp->G;
  Int KP;
  KP.SetInt32(1);
  for (int i = 0; i < 10; i++) {
    Gp[i] = GP;
    Kp[i] = KP;
    for (int j = 0; j < 16; j++)
      GP = secp->DoubleDirect(GP);
    KP.ShiftL(16);
  }

  THREAD_HANDLE threadIDs[10];
  for (int i = 0; i < 10; i++) {
    params[i].threadId = i;
    Rand(&seed,&params[i].localSeed);
    threadIDs[i] = LaunchThread(_InitKey,params+i);
  }
  JoinThreads(threadIDs,10);
  FreeHandles(threadIDs,10);
  printf("Done\n");


  nbFull = colSize / 16 ; // Number of full word
  int leftBit = colSize % 16;
  colMask = (1 << (16 - leftBit)) - 1;
  colMask = ~colMask;
#ifdef WIN64
  colMask = _byteswap_ushort(colMask);
#else
  colMask = __builtin_bswap16(colMask);
#endif

  this->hashTable.SetParam(n,nbFull,colMask);

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  printf("Start %s", ctimeBuff);

}

// ----------------------------------------------------------------------------

void BTCCollider::Check(std::vector<int> gpuId, std::vector<int> gridSize) {

  if(initDPSize<0)
    initDPSize = colSize/3;
  SetDP(initDPSize);

  // Check Int lib
  //Int::Check();

  // Check SSE and CPU group
  hash160_t *x = new hash160_t[CPU_GRP_SIZE];
  hash160_t *xc = new hash160_t[CPU_GRP_SIZE];

  for (int i = 0; i < CPU_GRP_SIZE; i++) {
    Rand(&seed, &x[i]);
    xc[i] = x[i];
  }

  for (int i = 0; i < CPU_GRP_SIZE; i++) {
    xc[i] = F(xc[i]);
  }

  IntGroup *grp = new IntGroup(CPU_GRP_SIZE);
  Point *pts = new Point[CPU_GRP_SIZE];
  Int *dInv = new Int[CPU_GRP_SIZE];
  FGroup(grp, pts, dInv, x);

  bool ok = true;
  int i = 0;
  while (ok && i < CPU_GRP_SIZE) {
    ok = (hashTable.compareHash(&x[i], &xc[i]) == 0);
    if (ok) i++;
  }

  if (ok) {
    printf("CPU Group OK!\n");
  } else {
    printf("CPU Group Not OK at %d!\n", i);
    printf("Hg=%s\n", GetHex(x[i]).c_str());
    printf("Hc=%s\n", GetHex(xc[i]).c_str());
  }

#ifdef WITHGPU
  // Check gpu
  if (useGpu) {

    printf("GPU allocate memory:");
    int x = gridSize[0];
    int y = gridSize[1];
    if (!GPUEngine::GetGridSize(gpuId[0], &x, &y)) {
      return;
    }

    GPUEngine h(x,y, gpuId[0], 65536);
    printf(" done\n");
    printf("GPU: %s\n", h.deviceName.c_str());
    printf("GPU: %.1f MB\n", h.GetMemory()/1048576.0);

    int nbH = h.GetNbThread() * GPU_GRP_SIZE;
    hash160_t *iHash = (hash160_t *)malloc(nbH *sizeof(hash160_t));
    for(int i=0;i<nbH;i++)
      Rand(&seed, &iHash[i]);

    h.SetMasks(colMask,dMask,nbFull);

    printf("GPU SetKeys:");
    h.SetKeys(pub);
    printf(" done\n");

    printf("GPU SetStartingHashes:");
    if (!h.SetStartingHashes((uint64_t *)iHash)) {
      printf(" failed !");
      return;
    }
    printf(" done\n");

    HashTable *h1 = new HashTable();
    HashTable *h2 = new HashTable();
    h1->SetParam(colSize,nbFull,colMask);
    h2->SetParam(colSize, nbFull, colMask);
 
    vector<ITEM> hashFound;
    h.Launch(hashFound);
    for (int i = 0; i < (int)hashFound.size(); i++)
      h1->AddHash((hash160_t *)(hashFound[i].h1), (hash160_t *)(hashFound[i].h2));
    printf("GPU found %d items\n", h1->GetNbItem());

    int nbF=0;
    hash160_t *sHash = (hash160_t *)malloc(nbH * sizeof(hash160_t));
    memcpy(sHash,iHash, nbH * sizeof(hash160_t));
    for (int run = 0; run < NB_RUN; run++) {
      for (int i = 0; i < nbH; i++) {
        iHash[i] = F(iHash[i]);
        if (IsDP(iHash[i])) {
          nbF++;
          h2->AddHash(&sHash[i],&iHash[i]);
          sHash[i] = iHash[i];
        }
      }
      printf("CPU R%d found %d items\n", run, h2->GetNbItem());
    }

    if( !h1->compare(h2) )
      return;

    printf("GPU/CPU ok\n");

  }
#endif

}

// ----------------------------------------------------------------------------
#ifdef WIN64

THREAD_HANDLE BTCCollider::LaunchThread(LPTHREAD_START_ROUTINE func, TH_PARAM *p) {
  p->obj = this;
  return CreateThread(NULL, 0, func, (void*)(p), 0, NULL);
}
void  BTCCollider::JoinThreads(THREAD_HANDLE *handles,int nbThread) {
  WaitForMultipleObjects(nbThread, handles, TRUE, INFINITE);
}
void  BTCCollider::FreeHandles(THREAD_HANDLE *handles, int nbThread) {
  for (int i = 0; i < nbThread; i++)
    CloseHandle(handles[i]);
}
#else

THREAD_HANDLE BTCCollider::LaunchThread(void *(*func) (void *), TH_PARAM *p) {
  THREAD_HANDLE h;
  p->obj = this;
  pthread_create(&h, NULL, func, (void*)(p));
  return h;
}
void  BTCCollider::JoinThreads(THREAD_HANDLE *handles, int nbThread) {
  for (int i = 0; i < nbThread; i++)
    pthread_join(handles[i], NULL);
}
void  BTCCollider::FreeHandles(THREAD_HANDLE *handles, int nbThread) {
}
#endif

// ----------------------------------------------------------------------------

void BTCCollider::SetDP(int size) {

  // Mask for distinguised point
  dpSize = size;
  if (dpSize == 0) {
    dMask = 0;
  } else {
    if (dpSize > 64) dpSize = 64;
    dMask = (1ULL << (64 - dpSize)) - 1;
    dMask = ~dMask;
  }

#ifdef WIN64
  printf("DP size: %d [0x%016I64X]\n", dpSize, dMask);
#else
  printf("DP size: %d [0x%" PRIx64 "]\n", dpSize, dMask);
#endif

  dMask = _byteswap_uint64(dMask);

}

// ----------------------------------------------------------------------------

void BTCCollider::InitKey(TH_PARAM *p) {

  Int k;
  Rand(&p->localSeed, &k);
  Point sp = secp->ComputePublicKey(&k);
  int id = p->threadId;
  pub[id][0] = sp;
  priv[id][0].Set(&k);
  for (int j = 1; j < 65536; j++) {
    k.ModAddK1order(&k,&Kp[id]);
    sp = secp->AddDirect(sp,Gp[id]);
    pub[id][j] = sp;
    priv[id][j].Set(&k);
  }

}

// ----------------------------------------------------------------------------

void BTCCollider::Rand(Int *seed, Int *i) {
  seed->AddOne();
  sha256((uint8_t *)seed->bits64,32,(uint8_t *)i->bits64);
  i->bits64[4] = 0;
}

void BTCCollider::Rand(Int *seed, hash160_t *i) {
  seed->AddOne();
  uint8_t digest[32];
  sha256((uint8_t *)seed->bits64, 32, digest);
  memcpy(i->i8,digest,20);
}

// ----------------------------------------------------------------------------

Int BTCCollider::GetPrivKey(hash160_t x) {

  Int k = priv[0][x.i16[0]];

  int i;
  for (i = 1; i < nbFull; i++)
    k.ModAddK1order(&k,&priv[i][x.i16[i]]);

  if (colMask)
    k.ModAddK1order(&k, &priv[i][x.i16[i] & colMask]);

  return k;

}

// ----------------------------------------------------------------------------

void  BTCCollider::AddGroup(IntGroup *grp, hash160_t *x, Point *p1, Int *dx, int i,uint16_t colMask) {

  // Affine coord
  Int dy;
  Int rx;
  Int _s;
  Int _p;

  for (int g = 0; g < CPU_GRP_SIZE; g++) {
    Point *p2 = &pub[i][x[g].i16[i] & colMask];
    dx[g].Set(&p2->x);
    dx[g].ModSub(&p1[g].x);
  }
  grp->Set(dx);
  grp->ModInv();

  for (int g = 0; g < CPU_GRP_SIZE; g++) {
    Point *p2 = &pub[i][x[g].i16[i] & colMask];
    dy.ModSub(&p2->y, &p1[g].y);
    _s.ModMulK1(&dy, &dx[g]);
    _p.ModSquareK1(&_s);

    rx.ModSub(&_p, &p1[g].x);
    rx.ModSub(&p2->x);

    p1[g].y.ModSub(&p2->x, &rx);
    p1[g].y.ModMulK1(&_s);
    p1[g].y.ModSub(&p2->y);
    p1[g].x.Set(&rx);
  }

}

// ----------------------------------------------------------------------------

void BTCCollider::FGroup(IntGroup *grp,Point *pts, Int *di,hash160_t *x) {

  // Perform x = F(x) for a group

  for (int g = 0; g < CPU_GRP_SIZE; g++)
    pts[g] = pub[0][x[g].i16[0]];

#ifdef CPU_AFFINE

  // Affine coordinates
  int i;
  for (i = 1; i < nbFull; i++)
    AddGroup(grp, x, pts, di, i, 0xFFFF);
  
  if (colMask)
    AddGroup(grp, x, pts, di, i, colMask);

#else

  // Projective coordinates

  for (int g = 0; g < CPU_GRP_SIZE; g++) {
    int i;
    for (i = 1; i < nbFull; i++)
      pts[g] = secp->Add2(pts[g], pub[i][x[g].i16[i]]);
    if (colMask)
      pts[g] = secp->Add2(pts[g], pub[i][x[g].i16[i] & colMask]);
    di[g].Set(&pts[g].z);
  }
  grp->Set(di);
  grp->ModInv();
  for (int g = 0; g < CPU_GRP_SIZE; g++) {
    pts[g].x.ModMulK1(&di[g]);
    pts[g].y.ModMulK1(&di[g]);
  }

#endif

  for (int g = 0; g < CPU_GRP_SIZE; g+=4) {
    secp->GetHash160(P2PKH, true,
      pts[g], pts[g+1], pts[g+2], pts[g+3],
      x[g].i8, x[g+1].i8, x[g+2].i8, x[g+3].i8);
  }

}

// ----------------------------------------------------------------------------

hash160_t BTCCollider::F(hash160_t x) {

  Point p = pub[0][x.i16[0]];

  int i;
  for (i = 1; i < nbFull; i++)
    p = secp->Add2(p,pub[i][x.i16[i]]);

  if (colMask)
    p = secp->Add2(p, pub[i][x.i16[i] & colMask]);


  p.Reduce();
  hash160_t ret;
  secp->GetHash160(P2PKH,true,p,ret.i8);
  return ret;

}

// ----------------------------------------------------------------------------

bool BTCCollider::IsDP(hash160_t x) {

  return (x.i64[0] & dMask)==0;

}

bool BTCCollider::IsEqual(hash160_t x1, hash160_t x2) {

  int i;

  for (i = 0; i < nbFull; i++)
    if(x1.i16[i]!=x2.i16[i])
      return false;

  if (colMask) {
    if ((x1.i16[i] & colMask) != (x2.i16[i] & colMask))
      return false;
  }

  return true;

}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindCollisionCPU(LPVOID lpParam) {
#else
void *_FindCollisionCPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindCollisionCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _UndistinguishCPU(LPVOID lpParam) {
#else
void *_UndistinguishCPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->UndistinguishCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _FindCollisionGPU(LPVOID lpParam) {
#else
void *_FindCollisionGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindCollisionGPU(p);
  return 0;
}

// ----------------------------------------------------------------------------

void  BTCCollider::Lock() {

#ifdef WIN64
  WaitForSingleObject(ghMutex, INFINITE);
#else
  pthread_mutex_lock(&ghMutex);
#endif

}

void  BTCCollider::Unlock() {

#ifdef WIN64
  ReleaseMutex(ghMutex);
#else
  pthread_mutex_unlock(&ghMutex);
#endif

}

// ----------------------------------------------------------------------------

void BTCCollider::UndistinguishCPU(TH_PARAM *ph) {

  int thId = ph->threadId;
  counters[thId] = 0;

  hash160_t x = ph->start;

  while (!endOfSearch) {

    // Replay random walk
    hash160_t y = F(x);
    while (!endOfSearch && !IsDP(y)) {
      y = F(y);
      counters[thId]++;
    }

    Lock();

    if (!endOfSearch) {

      if (hashTable.AddHash(&x,&y)==COLLISION) {
        // Collision found
        endOfSearch = true;
        //printf("C");
      } else {
        //printf(".");
      }

    }

    Unlock();

    if (IsEqual(y, ph->end)) {
      // Reached end of cycle
      //printf("S");
      return;
    }

    x = y;

  }

}

void BTCCollider::FindCollisionCPU(TH_PARAM *ph) {

  int thId = ph->threadId;
  counters[thId] = 0;

  ph->hasStarted = true;

  IntGroup *grp = new IntGroup(CPU_GRP_SIZE);
  hash160_t *x = (hash160_t *)malloc(CPU_GRP_SIZE * sizeof(hash160_t));
  hash160_t *y = (hash160_t *)malloc(CPU_GRP_SIZE * sizeof(hash160_t));
  Point *pts = new Point[CPU_GRP_SIZE];
  Int *dInv = new Int[CPU_GRP_SIZE];

  for (int g = 0; g < CPU_GRP_SIZE; g++)
    Rand(&ph->localSeed, &x[g]);
  memcpy(y, x, CPU_GRP_SIZE * sizeof(hash160_t));

  while (!endOfSearch) {

    // Random walk
    FGroup(grp,pts,dInv,y);
    counters[thId]+=CPU_GRP_SIZE;

    for (int g = 0; g < CPU_GRP_SIZE; g++) {
      if (IsDP(y[g])) {
        Lock();
        if (!endOfSearch) {
          int cStatus = hashTable.AddHash(&x[g], &y[g]);
          switch (cStatus) {
            case COLLISION:
              endOfSearch = true;
              break;
            case FALSE_COLLISION:
              // Reset bad random walk
              Rand(&ph->localSeed,&y[g]);
              break;
          }
        }
        Unlock();
        x[g] = y[g];
      }
    }

  }

  free(x);
  free(y);
  delete[] pts;
  delete[] dInv;
  delete grp;
  ph->isRunning = false;

}

// ----------------------------------------------------------------------------


void BTCCollider::FindCollisionGPU(TH_PARAM *ph) {

  bool ok = true;

#ifdef WITHGPU

  int thId = ph->threadId;
  counters[thId] = 0;

  GPUEngine h(ph->gridSizeX, ph->gridSizeY, ph->gpuId, 65536);
  printf("GPU: %s (%.1f MB used)\n", h.deviceName.c_str(), h.GetMemory() / 1048576.0);

  int nbH = h.GetNbThread() * GPU_GRP_SIZE;
  hash160_t *iHash = (hash160_t *)malloc(nbH * sizeof(hash160_t));
  for (int i = 0; i < nbH; i++)
    Rand(&ph->localSeed, &iHash[i]);

  h.SetMasks(colMask, dMask, nbFull);
  h.SetKeys(pub);

  ph->hasStarted = true;

  if (!h.SetStartingHashes((uint64_t *)iHash)) {
    printf("SetStartingHashes failed !");
    return;
  }
  vector<ITEM> hashFound;

  while(!endOfSearch) {

    h.Launch(hashFound);
    counters[thId] += nbH * NB_RUN;

    Lock();
    for (int i = 0;!endOfSearch && i < (int)hashFound.size(); i++)
      if( hashTable.AddHash((hash160_t *)(hashFound[i].h1), (hash160_t *)(hashFound[i].h2))==COLLISION )
        endOfSearch = true;
    Unlock();

  }

#else
  ph->hasStarted = true;
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

  ph->isRunning = false;

}

// ----------------------------------------------------------------------------

bool BTCCollider::isAlive(TH_PARAM *p) {

  bool isAlive = true;
  int total = nbCPUThread + nbGPUThread;
  for(int i=0;i<total;i++)
    isAlive = isAlive && p[i].isRunning;

  return isAlive;

}

// ----------------------------------------------------------------------------

bool BTCCollider::hasStarted(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
    hasStarted = hasStarted && p[i].hasStarted;

  return hasStarted;

}

// ----------------------------------------------------------------------------

uint64_t BTCCollider::getGPUCount() {

  uint64_t count = 0;
  for(int i=0;i<nbGPUThread;i++)
    count += counters[0x80L+i];
  return count;

}

uint64_t BTCCollider::getCPUCount() {

  uint64_t count = 0;
  for(int i=0;i<nbCPUThread;i++)
    count += counters[i];
  return count;

}

// ----------------------------------------------------------------------------

string BTCCollider::GetTimeStr(double dTime) {

  char tmp[256];

  double nbDay = dTime / 86400.0;
  if (nbDay >= 1) {

    double nbYear = nbDay / 365.0;
    if (nbYear > 1) {
      if (nbYear < 5)
        sprintf(tmp, "%.1fy", nbYear);
      else
        sprintf(tmp, "%gy", nbYear);
    } else {
      sprintf(tmp, "%.1fd", nbDay);
    }

  } else {

    int iTime = (int)dTime;
    int nbHour = (int)((iTime % 86400) / 3600);
    int nbMin = (int)(((iTime % 86400) % 3600) / 60);
    int nbSec = (int)(iTime % 60);

    if (nbHour == 0) {
      if (nbMin == 0) {
        sprintf(tmp, "%02d s", nbSec);
      } else {
        sprintf(tmp, "%02d:%02d", nbMin, nbSec);
      }
    } else {
      sprintf(tmp, "%02d:%02d:%02d", nbHour, nbMin, nbSec);
    }

  }

  return string(tmp);

}

// ----------------------------------------------------------------------------

void BTCCollider::Search(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize) {

  double t0;
  double t1;
  nbCPUThread = nbThread;
  nbGPUThread = (useGpu?(int)gpuId.size():0);
  double avgI = sqrt(M_PI/2.0)*pow(2.0,(double)colSize/2.0);

  TH_PARAM *params = (TH_PARAM *)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
  THREAD_HANDLE *thHandles = (THREAD_HANDLE *)malloc((nbCPUThread + nbGPUThread) * sizeof(THREAD_HANDLE));

  endOfSearch = false;
  memset(params, 0, (nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
  memset(counters, 0, sizeof(counters));
  hashTable.Reset();
  printf("Number of CPU thread: %d\n", nbCPUThread);
  uint64_t totalRW = nbCPUThread * CPU_GRP_SIZE;

#ifdef WITHGPU

  for (int i = 0; i < nbGPUThread; i++) {
    int x = gridSize[2 * i];
    int y = gridSize[2 * i + 1];
    if (!GPUEngine::GetGridSize(gpuId[i], &x, &y)) {
      return;
    } else {
      params[nbCPUThread + i].gridSizeX = x;
      params[nbCPUThread + i].gridSizeY = y;
    }
    totalRW += GPU_GRP_SIZE*x*y;
  }

#endif

  // Compute optimal distinguished bits number.
  // If dp is too large comparing to the total number of parallel random walks
  // an overload appears due to the fact that computed paths become too short
  // and decrease significantly the probability that distiguised points collide 
  // inside the centralized hash table.
  int optimalDP = (int)((double)colSize / 2.0 - log2((double)totalRW) - 1);
  if (optimalDP < 0) optimalDP = 0;
  printf("Number of random walk: 2^%.2f (Max DP=%d)\n", log2((double)totalRW), optimalDP);

  if (initDPSize > optimalDP) {
    printf("Warning, DP is too large, it may cause significant overload.\n");
    printf("Hint: decrease number of threads, gridSize, or decrese dp using -d.\n");
  }
  if (initDPSize < 0)
    initDPSize = optimalDP;

  SetDP(initDPSize);

#ifdef WIN64
  ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
  ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif

  // Launch CPU threads
  for (int i = 0; i < nbCPUThread; i++) {
    params[i].threadId = i;
    params[i].isRunning = true;
    Rand(&seed, &params[i].localSeed);
    thHandles[i] = LaunchThread(_FindCollisionCPU, params + i);
    totalRW += CPU_GRP_SIZE;
  }

  // Launch GPU threads
  for (int i = 0; i < nbGPUThread; i++) {
    params[nbCPUThread + i].threadId = 0x80L + i;
    params[nbCPUThread + i].isRunning = true;
    Rand(&seed, &params[nbCPUThread + i].localSeed);
    params[nbCPUThread + i].gpuId = gpuId[i];
    thHandles[nbCPUThread + i] = LaunchThread(_FindCollisionGPU, params + (nbCPUThread + i));
  }

#ifndef WIN64
  setvbuf(stdout, NULL, _IONBF, 0);
#endif

  uint64_t lastCount = 0;
  uint64_t gpuCount = 0;
  uint64_t lastGPUCount = 0;

  // Key rate smoothing filter
#define FILTER_SIZE 8
  double lastkeyRate[FILTER_SIZE];
  double lastGpukeyRate[FILTER_SIZE];
  uint32_t filterPos = 0;

  double keyRate = 0.0;
  double gpuKeyRate = 0.0;

  memset(lastkeyRate, 0, sizeof(lastkeyRate));
  memset(lastGpukeyRate, 0, sizeof(lastkeyRate));

  // Wait that all threads have started
  while (!hasStarted(params)) {
    Timer::SleepMillis(50);
  }


  t0 = Timer::get_tick();
  startTime = t0;

  while (isAlive(params)) {

    int delay = 2000;
    while (isAlive(params) && delay > 0) {
      Timer::SleepMillis(50);
      delay -= 50;
    }

    gpuCount = getGPUCount();
    uint64_t count = getCPUCount() + gpuCount;

    t1 = Timer::get_tick();
    keyRate = (double)(count - lastCount) / (t1 - t0);
    gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
    lastkeyRate[filterPos%FILTER_SIZE] = keyRate;
    lastGpukeyRate[filterPos%FILTER_SIZE] = gpuKeyRate;
    filterPos++;

    // KeyRate smoothing
    double avgKeyRate = 0.0;
    double avgGpuKeyRate = 0.0;
    uint32_t nbSample;
    for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
      avgKeyRate += lastkeyRate[nbSample];
      avgGpuKeyRate += lastGpukeyRate[nbSample];
    }
    avgKeyRate /= (double)(nbSample);
    avgGpuKeyRate /= (double)(nbSample);

    if (isAlive(params)) {

      //double cBit = 16 + 2.0*log2((double)hashTable.GetNbItem());
      //double tTo80 = (t1-startTime) * pow(2.0,(80.0 - cBit)/2.0);

      printf("\r[%.1f Mips][GPU %.1f Mips][Cnt 2^%.2f][T %s][Tavg %s][hSize %.1fMB]  ",
        avgKeyRate / 1000000.0, avgGpuKeyRate / 1000000.0,
        log2((double)count),
        GetTimeStr(t1 - startTime).c_str(),
        GetTimeStr(avgI / avgKeyRate).c_str(),
        hashTable.GetSizeMB());

    }

    lastCount = count;
    lastGPUCount = gpuCount;
    t0 = t1;

  }

  JoinThreads(thHandles, nbCPUThread + nbGPUThread);
  FreeHandles(thHandles, nbCPUThread + nbGPUThread);

  printf("\n");

  if (dpSize > 0) {

    // Undistinguish
    printf("Undistinguishing\n");
    do {

      hash160_t a;
      hash160_t b;
      hash160_t e;
      hashTable.getCollision(&a, &b, &e);
      hashTable.Reset();
      SetDP(dpSize / 2);
      TH_PARAM p1;
      TH_PARAM p2;
      p1.start = a;
      p1.end = e;
      p1.threadId = 0;
      p2.start = b;
      p2.end = e;
      p2.threadId = 1;
      THREAD_HANDLE th[2];
      endOfSearch = false;
      //printf("A=%s\n", GetHex(a).c_str());
      //printf("B=%s\n", GetHex(b).c_str());
      //printf("E=%s\n", GetHex(e).c_str());
      th[0] = LaunchThread(_UndistinguishCPU, &p1);
      th[1] = LaunchThread(_UndistinguishCPU, &p2);
      JoinThreads(th, 2);
      FreeHandles(th, 2);
      //printf("\n");

    } while (dpSize != 0);

  }

  hash160_t a;
  hash160_t b;
  hashTable.getCollision(&a, &b, NULL);

  Int k1 = GetPrivKey(a);
  Int k2 = GetPrivKey(b);
  Point p1 = secp->ComputePublicKey(&k1);
  Point p2 = secp->ComputePublicKey(&k2);
  hash160_t h1;
  hash160_t h2;
  secp->GetHash160(P2PKH, true, p1, h1.i8);
  secp->GetHash160(P2PKH, true, p2, h2.i8);

  double totalTime = Timer::get_tick() - startTime;

  printf("[Collision Found: %d bits][Cnt 2^%.2f][T %s]\n",
    hashTable.getCollisionSize(&h1, &h2),
    log2((double)lastCount),
    GetTimeStr(totalTime).c_str());

  printf("H1=%s\n", GetHex(h1).c_str());
  printf("H2=%s\n", GetHex(h2).c_str());
  printf("Priv (WIF): p2pkh:%s\n", secp->GetPrivAddress(true, k1).c_str());
  printf("Priv (WIF): p2pkh:%s\n", secp->GetPrivAddress(true, k2).c_str());
  printf("Add1: %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());
  printf("Add2: %s\n", secp->GetAddress(P2PKH, true, h2.i8).c_str());

}

// ----------------------------------------------------------------------------

string BTCCollider::GetHex(hash160_t x) {

  string ret;

  char tmp[128];
  for (int i = 0; i < 20; i++) {
    sprintf(tmp,"%02X",x.i8[i]);
    ret.append(tmp);
  }

  return ret;

}
