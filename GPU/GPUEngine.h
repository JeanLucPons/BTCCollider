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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../SECP256k1.h"

#define ITEM_SIZE   40
#define ITEM_SIZE32 (ITEM_SIZE/4)
#define _64K 65536

#define GPU_AFFINE
#define GPU_GRP_SIZE 128
#define NB_RUN 4

typedef struct {
  uint8_t *h1;
  uint8_t *h2;
} ITEM;

class GPUEngine {

public:

  GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound); 
  ~GPUEngine();
  void SetMasks(uint16_t colMask,uint64_t dpMask,uint16_t nbFull);
  void SetKeys(Int *p);
  void SetSearchType(int searchType);
  void SetExtraPoint(bool extraPoint);
  bool SetStartingHashes(uint64_t *sHash, uint64_t *cHash);
  bool GetHashes(uint64_t *sHash, uint64_t *cHash);
  bool Launch(std::vector<ITEM> &hashFound,bool spinWait=false);
  int GetNbThread();
  int GetGroupSize();
  int GetMemory();

  std::string deviceName;

  static void *AllocatePinnedMemory(size_t size);
  static void FreePinnedMemory(void *buff);
  static void PrintCudaInfo();
  static bool GetGridSize(int gpuId, int *x, int *y);

private:

  bool callKernel();

  int nbThread;
  int nbThreadPerGroup;
  int inputHashSize;
  int keySize;
  uint64_t *inputKey;
  uint64_t *inputKeyPinned;
  uint64_t *inputHash;
  uint64_t *inputHashPinned;
  uint32_t *outputHash;
  uint32_t *outputHashPinned;
  bool initialised;
  uint32_t searchType;
  bool littleEndian;
  bool lostWarning;
  bool useExtraPoints;
  uint32_t maxFound;
  uint32_t outputSize;
  uint64_t dpMask;
  uint16_t colMask;
  uint16_t nbFull;

};

#endif // GPUENGINEH
