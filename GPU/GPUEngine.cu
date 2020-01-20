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

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_hash(uint64_t *keys, uint64_t *hashes, uint32_t maxFound, uint32_t *found, uint64_t dpMask, uint16_t colMask, uint16_t nbFull) {

  int xPtr = (blockIdx.x*blockDim.x*GPU_GRP_SIZE) * 6;
  ComputeHash(keys, hashes + xPtr, maxFound, found, dpMask, colMask, nbFull);

}

__global__ void comp_hash_p2sh(uint64_t *keys, uint64_t *hashes, uint32_t maxFound, uint32_t *found, uint64_t dpMask, uint16_t colMask, uint16_t nbFull) {

  int xPtr = (blockIdx.x*blockDim.x*GPU_GRP_SIZE) * 6;
  ComputeHashP2SH(keys, hashes + xPtr, maxFound, found, dpMask, colMask, nbFull);

}

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1} };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound) {

  // Initialise CUDA
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
  gpuId,deviceProp.name,deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                      nbThread / nbThreadPerGroup,
                      nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  /*
  size_t stackSize = 49152;
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  size_t heapSize = ;
  err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(0);
  }

  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  printf("Stack Size %lld\n", size);
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  printf("Heap Size %lld\n", size);
  */

  // Allocate memory

  // Input keys (see BTCCollider.cpp)
  keySize = 10 * _64K * 32 * 2;
  err = cudaMalloc((void **)&inputKey, keySize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputKeyPinned, keySize, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  // OutputHash
  err = cudaMalloc((void **)&outputHash, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputHashPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  // InputHash (hash160 sotred on 3*64bit)
  inputHashSize = GPU_GRP_SIZE * nbThread * 48;
  err = cudaMalloc((void **)&inputHash, inputHashSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputHashPinned, inputHashSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  searchType = P2PKH;
  initialised = true;

}

int GPUEngine::GetMemory() {
  return keySize + inputHashSize + outputSize;
}


int GPUEngine::GetGroupSize() {
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId, int *x, int *y) {

  if ( *x <= 0 || *y <= 0 ) {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
      printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
      return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
      printf("GPUEngine: There are no available device(s) that support CUDA\n");
      return false;
    }

    if (gpuId >= deviceCount) {
      printf("GPUEngine::GetGridSize() Invliad gpuId\n");
      return false;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);

    if(*x<=0) *x = 2 * deviceProp.multiProcessorCount;
    if(*y<=0) *y = 2 * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    if(*y<=0) *y = 128;

  }

  return true;

}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
     NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i=0;i<deviceCount;i++) {

    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      deviceProp.major, deviceProp.minor,(double)deviceProp.totalGlobalMem/1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

GPUEngine::~GPUEngine() {

  cudaFree(inputKey);
  cudaFree(inputHash);
  cudaFreeHost(outputHashPinned);
  cudaFree(outputHash);

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetSearchType(int searchType) {
  this->searchType = searchType;
}

bool GPUEngine::SetStartingHashes(uint64_t *h) {

  lostWarning = false;

  // Sets the starting hash of each thread
  int gSize = 6 * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * 6;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;

  for (int b = 0; b < nbBlock; b++) {
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
      for (int t = 0; t < nbThreadPerGroup; t++) {
        // Start hash
        inputHashPinned[b * blockSize + g * strideSize + t + 0* nbThreadPerGroup] = h[0];
        inputHashPinned[b * blockSize + g * strideSize + t + 1* nbThreadPerGroup] = h[1];
        inputHashPinned[b * blockSize + g * strideSize + t + 2* nbThreadPerGroup] = h[2];
        // End hash
        inputHashPinned[b * blockSize + g * strideSize + t + 3 * nbThreadPerGroup] = h[0];
        inputHashPinned[b * blockSize + g * strideSize + t + 4 * nbThreadPerGroup] = h[1];
        inputHashPinned[b * blockSize + g * strideSize + t + 5 * nbThreadPerGroup] = h[2];
        h+=3;
      }
    }
  }

  // Fill device memory
  cudaMemcpy(inputHash, inputHashPinned, inputHashSize, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputHashPinned);
  inputHashPinned = NULL;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetStartHashes: %s\n", cudaGetErrorString(err));
  }

  return callKernel();

}

bool GPUEngine::callKernel() {

  // Reset nbFound
  cudaMemset(outputHash,0,4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  if (searchType == P2SH) {

    comp_hash_p2sh << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
        (inputKey, inputHash, maxFound, outputHash, dpMask, colMask, nbFull);

  } else {

    // P2PKH or BECH32
    comp_hash << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
      (inputKey, inputHash, maxFound, outputHash, dpMask, colMask, nbFull);
    
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;

}

void GPUEngine::SetMasks(uint16_t colMask, uint64_t dpMask, uint16_t nbFull) {
  this->colMask = colMask;
  this->dpMask = dpMask;
  this->nbFull = nbFull;
}

void GPUEngine::SetKeys(Point p[10][65536]) {

  // Sets the base keys for mapping

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 65536; j++) {

      inputKeyPinned[8 * ((i*_64K) + j) + 0] = p[i][j].x.bits64[0];
      inputKeyPinned[8 * ((i*_64K) + j) + 1] = p[i][j].x.bits64[1];
      inputKeyPinned[8 * ((i*_64K) + j) + 2] = p[i][j].x.bits64[2];
      inputKeyPinned[8 * ((i*_64K) + j) + 3] = p[i][j].x.bits64[3];

      inputKeyPinned[8 * ((i*_64K) + j) + 4] = p[i][j].y.bits64[0];
      inputKeyPinned[8 * ((i*_64K) + j) + 5] = p[i][j].y.bits64[1];
      inputKeyPinned[8 * ((i*_64K) + j) + 6] = p[i][j].y.bits64[2];
      inputKeyPinned[8 * ((i*_64K) + j) + 7] = p[i][j].y.bits64[3];

    }
  }

  // Fill device memory
  cudaMemcpy(inputKey, inputKeyPinned, keySize, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputKeyPinned);
  inputKeyPinned = NULL;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
  }

}

bool GPUEngine::Launch(std::vector<ITEM> &hashFound,bool spinWait) {


  hashFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputHashPinned, outputHash, outputSize, cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputHashPinned, outputHash, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputHashPinned[0];
  if (nbFound > maxFound) {
    // prefix has been lost
    if (!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less threads (-g)\n", (nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }
  
  // When can perform a standard copy, the kernel is eneded
  cudaMemcpy(outputHashPinned, outputHash, nbFound*ITEM_SIZE + 4 , cudaMemcpyDeviceToHost);
  
  for (uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputHashPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;
    it.h1 = ((uint8_t *)(itemPtr));
    it.h2 = ((uint8_t *)(itemPtr))+20;
    hashFound.push_back(it);
  }

  return callKernel();

}
