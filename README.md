# BTCCollider

BTCCollider generates BTC p2pkh address pair (and their corresponding private keys) that 
share the same prefix. It looks for a pair of distinct HASH160 starting with the same bits 
(a partial collision) using the so called "distinguished point" method which allows 
to efficiently take benefit of the birthday paradox using parallel calculations.
BTCCollider supports multi GPU using CUDA.

# Usage

```
BTCCollider [-check] [-v] [-gpu]
            [-gpuId gpuId1[,gpuId2,...]] [-g gridSize1[,gridSize2,...]]
            [-o outputfile] [-s collisionSize] [-t nbThread] [-d dpBit]
            [-nosse] [-check]

 -v: Print version
 -gpu: Enable gpu calculation
 -o outputfile: Output results to the specified file
 -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0
 -g gridSize1x,gridSize1y,gridSize1x,gridSize1y, ...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)
 -s: Specify size of the collision in bit (default is 40)
 -d: Specify number of leading zeros for the DP method (default is auto)
 -t threadNumber: Specify number of CPU thread, default is number of core
 -l: List cuda enabled devices
 -check: Check CPU and GPU kernel vs CPU
```
 
Example (Windows, Intel Core i7-4770 3.4GHz 8 multithreaded cores, GeForce GTX 1050 Ti):
```
C:\C++\BTCCollider\x64\Release>BTCCollider.exe -gpu -s 64
BTCCollider v1.0
Collision: 64 bits
Seed: 2F2FA6C636C8F57E1EE31C77C81A272A325F9510B4B327DA6499E3BC220171D5
Initializing:Done
Start Mon Jan 20 16:03:22 2020
Number of CPU thread: 7
Number of random walk: 2^18.59 (Max DP=12)
DP size: 12 [0xFFF0000000000000]
GPU: GPU #0 GeForce GTX 1050 Ti (6x128 cores) Grid(12x256) (60.5 MB used)
[30.5 Mips][GPU 26.3 Mips][Cnt 2^32.84][T 04:12][Tavg 02:56][hSize 103.9MB]
Undistinguishing
DP size: 6 [0xFC00000000000000]
DP size: 3 [0xE000000000000000]
DP size: 1 [0x8000000000000000]
DP size: 0 [0x0000000000000000]
[Collision Found: 64 bits][Cnt 2^32.84][T 04:13]
H1=3102614E204BAC3B62365B3955A0DF6D98E35169
H2=3102614E204BAC3BDA05B6A31D17868EADFF44DE
Priv (WIF): p2pkh:KwWbLNMxzAgwEXRbxZUJufCVbsBizpfkYvCzRJ1ARnVYs5x818Zq
Priv (WIF): p2pkh:L1ezmtBrzoSWEYVTfRktzR6bNtHUDpyE15eKofVPttxcw8SFgSBU
Add1: 15U8xmmR6AGJs4ioYbPPwZqiFugJVZfQcy
Add2: 15U8xmmR6AGK7rozwS9eDqepBe7zHrufuX
```

# Compilation

## Windows

Install CUDA SDK and open BTCCollider.sln in Visual C++ 2017.\
You may need to reset your *Windows SDK version* in project properties.\
In Build->Configuration Manager, select the *Release* configuration.\
Build and enjoy.\
\
Note: The current release has been compiled with CUDA SDK 10.0, if you have a different release of the CUDA SDK, you may need to update CUDA SDK paths in BTCCollider.vcxproj using a text editor. The current nvcc option are set up to architecture starting at 3.0 capability, for older hardware, add the desired compute capabilities to the list in GPUEngine.cu properties, CUDA C/C++, Device, Code Generation.

## Linux

Install CUDA SDK.\
Depending on the CUDA SDK version and on your Linux distribution you may need to install an older g++ (just for the CUDA SDK).\
Edit the makefile and set up the good CUDA SDK path and appropriate compiler for nvcc. 

```
CUDA       = /usr/local/cuda-8.0
CXXCUDA    = /usr/bin/g++-4.8
```

You can enter a list of architecture (refer to nvcc documentation) if you have several GPU with different architecture. Compute capability 2.0 (Fermi) is deprecated for recent CUDA SDK.
BTCCollider need to be compiled and linked with a recent gcc (>=7). The current release has been compiled with gcc 7.3.0.\
Go to the BTCCollider directory. ccap is the desired compute capability.

```
$ g++ -v
gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)
$ make all (for build without CUDA support)
or
$ make gpu=1 ccap=20 all
```
Runnig BTCCollider (Intel(R) Xeon(R) CPU, 8 cores,  @ 2.93GHz, Quadro 600 (x2))
```
$export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
```

# License

BTCCollider is licensed under GPLv3.

