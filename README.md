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
C:\C++\BTCCollider\x64\Release>BTCCollider.exe -t 6 -gpu -s 80
BTCCollider v1.0
Collision: 80 bits
Seed: 305ABC344F924BD8B268905F99ADBD403D9A45BB220C5948731740DDE0E3A41A
Initializing:Done
Start Tue Jan 21 16:02:07 2020
Number of CPU thread: 6
Number of random walk: 2^18.59 (Max DP=20)
DP size: 20 [0xFFFFF00000000000]
GPU: GPU #0 GeForce GTX 1050 Ti (6x128 cores) Grid(12x256) (60.5 MB used)
[27.9 Mips][GPU 24.8 Mips][Cnt 2^39.44][T 07:24:33][Tavg 13:42:43][hSize 41.3MB]
Undistinguishing
DP size: 10 [0xFFC0000000000000]
DP size: 5 [0xF800000000000000]
DP size: 2 [0xC000000000000000]
DP size: 1 [0x8000000000000000]
DP size: 0 [0x0000000000000000]
[Collision Found: 82 bits][Cnt 2^39.44][T 07:24:40]
H1=FD53BD2E39ECB0F2D6AE441DE23ED586CA57C7B3
H2=FD53BD2E39ECB0F2D6AE72F3F4D6A824A065915B
Priv (WIF): p2pkh:KzNuYQf9X5sbXtr5mgEeBMx7LK2K6QRK73pMWqMhVAGpoWLEowLA
Priv (WIF): p2pkh:KzNuYQf9X5sbXtr5mgEeBMx7LK2K6QRFqBMftDEYHFzssVsN2Shn
Add1: 1Q6UGpZ6oKmCSZF1m7W1r3pp7syt1JZwMe
Add2: 1Q6UGpZ6oKmCSZFJynpzcRPJWJ2ur5Ujb5
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

# Note of DP method tradeoff

This picture illustrates the overhead you get according to the number of random walk and the number of distinguiseg bits.
All experimental points (red points) are an average on 1000 collisions.
The blue curve is an experimental fit with Z+Z.pow(nbWalk*pow(2.0,dp-20)/2,2/5), this fit works rather well when dp > 11. Z=sqrt(PI/2.2^40).
The green curve is the average of the birthday paradox without using DP method (DP0).
Significant overhead appear when dp > 2^20 - log2(nbWalk), for 40 bits collision.

![JSSHTerminal](img/hash160_col40.jpg)

# License

BTCCollider is licensed under GPLv3.

