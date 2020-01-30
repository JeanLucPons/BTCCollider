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

// CUDA Kernel main function

#ifdef GPU_AFFINE

__device__ void AddGroup(uint64_t *keys, uint64_t px[GPU_GRP_SIZE][4], uint64_t py[GPU_GRP_SIZE][4], uint16_t x[GPU_GRP_SIZE][12], int i, uint16_t colMask) {

  uint64_t dx[GPU_GRP_SIZE][4];
  uint64_t dy[4];
  uint64_t rx[4];
  uint64_t _s[4];
  uint64_t _p[4];
  uint64_t _kx[4];
  uint64_t _ky[4];
  uint16_t kIdx;

  for (int g = 0; g < GPU_GRP_SIZE; g++) {    
    kIdx = x[g][i] & colMask;
    LoadKeyX(dx[g], i, kIdx);
    ModSub256(dx[g],px[g]);
  }
  _ModInvGrouped(dx);

  for (int g = 0; g < GPU_GRP_SIZE; g++) {

    kIdx = x[g][i] & colMask;
    LoadKeyX(_kx, i, kIdx);
    LoadKeyY(_ky, i, kIdx);

    ModSub256(dy,_ky,py[g]);
    _ModMult(_s,dy,dx[g]);
    _ModSqr(_p,_s);

    ModSub256(rx,_p,px[g]);
    ModSub256(rx,_kx);
    ModSub256(py[g],_kx,rx);
    _ModMult(py[g], _s);
    ModSub256(py[g],_ky);
    Load256(px[g],rx);

  }

}

#else 

__device__ void Add(uint64_t p1x[4], uint64_t p1y[4], uint64_t p1z[4],
  uint64_t p2x[4], uint64_t p2y[4]) {

  uint64_t u[4];
  uint64_t v[4];
  uint64_t u1[4];
  uint64_t v1[4];
  uint64_t vs2[4];
  uint64_t vs3[4];
  uint64_t us2[4];
  uint64_t a[4];
  uint64_t us2w[4];
  uint64_t vs2v2[4];
  uint64_t vs3u2[4];
  uint64_t _2vs2v2[4];
  uint64_t rx[4];
  uint64_t ry[4];
  uint64_t rz[4];

  _ModMult(u1, p2y, p1z);
  _ModMult(v1, p2x, p1z);
  ModSub256(u, u1, p1y);
  ModSub256(v, v1, p1x);
  _ModSqr(us2, u);
  _ModSqr(vs2, v);
  _ModMult(vs3, vs2, v);
  _ModMult(us2w, us2, p1z);
  _ModMult(vs2v2, vs2, p1x);
  ModAdd256(_2vs2v2, vs2v2, vs2v2);
  ModSub256(a, us2w, vs3);
  ModSub256(a, _2vs2v2);
  _ModMult(rx, v, a);
  _ModMult(vs3u2, vs3, p1y);
  ModSub256(ry, vs2v2, a);
  _ModMult(ry, u);
  ModSub256(ry, vs3u2);
  _ModMult(rz, vs3, p1z);

  Load256(p1x, rx);
  Load256(p1y, ry);
  Load256(p1z, rz);

}

#endif

// -----------------------------------------------------------------------------------------

#define HASHOK(h)  ((h[0] & 0x0080)==0)

__device__ void ComputeHash(uint64_t *keys, uint64_t *hashes,uint32_t maxFound, uint32_t *out,uint64_t dpMask,uint16_t colMask, uint16_t nbFull, bool extraPoints) {

  // Perform x = F(x) for a group
  uint64_t px[GPU_GRP_SIZE][4];
  uint64_t py[GPU_GRP_SIZE][4];
  uint16_t x[GPU_GRP_SIZE][12];

  __syncthreads();
  LoadHash(x, hashes);

  for (int run = 0; run < NB_RUN; run++) {

    __syncthreads();

#ifdef GPU_AFFINE

    for (int g = 0; g < GPU_GRP_SIZE; g++) {
      LoadKeyX(px[g], 0, x[g][0]);
      LoadKeyY(py[g], 0, x[g][0]);
    }

    int i;
    for (i = 1; i < nbFull; i++)
      AddGroup(keys, px, py, x, i, 0xFFFF);

    if (colMask)
      AddGroup(keys, px, py, x, i, colMask);

#else

    uint64_t pz[GPU_GRP_SIZE][4];
    uint64_t _kx[4];
    uint64_t _ky[4];

    for (int g = 0; g < GPU_GRP_SIZE; g++) {

      LoadKeyX(px[g], 0, x[g][0]);
      LoadKeyY(py[g], 0, x[g][0]);
      Load256(pz[g],_1);

      int i;
      for (i = 1; i < nbFull; i++) {
        LoadKeyX(_kx, i, x[g][i]);
        LoadKeyY(_ky, i, x[g][i]);
        Add(px[g], py[g], pz[g], _kx, _ky);
      }

      if (colMask) {
        LoadKeyX(_kx, i, x[g][i] & colMask);
        LoadKeyY(_ky, i, x[g][i] & colMask);
        Add(px[g], py[g], pz[g], _kx, _ky);
      }

    }

    _ModInvGrouped(pz);
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
      _ModMult(px[g], pz[g]);
      _ModMult(py[g], pz[g]);
    }

#endif

    for (int g = 0; g < GPU_GRP_SIZE; g++) {

      uint8_t isOdd = (uint8_t)(py[g][0] & 0x1);

      __syncthreads();
      _GetHash160Comp(px[g], isOdd, (uint8_t *)(x[g]));

      if (extraPoints) {

        uint64_t xe1[4];
        uint64_t xe2[4];

        if (HASHOK(x[g])) goto checkdp;

        _GetHash160Comp(px[g], !isOdd, (uint8_t *)(x[g]));
        if (HASHOK(x[g])) goto checkdp;

        Load256(xe1, px[g]);
        _ModMult(xe1, _beta);
        _GetHash160Comp(xe1, isOdd, (uint8_t *)(x[g]));
        if (HASHOK(x[g])) goto checkdp;

        _GetHash160Comp(xe1, !isOdd, (uint8_t *)(x[g]));
        if (HASHOK(x[g])) goto checkdp;

        Load256(xe2, px[g]);
        _ModMult(xe2, _beta2);
        _GetHash160Comp(xe2, isOdd, (uint8_t *)(x[g]));
        if (HASHOK(x[g])) goto checkdp;

        _GetHash160Comp(xe2, !isOdd, (uint8_t *)(x[g]));

      }

    checkdp:

      if ((*((uint64_t *)x[g]) & dpMask) == 0) {

        // Distinguished point
        uint16_t s[12];
        LoadStartHash(s, hashes, g);
        uint32_t pos = atomicAdd(out, 1);
        if (pos < maxFound)
          OutputHash(s,x[g]);
        StoreStartHash(x[g], hashes, g);

      }

    }

  }

  __syncthreads();
  StoreHash(x,hashes);

}

__device__ void ComputeHashP2SH(uint64_t *keys, uint64_t *hashes, uint32_t maxFound, uint32_t *out, uint64_t dpMask, uint16_t colMask, uint16_t nbFull, bool extraPoints) {

}
