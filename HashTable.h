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

#ifndef HASHTABLEH
#define HASHTABLEH

#include <string>

#define HASH_SIZE 65536

union hash160_s {

  uint8_t i8[20];
  uint16_t i16[10];
  uint32_t i32[5];
  uint64_t i64[3];

};


typedef union hash160_s hash160_t;

typedef struct ENTRY {

  hash160_t start;
  hash160_t end;

} ENTRY;

typedef struct {

  uint32_t   nbItem;
  uint32_t   maxItem;
  ENTRY    **items;

} HASH_ENTRY;

#define NO_COLLISION 0
#define COLLISION 1
#define FALSE_COLLISION 2

class HashTable {

public:

  HashTable();
  void SetParam(int n,int nbFull,uint16_t colMask);
  int AddHash(hash160_t *start, hash160_t *end);
  void SaveTable(FILE *f);
  void LoadTable(FILE *f);
  std::string GetHashStr(hash160_t *h);
  void PrintTable(int limit=0);
  int GetNbItem();
  int getCollisionSize(hash160_t *h1, hash160_t *h2);
  void getCollision(hash160_t *a, hash160_t *b, hash160_t *e);
  void Reset();
  bool hashCollide(hash160_t *h1, hash160_t *h2);
  int  compareHash(hash160_t *h1, hash160_t *h2);
  bool compare(HashTable *ht);
  uint16_t getHash(hash160_t *h);
  ENTRY *CreateEntry(hash160_t *h1, hash160_t *h2);
  double GetSizeMB();

private:


  HASH_ENTRY E[HASH_SIZE];

  int       cSize;
  int       nbFull;
  int       totalItem;
  uint16_t  colMask;
  hash160_t a;
  hash160_t b;
  hash160_t e;

};

#endif // HASHTABLEH
