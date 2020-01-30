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

#include "HashTable.h"
#include <stdio.h>
#include "Int.h"
#ifndef WIN64
#include <string.h>
#endif

#define safe_free(x) if(x) {free(x);x=NULL;}
#define GET(hash,id) E[hash].items[id]

HashTable::HashTable() {

  totalItem = 0;
  memset(E,0,sizeof(E));

}

void HashTable::Reset() {

  for (uint32_t h = 0; h < HASH_SIZE; h++) {
    for(uint32_t i=0;i<E[h].nbItem;i++)
      free(E[h].items[i]);
    safe_free(E[h].items);
    E[h].maxItem = 0;
    E[h].nbItem = 0;
  }
  totalItem = 0;

}

uint16_t HashTable::getHash(hash160_t *h) {

  uint16_t s = 0;

  int i;
  for(i=0;i<nbFull;i++)
    s+=h->i16[i];

  if(colMask)
    s+=h->i16[i] & colMask;

  return s % HASH_SIZE;

}

void HashTable::SetParam(int n,int nbFull,uint16_t colMask) {
  this->cSize = n;
  this->nbFull = nbFull;
  this->colMask = colMask;
}

int HashTable::GetNbItem() {
  return totalItem;
}

ENTRY *HashTable::CreateEntry(hash160_t *h1, hash160_t *h2) {
  ENTRY *e = (ENTRY *)malloc(sizeof(ENTRY));
  e->start = *h1;
  e->end = *h2;
  return e;
}


void HashTable::getCollision(hash160_t *a, hash160_t *b, hash160_t *e) {
  *a = this->a;
  *b = this->b;
  if(e) *e = this->e;
}

#define ADD_ENTRY(entry) {                 \
  /* Shift the end of the index table */   \
  for (int i = E[h].nbItem; i > st; i--)   \
    E[h].items[i] = E[h].items[i - 1];     \
  E[h].items[st] = entry;                  \
  E[h].nbItem++;                           \
  totalItem++;}

int HashTable::AddHash(hash160_t *start, hash160_t *end) {

  uint16_t h = getHash(end);

  if (E[h].maxItem == 0) {
    E[h].maxItem = 16;
    E[h].items = (ENTRY **)malloc(sizeof(ENTRY *) * E[h].maxItem);
  }

  if(E[h].nbItem==0) {
    E[h].items[0] = CreateEntry(start, end);
    E[h].nbItem = 1;
    totalItem++;
    return false;
  }

  if (E[h].nbItem >= E[h].maxItem - 1) {
    // We need to reallocate
    E[h].maxItem += 16;
    ENTRY **nitems = (ENTRY **)malloc(sizeof(ENTRY *) * E[h].maxItem);
    memcpy(nitems, E[h].items, sizeof(ENTRY *) * E[h].nbItem);
    free(E[h].items);
    E[h].items = nitems;
  }

  // Search insertion position
  int st, ed, mi;
  st=0; ed = E[h].nbItem-1;
  while(st<=ed) {
    mi = (st+ed)/2;
	  int comp = compareHash(end,&GET(h,mi)->end);
	  if(comp<0) {
		  ed = mi - 1;
	  } else {
	    st = mi + 1;
	  }
  }

  // Collision check
  if (st > 0) {
    if (hashCollide(&GET(h, st - 1)->end, end)) {
      if (hashCollide(&GET(h, st - 1)->start, start)) {
        printf("\nFalse collision\n");
        return FALSE_COLLISION;
      }
      ENTRY *entry = CreateEntry(start,end);
      ADD_ENTRY(entry);
      a = GET(h, st-1)->start;
      b = GET(h, st)->start;
      e = GET(h, st)->end;
      return COLLISION;
    }
  }
  if (st < (int)E[h].nbItem ) {
    if (hashCollide(&GET(h, st)->end, end)) {
      if (hashCollide(&GET(h, st)->start, start)) {
        printf("\nFalse collision\n");
        return FALSE_COLLISION;
      }
      ENTRY *entry = CreateEntry(start, end);
      ADD_ENTRY(entry);
      a = GET(h, st)->start;
      b = GET(h, st+1)->start;
      e = GET(h, st)->end;
      return COLLISION;
    }
  }

  ENTRY *entry = CreateEntry(start, end);
  ADD_ENTRY(entry);
  return NO_COLLISION;

}

double HashTable::GetSizeMB() {

  uint64_t byte = 0;
  for (int h = 0; h < HASH_SIZE; h++) {
    if (E[h].nbItem > 0) {
      byte += sizeof(ENTRY *) * E[h].maxItem;
      byte += sizeof(ENTRY) * E[h].nbItem;
    }
  }

  return (double)byte / (1024.0*1024.0);

}


void HashTable::SaveTable(FILE *f) {

  for (uint32_t h = 0; h < HASH_SIZE; h++) {
    fwrite(&E[h].nbItem, sizeof(uint32_t), 1, f);
    fwrite(&E[h].maxItem, sizeof(uint32_t), 1, f);
    for (uint32_t i = 0; i < E[h].nbItem; i++) {
      fwrite(&E[h].items[i]->start, 20, 1, f);
      fwrite(&E[h].items[i]->end, 20, 1, f);
    }
  }

}

void HashTable::LoadTable(FILE *f) {

  Reset();
  for (uint32_t h = 0; h < HASH_SIZE; h++) {
    fread(&E[h].nbItem, sizeof(uint32_t), 1, f);
    fread(&E[h].maxItem, sizeof(uint32_t), 1, f);
    if (E[h].maxItem > 0) {
      // Allocate indexes
      E[h].items = (ENTRY **)malloc(sizeof(ENTRY *) * E[h].maxItem);
    }
    for (uint32_t i = 0; i < E[h].nbItem; i++) {
      ENTRY *e = (ENTRY *)malloc(sizeof(ENTRY));
      fread(&e->start, 20, 1, f);
      fread(&e->end, 20, 1, f);
      E[h].items[i] = e;
    }
    totalItem += E[h].nbItem;
  }

  printf("HashTable::LoadTable(): %d items loaded\n",totalItem);

}

std::string HashTable::GetHashStr(hash160_t *h1) {

  char tmp2[3];
  char tmp[256];
  tmp[0] = 0;
  for (int i = 0; i < 20; i++) {
#ifdef WIN64
    sprintf_s(tmp2,3,"%02X",h1->i8[i]);
#else
    sprintf(tmp2,"%02X",h1->i8[i]);
#endif
    strcat(tmp,tmp2);
  }
  return std::string(tmp);

}

void HashTable::PrintTable(int limit) {

  int curItem=0;
  if(limit<=0) limit = totalItem;

  for (int h = 0; h < HASH_SIZE; h++) {
    if (E[h].nbItem > 0) {
      printf("ENTRY: %04X\n",h);
      for (uint32_t i = 0; i < E[h].nbItem && curItem<limit; i++) {
        //printf("%02d: S:%s\n", i, GetHashStr(&GET(h,i).start).c_str());
        //printf("    E:%s\n", GetHashStr(&GET(h, i).end).c_str());
        printf("%02d: E:%s\n", i, GetHashStr(&GET(h,i)->start).c_str());
      }

    }
  }


}

bool HashTable::compare(HashTable *ht) {

  if (ht->GetNbItem() != GetNbItem()) {
    printf("Item number not equal !\n");
    return false;
  }

  for (int h = 0; h < HASH_SIZE; h++) {
    
    if (ht->E[h].nbItem != E[h].nbItem) {
      printf("[H%04X] Item number not equal !\n",h);
      return false;
    }

    bool equal = true;
    uint32_t i = 0;
    while (equal && i < E[h].nbItem) {
      equal = (compareHash(&GET(h,i)->start , &ht->GET(h,i)->start)==0) &&
              (compareHash(&GET(h, i)->end, &ht->GET(h, i)->end) == 0);
      if(equal) i++;
    }

    if (!equal) {
      printf("Unexpected hash found at [%04X][%d] !\n", h,i);
      printf("H1 H %s\n", GetHashStr(&GET(h, i)->start).c_str());
      printf("H1 E %s\n", GetHashStr(&GET(h, i)->end).c_str());
      printf("H2 H %s\n", GetHashStr(&ht->GET(h, i)->start).c_str());
      printf("H2 E %s\n", GetHashStr(&ht->GET(h, i)->end).c_str());
      return false;
    }

  }

  return true;

}

bool HashTable::hashCollide(hash160_t *h1, hash160_t *h2) {

  int i=0;
  for(i=0;i<nbFull;i++)
    if(h1->i16[i]!=h2->i16[i])
      return false;

  if( colMask )
    if( (h1->i16[i] & colMask) != (h2->i16[i] & colMask) )
      return false;

  return true;

}

int HashTable::getCollisionSize(hash160_t *h1, hash160_t *h2) {

  int i;
  int size = 0;

  for (i = 0; i < 20; i++) {
    if (h1->i8[i] != h2->i8[i])
      break;
    size += 8;
  }

  if (i < 20) {
    
    uint8_t m = 0x80;
    uint8_t b1=0;
    uint8_t b2=0;
    while (b1 == b2) {
      b1 = h1->i8[i] & m;
      b2 = h2->i8[i] & m;
      if (b1 == b2) {
        size++;
        m = m >> 1;
      }
    }

  }

  return size;

}

int HashTable::compareHash(hash160_t *h1, hash160_t *h2) {

  uint32_t *a = h1->i32;
  uint32_t *b = h2->i32;
  int i;

  for(i=0;i<5;i++) {
    if( a[i]!=b[i] )
		break;
  }

  if(i<5) {
#ifdef WIN64
    uint32_t ai = _byteswap_ulong(a[i]);
    uint32_t bi = _byteswap_ulong(b[i]);
#else
    uint32_t ai = __builtin_bswap32(a[i]);
    uint32_t bi = __builtin_bswap32(b[i]);
#endif
    if( ai>bi ) return 1;
    else        return -1;
  } else {
    return 0;
  }

}

