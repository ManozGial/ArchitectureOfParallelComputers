#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<assert.h>
#include<math.h>
#include<xmmintrin.h> //SSE

#include<emmintrin.h> //SSE2
#include<mpi.h>

int N;
float * mVec;
float * nVec;
float * RVec;
float * LVec;
float * CVec;
float * FVec;
int world_size,world_rank;

void original(int n);
void simd(int n);
void mpi(int n);

double gettime(void);

