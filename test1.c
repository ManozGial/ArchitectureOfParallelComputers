#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<assert.h>
//eisai malakas



#include<xmmintrin.h> //SSE

#include<emmintrin.h> //SSE2




double gettime(void)
{
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

float randpval ()
{
	int vr = rand();
	int vm = rand()%vr;
	float r = ((float)vm)/(float)vr;
	assert(r>=0.0 && r<=1.00001);
	return r;
}


int main(int argc, char ** argv)
{
	int N = atoi(argv[1]);
	int iters = 1000;
	srand(1);

	float * mVec = (float*)malloc(sizeof(float)*N);
	assert(mVec!=NULL);

	float * nVec = (float*)malloc(sizeof(float)*N);
	assert(nVec!=NULL);
	
	float * LVec = (float*)malloc(sizeof(float)*N);
	assert(LVec!=NULL);
	
	float * RVec = (float*)malloc(sizeof(float)*N);
	assert(RVec!=NULL);

	float * CVec = (float*)malloc(sizeof(float)*N);
	assert(CVec!=NULL);

	float * FVec = (float*)malloc(sizeof(float)*N);
	assert(FVec!=NULL);



for(int i=0;i<N;i++)
{
	mVec[i] = (float)(2+rand()%10);
	nVec[i] = (float)(2+rand()%10);
	LVec[i] = 0.0;
	for(int j=0;j<mVec[i];j++)
	{
		LVec[i] += randpval();
	}
	RVec[i] = 0.0;
	for(int j=0;j<nVec[i];j++)
	{
		RVec[i] += randpval();
	}
	CVec[i] = 0.0;
	for(int j=0;j<mVec[i]*nVec[i];j++)
	{
		CVec[i] += randpval();
	}
	FVec[i] = 0.0;
	
	assert(mVec[i]>=2.0 && mVec[i]<=12.0);
	assert(nVec[i]>=2.0 && nVec[i]<=12.0);
	assert(LVec[i]>0.0 && LVec[i]<=1.0*mVec[i]);
	assert(RVec[i]>0.0 && RVec[i]<=1.0*nVec[i]);
	assert(CVec[i]>0.0 && CVec[i]<=1.0*mVec[i]*nVec[i]);
}
	float maxF = 0.0f;
	double timeTotal = 0.0f;

/******************************************************************************************/
	int extra = N%4;
	__m128 v1= _mm_set_ps1(1.0);
	__m128 v2= _mm_set_ps1(2.0);
	__m128 v3= _mm_set_ps1(0.01);

	for(int j=0;j<iters;j++)
	{
		double time0=gettime();

		

		int i =0 ;


		for(i=0;i<N-4;i+4)
		{	
			//__m128 F_vector=_mm_load_ps(&FVec[i]);
			__m128 C_vector=_mm_load_ps(&CVec[i]);
			__m128 L_vector=_mm_load_ps(&LVec[i]);
			__m128 R_vector=_mm_load_ps(&RVec[i]);
			__m128 m_vector=_mm_load_ps(&mVec[i]);
			__m128 n_vector=_mm_load_ps(&nVec[i]);  


			__m128 num_0 = __mm_add_ps(L_vector, R_vector);

			__m128 tmp1 =_mm_sub_ps(m_vector, v1);
			__m128 tmp2 =_mm_mul_ps(m_vector,tmp1);
			__m128 num_1=_mm_div_ps(tmp2, v2);
			
			tmp1  =_mm_sub_ps( nvector, v1);
			tmp2  =_mm_mul_ps(n_vector,tmp1);
			num_2 =_mm_div_ps(tmp2, v2);

			__m128 tmp3 =_mm_add_ps(num_1,num_2);
			__m128 num  =_mm_div_ps(num_0,tmp3);  


			tmp1 		 	= _mm_sub_ps(C_vector,L_vector);
			__m128 den_0 	= _mm_sub_ps(tmp1,R_vector);
			__m128 den_1 	= _mm_mul_ps(m_vector,n_vector);
			__m128 den   	= _mm_div_ps(den_0,den_1);
			tmp2 		 	= _mm_add_ps(den,v3);
			tmp3		 	= _mm_div_ps(num,tmp2);
			__m128 F_vector = _mm_store_ps(&FVec[i],tmp3); 

			maxF = FVec[i]>maxF?FVec[i]:maxF;
			maxF = FVec[i+1]>maxF?FVec[i+1]:maxF;
			maxF = FVec[i+2]>maxF?FVec[i+2]:maxF;
			maxF = FVec[i+3]>maxF?FVec[i+3]:maxF;

/*
			float num_0 = LVec[i]+RVec[i];
			float num_1 = mVec[i]*(mVec[i]-1.0)/2.0;
			float num_2 = nVec[i]*(nVec[i]-1.0)/2.0;
			float num = num_0/(num_1+num_2);
			
			float den_0 = CVec[i]-LVec[i]-RVec[i];
			float den_1 = mVec[i]*nVec[i];
			float den = den_0/den_1;
			FVec[i] = num/(den+0.01);
*/
			
		}
/*******************************************************************************************/

			double time1=gettime();
			timeTotal += time1-time0;
	}
	printf("Time %f Max %f\n", timeTotal/iters, maxF);

	free(mVec);
	free(nVec);
	free(LVec);
	free(RVec);
	free(CVec);
	free(FVec);
}
