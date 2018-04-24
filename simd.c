#include"Project2_header.h"




void simd(int N)
{
	int iters = 1000;
	float maxF = 0.0f;
	double timeTotal = 0.0f;

/******************************************************************************************/
	__m128 v1= _mm_set_ps1(1.0);
	__m128 v2= _mm_set_ps1(2.0);
	__m128 v3= _mm_set_ps1(0.01);
	__m128 maxf_vector =_mm_set_ps1(maxF);
	__m128 Fvector =_mm_set_ps1(0.0); 


	__m128 num_2;
	__m128 tmp3 ;
	__m128 num  ;

	__m128 den_0;
	__m128 den_1;
	__m128 den  ;

	__m128 cmp_tmp;

	__m128 C_vector;
	__m128 L_vector;
	__m128 R_vector;
	__m128 m_vector;
	__m128 n_vector;

	__m128 tmp1;
	__m128 tmp2;
	__m128 num_1;
	__m128 num_0;

	for(int j=0;j<iters;j++)
	{
		double time0=gettime();

		

		int i =0 ;


		for(i=0;i<N-4;i+=4)
		{	
			
			 C_vector=_mm_load_ps(&CVec[i]);
			 L_vector=_mm_load_ps(&LVec[i]);
			 R_vector=_mm_load_ps(&RVec[i]);
			 m_vector=_mm_load_ps(&mVec[i]);
			 n_vector=_mm_load_ps(&nVec[i]);  


			 num_0 = _mm_add_ps(L_vector, R_vector);

			 tmp1 =_mm_sub_ps(m_vector, v1);
			 tmp2 =_mm_mul_ps(m_vector,tmp1);
			 num_1=_mm_div_ps(tmp2, v2);
			
			 tmp1  =_mm_sub_ps( n_vector, v1);
			 tmp2  =_mm_mul_ps(n_vector,tmp1);
			 num_2 =_mm_div_ps(tmp2, v2);

			 tmp3 =_mm_add_ps(num_1,num_2);
			 num  =_mm_div_ps(num_0,tmp3);  


			 tmp1  = _mm_sub_ps(C_vector,L_vector);
			 den_0 = _mm_sub_ps(tmp1,R_vector);
			 den_1 = _mm_mul_ps(m_vector,n_vector);
			 den   = _mm_div_ps(den_0,den_1);
			 tmp2  = _mm_add_ps(den,v3);
			 Fvector  = _mm_div_ps(num,tmp2);
			 _mm_store_ps(&FVec[i],Fvector); 


			 maxf_vector = _mm_max_ps(Fvector, maxf_vector);
			
		}

		maxF = maxf_vector[0]>maxf_vector[1]?maxf_vector[0]:maxf_vector[1];
		maxF = maxf_vector[2]>maxF?maxf_vector[2]:maxF;
		maxF = maxf_vector[3]>maxF?maxf_vector[3]:maxF;

		for (int z=i; z<N; z++){
			float num_0 = LVec[z]+ RVec[z];
			float num_1 = mVec[z]*(mVec[z]-1.0)/2.0;
			float num_2 = nVec[z]*(nVec[z]-1.0)/2.0;
			float num = num_0/(num_1+num_2);
			
			float den_0 = CVec[z]-LVec[z]-RVec[z];
			float den_1 = mVec[z]*nVec[z];
			float den = den_0/den_1;
			FVec[z] = num/(den+0.01);
			maxF = FVec[z]>maxF?FVec[z]:maxF;
		}

/*******************************************************************************************/

			double time1=gettime();
			timeTotal += time1-time0;
	}
	printf("SIMD :  Time %f Max %f\n", timeTotal/iters, maxF);


}
