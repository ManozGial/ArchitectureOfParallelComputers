#include"Project2_header.h"




void mpi(int N)
{	
	MPI_Barrier(MPI_COMM_WORLD);
	
	int iters = 1000;
	float maxF = 0.0f;
	double timeTotal = 0.0f;

/******************************************************************************************/
	float tmpvar =1.0;
	__m128 v1= _mm_set1_ps(tmpvar);
	tmpvar=2.0;
	__m128 v2= _mm_set1_ps(tmpvar);
	tmpvar=0.01;
	__m128 v3= _mm_set1_ps(tmpvar);

	__m128 maxf_vector =_mm_set_ps1(maxF);
	__m128 Fvector =_mm_set_ps1(0.0); 


	__m128 tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
	__m128 C_vector;
	__m128 L_vector;
	__m128 R_vector;

	__m128 num_0,num_1,num_2, num;
	__m128 den_0,den_1,den;

	__m128 m_vector;
	__m128 n_vector;


	float maxf0,maxf1,maxf2,maxf3,tt1,tt2,tt3;

	int low = 0;
	int high;
	low = ((int)ceil((double)N/(double)  world_size))*world_rank;
	high = ((int) ceil((double)N/(double)  world_size))*(world_rank+1)-1;

	if(high>=N) 
	   high=N-1;

	double time0,time1;
	for(int j=0;j<iters;j++)
	{
		time0=gettime();

		
		
		int i;

		for(i=low;i <= high - 4; i+=4)
		{	
				 C_vector=_mm_loadu_ps(&CVec[i]);
				 L_vector=_mm_loadu_ps(&LVec[i]);
				 R_vector=_mm_loadu_ps(&RVec[i]);

				

				num_0 = _mm_add_ps(L_vector, R_vector);
				

				tmp6 	= _mm_sub_ps(C_vector,L_vector);
			        den_0 	= _mm_sub_ps(tmp6,R_vector);

				 m_vector=_mm_loadu_ps(&mVec[i]);
				 n_vector=_mm_loadu_ps(&nVec[i]);  
			

				 tmp1 =_mm_sub_ps(m_vector, v1);
				 tmp2 =_mm_mul_ps(m_vector,tmp1);
				 num_1=_mm_div_ps(tmp2, v2);


				tmp3  =_mm_sub_ps( n_vector, v1);
				tmp4  =_mm_mul_ps(n_vector,tmp3);
				num_2 =_mm_div_ps(tmp4, v2);


				tmp5  =_mm_add_ps(num_1,num_2);
		

				den_1 = _mm_mul_ps(m_vector,n_vector);
				
		
				num  =_mm_div_ps(num_0,tmp5);

				den  = _mm_div_ps(den_0,den_1);
				tmp7 = _mm_add_ps(den,v3);
				Fvector = _mm_div_ps(num,tmp7);
				_mm_storeu_ps(&FVec[i],Fvector);
					

				maxf_vector = _mm_max_ps(Fvector, maxf_vector); 	
		

		}
		maxF = maxf_vector[0]>maxf_vector[1]?maxf_vector[0]:maxf_vector[1];
		maxF = maxf_vector[2]>maxF?maxf_vector[2]:maxF;
		maxF = maxf_vector[3]>maxF?maxf_vector[3]:maxF;

		for (int z=i; z<=high; z++){
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
			
			time1=gettime();
			timeTotal += time1-time0;
		//}

		if((j == (iters-1)) && (world_rank!=0)){
			time0=gettime();		
			MPI_Send(&maxF ,1 ,MPI_FLOAT ,0 ,world_rank ,MPI_COMM_WORLD);
					
		}

	}/// telozz tou megalou for
	
	if(world_rank==0){
			float maxf_temp;
		
			for(int i=1 ;i<world_size ;i++){
				MPI_Recv(&maxf_temp ,1 ,MPI_FLOAT ,i ,i ,MPI_COMM_WORLD,MPI_STATUS_IGNORE);			
				maxF= maxF<maxf_temp?maxf_temp:maxF;
				

			}

			double time1=gettime();
			timeTotal += time1-time0;	
			printf("MPI : Time %f Max %f\n", timeTotal/iters, maxF);
		}
	
	
}
