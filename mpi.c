#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<assert.h>

#include<xmmintrin.h> //SSE

#include<emmintrin.h> //SSE2
#include<mpi.h>



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
	float tmpvar =1.0;
	__m128 v1= _mm_set1_ps(tmpvar);
	tmpvar=2.0;
	__m128 v2= _mm_set1_ps(tmpvar);
	tmpvar=0.01;
	__m128 v3= _mm_set1_ps(tmpvar);
/*
//Initialazing ***********ATTENTION!!!*******
// may need to get inside loop
*/	int world_size;
	int world_rank;

	//MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
			
	__m128 tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
	__m128 C_vector;
	__m128 L_vector;
	__m128 R_vector;

	__m128 num_0,num_1,num_2, num;
	__m128 den_0,den_1,den;

	__m128 m_vector;
	__m128 n_vector;

	__m128** arvec=  (__m128**)malloc(2*sizeof(__m128*));
	__m128** arvec2= (__m128**)malloc(2*sizeof(__m128*));
	
	MPI_Datatype int_array;
	MPI_Type_contiguous(5,MPI_INT,&int_array);
	
	
	for(int j=0;j<iters;j++)
	{
		double time0=gettime();

		

		int i;


		for(i=0;i<N-4;i+=4)
		{	
			if (world_rank==0){
				 C_vector=_mm_load_ps(&CVec[i]);
				 L_vector=_mm_load_ps(&LVec[i]);
				 R_vector=_mm_load_ps(&RVec[i]);



				num_0 = _mm_add_ps(L_vector, R_vector);
				

				tmp6 	= _mm_sub_ps(C_vector,L_vector);
			    den_0 	= _mm_sub_ps(tmp6,R_vector);
			    *arvec[0] = den_0;
			    *arvec[1] = num_0;
				MPI_Send(arvec,2,int_array,2,i,MPI_COMM_WORLD);
			}
			else if((world_rank==1)){
				 m_vector=_mm_load_ps(&mVec[i]);
				 n_vector=_mm_load_ps(&nVec[i]);  
			

				 tmp1 =_mm_sub_ps(m_vector, v1);
				 tmp2 =_mm_mul_ps(m_vector,tmp1);
				 num_1=_mm_div_ps(tmp2, v2);


				tmp3  =_mm_sub_ps( n_vector, v1);
				tmp4  =_mm_mul_ps(n_vector,tmp3);
				num_2 =_mm_div_ps(tmp4, v2);


				tmp5  =_mm_add_ps(num_1,num_2);
		

				den_1 = _mm_mul_ps(m_vector,n_vector);
				
				*arvec2[0] = tmp5;
			    *arvec2[1] = den_1;
				MPI_Send(arvec,2,int_array,2,i,MPI_COMM_WORLD);
			}


			if(world_rank==2){
				MPI_Recv(arvec,2,int_array,0,i,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(arvec2,2,int_array,1,i,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				tmp5=*arvec2[0];
				den_1=*arvec2[1];

				den_0=*arvec[0];
				num_0=*arvec[1];

				num  =_mm_div_ps(num_0,tmp5);

				den  = _mm_div_ps(den_0,den_1);
				tmp7 = _mm_add_ps(den,v3);
				tmp8 = _mm_div_ps(num,tmp7);
				_mm_store_ps(&FVec[i],tmp8); 

				maxF = FVec[i]  >maxF?FVec[i]  :maxF;
				maxF = FVec[i+1]>maxF?FVec[i+1]:maxF;
				maxF = FVec[i+2]>maxF?FVec[i+2]:maxF;
				maxF = FVec[i+3]>maxF?FVec[i+3]:maxF;	
			}
			
			


	


		}
		MPI_Finalize();

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
	printf("Time %f Max %f\n", timeTotal/iters, maxF);

	free(mVec);
	free(nVec);
	free(LVec);
	free(RVec);
	free(CVec);
	free(FVec);
}
