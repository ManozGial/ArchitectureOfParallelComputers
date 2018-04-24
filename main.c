#include"Project2_header.h"

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


int main()
{
	//int world_size;
	//int world_rank;
	//int N;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
if(world_rank==0){	
	
	printf("---------\nMain\n---------\n\n");


	
	printf("Give a number N that represents the DNA Positions : ");
	scanf("%d",&N);
	printf("\n----------\nResults\n---------\n\n");
	for(int j=1;j<world_size;j++){
		MPI_Send(&N ,1 ,MPI_INT ,j ,j ,MPI_COMM_WORLD);
	}

}
	if(world_rank!=0)
		MPI_Recv(&N ,1 ,MPI_INT ,0 ,world_rank ,MPI_COMM_WORLD,MPI_STATUS_IGNORE);



	srand(1);

	mVec = (float*)malloc(sizeof(float)*N);
	assert(mVec!=NULL);

	nVec = (float*)malloc(sizeof(float)*N);
	assert(nVec!=NULL);
	
	LVec = (float*)malloc(sizeof(float)*N);
	assert(LVec!=NULL);
	
	RVec = (float*)malloc(sizeof(float)*N);
	assert(RVec!=NULL);

	CVec = (float*)malloc(sizeof(float)*N);
	assert(CVec!=NULL);

	FVec = (float*)malloc(sizeof(float)*N);
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
	
	
		
	mpi(N);


	
MPI_Finalize();

	if(world_rank==0){
		simd(N);
		original(N);	
	}

	free(mVec);
	free(nVec);
	free(LVec);
	free(RVec);
	free(CVec);
	free(FVec);
}


