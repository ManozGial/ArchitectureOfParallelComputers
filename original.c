#include"Project2_header.h"


void original(int N)
{	
	int iters = 1000;
	float maxF = 0.0f;
	double timeTotal = 0.0f;
	for(int j=0;j<iters;j++)
	{
		double time0=gettime();
		for(int i=0;i<N;i++)
		{
			float num_0 = LVec[i]+RVec[i];
			float num_1 = mVec[i]*(mVec[i]-1.0)/2.0;
			float num_2 = nVec[i]*(nVec[i]-1.0)/2.0;
			float num = num_0/(num_1+num_2);
			float den_0 = CVec[i]-LVec[i]-RVec[i];
			float den_1 = mVec[i]*nVec[i];
			float den = den_0/den_1;
			FVec[i] = num/(den+0.01);

			maxF = FVec[i]>maxF?FVec[i]:maxF;
		}
			double time1=gettime();
			timeTotal += time1-time0;
	}
	printf("Original : Time %f Max %f\n", timeTotal/iters, maxF);
}
