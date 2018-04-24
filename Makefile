all:
	
	mpicc -lm -o Project2 main.c original.c simd.c mpi.c
	
	
