use 5.006;
use strict;
use warnings;

our $VERSION = 0.09;

use Test::More;

use Inline CUDA => Config =>
	clean_after_build => 0,
	BUILD_NOISY => 1,
	warnings => 10,
;

use Inline CUDA => <<'EOC';

#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

#define N 1000

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function adds the elements of two vectors (element-wise)
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//
__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = a[i]+b[i];
	}
}

int main() {
	cudaError_t err;
	//
	// Create int arrays on the CPU.
	// ('h' stands for "host".)
	//
	int ha[N], hb[N];

	//
	// Create corresponding int arrays on the GPU.
	// ('d' stands for "device".)
	//
	int *da, *db;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMalloc() has failed for %zu bytes for da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	if( (err=cudaMalloc((void **)&db, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

	//
	// Initialise the input data on the CPU.
	//
	for (int i = 0; i<N; ++i) {
		ha[i] = i;
	}

	//
	// Copy input data to array on GPU.
	//
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

	//
	// Launch GPU code with N threads, one per
	// array element.
	//
	add<<<N, 1>>>(da, db);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "main(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return 1; }

	//
	// Copy output array from GPU back to CPU.
	//
	if( (err=cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for %zu bytes for db->ha: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

/*
	for (int i = 0; i<N; ++i) {
		printf("%d\n", hb[i]);
	}
*/
	//
	// Free up the arrays on the GPU.
	//
	if( (err=cudaFree(da)) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaFree() has failed for da: %s\n", cudaGetErrorString(err)); return 1; }
	if( (err=cudaFree(db)) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaFree() has failed for db: %s\n", cudaGetErrorString(err)); return 1; }

	return 0;
}

EOC

my $retcode = main();

is($retcode, 0, "main() : called.");
done_testing();
