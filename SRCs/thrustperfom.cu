/********************************************************************
*  sample.cu
*  This is a example for using thrust in CUDA programming.
*
*  Written by: Wayne Wood
*  Manchester, UK
*  22/05/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>

#include <thrust/version.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <iostream>

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/

#if __DEVICE_EMULATION__

bool InitCUDA(void) { return true; }

#else

bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif

/************************************************************************/
/* raw CUDA routines                                                    */
/************************************************************************/

long DATA_SIZE = 1 * 1024 * 1024; // 1 M

int * data;

// generate random number ranged in [0, 9]
void GenerateNumbers(int * number, int size)
{
	srand(time(NULL));
	for (int i = 0; i < size; i++) {
		number[i] = rand() % 10;
	}
}

#define BLOCK_NUM	32
#define THREAD_NUM	512

__global__ static void sumOfSquares(int * num, int * result, clock_t * time,
									int DATA_SIZE)
{
	extern __shared__ int shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	if (tid == 0) time[bid] = clock();

	shared[tid] = 0;
	for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
		shared[tid] += num[i] * num[i];
	}

	__syncthreads();
	int offset = THREAD_NUM / 2;
	while (offset > 0) {
		if (tid < offset) {
			shared[tid] += shared[tid + offset];
		}
		offset >>= 1;
		__syncthreads();
	}

	if (tid == 0) {
		result[bid] = shared[0];
		time[bid + BLOCK_NUM] = clock();
	}
}

/************************************************************************/
/* helper routines for thrust                                           */
/************************************************************************/

// define functor for
// random number ranged in [0, 9]
class random
{
public:
	int operator() ()
	{
		return rand() % 10;
	}
};

// define transformation f(x) -> x^2
template <typename T>
struct square
{
	__host__ __device__
		T operator() (T x)
	{
		return x * x;
	}
};

/************************************************************************/
/* The main routine                                                     */
/************************************************************************/

int main(int argc, char* argv[])
{
	if (!InitCUDA()) {
		return 0;
	}

	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << std::endl;
	std::cout << std::endl;

	thrust::host_vector<int> vec_data;

	// for timer
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	LARGE_INTEGER elapsed_time_start, elapsed_time_end;
	double elapsed_time;

	// output file
	FILE * optr = fopen("testResults.csv", "w");
	if (!optr) {
		std::cout << "cannot open file!" << std::endl;
		return 1;
	}

	fprintf(optr,"Size (M),GPU Time,CPU Time,GPU thrust,CPU thrust\n");

	for (int h = 0; h < 6; h++, DATA_SIZE *= 2)
	{
		printf("Data size = %dM\n", DATA_SIZE / (1024 * 1024));
		fprintf(optr, "%d,", DATA_SIZE / (1024 * 1024));

		//////////////////////////////////////////////////////////////////
		// raw CUDA
		//////////////////////////////////////////////////////////////////

		data = new int[DATA_SIZE];
		GenerateNumbers(data, DATA_SIZE);

		//
		// calculation on GPU
		//

		QueryPerformanceCounter(&elapsed_time_start);

		int * gpudata, * result;
		clock_t * time;
		cudaMalloc((void **) &gpudata, sizeof(int) * DATA_SIZE);
		cudaMalloc((void **) &result, sizeof(int) * THREAD_NUM * BLOCK_NUM);
		cudaMalloc((void **) &time, sizeof(clock_t) * BLOCK_NUM * 2);
		cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

		int sum[BLOCK_NUM];
		sumOfSquares<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int)>>>
			(gpudata, result, time, DATA_SIZE);
		cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);

		int final_sum = 0;
		for (int i = 0; i < BLOCK_NUM; i++) {
			final_sum += sum[i];
		}
		QueryPerformanceCounter(&elapsed_time_end);

		cudaFree(gpudata);
		cudaFree(result);

		clock_t time_used[BLOCK_NUM * 2];
		cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
		cudaFree(time);

		clock_t min_start, max_end;
		min_start = time_used[0];
		max_end = time_used[BLOCK_NUM];
		for (int i = 1; i < BLOCK_NUM; i++) {
			if (min_start > time_used[i])
				min_start = time_used[i];
			if (max_end < time_used[i + BLOCK_NUM])
				max_end = time_used[i + BLOCK_NUM];
		}

		elapsed_time = (double)(elapsed_time_end.QuadPart - elapsed_time_start.QuadPart)
			/ frequency.QuadPart;

		// elapsed_time = (double)(max_end - min_start) / CLOCKS_PER_SEC;
		printf("sum (on GPU): %d; time: %lf (core clocks: %d)\n", final_sum, elapsed_time,
			max_end - min_start);
		fprintf(optr, "%f,", elapsed_time);

		//
		// calculation on CPU
		//

		QueryPerformanceCounter(&elapsed_time_start);

		final_sum = 0;
		for (int i = 0; i < DATA_SIZE; i++) {
			final_sum += data[i] * data[i];
		}

		QueryPerformanceCounter(&elapsed_time_end);
		elapsed_time = (double)(elapsed_time_end.QuadPart - elapsed_time_start.QuadPart)
			/ frequency.QuadPart;

		printf("sum (on CPU): %d; time: %lf\n", final_sum, elapsed_time);
		fprintf(optr, "%f,", elapsed_time);

		free(data);

		//////////////////////////////////////////////////////////////////
		// with thrust support
		//////////////////////////////////////////////////////////////////

		std::cout << "if with thrust support," << std::endl;

		//
		// calculation on GPU
		//

		vec_data.resize(DATA_SIZE);
		//srand(time(NULL));
		thrust::generate(vec_data.begin(), vec_data.end(), random());

		QueryPerformanceCounter(&elapsed_time_start);

		thrust::device_vector<int> vec_gpudata = vec_data;

		final_sum = thrust::transform_reduce(vec_gpudata.begin(), vec_gpudata.end(),
			square<int>(), 0, thrust::plus<int>());

		QueryPerformanceCounter(&elapsed_time_end);
		elapsed_time = (double)(elapsed_time_end.QuadPart - elapsed_time_start.QuadPart)
			/ frequency.QuadPart;

		printf("sum (on GPU): %d; time: %lf\n", final_sum, elapsed_time);
		fprintf(optr, "%f,", elapsed_time);

		//
		// calculation on CPU
		//

		QueryPerformanceCounter(&elapsed_time_start);

		final_sum = 0;
		for (int i = 0; i < DATA_SIZE; i++) {
			final_sum += vec_data[i] * vec_data[i];
		}

		QueryPerformanceCounter(&elapsed_time_end);
		elapsed_time = (double)(elapsed_time_end.QuadPart - elapsed_time_start.QuadPart)
			/ frequency.QuadPart;

		printf("sum (on CPU): %d; time: %lf\n", final_sum, elapsed_time);
		fprintf(optr, "%f\n", elapsed_time);

		std::cout << std::endl;
	}

	fclose(optr);

	return 0;
}
