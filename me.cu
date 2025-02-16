#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "me.h"
#include "tables.h"

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                             \
              __FILE__, __LINE__, cudaGetErrorString(err));                \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

/* Global Variables */
int w_y, h_y, w_uv, h_uv;                 // Height and width
int mb_cols_y, mb_rows_y, mb_cols_uv, mb_rows_uv; // Columns and rows
int mem_size_y, mem_size_uv, mem_size_mbs_y, mem_size_mbs_uv;
int range;

uint8_t *d_in_org_Y, *d_in_org_U, *d_in_org_V;    // Pointer to org input on GPU
uint8_t *d_in_ref_Y, *d_in_ref_U, *d_in_ref_V;    // Pointer to ref input on GPU
struct macroblock *d_mbs_Y, *d_mbs_U, *d_mbs_V;   // Pointer to macroblocks on GPU
uint8_t *d_out_Y, *d_out_U, *d_out_V;             // Pointer to output from GPU

extern __global__ void me_kernel(const uint8_t *d_orig, uint8_t *d_ref,
struct macroblock *d_mbs, int range, int w, int h, int mb_cols, int mb_rows);

extern __global__ void mc_kernel(uint8_t *d_out, const uint8_t *d_ref,
const struct macroblock *d_mbs, int w, int h, int mb_cols, int mb_rows);

static cudaStream_t stream[3];

__host__ void gpu_init(struct c63_common *cm)
{
  // Set the height and width of the frame components
  w_y = cm->padw[Y_COMPONENT];
  h_y = cm->padh[Y_COMPONENT];
  w_uv = cm->padw[U_COMPONENT];
  h_uv = cm->padh[U_COMPONENT];

  // Set the column and rows for the frame components
  mb_cols_y = cm->mb_cols;
  mb_rows_y = cm->mb_rows;
  mb_cols_uv = mb_cols_y/2;
  mb_rows_uv = mb_rows_y/2;

  // Calculate the size of input and output memory
  mem_size_y = w_y*h_y*sizeof(uint8_t);
  mem_size_uv = w_uv*h_uv*sizeof(uint8_t);
  mem_size_mbs_y = mb_cols_y*mb_rows_y*sizeof(struct macroblock);
  mem_size_mbs_uv = mb_cols_uv*mb_rows_uv*sizeof(struct macroblock);

  range = cm->me_search_range;

  // Allocate memory for input
  CUDA_CHECK(cudaMalloc((void **) &d_in_org_Y, mem_size_y));
  CUDA_CHECK(cudaMalloc((void **) &d_in_org_U, mem_size_uv));
  CUDA_CHECK(cudaMalloc((void **) &d_in_org_V, mem_size_uv));

  CUDA_CHECK(cudaMalloc((void **) &d_in_ref_Y, mem_size_y));
  CUDA_CHECK(cudaMalloc((void **) &d_in_ref_U, mem_size_uv));
  CUDA_CHECK(cudaMalloc((void **) &d_in_ref_V, mem_size_uv));

  // Allocate memory for macroblock offsets
  CUDA_CHECK(cudaMalloc((void**) &d_mbs_Y, mem_size_mbs_y));
  CUDA_CHECK(cudaMalloc((void**) &d_mbs_U, mem_size_mbs_uv));
  CUDA_CHECK(cudaMalloc((void**) &d_mbs_V, mem_size_mbs_uv));

  // Allocate memory for output
  CUDA_CHECK(cudaMalloc((void**) &d_out_Y, mem_size_y));
  CUDA_CHECK(cudaMalloc((void**) &d_out_U, mem_size_uv));
  CUDA_CHECK(cudaMalloc((void**) &d_out_V, mem_size_uv));

  // Create streams
  for (int i = 0; i < 3; i++) 
    CUDA_CHECK(cudaStreamCreate(&stream[i]));
}

__host__ void gpu_cleanup()
{
  // Free all memory used on the GPu
  CUDA_CHECK(cudaFree(d_in_org_Y));
  CUDA_CHECK(cudaFree(d_in_org_U));
  CUDA_CHECK(cudaFree(d_in_org_V));

  CUDA_CHECK(cudaFree(d_in_ref_Y));
  CUDA_CHECK(cudaFree(d_in_ref_U));
  CUDA_CHECK(cudaFree(d_in_ref_V));

  CUDA_CHECK(cudaFree(d_mbs_Y));
  CUDA_CHECK(cudaFree(d_mbs_U));
  CUDA_CHECK(cudaFree(d_mbs_V));

  CUDA_CHECK(cudaFree(d_out_Y));
  CUDA_CHECK(cudaFree(d_out_U));
  CUDA_CHECK(cudaFree(d_out_V));

  // Destroy streams
  for(int i = 0; i < 3; i++)
    CUDA_CHECK(cudaStreamDestroy(stream[i]));
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpyAsync(d_in_org_Y, cm->curframe->orig->Y, mem_size_y, cudaMemcpyHostToDevice, stream[0]));
  CUDA_CHECK(cudaMemcpyAsync(d_in_ref_Y, cm->refframe->recons->Y, mem_size_y, cudaMemcpyHostToDevice, stream[0]));
  CUDA_CHECK(cudaMemcpyAsync(d_in_org_U, cm->curframe->orig->U, mem_size_uv, cudaMemcpyHostToDevice, stream[1]));
  CUDA_CHECK(cudaMemcpyAsync(d_in_ref_U, cm->refframe->recons->U, mem_size_uv, cudaMemcpyHostToDevice, stream[1]));
  CUDA_CHECK(cudaMemcpyAsync(d_in_org_V, cm->curframe->orig->V, mem_size_uv, cudaMemcpyHostToDevice, stream[2]));
  CUDA_CHECK(cudaMemcpyAsync(d_in_ref_V, cm->refframe->recons->V, mem_size_uv, cudaMemcpyHostToDevice, stream[2]));
  
  /* Set dimentions for grid and blocks */
  // Blocks correspond to macroblocks in frame
  dim3 block_grid_y(mb_cols_y, mb_rows_y, 1);
  dim3 block_grid_uv(mb_cols_uv, mb_rows_uv, 1);

  // Threads correspond to each candidate reference
  dim3 thread_grid_y(2*range, 2*range, 1);
  dim3 thread_grid_uv(range, range, 1);
  
  /* Motion estimation for Luma (Y) */
  me_kernel<<<block_grid_y, thread_grid_y, 0, stream[0]>>>(
    d_in_org_Y, d_in_ref_Y, d_mbs_Y, range, w_y, h_y, mb_cols_y, mb_rows_y);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 1 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // Copy motion vectors back to host
  CUDA_CHECK(cudaMemcpyAsync(cm->curframe->mbs[Y_COMPONENT], d_mbs_Y, mem_size_mbs_y, cudaMemcpyDeviceToHost, stream[0]));

  /* Motion estimation for Chroma (U) */
  me_kernel<<<block_grid_uv, thread_grid_uv, 0, stream[1]>>>(
    d_in_org_U, d_in_ref_U, d_mbs_U, range/2, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 2 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // Copy motion vectors back to host
  CUDA_CHECK(cudaMemcpyAsync(cm->curframe->mbs[U_COMPONENT], d_mbs_U, mem_size_mbs_uv, cudaMemcpyDeviceToHost, stream[1]));

  /* Motion estimation for Chroma (V) */
  me_kernel<<<block_grid_uv, thread_grid_uv, 0, stream[2]>>>(
    d_in_org_V, d_in_ref_V, d_mbs_V, range/2, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 3 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // Copy motion vectors back to host
  CUDA_CHECK(cudaMemcpyAsync(cm->curframe->mbs[V_COMPONENT], d_mbs_V, mem_size_mbs_uv, cudaMemcpyDeviceToHost, stream[2]));

}

void c63_motion_compensate(struct c63_common *cm)
{
  /* Set dimentions for grid and blocks */
  // Each block correspond to one macroblock
  dim3 block_grid_y(mb_cols_y, mb_rows_y);
  dim3 block_grid_uv(mb_cols_uv, mb_rows_uv);

  // Each thread correspond to one pixel in each block
  dim3 thread_grid(8, 8);

  /* Motion compensation for Luma (Y) */
  mc_kernel<<<block_grid_y, thread_grid, 0, stream[0]>>>(
    d_out_Y, d_in_ref_Y, d_mbs_Y, w_y, h_y, mb_cols_y, mb_rows_y);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 4 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // Copy results back to host
  CUDA_CHECK(cudaMemcpyAsync(cm->curframe->predicted->Y, d_out_Y, mem_size_y, cudaMemcpyDeviceToHost, stream[0]));

  /* Motion estimation for Chroma (U) */
  mc_kernel<<<block_grid_uv, thread_grid, 0, stream[1]>>>(
    d_out_U, d_in_ref_U, d_mbs_U, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 5 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // Copy results back to host
  CUDA_CHECK(cudaMemcpyAsync(cm->curframe->predicted->U, d_out_U, mem_size_uv, cudaMemcpyDeviceToHost, stream[1]));

  /* Motion estimation for Chroma (V) */
  mc_kernel<<<block_grid_uv, thread_grid, 0, stream[2]>>>(
    d_out_V, d_in_ref_V, d_mbs_V, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 6 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // Copy results back to host
  CUDA_CHECK(cudaMemcpyAsync(cm->curframe->predicted->V, d_out_V, mem_size_uv, cudaMemcpyDeviceToHost, stream[2]));

  // Ensure predicted is copied back to host before continuing with DCT
  cudaDeviceSynchronize();
}
