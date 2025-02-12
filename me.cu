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

/* Global Variables */
int w_y, h_y, w_uv, h_uv;                 // Height and width
int mb_cols_y, mb_rows_y, mb_cols_uv, mb_rows_uv; // Columns and rows
int mem_size_y, mem_size_uv, mem_size_mbs_y, mem_size_mbs_uv;
int range;

uint8_t *d_in_org_Y, *d_in_org_U, *d_in_org_V;    // Pointer to org input on GPU
uint8_t *d_in_ref_Y, *d_in_ref_U, *d_in_ref_V;    // Pointer to ref input on GPU
struct macroblock *d_mbs_Y, *d_mbs_U, *d_mbs_V;         // Pointer to macroblocks on GPU
uint8_t *d_out_Y, *d_out_U, *d_out_V;             // Pointer to output from GPU

extern __global__ void me_kernel(const uint8_t *d_orig, uint8_t *d_ref,
struct macroblock *d_mbs, int range, int w, int h, int mb_cols, int mb_rows);

extern __global__ void mc_kernel(uint8_t *d_out, const uint8_t *d_ref,
const struct macroblock *d_mbs, int w, int h, int mb_cols, int mb_rows);

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
  cudaMalloc((void **) &d_in_org_Y, mem_size_y);
  cudaMalloc((void **) &d_in_org_U, mem_size_uv);
  cudaMalloc((void **) &d_in_org_V, mem_size_uv);

  cudaMalloc((void **) &d_in_ref_Y, mem_size_y);
  cudaMalloc((void **) &d_in_ref_U, mem_size_uv);
  cudaMalloc((void **) &d_in_ref_V, mem_size_uv);

  // Allocate memory for macroblock offsets
  cudaMalloc((void**) &d_mbs_Y, mem_size_mbs_y);
  cudaMalloc((void**) &d_mbs_U, mem_size_mbs_uv);
  cudaMalloc((void**) &d_mbs_V, mem_size_mbs_uv);

  // Allocate memory for output
  cudaMalloc((void**) &d_out_Y, mem_size_y);
  cudaMalloc((void**) &d_out_U, mem_size_uv);
  cudaMalloc((void**) &d_out_V, mem_size_uv);
}

__host__ void gpu_cleanup()
{
  cudaFree(d_in_org_Y);
  cudaFree(d_in_org_U);
  cudaFree(d_in_org_V);

  cudaFree(d_in_ref_Y);
  cudaFree(d_in_ref_U);
  cudaFree(d_in_ref_V);

  cudaFree(d_mbs_Y);
  cudaFree(d_mbs_U);
  cudaFree(d_mbs_V);

  cudaFree(d_out_Y);
  cudaFree(d_out_U);
  cudaFree(d_out_V);
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  
  // Copy data to device
  cudaMemcpy(d_in_org_Y, cm->curframe->orig->Y, mem_size_y, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_ref_Y, cm->refframe->recons->Y, mem_size_y, cudaMemcpyHostToDevice);
  
  // Set dimentions for grid and blocks
  dim3 block_grid_y(mb_cols_y, mb_rows_y, 1);
  dim3 block_grid_uv(mb_cols_uv, mb_rows_uv, 1);

  dim3 thread_grid_y(2*range, 2*range, 1);
  dim3 thread_grid_uv(range, range, 1);

  /* Motion estimation for Luma (Y) */
  me_kernel<<<block_grid_y, thread_grid_y>>>(d_in_org_Y, d_in_ref_Y, d_mbs_Y, 
  range, w_y, h_y, mb_cols_y, mb_rows_y);

  cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], d_mbs_Y, mem_size_mbs_y, cudaMemcpyDeviceToHost);

  cudaMemcpy(d_in_org_U, cm->curframe->orig->U, mem_size_uv, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_ref_U, cm->refframe->recons->U, mem_size_uv, cudaMemcpyHostToDevice);

  me_kernel<<<block_grid_uv, thread_grid_uv>>>(d_in_org_U, d_in_ref_U, d_mbs_U, 
  range/2, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  cudaMemcpy(cm->curframe->mbs[U_COMPONENT], d_mbs_U, mem_size_mbs_uv, cudaMemcpyDeviceToHost);

  cudaMemcpy(d_in_org_V, cm->curframe->orig->V, mem_size_uv, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_ref_V, cm->refframe->recons->V, mem_size_uv, cudaMemcpyHostToDevice);

  me_kernel<<<block_grid_uv, thread_grid_uv>>>(d_in_org_V, d_in_ref_V, d_mbs_V, 
  range/2, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

}

void c63_motion_compensate(struct c63_common *cm)
{
  dim3 block_grid_y(mb_cols_y, mb_rows_y);
  dim3 block_grid_uv(mb_cols_uv, mb_rows_uv);

  dim3 thread_grid(8, 8);

  mc_kernel<<<block_grid_y, thread_grid>>>(d_out_Y, d_in_ref_Y, d_mbs_Y, 
  w_y, h_y, mb_cols_y, mb_rows_y);

  mc_kernel<<<block_grid_uv, thread_grid>>>(d_out_U, d_in_ref_U, d_mbs_U, 
  w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  mc_kernel<<<block_grid_uv, thread_grid>>>(d_out_V, d_in_ref_V, d_mbs_V, 
  w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  cudaMemcpy(cm->curframe->predicted->Y, d_out_Y, mem_size_y, cudaMemcpyDeviceToHost);
  cudaMemcpy(cm->curframe->predicted->U, d_out_U, mem_size_uv, cudaMemcpyDeviceToHost);
  cudaMemcpy(cm->curframe->predicted->V, d_out_V, mem_size_uv, cudaMemcpyDeviceToHost);
}
