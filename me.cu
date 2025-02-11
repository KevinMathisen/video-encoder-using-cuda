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

#include <cuda_runtime.h>

#include "me.h"
#include "tables.h"


__device__ void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  *result = 0;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      *result += abs((int)(block2[v*stride+u] - block1[v*stride+u]));
    }
  }
}


__global__ void me_block_8x8_kernel(struct c63_common *cm, int mb_cols, int mb_rows,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
  int mb_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (mb_x >= mb_cols || mb_y >= mb_rows) return;

  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      int sad;
      sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);

      if (sad < best_sad)
      {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  mb->use_mv = 1;
}


void c63_motion_estimate(struct c63_common *cm)
{
  int mb_cols = cm->mb_cols;
  int mb_rows = cm->mb_rows;

  uint8_t *d_orig_Y, *d_recons_Y;
  uint8_t *d_orig_U, *d_recons_U;
  uint8_t *d_orig_V, *d_recons_V;

  // Allocate memory on the GPU
  cudaMalloc((void**)&d_orig_Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t));
  cudaMalloc((void**)&d_recons_Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t));
  cudaMalloc((void**)&d_orig_U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t));
  cudaMalloc((void**)&d_recons_U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t));
  cudaMalloc((void**)&d_orig_V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t));
  cudaMalloc((void**)&d_recons_V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t));

  // Copy data from host to device
  cudaMemcpy(d_orig_Y, cm->curframe->orig->Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_recons_Y, cm->refframe->recons->Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orig_U, cm->curframe->orig->U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_recons_U, cm->refframe->recons->U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orig_V, cm->curframe->orig->V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_recons_V, cm->refframe->recons->V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t), cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim((mb_cols + blockDim.x - 1) / blockDim.x, (mb_rows + blockDim.y - 1) / blockDim.y);

  /* Luma */
  me_block_8x8_kernel<<<gridDim, blockDim>>>(cm, mb_cols, mb_rows, d_orig_Y, d_recons_Y, Y_COMPONENT);

  /* Chroma */
  mb_cols /= 2;
  mb_rows /= 2;

  me_block_8x8_kernel<<<gridDim, blockDim>>>(cm, mb_cols, mb_rows, d_orig_U, d_recons_U, U_COMPONENT);
  me_block_8x8_kernel<<<gridDim, blockDim>>>(cm, mb_cols, mb_rows, d_orig_V, d_recons_V, V_COMPONENT);

  cudaDeviceSynchronize();

  // Free GPU memory
  cudaFree(d_orig_Y);
  cudaFree(d_recons_Y);
  cudaFree(d_orig_U);
  cudaFree(d_recons_U);
  cudaFree(d_orig_V);
  cudaFree(d_recons_V);
}





/* CUDA kernel for motion compensation for 8x8 block */
__global__ void mc_block_8x8_kernel(struct c63_common *cm, int mb_cols, int mb_rows,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
  int mb_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (mb_x >= mb_cols || mb_y >= mb_rows) return;

  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int y, x;
  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_cols = cm->mb_cols;
  int mb_rows = cm->mb_rows;

  dim3 blockDim(16, 16);
  dim3 gridDim((mb_cols + blockDim.x - 1) / blockDim.x, (mb_rows + blockDim.y - 1) / blockDim.y);

  /* Luma */
  mc_block_8x8_kernel<<<gridDim, blockDim>>>(cm, mb_cols, mb_rows,
      cm->curframe->predicted->Y, cm->refframe->recons->Y, Y_COMPONENT);

  /* Chroma */
  mb_cols /= 2;
  mb_rows /= 2;

  mc_block_8x8_kernel<<<gridDim, blockDim>>>(cm, mb_cols, mb_rows,
      cm->curframe->predicted->U, cm->refframe->recons->U, U_COMPONENT);
  mc_block_8x8_kernel<<<gridDim, blockDim>>>(cm, mb_cols, mb_rows,
      cm->curframe->predicted->V, cm->refframe->recons->V, V_COMPONENT);

  cudaDeviceSynchronize();
}
