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

extern __global__ void me_kernel(const uint8_t *d_orig, uint8_t *d_ref,
struct macroblock *d_mbs, int range, int w, int h, int mb_cols, int mb_rows);

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  
  /* Motion estimation for Luma (Y) */
  {
    // Variables needed for Y
    int w = cm->padw[Y_COMPONENT], h = cm->padh[Y_COMPONENT];
    int cols = cm->mb_cols, rows = cm->mb_rows;
    int range = cm->me_search_range;

    uint8_t *d_orig_y = NULL;
    uint8_t *d_ref_y  = NULL;
    struct macroblock *d_mbs_y = NULL;

    // Allocate and copy memory
    CUDA_CHECK(cudaMalloc((void**)&d_orig_y, w*h*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_ref_y, w*h*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_mbs_y, cols*rows*sizeof(struct macroblock)));

    CUDA_CHECK(cudaMemcpy(d_orig_y, cm->curframe->orig->Y, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_y, cm->refframe->recons->Y, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(cols, rows, 1);
    dim3 block(2*range, 2*range, 1);

    me_kernel<<<grid, block>>>(d_orig_y, d_ref_y, d_mbs_y, 
      range, w, h, cols, rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back results and free memory
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], d_mbs_y, cols*rows*sizeof(struct macroblock), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_orig_y));
    CUDA_CHECK(cudaFree(d_ref_y));
    CUDA_CHECK(cudaFree(d_mbs_y));
  }

  /* Motion estimation for Chroma (U) */
  {
    // Variables needed for U
    int w = cm->padw[U_COMPONENT], h = cm->padh[U_COMPONENT];
    int cols = cm->mb_cols/2, rows = cm->mb_rows/2;
    int range = cm->me_search_range/2;

    uint8_t *d_orig_u = NULL;
    uint8_t *d_ref_u  = NULL;
    struct macroblock *d_mbs_u = NULL;

    // Allocate and copy memory
    cudaMalloc((void**)&d_orig_u, w*h*sizeof(uint8_t));
    cudaMalloc((void**)&d_ref_u, w*h*sizeof(uint8_t));
    cudaMalloc((void**)&d_mbs_u, cols*rows*sizeof(struct macroblock));

    cudaMemcpy(d_orig_u, cm->curframe->orig->U, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_u, cm->refframe->recons->U, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid(cols, rows, 1);
    dim3 block(2*range, 2*range, 1);

    me_kernel<<<grid, block>>>(d_orig_u, d_ref_u, d_mbs_u, 
      range, w, h, cols, rows);
    cudaDeviceSynchronize();

    // Copy back results and free memory
    cudaMemcpy(cm->curframe->mbs[U_COMPONENT], d_mbs_u, cols*rows*sizeof(struct macroblock), cudaMemcpyDeviceToHost);

    cudaFree(d_orig_u);
    cudaFree(d_ref_u);
    cudaFree(d_mbs_u);
  }

  /* Motion estimation for Chroma (V) */
  {
    // Variables needed for V
    int w = cm->padw[V_COMPONENT], h = cm->padh[V_COMPONENT];
    int cols = cm->mb_cols/2, rows = cm->mb_rows/2;
    int range = cm->me_search_range/2;

    uint8_t *d_orig_v = NULL;
    uint8_t *d_ref_v  = NULL;
    struct macroblock *d_mbs_v = NULL;

    // Allocate and copy memory
    cudaMalloc((void**)&d_orig_v, w*h*sizeof(uint8_t));
    cudaMalloc((void**)&d_ref_v, w*h*sizeof(uint8_t));
    cudaMalloc((void**)&d_mbs_v, cols*rows*sizeof(struct macroblock));

    cudaMemcpy(d_orig_v, cm->curframe->orig->V, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_v, cm->refframe->recons->V, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid(cols, rows, 1);
    dim3 block(2*range, 2*range, 1);

    me_kernel<<<grid, block>>>(d_orig_v, d_ref_v, d_mbs_v, 
      range, w, h, cols, rows);
    cudaDeviceSynchronize();

    // Copy back results and free memory
    cudaMemcpy(cm->curframe->mbs[V_COMPONENT], d_mbs_v, cols*rows*sizeof(struct macroblock), cudaMemcpyDeviceToHost);

    cudaFree(d_orig_v);
    cudaFree(d_ref_v);
    cudaFree(d_mbs_v);
  }

}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
