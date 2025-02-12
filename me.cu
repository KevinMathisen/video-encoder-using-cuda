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

extern __global__ void me_kernel(const uint8_t *d_orig, uint8_t *d_ref,
struct macroblock *d_mbs, int range, int w, int h, int mb_cols, int mb_rows);

static void run_me_kernel(const uint8_t *orig, const uint8_t *ref, struct macroblock *mbs,
int range, int w, int h, int mb_cols, int mb_rows)
{
  // Allocate and copy memory
  uint8_t *d_orig_y, *d_ref_y;
  struct macroblock *d_mbs_y;

  cudaMalloc((void**)&d_orig_y, w*h*sizeof(uint8_t));
  cudaMalloc((void**)&d_ref_y, w*h*sizeof(uint8_t));
  cudaMalloc((void**)&d_mbs_y, mb_cols*mb_rows*sizeof(struct macroblock));

  cudaMemcpy(d_orig_y, orig, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_y, ref, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 grid(mb_cols, mb_rows, 1);
  dim3 block(2*range, 2*range, 1);

  me_kernel<<<grid, block>>>(d_orig_y, d_ref_y, d_mbs_y, range, w, h, mb_cols, mb_rows);
  cudaDeviceSynchronize();

  // Copy back results and free memory
  cudaMemcpy(mbs, d_mbs_y, mb_cols*mb_rows*sizeof(struct macroblock), cudaMemcpyDeviceToHost);

  cudaFree(d_orig_y);
  cudaFree(d_ref_y);
  cudaFree(d_mbs_y);
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  
  /* Motion estimation for Luma (Y) */
  run_me_kernel(cm->curframe->orig->Y, cm->refframe->recons->Y,
  cm->curframe->mbs[Y_COMPONENT], cm->me_search_range, 
  cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->mb_cols, cm->mb_rows);

  /* Motion estimation for Chroma (U) */
  run_me_kernel(cm->curframe->orig->U, cm->refframe->recons->U,
  cm->curframe->mbs[U_COMPONENT], cm->me_search_range/2, 
  cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->mb_cols/2, cm->mb_rows/2);

  /* Motion estimation for Chroma (V) */
  run_me_kernel(cm->curframe->orig->V, cm->refframe->recons->V,
  cm->curframe->mbs[V_COMPONENT], cm->me_search_range/2, 
  cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->mb_cols/2, cm->mb_rows/2);
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
