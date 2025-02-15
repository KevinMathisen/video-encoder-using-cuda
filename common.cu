#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "common.h"

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  // Free pinned memory
  cudaFreeHost(f->recons->Y);
  cudaFreeHost(f->recons->U);
  cudaFreeHost(f->recons->V);
  free(f->recons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  // Free pinned memory
  cudaFreeHost(f->predicted->Y);
  cudaFreeHost(f->predicted->U);
  cudaFreeHost(f->predicted->V);
  free(f->predicted);

  free(f->mbs[Y_COMPONENT]);
  free(f->mbs[U_COMPONENT]);
  free(f->mbs[V_COMPONENT]);

  free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  frame *f = (frame*)malloc(sizeof(struct frame));

  f->orig = image;

  // Use pinned memory for reconstructed, as this will be used to encode next frame
  f->recons = (yuv_t*)malloc(sizeof(yuv_t));
  cudaHostAlloc((void**)&(f->recons->Y), cm->ypw * cm->yph * sizeof(uint8_t), cudaHostAllocDefault);
  cudaHostAlloc((void**)&(f->recons->U), cm->upw * cm->uph * sizeof(uint8_t), cudaHostAllocDefault);
  cudaHostAlloc((void**)&(f->recons->V), cm->vpw * cm->vph * sizeof(uint8_t), cudaHostAllocDefault);

  // Use pinned memory for predicted, as this will be written to from the GPU
  f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
  cudaHostAlloc((void**)&(f->predicted->Y), cm->ypw * cm->yph * sizeof(uint8_t), cudaHostAllocDefault);
  cudaHostAlloc((void**)&(f->predicted->U), cm->upw * cm->uph * sizeof(uint8_t), cudaHostAllocDefault);
  cudaHostAlloc((void**)&(f->predicted->V), cm->vpw * cm->vph * sizeof(uint8_t), cudaHostAllocDefault);

  f->predicted->Y = (uint8_t*)calloc(cm->ypw * cm->yph, sizeof(uint8_t));
  f->predicted->U = (uint8_t*)calloc(cm->upw * cm->uph, sizeof(uint8_t));
  f->predicted->V = (uint8_t*)calloc(cm->vpw * cm->vph, sizeof(uint8_t));

  f->residuals = (dct_t*)malloc(sizeof(dct_t));
  f->residuals->Ydct = (int16_t*)calloc(cm->ypw * cm->yph, sizeof(int16_t));
  f->residuals->Udct = (int16_t*)calloc(cm->upw * cm->uph, sizeof(int16_t));
  f->residuals->Vdct = (int16_t*)calloc(cm->vpw * cm->vph, sizeof(int16_t));

  f->mbs[Y_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}
