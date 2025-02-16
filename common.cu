#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  free(f->recons->Y);
  free(f->recons->U);
  free(f->recons->V);
  free(f->recons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  free(f->predicted->Y);
  free(f->predicted->U);
  free(f->predicted->V);
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

  f->recons = (yuv_t*)malloc(sizeof(yuv_t));
  f->recons->Y = (uint8_t*)malloc(cm->ypw * cm->yph);
  f->recons->U = (uint8_t*)malloc(cm->upw * cm->uph);
  f->recons->V = (uint8_t*)malloc(cm->vpw * cm->vph);

  f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
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
