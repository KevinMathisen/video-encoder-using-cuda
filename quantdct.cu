#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "common.h"
#include "tables.h"

#define ISQRT2 0.70710678118654f

static void dct_2d( const float* in, float* out )
{
    //Loop through all elements of the block
    for(int v = 0; v < 8; v++)
    {
        for(int u = 0; u < 8; u++)
        {
            /* Compute the DCT */
            float dct = 0;
            for(int y = 0; y < 8; y++)
            {
                for(int x = 0; x < 8; x++)
                {
                    dct += in[y*8+x] * dctlookup[x][u] * dctlookup[y][v];
                }
            }

            out[v*8+u] = dct;
        }
    }
}

static void idct_2d( const float* in, float* out )
{
    //Loop through all elements of the block
    for(int v = 0; v < 8; v++)
    {
        for(int u = 0; u < 8; u++)
        {
            /* Compute the iDCT */
            float dct = 0;
            for(int y = 0; y < 8; y++)
            {
                for(int x = 0; x < 8; x++)
                {
                    dct += in[y*8+x] * dctlookup[u][x] * dctlookup[v][y];
                }
            }

            out[v*8+u] = dct;
        }
    }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

static void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  for( int i = 0; i < 64; i++ ) { mb[i] = in_data[i]; }

  dct_2d(mb, mb2);
  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for( int i = 0; i < 64; i++ ) { out_data[i] = mb2[i]; }
}

static void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  for( int i = 0; i < 64; i++ ) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);
  idct_2d(mb, mb2);

  for( int i = 0; i < 64; i++ ) { out_data[i] = mb2[i]; }
}

static void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
    int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the dequantization and iDCT */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    dequant_idct_block_8x8(in_data+(x*8), block, quantization);

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];

        if (tmp < 0) { tmp = 0; }
        else if (tmp > 255) { tmp = 255; }

        out_data[i*w+j+x] = tmp;
      }
    }
  }
}

static void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
    int16_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the DCT and quantization */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions. */
    dct_quant_block_8x8(block, out_data+(x*8), quantization);
  }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y,
        out_data+y*width, quantization);
  }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dct_quantize_row(in_data+y*width, prediction+y*width, width, height,
        out_data+y*width, quantization);
  }
}

