#ifndef QUANT_DCT_H
#define QUANT_DCT_H

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization);

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization);

#endif

