#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>
#include "me.h"
#include "c63.h"

/**
 * Compute the sum of absolute differences (SAD) for an 8x8 block
 * using texture fetches from the original and reference frames.
 *
 * @param tex_orig   Texture object for the original frame.
 * @param tex_ref    Texture object for the reference frame.
 * @param orig_x     X-coordinate of the top-left corner of the block in the original frame.
 * @param orig_y     Y-coordinate of the top-left corner of the block in the original frame.
 * @param ref_x      X-coordinate of the top-left corner of the block in the reference frame.
 * @param ref_y      Y-coordinate of the top-left corner of the block in the reference frame.
 *
 * @return           Sum of absolute differences over the 8x8 block.
 */
__device__ int sad_block_8x8_tex_device(cudaTextureObject_t tex_orig, cudaTextureObject_t tex_ref,
                                          int orig_x, int orig_y, int ref_x, int ref_y)
{
    int u, v;
    int result = 0;
    for (v = 0; v < 8; ++v)
    {
        for (u = 0; u < 8; ++u)
        {
            // Fetch pixels using texture fetches.
            uint8_t orig_pixel = tex2D<uint8_t>(tex_orig, orig_x + u, orig_y + v);
            uint8_t ref_pixel  = tex2D<uint8_t>(tex_ref, ref_x + u, ref_y + v);
            result += abs((int)ref_pixel - (int)orig_pixel);
        }
    }
    return result;
}

/**
 * Kernel for performing motion estimation on macroblocks.
 *
 * @param tex_orig   Texture object for the original frame.
 * @param tex_ref    Texture object for the reference frame.
 * @param d_mbs      Output macroblock data (motion vectors).
 * @param range      Search range.
 * @param w          Width of the frame.
 * @param h          Height of the frame.
 * @param mb_cols    Number of macroblock columns.
 * @param mb_rows    Number of macroblock rows.
 */
__global__ void me_kernel(cudaTextureObject_t tex_orig, cudaTextureObject_t tex_ref,
                            struct macroblock *d_mbs, int range, int w, int h, int mb_cols, int mb_rows)
{
    // Macroblock indices.
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    // Return if this macroblock is outside the valid range.
    if (mb_x >= mb_cols || mb_y >= mb_rows)
        return;

    // Determine the top-left corner of the search area in the reference frame.
    int search_left = mb_x * 8 - range;
    int search_top  = mb_y * 8 - range;

    // Determine the candidate position within the search area using the thread indices.
    int x = search_left + threadIdx.x;
    int y = search_top + threadIdx.y;

    // Coordinates for the original block.
    int mx = mb_x * 8;
    int my = mb_y * 8;

    int sad_value = INT_MAX;
    // Only compute SAD if the candidate block is within the frame bounds.
    if (x >= 0 && x <= w - 8 && y >= 0 && y <= h - 8)
    {
        sad_value = sad_block_8x8_tex_device(tex_orig, tex_ref, mx, my, x, y);
    }

    // Allocate shared memory for SAD and corresponding motion vector components.
    __shared__ int s_sad[1024];
    __shared__ int s_mv_x[1024];
    __shared__ int s_mv_y[1024];

    // Flatten the thread index.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    s_sad[tid] = sad_value;
    s_mv_x[tid] = x - mx;
    s_mv_y[tid] = y - my;

    __syncthreads();

    // Reduce across the threads to find the minimum SAD value.
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (s_sad[tid + stride] < s_sad[tid])
            {
                s_sad[tid] = s_sad[tid + stride];
                s_mv_x[tid] = s_mv_x[tid + stride];
                s_mv_y[tid] = s_mv_y[tid + stride];
            }
        }
        __syncthreads();
    }

    // The first thread writes the best motion vector for this macroblock.
    if (tid == 0)
    {
        struct macroblock *mb = &d_mbs[mb_y * mb_cols + mb_x];
        mb->mv_x = s_mv_x[0];
        mb->mv_y = s_mv_y[0];
        mb->use_mv = 1; // Assume the motion vector is always beneficial.
    }
}

/**
 * Kernel for performing motion compensation using the motion vectors.
 *
 * @param d_out     Output predicted frame buffer.
 * @param tex_ref   Texture object for the reference frame.
 * @param d_mbs     Macroblock motion vector data.
 * @param w         Width of the frame.
 * @param h         Height of the frame.
 * @param mb_cols   Number of macroblock columns.
 * @param mb_rows   Number of macroblock rows.
 */
__global__ void mc_kernel(uint8_t *d_out, cudaTextureObject_t tex_ref,
                           const struct macroblock *d_mbs, int w, int h, int mb_cols, int mb_rows)
{
    // Macroblock indices.
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    if (mb_x >= mb_cols || mb_y >= mb_rows)
        return;

    // Determine the pixel coordinates in the output frame.
    int x = mb_x * 8 + threadIdx.x;
    int y = mb_y * 8 + threadIdx.y;

    if (x >= w || y >= h)
        return;

    // Retrieve the macroblock data.
    struct macroblock mb = d_mbs[mb_y * mb_cols + mb_x];
    if (!mb.use_mv)
        return;

    // Compute the corresponding pixel coordinates in the reference frame.
    int ref_x = x + mb.mv_x;
    int ref_y = y + mb.mv_y;

    // Fetch the pixel from the reference texture.
    uint8_t pixel = tex2D<uint8_t>(tex_ref, ref_x, ref_y);

    // Write the pixel to the output frame.
    d_out[y * w + x] = pixel;
}
