#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>
#include "me.h"
#include "c63.h"

/**
 * Function for calculating sum of absolute difference between 
 * a block in orgin frame and a block in ref frame
 * 
 * @param orig      Pointer to top left corner of 8x8 block in orig
 * @param ref       Pointer to top left corner of 8x8 block in ref
 * @param stride    Width of frame, used for indexing
 * 
 * @return          Sum of absolute difference between blocks
 */
__device__ __forceinline__ int sad_block_8x8_device(const uint8_t *block1, uint8_t *block2, int stride)
{
    int u, v;
    int result = 0;

    for (v = 0; v < 8; ++v)
    {
        for (u = 0; u < 8; ++u)
        {
            result += abs(block2[v*stride+u] - block1[v*stride+u]);
        }
    }
    return result;
}

/**
 * Kernel for doing motion estimation on a given macroblock, and finding the
 * offset with the smallest sad to use in the encoding. 
 * 
 * @param d_orig    Frame we are encoding
 * @param d_ref     Frame we are using as reference for finding residuals
 * @param d_mbs     Where we store offset for each macroblock
 * @param range     Search range, i.e. how much to search in reference. Is halved for u and v
 * @param w         width of frame
 * @param h         height of frame
 * @param mb_cols   Number of columns
 * @param mb_rows   Number of rows
 */
__global__ void me_kernel(const uint8_t *d_orig, uint8_t *d_ref,
    struct macroblock *d_mbs, int range, int w, int h,
    int mb_cols, int mb_rows)
{
    // Macroblock indices from grid dimensions.
    int mb_x = blockIdx.x, mb_y = blockIdx.y;
    if (mb_x >= mb_cols || mb_y >= mb_rows)
        return;

    // Calculate search area top left corner in reference frame.
    int search_left = mb_x * 8 - range;
    int search_top  = mb_y * 8 - range;

    // Determine candidate block position using thread indices.
    int x = search_left + threadIdx.x;
    int y = search_top  + threadIdx.y;

    // Calculate starting point of the original block.
    int mx = mb_x * 8;
    int my = mb_y * 8;

    int sad_value = INT_MAX;
    // Only compute SAD if candidate block is within valid bounds.
    if (x >= 0 && x <= w - 8 && y >= 0 && y <= h - 8) {
         sad_value = sad_block_8x8_device(d_orig + my * w + mx,
                    d_ref + y * w + x, w);
    }

    // Shared memory arrays for reduction (assumes blockDim.x * blockDim.y = 1024).
    __shared__ int s_sad[1024];
    __shared__ int s_mv_x[1024];
    __shared__ int s_mv_y[1024];

    // Compute a unique thread id within the block.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Store each thread's SAD value and motion vector (offset relative to original).
    s_sad[tid]  = sad_value;
    s_mv_x[tid] = x - mx;
    s_mv_y[tid] = y - my;

    __syncthreads();

    // Full block reduction using shared memory.
    // Reduce 1024 values down to 1.
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_sad[tid + stride] < s_sad[tid]) {
                s_sad[tid]  = s_sad[tid + stride];
                s_mv_x[tid] = s_mv_x[tid + stride];
                s_mv_y[tid] = s_mv_y[tid + stride];
            }   
        }
        __syncthreads();
    }

    // The thread with tid 0 writes the final result.
    if (tid == 0) {
        int mb_index = mb_y * mb_cols + mb_x;
        d_mbs[mb_index].sad  = s_sad[0];
        d_mbs[mb_index].mv_x = s_mv_x[0];
        d_mbs[mb_index].mv_y = s_mv_y[0];
    }
}


/**
 * Kernel for doing motion compensation, using the offset found in ME for a block
 * to copy a single pixel in the block from the reference to predicted (output)
 * 
 * @param d_out     Where we will place predicted
 * @param d_ref     Reference we will copy from
 * @param d_mbs     Block offsets
 * @param w         Width of pixels
 * @param h         Height of pixels
 * @param mb_cols   Number of columns 
 * @param mb_rows   Number of rows
 */
__global__ void mc_kernel(uint8_t *d_out, const uint8_t *d_ref,
const struct macroblock *d_mbs, int w, int h, int mb_cols, int mb_rows)
{
    // Macroblock index from the grid
    int mb_x = blockIdx.x, mb_y = blockIdx.y;

    // Return if outside of valid blocks
    if (mb_x >= mb_cols || mb_y >= mb_rows) return;

    // Pixel coordinates in original frame
    int x = mb_x*8 + threadIdx.x, y = mb_y*8 + threadIdx.y;

    // Return if pixel out of bounds
    if (x >= w || y >= h) return;

    // Get macroblock offset
    struct macroblock mb = d_mbs[mb_y*mb_cols + mb_x];

    // check if we should use mv, although redundant
    if (!mb.use_mv) return;

    // Compute pixel coordinates in reference
    int ref_x = x + mb.mv_x, ref_y = y + mb.mv_y;
    // Could check if reference is out of bounds, but should not be possible

    // Copy pixel to predicted frame
    d_out[y*w + x] = d_ref[ref_y*w + ref_x];

}
