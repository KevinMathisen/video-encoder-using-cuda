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
struct macroblock *d_mbs, int range, int w, int h, int mb_cols, int mb_rows)
{
    // Macroblock index from the grid
    int mb_x = blockIdx.x, mb_y = blockIdx.y;

    // Return if outside of valid blocks
    if (mb_x >= mb_cols || mb_y >= mb_rows) return;

    // Calculate left top corner for search area in reference frame
    int search_left = mb_x*8-range, search_top = mb_y*8-range;

    // Calculate where the thread should then start, 
    // i.e. use the thread index to calculcate where in the search area it is
    int x = search_left + threadIdx.x, y = search_top + threadIdx.y;

    // Find where orig block starts
    int mx = mb_x * 8, my = mb_y * 8;

    int sad_value = INT_MAX;

    // If we are within bounds of reference frame 
    // (Does not support partial frame bounds) 
    if (x >= 0 && x <= w-8 && y >= 0 && y <= h-8) 
    {
        sad_value = sad_block_8x8_device(d_orig + my*w+mx, d_ref+y*w+x, w);
    }

    // Next we need to find the lowest sad_value and its offset
    // Store (sad, mv_x, mv_y) in shared memory for each thread

    // Shared memory for storing sad, mv_x, and my_y:
    __shared__ int s_sad[1024];
    __shared__ int s_mv_x[1024];
    __shared__ int s_mv_y[1024];

    // Get the thread index used to access shared memory
    int tid = threadIdx.y*blockDim.x + threadIdx.x;

    s_sad[tid] = sad_value;
    s_mv_x[tid] = x-mx;
    s_mv_y[tid] = y-my;

    // After storing each thread's SAD and motion vector into shared memory…
    __syncthreads();

    // Instead of a full block-wide reduction, perform warp-level reduction first.
    int lane   = tid & 31;     // Lane index within the warp (0-31)
    int warpId = tid >> 5;     // Warp index

    // Load the thread's value from shared memory
    int local_sad  = s_sad[tid];
    int local_mv_x = s_mv_x[tid];
    int local_mv_y = s_mv_y[tid];

    // Use warp shuffle to reduce within each warp.
    // Note: 0xFFFFFFFF is a mask for all active threads.
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_sad  = __shfl_down_sync(0xFFFFFFFF, local_sad, offset);
        int other_mv_x = __shfl_down_sync(0xFFFFFFFF, local_mv_x, offset);
        int other_mv_y = __shfl_down_sync(0xFFFFFFFF, local_mv_y, offset);
        
        if (other_sad < local_sad) {
            local_sad  = other_sad;
            local_mv_x = other_mv_x;
            local_mv_y = other_mv_y;
        }
    }

    // Each warp’s first lane writes its reduced result to shared memory.
    __shared__ int warp_sad[32];
    __shared__ int warp_mv_x[32];
    __shared__ int warp_mv_y[32];
    if (lane == 0) {
        warp_sad[warpId]  = local_sad;
        warp_mv_x[warpId] = local_mv_x;
        warp_mv_y[warpId] = local_mv_y;
    }
    __syncthreads();

    // Now, only the first warp (threads with tid < 32) participate in reducing the 32 warp results.
    if (tid < 32) {
        int final_sad  = warp_sad[lane];
        int final_mv_x = warp_mv_x[lane];
        int final_mv_y = warp_mv_y[lane];
        
        // Again, use warp-level reduction among these 32 values.
        for (int offset = 16; offset > 0; offset /= 2) {
            int other_sad  = __shfl_down_sync(0xFFFFFFFF, final_sad, offset);
            int other_mv_x = __shfl_down_sync(0xFFFFFFFF, final_mv_x, offset);
            int other_mv_y = __shfl_down_sync(0xFFFFFFFF, final_mv_y, offset);
            
            if (other_sad < final_sad) {
                final_sad  = other_sad;
                final_mv_x = other_mv_x;
                final_mv_y = other_mv_y;
            }
        }
        
        // The first lane of the first warp writes the final best motion vector.
        if (lane == 0) {
            int mb_index = mb_y * mb_cols + mb_x;
            d_mbs[mb_index].sad  = final_sad;
            d_mbs[mb_index].mv_x = final_mv_x;
            d_mbs[mb_index].mv_y = final_mv_y;
        }
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
