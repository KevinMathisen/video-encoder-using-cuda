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
__device__ __forceinline__ int sad_block_8x8_device(const uint8_t share_orig[8][8], 
    const uint8_t share_ref[32][32], int ref_x, int ref_y)
{
    int u, v;
    int result = 0;

    for (v = 0; v < 8; ++v)
    {
        for (u = 0; u < 8; u+=4)
        {
            // Load 4 bytes at a time for memory coalescing
            result += abs(share_ref[ref_y + v][ref_x+u] - share_orig[v][u]);
            result += abs(share_ref[ref_y + v][ref_x+u+1] - share_orig[v][u+1]);
            result += abs(share_ref[ref_y + v][ref_x+u+2] - share_orig[v][u+2]);
            result += abs(share_ref[ref_y + v][ref_x+u+3] - share_orig[v][u+3]);
        }
    }
    return result;
}

/**
 * Uses warp level primitive to find the smallest sad value and its offset for a warp
 * 
 * @param sad   SAD value of a thread/candidate 
 * @param mv_x  X motion vector offset for candidate
 * @param mv_y  Y motion vector offset for candidate
 */
__device__ __forceinline__ void warp_min_reduction(int &sad, int &mv_x, int &mv_y)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // Get the value to compare with
        int sad_compare = __shfl_xor_sync(0xFFFFFFFF, sad, offset);   // (assume 32 lanes in each warp because we 
        int mv_x_compare = __shfl_xor_sync(0xFFFFFFFF, mv_x, offset); //  have search range 16 so 1025 threads.
        int mv_y_compare = __shfl_xor_sync(0xFFFFFFFF, mv_y, offset); //  could use __activemask() instead of 0xFFFFFFFF)

        if (sad_compare < sad) { // Update values to the one with smallest sad
            sad = sad_compare;
            mv_x = mv_x_compare;
            mv_y = mv_y_compare;
        }
    }
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

    // Find where orig block starts
    int mx = mb_x * 8, my = mb_y * 8;

    // Allocate shared memory for original 8x8 block and 32x32 reference block
    __shared__ uint8_t share_orig[8][8];
    __shared__ uint8_t share_ref[32][32];

    // Thread index to identify which candidate
    int tid_x = threadIdx.x, tid_y = threadIdx.y;

    // load original 8x8 block into shared memory
    if (tid_x < 8 && tid_y < 8)
        share_orig[tid_y][tid_x] = d_orig[(my+tid_y)*w + (mx+tid_x)];
    
    // Calculate left top corner for search area in reference frame
    int search_left = mb_x*8-range, search_top = mb_y*8-range;

    // Calculate where the thread should then start, 
    // i.e. use the thread index to calculcate where in the search area it is
    int x = search_left + tid_x, y = search_top + tid_y;

    // load 32x32 part of reference frame we use to compare into shared memory
    if (x >= 0 && x < w && y >= 0 && y < h)
        share_ref[tid_y][tid_x] = d_ref[y*w+x];
    else
        share_ref[tid_y][tid_x] = 0; // Set reference outside of frame to 0

    __syncthreads(); // ensure orig and ref is in shared memory before continuing

    int sad_value = INT_MAX;

    // If we are within bounds of reference frame 
    // (Does not support partial frame bounds) 
    if (x >= 0 && x <= w-8 && y >= 0 && y <= h-8) 
    {
        sad_value = sad_block_8x8_device(share_orig, share_ref, tid_x, tid_y);
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

    __syncthreads(); // Ensure all thread values are in shared memory before reduce

    // Use reduction to find the minimum sad
    // Start with half of the blocks, then 1/4, 1/8, etc until we only have one left
    for (int stride = blockDim.x*blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) // only use threads before stride
        {
            if (s_sad[tid + stride] < s_sad[tid]) // Check if we should move data to left
            {
                s_sad[tid] = s_sad[tid+stride];
                s_mv_x[tid] = s_mv_x[tid+stride];
                s_mv_y[tid] = s_mv_y[tid+stride];
            }
        }
        __syncthreads(); // Ensure all threads have copied if needed
    }

    // thread 0 has the smallest sad, return its offset
    if (tid == 0)
    {
        struct macroblock *mb = &d_mbs[mb_y*mb_cols + mb_x];
        mb->mv_x = s_mv_x[0];
        mb->mv_y = s_mv_y[0];
        mb->use_mv = 1; // always assume MV to be beneficial
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