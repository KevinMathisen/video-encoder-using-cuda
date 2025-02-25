#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>
#include "me.h"
#include "c63.h"

/**
 * Function for calculating sum of absolute difference between 
 * a block in orgin frame and a block in ref frame
 * 
 * @param share_orig    Original block to encode
 * @param share_ref     Reference block to compare and calculcate SAD with
 * @param ref_x         Start of reference block in x
 * @param ref_y         Start of reference block in y
 * 
 * @return          Sum of absolute difference between blocks
 */
__device__ __forceinline__ int sad_block_8x8_device(const uint8_t share_orig[8][8], 
    const uint8_t share_ref[40][40], int ref_x, int ref_y)
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

    // Allocate shared memory for original 8x8 block and 40x40 reference block
    //   (40x40 as this covers all pixels when search range is 16 and block size 8x8)
    //   (aka (16x2+8)x(16x2+8), results in 24x24 for search range 8 )
    __shared__ uint8_t share_orig[8][8];
    __shared__ uint8_t share_ref[40][40];

    // Thread index to identify which candidate
    int tid_x = threadIdx.x, tid_y = threadIdx.y;

    // Use 64 threads to load original 8x8 block into shared memory
    if (tid_x < 8 && tid_y < 8)
        share_orig[tid_y][tid_x] = d_orig[(my+tid_y)*w + (mx+tid_x)];
    
    // Calculate left top corner for search area in reference frame
    int search_left = mb_x*8-range, search_top = mb_y*8-range;

    // Calculate where the thread should then start, 
    // i.e. use the thread index to calculcate where in the search area it is
    int x = search_left + tid_x, y = search_top + tid_y;

    // For each thread: load main part of reference frame we use to compare into shared memory
    //  (16 search range -> 32x32, 8 search range -> 16x16)
    if (x >= 0 && x < w && y >= 0 && y < h)
        share_ref[tid_y][tid_x] = d_ref[y*w+x];
    else
        share_ref[tid_y][tid_x] = 0; // Set reference outside of frame to 0

    int main_ref_edge = range*2;

    /* Load "edge" of the reference frame, where the width of it is always 7 (as block size is 8x8)
           Results in filling 40x40 for search range 16, and 24x24 for search range 8 */

    // Load "right" edge
    if (tid_x < 7 && x + main_ref_edge < w) 
        share_ref[tid_y][tid_x + main_ref_edge] = (y >= 0 && y < h) ? d_ref[y*w + (x+main_ref_edge)] : 0;
    // Load "bottom" edge
    if (tid_y < 7 && y + main_ref_edge < h) 
        share_ref[tid_y + main_ref_edge][tid_x] = (x >= 0 && x < w) ? d_ref[(y+main_ref_edge)*w + x] : 0;
    // Load "right-bottom" corner
    if (tid_x < 7 && tid_y < 7 && x + main_ref_edge < w && y + main_ref_edge < h) 
        share_ref[tid_y + main_ref_edge][tid_x + main_ref_edge] = d_ref[(y+main_ref_edge)*w + (x+main_ref_edge)];

    /* Ensure orig and ref is in shared memory before continuing */
    __syncthreads();

    int sad_value = INT_MAX;

    // If we are within bounds of reference frame 
    // (Does not support partial frame bounds) 
    if (x >= 0 && x <= w-8 && y >= 0 && y <= h-8) 
    {
        sad_value = sad_block_8x8_device(share_orig, share_ref, tid_x, tid_y);
    }

    /* Next we need to find the lowest sad_value and its offset 
        Use warp level reduction for this */

    int tid = tid_y * blockDim.x + tid_x;
    int lane = tid%32;      // index of thread in warp
    int warp_id = tid/32;   // index of warp

    // Calculate motion vector offset for thread/candidate
    int mv_x = x-mx, mv_y = y-my;

    /* Find lowest sad for each warp
        Do this by doing reduction with shfl_down_sync to end up with  
        lowest SAD and its offset in lane 0 of each warp */
    for (int offset = 16; offset > 0; offset /= 2) 
    {
        int sad_compare = __shfl_down_sync(0xFFFFFFFF, sad_value, offset);  // (assume 32 lanes in each warp because we 
        int mv_x_compare = __shfl_down_sync(0xFFFFFFFF, mv_x, offset);      //  have search range 16/8 -> 1024/256 threads.
        int mv_y_compare = __shfl_down_sync(0xFFFFFFFF, mv_y, offset);      //  could use __activemask() instead of 0xFFFFFFFF)

        if (lane < offset && sad_compare < sad_value) 
        {
            sad_value = sad_compare;
            mv_x = mv_x_compare;
            mv_y = mv_y_compare;
        }
    }

    /* Ensure all warps are done finding their best SAD */
    __syncthreads();

    /* Now we need to find best SAD from the remaning ones! */

    // Find amount of warps
    int num_warps = (blockDim.x*blockDim.y)/32;

    __shared__ int warp_sad[32];
    __shared__ int warp_mv_x[32];
    __shared__ int warp_mv_y[32];

    if (lane == 0) // use best sad from warp
    {
        warp_sad[warp_id] = sad_value;
        warp_mv_x[warp_id] = mv_x;
        warp_mv_y[warp_id] = mv_y;
    }
        
    /* Ensure all warps have written their minimum */
    __syncthreads();

    /* Final reduction using only first warp */
    if (warp_id == 0) 
    {
        // Each thread/lane in first warp will retreive best SAD from each warp
        sad_value = (lane < num_warps) ? warp_sad[lane] : INT_MAX;
        mv_x = (lane < num_warps) ? warp_mv_x[lane] : 0;
        mv_y = (lane < num_warps) ? warp_mv_y[lane] : 0;

        // Find lowest sad for remaining warp values
        for (int offset = num_warps/2; offset > 0; offset /= 2) 
        {
            int sad_compare = __shfl_down_sync(0xFFFFFFFF, sad_value, offset);  // (assume 32 lanes in each warp because we 
            int mv_x_compare = __shfl_down_sync(0xFFFFFFFF, mv_x, offset);      //  have search range 16/8 -> 1024/256 threads.
            int mv_y_compare = __shfl_down_sync(0xFFFFFFFF, mv_y, offset);      //  could use __activemask() instead of 0xFFFFFFFF)

            if (lane < offset && sad_compare < sad_value) 
            {
                sad_value = sad_compare;
                mv_x = mv_x_compare;
                mv_y = mv_y_compare;
            }
        }
    }

    // Thread 0 has the smallest sad, return its offset
    if (tid == 0)
    {
        struct macroblock *mb = &d_mbs[mb_y*mb_cols + mb_x];
        mb->mv_x = mv_x;
        mb->mv_y = mv_y;
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