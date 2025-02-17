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

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                              \
              __FILE__, __LINE__, cudaGetErrorString(err));                 \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

/* Global Variables */
int w_y, h_y, w_uv, h_uv;                 // Width and height of the Y and UV components
int mb_cols_y, mb_rows_y, mb_cols_uv, mb_rows_uv; // Macroblock grid dimensions
int mem_size_y, mem_size_uv, mem_size_mbs_y, mem_size_mbs_uv;
int range;

// Instead of device pointers for the input images, we now use CUDA arrays...
cudaArray *d_in_org_Y_array = NULL;
cudaArray *d_in_org_U_array = NULL;
cudaArray *d_in_org_V_array = NULL;
cudaArray *d_in_ref_Y_array = NULL;
cudaArray *d_in_ref_U_array = NULL;
cudaArray *d_in_ref_V_array = NULL;

// ...and bind them to texture objects for optimized access.
cudaTextureObject_t tex_org_Y = 0;
cudaTextureObject_t tex_org_U = 0;
cudaTextureObject_t tex_org_V = 0;
cudaTextureObject_t tex_ref_Y = 0;
cudaTextureObject_t tex_ref_U = 0;
cudaTextureObject_t tex_ref_V = 0;

// Macroblocks and output buffers remain allocated in unified memory.
struct macroblock *d_mbs_Y, *d_mbs_U, *d_mbs_V;  // Macroblocks on GPU
uint8_t *d_out_Y, *d_out_U, *d_out_V;              // Output image buffers on GPU

/* 
 * Modified kernel declarations.
 * Note: The kernels must be updated to accept texture objects instead of 
 * raw pointers. For example, inside the kernel, you might use:
 *   uint8_t pixel = tex2D<uint8_t>(tex_orig, x, y);
 */
extern __global__ void me_kernel(cudaTextureObject_t tex_orig,
                                   cudaTextureObject_t tex_ref,
                                   struct macroblock *d_mbs, int range,
                                   int w, int h, int mb_cols, int mb_rows);

extern __global__ void mc_kernel(uint8_t *d_out,
                                   cudaTextureObject_t tex_ref,
                                   const struct macroblock *d_mbs,
                                   int w, int h, int mb_cols, int mb_rows);

                                  


__host__ void gpu_init(struct c63_common *cm)
{
  // Set the dimensions for each frame component from the common structure.
  w_y = cm->padw[Y_COMPONENT];
  h_y = cm->padh[Y_COMPONENT];
  w_uv = cm->padw[U_COMPONENT];
  h_uv = cm->padh[U_COMPONENT];
                                   
  mb_cols_y = cm->mb_cols;
  mb_rows_y = cm->mb_rows;
  mb_cols_uv = mb_cols_y / 2;
  mb_rows_uv = mb_rows_y / 2;
                                   
  mem_size_y = w_y * h_y * sizeof(uint8_t);
  mem_size_uv = w_uv * h_uv * sizeof(uint8_t);
  mem_size_mbs_y = mb_cols_y * mb_rows_y * sizeof(struct macroblock);
  mem_size_mbs_uv = mb_cols_uv * mb_rows_uv * sizeof(struct macroblock);
                                   
  range = cm->me_search_range;
                                   
  // Create a channel descriptor for 8-bit data.
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
                                   
  // Allocate CUDA arrays for the original and reference images (Y, U, V)
  CUDA_CHECK(cudaMallocArray(&d_in_org_Y_array, &channelDesc, w_y, h_y));
  CUDA_CHECK(cudaMallocArray(&d_in_org_U_array, &channelDesc, w_uv, h_uv));
  CUDA_CHECK(cudaMallocArray(&d_in_org_V_array, &channelDesc, w_uv, h_uv));

  CUDA_CHECK(cudaMallocArray(&d_in_ref_Y_array, &channelDesc, w_y, h_y));
  CUDA_CHECK(cudaMallocArray(&d_in_ref_U_array, &channelDesc, w_uv, h_uv));
  CUDA_CHECK(cudaMallocArray(&d_in_ref_V_array, &channelDesc, w_uv, h_uv));

  // Initialize the resource and texture descriptors to zero.
  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;

  memset(&resDesc, 0, sizeof(cudaResourceDesc));
  memset(&texDesc, 0, sizeof(cudaTextureDesc));


  /* Set fields in the descriptors */
  resDesc.resType = cudaResourceTypeArray;  // No cast needed if using C++
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // For original Y component.
  resDesc.res.array.array = d_in_org_Y_array;
  CUDA_CHECK(cudaCreateTextureObject(&tex_org_Y, &resDesc, &texDesc, NULL));

  // For original U component.
  resDesc.res.array.array = d_in_org_U_array;
  CUDA_CHECK(cudaCreateTextureObject(&tex_org_U, &resDesc, &texDesc, NULL));

  // For original V component.
  resDesc.res.array.array = d_in_org_V_array;
  CUDA_CHECK(cudaCreateTextureObject(&tex_org_V, &resDesc, &texDesc, NULL));

  // For reference Y component.
  resDesc.res.array.array = d_in_ref_Y_array;
  CUDA_CHECK(cudaCreateTextureObject(&tex_ref_Y, &resDesc, &texDesc, NULL));

  // For reference U component.
  resDesc.res.array.array = d_in_ref_U_array;
  CUDA_CHECK(cudaCreateTextureObject(&tex_ref_U, &resDesc, &texDesc, NULL));

  // For reference V component.
  resDesc.res.array.array = d_in_ref_V_array;
  CUDA_CHECK(cudaCreateTextureObject(&tex_ref_V, &resDesc, &texDesc, NULL));

  // Allocate memory for macroblocks (using unified memory for simplicity)
  CUDA_CHECK(cudaMallocManaged((void **) &d_mbs_Y, mem_size_mbs_y));
  CUDA_CHECK(cudaMallocManaged((void **) &d_mbs_U, mem_size_mbs_uv));
  CUDA_CHECK(cudaMallocManaged((void **) &d_mbs_V, mem_size_mbs_uv));

// Allocate memory for output images.
  CUDA_CHECK(cudaMallocManaged((void **) &d_out_Y, mem_size_y));
  CUDA_CHECK(cudaMallocManaged((void **) &d_out_U, mem_size_uv));
  CUDA_CHECK(cudaMallocManaged((void **) &d_out_V, mem_size_uv));
}   
                                   


__host__ void gpu_cleanup()
{
  // Destroy all texture objects.
  if (tex_org_Y) { CUDA_CHECK(cudaDestroyTextureObject(tex_org_Y)); tex_org_Y = 0; }
  if (tex_org_U) { CUDA_CHECK(cudaDestroyTextureObject(tex_org_U)); tex_org_U = 0; }
  if (tex_org_V) { CUDA_CHECK(cudaDestroyTextureObject(tex_org_V)); tex_org_V = 0; }
  if (tex_ref_Y) { CUDA_CHECK(cudaDestroyTextureObject(tex_ref_Y)); tex_ref_Y = 0; }
  if (tex_ref_U) { CUDA_CHECK(cudaDestroyTextureObject(tex_ref_U)); tex_ref_U = 0; }
  if (tex_ref_V) { CUDA_CHECK(cudaDestroyTextureObject(tex_ref_V)); tex_ref_V = 0; }

  // Free the CUDA arrays.
  if (d_in_org_Y_array) { CUDA_CHECK(cudaFreeArray(d_in_org_Y_array)); d_in_org_Y_array = NULL; }
  if (d_in_org_U_array) { CUDA_CHECK(cudaFreeArray(d_in_org_U_array)); d_in_org_U_array = NULL; }
  if (d_in_org_V_array) { CUDA_CHECK(cudaFreeArray(d_in_org_V_array)); d_in_org_V_array = NULL; }
  if (d_in_ref_Y_array) { CUDA_CHECK(cudaFreeArray(d_in_ref_Y_array)); d_in_ref_Y_array = NULL; }
  if (d_in_ref_U_array) { CUDA_CHECK(cudaFreeArray(d_in_ref_U_array)); d_in_ref_U_array = NULL; }
  if (d_in_ref_V_array) { CUDA_CHECK(cudaFreeArray(d_in_ref_V_array)); d_in_ref_V_array = NULL; }

  // Free macroblock and output memory.
  CUDA_CHECK(cudaFree(d_mbs_Y));
  CUDA_CHECK(cudaFree(d_mbs_U));
  CUDA_CHECK(cudaFree(d_mbs_V));

  CUDA_CHECK(cudaFree(d_out_Y));
  CUDA_CHECK(cudaFree(d_out_U));
  CUDA_CHECK(cudaFree(d_out_V));
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
    /* Copy the host frame data into the CUDA arrays using cudaMemcpy2DToArray.
     * This is done for each component.
     */
    CUDA_CHECK(cudaMemcpy2DToArray(d_in_org_Y_array, 0, 0,
                    cm->curframe->orig->Y, w_y * sizeof(uint8_t),
                    w_y * sizeof(uint8_t), h_y,
                    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy2DToArray(d_in_ref_Y_array, 0, 0,
                    cm->refframe->recons->Y, w_y * sizeof(uint8_t),
                    w_y * sizeof(uint8_t), h_y,
                    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy2DToArray(d_in_org_U_array, 0, 0,
                    cm->curframe->orig->U, w_uv * sizeof(uint8_t),
                    w_uv * sizeof(uint8_t), h_uv,
                    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy2DToArray(d_in_ref_U_array, 0, 0,
                    cm->refframe->recons->U, w_uv * sizeof(uint8_t),
                    w_uv * sizeof(uint8_t), h_uv,
                    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy2DToArray(d_in_org_V_array, 0, 0,
                    cm->curframe->orig->V, w_uv * sizeof(uint8_t),
                    w_uv * sizeof(uint8_t), h_uv,
                    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy2DToArray(d_in_ref_V_array, 0, 0,
                    cm->refframe->recons->V, w_uv * sizeof(uint8_t),
                    w_uv * sizeof(uint8_t), h_uv,
                    cudaMemcpyHostToDevice));

    // Set grid and block dimensions.
    dim3 block_grid_y(mb_cols_y, mb_rows_y, 1);
    dim3 block_grid_uv(mb_cols_uv, mb_rows_uv, 1);

    dim3 thread_grid_y(2 * range, 2 * range, 1);
    dim3 thread_grid_uv(range, range, 1);
    
    /* Launch the motion estimation kernels.
     * Note that we now pass the texture objects instead of raw pointers.
     */
    me_kernel<<<block_grid_y, thread_grid_y>>>(tex_org_Y, tex_ref_Y, d_mbs_Y,
                                            range, w_y, h_y, mb_cols_y, mb_rows_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel 1 launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    me_kernel<<<block_grid_uv, thread_grid_uv>>>(tex_org_U, tex_ref_U, d_mbs_U,
                                                 range / 2, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel 2 launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    me_kernel<<<block_grid_uv, thread_grid_uv>>>(tex_org_V, tex_ref_V, d_mbs_V,
                                                 range / 2, w_uv, h_uv, mb_cols_uv, mb_rows_uv);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel 3 launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy macroblock results back to host memory.
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], d_mbs_Y, mem_size_mbs_y, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[U_COMPONENT], d_mbs_U, mem_size_mbs_uv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[V_COMPONENT], d_mbs_V, mem_size_mbs_uv, cudaMemcpyDeviceToHost));
}

__host__ void c63_motion_compensate(struct c63_common *cm)
{
  dim3 block_grid_y(cm->mb_cols, cm->mb_rows);
  dim3 block_grid_uv(cm->mb_cols / 2, cm->mb_rows / 2);
  dim3 thread_grid(8, 8);

  /* Launch the motion compensation kernels.
   * Here the reference image is accessed via the texture objects.
   */
  mc_kernel<<<block_grid_y, thread_grid>>>(d_out_Y, tex_ref_Y, d_mbs_Y,
                                            w_y, h_y, mb_cols_y, mb_rows_y);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 4 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  mc_kernel<<<block_grid_uv, thread_grid>>>(d_out_U, tex_ref_U, d_mbs_U,
                                             w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 5 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  mc_kernel<<<block_grid_uv, thread_grid>>>(d_out_V, tex_ref_V, d_mbs_V,
                                             w_uv, h_uv, mb_cols_uv, mb_rows_uv);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel 6 launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  CUDA_CHECK(cudaMemcpy(cm->curframe->predicted->Y, d_out_Y, mem_size_y, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cm->curframe->predicted->U, d_out_U, mem_size_uv, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cm->curframe->predicted->V, d_out_V, mem_size_uv, cudaMemcpyDeviceToHost));
}
