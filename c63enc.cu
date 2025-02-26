#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "c63.h"
#include "c63_write.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                             \
              __FILE__, __LINE__, cudaGetErrorString(err));                \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static cudaStream_t stream[3]; // Cuda streams used in me.cu

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t* read_yuv(FILE *file, struct c63_common *cm)
{
  size_t len = 0;
  yuv_t *image = (yuv_t*)malloc(sizeof(*image));

  
  // Allocate pinned memory for Y component
  CUDA_CHECK(cudaHostAlloc((void**)&image->Y, 
    cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT] * sizeof(uint8_t), cudaHostAllocDefault));

  /* Read Y. The size of Y is the same as the size of the image. The indices
     represents the color component (0 is Y, 1 is U, and 2 is V) */
  len += fread(image->Y, 1, width*height, file);


  // Allocate pinned memory for U component
  CUDA_CHECK(cudaHostAlloc((void**)&image->U, 
    cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT] * sizeof(uint8_t), cudaHostAllocDefault));

  /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
     because (height/2)*(width/2) = (height*width)/4. */
  len += fread(image->U, 1, (width*height)/4, file);


  // Allocate pinned memory for V component
  CUDA_CHECK(cudaHostAlloc((void**)&image->V, 
    cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT] * sizeof(uint8_t), cudaHostAllocDefault));

  /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
  len += fread(image->V, 1, (width*height)/4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
    // Free pinned memory
    CUDA_CHECK(cudaFreeHost(image->Y));
    CUDA_CHECK(cudaFreeHost(image->U));
    CUDA_CHECK(cudaFreeHost(image->V));
    free(image);

    return NULL;
  }
  else if (len != width*height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

    // Free pinned memory
    CUDA_CHECK(cudaFreeHost(image->Y));
    CUDA_CHECK(cudaFreeHost(image->U));
    CUDA_CHECK(cudaFreeHost(image->V));
    free(image);

    return NULL;
  }

  return image;
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
  /* Advance to next frame */
  destroy_frame(cm->refframe);
  cm->refframe = cm->curframe;
  cm->curframe = create_frame(cm, image);

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
    
    // Apparantly memory is not zeroed when using cudaHostAlloc, so manually set it to 0 for keyframes
    //   (not needed for non-keyframes, as their predicted is set by the GPU)
    memset(cm->curframe->predicted->Y, 0, cm->ypw * cm->yph);
    memset(cm->curframe->predicted->U, 0, cm->upw * cm->uph);
    memset(cm->curframe->predicted->V, 0, cm->vpw * cm->vph);
  }
  else { cm->curframe->keyframe = 0; }

  if (!cm->curframe->keyframe)
  {
    /* Motion Estimation */
    c63_motion_estimate(cm, stream);

    /* Motion Compensation */
    // c63_motion_compensate(cm, stream);

    // Write previous frame while GPU does Motion Estimation and Motion Compensation for Y
    write_frame(cm);

    // Ensure motion vectors and predicted for Y is copied back to host before continuing with DCT
    cudaStreamSynchronize(stream[0]);
  }
  else if (cm->framenum != 0)
  {
    // Simply write previous frame for keyframes, as it can't be parallelized
    write_frame(cm);
  }

  /* DCT and Quantization */
  dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
      cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
      cm->quanttbl[Y_COMPONENT]);
  /* Reconstruct frame for inter-prediction */
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
      cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);

  // Ensure motion vectors and predicted for U is copied back to host 
  if (!cm->curframe->keyframe) cudaStreamSynchronize(stream[1]);

  dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
      cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
      cm->quanttbl[U_COMPONENT]);

  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
      cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);

  // Ensure motion vectors and predicted for V is copied back to host
  if (!cm->curframe->keyframe) cudaStreamSynchronize(stream[2]);

  dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
      cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
      cm->quanttbl[V_COMPONENT]);

  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
      cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  c63_common *cm = (c63_common*)calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
  destroy_frame(cm->curframe);
  free(cm);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  yuv_t *image;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      default:
        print_help();
        break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  /* INIT GPU - Gentlemen, please start you multiprocessors */
  gpu_init(cm, stream);

  while (1)
  {
    image = read_yuv(infile, cm);

    // Need to write last frame if end of file
    if (!image) { write_frame(cm); break; }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image);

    CUDA_CHECK(cudaFreeHost(image->Y));
    CUDA_CHECK(cudaFreeHost(image->U));
    CUDA_CHECK(cudaFreeHost(image->V));
    free(image);

    printf("Done!\n");

    ++numframes;

    // Need to write last frame if reached limit of frames
    if (limit_numframes && numframes >= limit_numframes) { write_frame(cm); break; }
  }

  /* Gentlemen, please clean up all memory used on the GPU */
  gpu_cleanup(stream);

  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  //int i, j;
  //for (i = 0; i < 2; ++i)
  //{
  //  printf("int freq[] = {");
  //  for (j = 0; j < ARRAY_SIZE(frequencies[i]); ++j)
  //  {
  //    printf("%d, ", frequencies[i][j]);
  //  }
  //  printf("};\n");
  //}

  return EXIT_SUCCESS;
}
