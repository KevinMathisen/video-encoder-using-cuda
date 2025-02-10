#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"

// Declarations
struct frame* create_frame(struct c63_common *cm, yuv_t *image);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif  /* C63_COMMON_H_ */
