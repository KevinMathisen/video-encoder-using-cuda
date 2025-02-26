#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"

// Declaration
void gpu_init(struct c63_common *cm, cudaStream_t stream[3]);

void gpu_cleanup(cudaStream_t stream[3]);

void c63_motion_estimate(struct c63_common *cm, cudaStream_t stream[3]);

void c63_motion_compensate(struct c63_common *cm, cudaStream_t stream[3]);

#endif  /* C63_ME_H_ */
