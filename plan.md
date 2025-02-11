# Plan/Pseudocode

## One thread for each sad_block call

### Preparing GPU
Allocate memory for the entire orig frame and the reference frame (for all of the component). We also need to allocate space for the marcoblock offsets for their best candidates (stored under cm->frame->mbs on the host). We also need to send the cm->me_search_range (to check if we are within bound in kernel), cm->padw, and cm->padh to the kernels. 

### Kernel

Launch grid equal to whats required from dimentions of frame.
Each grid should have threads based on the given search range, where each threads calculates the SAD for one candidate match from the reference frame.  If the search range is 16, which is the max and standard, there should be 1024 threads (256 for u and v component as search range is halved for these).

#### Kernel Logic
(Separate kernel for each y/u/v component, so hardcode their component)

Kernel should identify its block (and y/u/v component), and load the corresponding pixels from the current frame into shared memory. 
(for now keep reference frame in global memory)

Then identify which reference to compare with given the offset which is based on the thread x and y values. Here we also need to check we are within bounds using cm->padw and cm->padh.

For the given reference and current, calculate its SAD score. 

Then do a simple reduce of all the threads in the block to find the smallest SAD score and its offset. This offset should be written to global memory, where the location is given based on the block.

### Reading back to host
The offsets should then be read back to the host and be used for motion compensation.

