# Video Encoder using CUDA

**NB: When running cmake on server with GPU with compute capability 5.0”, include the flag** `-DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50"`

The repository contains native CUDA code that is meant for compilation with NVCC. 
The program can be used to encode a `.yuv` file using motion estimation, motion compensation, DCT, and quantization. 
Currently only motion estimation and motion compensation is offloaded to the GPU.

### Build
To build (ignore the flag for cmake when compiling for GPU with compute version != 5.0):
```
mkdir build
cd build
cmake -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50" ..
make
```

### Run
To encode a video:
```
./c63enc -w 352 -h 288 -f 10 -o foremanout.c63 foreman.yuv
```

To decode the c63 file (should be done from [this resitory](https://github.com/griwodz/in5050-codec63) without modifications):
```
./c63dec foremanout.c63 output.yuv
```

To get the prediction buffer (which can also be played using mplayer):
```
./c63pred foremanout.c63 predictionbuffer.yuv
```

Use mplayer or ffplay to playback raw YUV file:
```
mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
```

