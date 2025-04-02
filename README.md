
# VVC-Affine-GPU

A GPU based engine for the Affine Motion Estimation (Affine ME) of the Versatile Video Coding (VVC) standard. It is a GPU-fiendly version of the Affine ME bundled into the VTM encoder, implemented in OpenCL to seize the performance of GPUs from most manufacturers.


## Building the project

This project is based on C++ and OpenCL, and it has dependencies with libboost. A convenient makefile is supplied for building the project locally.



```bash
  cd VVC-Affine-GPU
  make
```
    
## Usage/Examples

The directory ```data/``` contains examples of input files for 1080p videos (1920x1080 samples). The execution requires one .csv file containing the original frame samples (i.e., the frame that we want to encode) and another .csv file with the reference frame samples. Both files must contain the same number of samples with the same number of columns and rows. To encode multiple frames in a single run, all original frames must be concatenated vertically on the same input file. The same goes for the reference frames.

A typical run with the sample data takes the following form:

```bash
./main -f 2 -s 1920x1080 -q 32 -o data/original_frames_1_2.csv -r data/reconstructed_frames_0_1.csv -l AffineME_decisions_log 

```

where 

| Parameter | Description|
| ------ | ------ |
| f | Number of input frames to be encoded|
| s | Resolution of the frames, in the form WidthxHeight|
| q | Quantization parameter (QP) used for the rate-distortion optimization|
| o | Input file for original frame samples |
| r | Input file for reference frame samples|
| l | Log of the Affine ME decisions. If left empty, no logs are created|
