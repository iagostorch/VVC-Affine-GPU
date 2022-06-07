/*
    THIS FILE CONTAINS SEVERAL CONSTANTS INHERITED FROM VTM-12.0 OR CREATED SPECIFICALLY FOR THE AFFINE KERNEL
    THEY ARE USED TO IMPROVE THE CLARITY AND AVOID MAGIC NUMBERS IN THE CODE
*/

// CONSTANT USED TO DEBUG INSTANCES OF THE KERNEL
#define target_gid 1023

// CONSTANTS INHERITED FROM VTM-12
#define MAX_CU_DEPTH 7
#define MV_FRACTIONAL_BITS_INTERNAL 4
#define MAX_CU_WIDTH 128
#define MAX_CU_HEIGHT 128
#define IF_FILTER_PREC 6
#define IF_INTERNAL_PREC 14 ///< Number of bits for internal precision
__constant int IF_INTERNAL_OFFS = 1 <<(IF_INTERNAL_PREC-1); // (1<<(IF_INTERNAL_PREC-1)) ///< Offset used internally
#define CLP_RNG_MAX 1023
#define CLP_RNG_MIN 0
#define CLP_RNG_BD 10
#define NTAPS_LUMA 8 // Number of taps for luma filter
__constant int MV_PRECISION_INTERNAL = 2 + MV_FRACTIONAL_BITS_INTERNAL;
__constant int MAX_CU_SIZE = 1<<MAX_CU_DEPTH;
// The following are used for AMVR, for cu.imv = 0, 1 and 2
#define AFFINE_MV_PRECISION_QUARTER 4
#define AFFINE_MV_PRECISION_SIXTEENTH 1
#define AFFINE_MV_PRECISION_INT 2

// CONSTANTS CREATED TO AFFINE KERNEL
#define SUBBLOCK_SIZE 4
#define PROF_PADDING 1

// Constants used to clip motion vectors in Clip3()
#define MV_BITS 18
__constant int MV_MAX =  (1 << (MV_BITS - 1)) - 1;
__constant int MV_MIN =  -(1 << (MV_BITS - 1));

// COEFFICIENTS FOR FILTERING OPERATIONS IN AFFINE MC
__constant int const m_lumaFilter4x4[16][8] =
{
  {  0, 0,   0, 64,  0,   0,  0,  0 },
  {  0, 1,  -3, 63,  4,  -2,  1,  0 },
  {  0, 1,  -5, 62,  8,  -3,  1,  0 },
  {  0, 2,  -8, 60, 13,  -4,  1,  0 },
  {  0, 3, -10, 58, 17,  -5,  1,  0 }, //1/4
  {  0, 3, -11, 52, 26,  -8,  2,  0 },
  {  0, 2,  -9, 47, 31, -10,  3,  0 },
  {  0, 3, -11, 45, 34, -10,  3,  0 },
  {  0, 3, -11, 40, 40, -11,  3,  0 }, //1/2
  {  0, 3, -10, 34, 45, -11,  3,  0 },
  {  0, 3, -10, 31, 47,  -9,  2,  0 },
  {  0, 2,  -8, 26, 52, -11,  3,  0 },
  {  0, 1,  -5, 17, 58, -10,  3,  0 }, //3/4
  {  0, 1,  -4, 13, 60,  -8,  2,  0 },
  {  0, 1,  -3,  8, 62,  -5,  1,  0 },
  {  0, 1,  -2,  4, 63,  -3,  1,  0 }
};

__constant int MAX_INT = 1<<30;
__constant long MAX_LONG = 1<<62;

// #############################################################
// TODO: Keep the next variables synchronized with constants.h
// #############################################################

#define ITEMS_PER_WG 256
#define CTU_WIDTH 128
#define CTU_HEIGHT 128
#define MAX_CUS_PER_CTU 64 // TODO: This is valid only for aligned CUs, and occurs for CUs 16x16

__constant int NUM_CU_SIZES = 12; // Number of CU sizes being supported. The kernel supports the N first sizes in WIDTH_LIST and HEIGHT_LIST
__constant int const WIDTH_LIST[12] = 
{
  128, //128x128
  128, //128x64
  64,  //64x128
  
  64,  //64x64
  64,  //64x32
  32,  //32x64

  32, //32x32

  64,  //64x16
  16,  //16x64

  32,  //32x16
  16,  //16x32

  16   //16x16
};
__constant int const HEIGHT_LIST[12] = 
{
  128, //128x128
  64,  //128x64
  128, //64x128
  
  64,  //64x64
  32,  //64x32
  64,  //32x64

  32, //32x32

  16,  //64x16
  64,  //16x64

  16,  //32x16
  32,  //16x32

  16   //16x16
};

// Number of aligned CUs inside one CTU, considering all affine block sizes
// TODO: If we decide to support more or fewer block sizes this must be updated
// 1* 128x128 + 2* 128x64 + 2* 64x128 + 4* 64x64 + ...
__constant int TOTAL_ALIGNED_CUS_PER_CTU = 201; 

// This list is used to help indexing the result (CPMVs, distortion) into the global array at the end of computation
// TODO: It is designed to deal with "aligned blocks" only, i.e., blocks positioned into (x,y) positions that are multiple of its dimensions
__constant int RETURN_STRIDE_LIST[12] = 
{
  0,    //128x128 -> first position
  1,    //128x64  -> first position after the ONE 128x128 CU
  3,    //64x128  -> first position after the TWO 128x64 CUs
  5,    //64x64   -> first position after the TWO 64x128 CUs
  9,    //64x32   -> first position after the FOUR 64x64 CUs
  17,   //32x64   -> each line is the previous index, plus the number of CUs in the previous index
  25,   //32x32

  41,   //64x16
  57,   //16x64
  73,   //32x16
  105,  //16x32
  137,  //16x16
};


// TODO: This macro controls some memory access of the kernel. When equal to 1 some memory access are done using vload/vstore. When 0, the access is done indexing the values one by one
// [!] DO NOT CHANGE IT TO ZERO: The indexing inside affine.cl was not corrected to support non vectorized memory access
#define VECTORIZED_MEMORY 1