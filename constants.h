/*
    THIS FILE CONTAINS SEVERAL CONSTANTS INHERITED FROM VTM-12.0 OR CREATED SPECIFICALLY FOR THE AFFINE KERNEL
    THEY ARE USED TO IMPROVE THE CLARITY AND AVOID MAGIC NUMBERS IN THE CODE
*/

// #############################################################
// The following enum is used to avoid using magic numbers when
// referring to different CU sizes. They are used to index the
// RETURN_STRIDE_LIST array
// #############################################################

// TODO: Keep the order synchronized with RETURN_STRIDE_LIST

enum cuSizeIdx{
  _128x128 = 0,
  _128x64  = 1,
  _64x128  = 2, 
  _64x64   = 3,
  _64x32   = 4,
  _32x64   = 5,
  _32x32   = 6,
  _64x16   = 7,
  _16x64   = 8,
  _32x16   = 9,
  _16x32   = 10,
  _16x16   = 11,
};

// These lambdas are valid when using low delay with a single reference frame. Improve this when using multiple reference frames
const float lambdas[4] = 
{
  17.583905,  // QP 22
  39.474532,  // QP 27
  78.949063,  // QP 32
  140.671239 // QP 37
};

// #############################################################
// The following variables keep the XY position of all aligned CUs
// inside one CTU, in RASTER ORDER. Only the supported block sizes
// are described here
// #############################################################

const int X_POS_128x64[2] = {0, 0};
const int Y_POS_128x64[2] = {0, 64};

const int X_POS_64x128[2] = {0, 64};
const int Y_POS_64x128[2] = {0, 0};

const int X_POS_64x64[4] = {0, 64, 0, 64};
const int Y_POS_64x64[4] = {0, 0, 64, 64};

const int X_POS_64x32[8] = {0, 64, 0,  64, 0,  64, 0,  64};
const int Y_POS_64x32[8] = {0, 0,  32, 32, 64, 64, 96, 96};

const int X_POS_32x64[8] = {0, 32, 64, 96, 0,  32, 64, 96};
const int Y_POS_32x64[8] = {0, 0,  0,  0,  64, 64, 64, 64};

const int X_POS_32x32[16] = {0, 32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96};
const int Y_POS_32x32[16] = {0, 0,  0,  0,  32, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 96};

const int X_POS_64x16[16] = {0, 64, 0,  64, 0,  64, 0,  64, 0,  64, 0,  64,  0, 64, 0,   64};
const int Y_POS_64x16[16] = {0, 0,  16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 112, 112}; 

const int X_POS_16x64[16] = {0, 16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112};
const int Y_POS_16x64[16] = {0, 0,  0,  0,  0,  0,  0,  0,   64, 64, 64, 64, 64, 64, 64, 64};

const int X_POS_32x16[32] = {0, 32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,   32,  64,  96};
const int Y_POS_32x16[32] = {0, 0,  0,  0,  16, 16, 16 ,16, 32, 32, 32, 32, 48, 48, 48, 48, 64, 64, 64, 64, 80, 80, 80, 80, 96, 96, 96, 96, 112, 112, 112, 112};

const int X_POS_16x32[32] = {0, 16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112}; 
const int Y_POS_16x32[32] = {0, 0,  0,  0,  0,  0,  0,  0,   32, 32, 32, 32, 32, 32, 32, 32,  64, 64, 64, 64, 64, 64, 64, 64,  96, 96, 96, 96, 96, 96, 96, 96};

const int X_POS_16x16[64] = {0, 16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,   16,  32,  48,  64,  80,  96,  112};
const int Y_POS_16x16[64] = {0, 0,  0,  0,  0,  0,  0,  0,   16, 16, 16, 16, 16, 16, 16, 16,  32, 32, 32, 32, 32, 32, 32, 32,  48, 48, 48, 48, 48, 48, 48, 48,  64, 64, 64, 64, 64, 64, 64, 64,  80, 80, 80, 80, 80, 80, 80, 80,  96, 96, 96, 96, 96, 96, 96, 96,  112, 112, 112, 112, 112, 112, 112, 112};



// #############################################################
// TODO: Keep the next variables synchronized with constants.cl
// #############################################################

# define ITEMS_PER_WG 256

#define CTU_WIDTH 128
#define CTU_HEIGHT 128
#define MAX_CUS_PER_CTU 64 // This occurs for CUs 16x16

int NUM_CU_SIZES = 12; // Number of CU sizes being supported. The kernel supports the N first sizes in WIDTH_LIST and HEIGHT_LIST
int const WIDTH_LIST[12] = 
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
int const HEIGHT_LIST[12] = 
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
int TOTAL_ALIGNED_CUS_PER_CTU = 201; // 1* 128x128 + 2* 128x64 + 2* 64x128 + 4* 64x64 + ...

// This list is used to help indexing the result (CPMVs, distortion) into the global array at the end of computation
// TODO: It is designed to deal with "aligned blocks" only, i.e., blocks positioned into (x,y) positions that are multiple of its dimensions
int const RETURN_STRIDE_LIST[12] = 
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
