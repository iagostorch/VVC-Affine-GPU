/*
    THIS FILE CONTAINS SEVERAL CONSTANTS INHERITED FROM VTM-12.0 OR CREATED SPECIFICALLY FOR THE AFFINE KERNEL
    THEY ARE USED TO IMPROVE THE CLARITY AND AVOID MAGIC NUMBERS IN THE CODE
*/

#define ALLOW_EARLY_TERM_GRADIENT 1

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
#define MAX_HA_CUS_PER_CTU 32 // TODO: This is valid only for HALF-ALIGNED CUs considering a SINGLE GROUP, and occurs for CUs 16x16

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
__constant int TOTAL_HALF_ALIGNED_CUS_PER_CTU = 224; 

__constant int HA_NUM_CU_SIZES = 18; // Number of HALF-ALIGNED CU sizes being supported. 

// This list is used to help indexing the result (CPMVs, distortion) into the global array at the end of computation
// This list is designed to deal with "aligned blocks" only, i.e., blocks positioned into (x,y) positions that are multiple of its dimensions. Half-aligned return strides are described further
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


// #############################################################
// The following variables keep the XY position of all HALF-ALIGNED CUs
// inside one CTU, in RASTER ORDER. Only the supported block sizes
// are described here
// #############################################################

__constant unsigned char HA_X_POS_64x32[4] = {0, 64, 0,  64}; // QT-TH
__constant unsigned char HA_Y_POS_64x32[4] = {16, 16,  80, 80};

__constant unsigned char HA_X_POS_32x64[4] = {16, 80, 16, 80}; // QT-TV
__constant unsigned char HA_Y_POS_32x64[4] = {0, 0,  64,  64};

__constant unsigned char HA_X_POS_64x16_G1[8] = {0, 64, 0,  64, 0,  64, 0,   64}; // QT-BH-TH
__constant unsigned char HA_Y_POS_64x16_G1[8] = {8, 8,  40, 40, 72, 72, 104, 104}; 

__constant unsigned char HA_X_POS_64x16_G2[4] = {0,  64, 0,  64}; // QT-TH-TH
__constant unsigned char HA_Y_POS_64x16_G2[4] = {24, 24, 88, 88}; 

__constant unsigned char HA_X_POS_16x64_G1[8] = {8, 40, 72, 104, 8,  40, 72, 104}; //QT-BV-TV
__constant unsigned char HA_Y_POS_16x64_G1[8] = {0, 0,  0,  0,   64, 64, 64, 64}; 

__constant unsigned char HA_X_POS_16x64_G2[4] = {24, 88, 24, 88}; // QT-TV-TV
__constant unsigned char HA_Y_POS_16x64_G2[4] = {0, 0, 64, 64};

__constant unsigned char HA_X_POS_32x32_G1[8] = {16, 80, 16, 80, 16, 80, 16, 80}; // QT-TV-BH
__constant unsigned char HA_Y_POS_32x32_G1[8] = {0,  0,  32, 32, 64, 64, 96, 96};

__constant unsigned char HA_X_POS_32x32_G2[8] = {0,  32, 64, 96, 0,  32, 64, 96}; // QT-TH-BV
__constant unsigned char HA_Y_POS_32x32_G2[8] = {16, 16, 16, 16, 80, 80, 80, 80};

__constant unsigned char HA_X_POS_32x16_G1[16] = {0, 32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,   32,  64,  96}; // QT-QT-TH
__constant unsigned char HA_Y_POS_32x16_G1[16] = {8, 8,  8,  8,  40, 40, 40, 40, 72, 72, 72, 72, 104, 104, 104, 104};

__constant unsigned char HA_X_POS_32x16_G2[8] = {0,  32, 64, 96, 0,  32, 64, 96}; // QT-BV-TH-TH
__constant unsigned char HA_Y_POS_32x16_G2[8] = {24, 24, 24, 24, 88, 88, 88, 88};

__constant unsigned char HA_X_POS_32x16_G3[16] = {16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16,  80}; // QT-TV-BH-BH
__constant unsigned char HA_Y_POS_32x16_G3[16] = {0,  0,  16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 112, 112};

__constant unsigned char HA_X_POS_16x32_G1[16] = {8, 40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104}; // QT-QT-TV
__constant unsigned char HA_Y_POS_16x32_G1[16] = {0, 0,  0,  0,   32, 32, 32, 32,  64, 64, 64, 64,  96, 96, 96, 96};

__constant unsigned char HA_X_POS_16x32_G2[8] = {24, 88, 24, 88, 24, 88, 24, 88}; // QT-BH-TV-TV
__constant unsigned char HA_Y_POS_16x32_G2[8] = {0,  0,  32, 32, 64, 64, 96, 96};

__constant unsigned char HA_X_POS_16x32_G3[16] = {0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112}; // QT-TH-BV-BV
__constant unsigned char HA_Y_POS_16x32_G3[16] = {16, 16, 16, 16, 16, 16, 16, 16,  80, 80, 80, 80, 80, 80, 80, 80};

__constant unsigned char HA_X_POS_16x16_G1[32] = {0, 16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,   16,  32,  48,  64 , 80,  96,  112}; // QT-QT-BV-TH
__constant unsigned char HA_Y_POS_16x16_G1[32] = {8, 8,  8,  8,  8,  8,  8,  8,   40, 40, 40, 40, 40, 40, 40, 40,  72, 72, 72, 72, 72, 72, 72, 72,  104, 104, 104, 104, 104, 104, 104, 104, };

__constant unsigned char HA_X_POS_16x16_G2[32] = {8, 40, 72, 104, 8, 40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,   40,  72,  104}; // QT-QT-BH-TV
__constant unsigned char HA_Y_POS_16x16_G2[32] = {0, 0,  0,  0,  16, 16, 16, 16,  32, 32, 32, 32,  48, 48, 48, 48,  64, 64, 64, 64,  80, 80, 80, 80,  96, 96, 96, 96,  112, 112, 112, 112};
                                                // First the coordinates from "original G3" then "original G4"
// __constant unsigned char HA_X_POS_16x16_G34[16] = {0, 48, 64, 112, 0, 48, 64, 112, 24, 88, 24, 88, 24, 88, 24, 88};
// __constant unsigned char HA_Y_POS_16x16_G34[16] = {24, 24, 24, 24, 88, 88, 88, 88, 0, 0, 48, 48, 64, 64, 112, 112};

// __constant unsigned char HA_X_POS_16x16_G3[8] = {0, 48, 64, 112, 0, 48, 64, 112}; // QT-TH-TH-TV
// __constant unsigned char HA_Y_POS_16x16_G3[8] = {24, 24, 24, 24, 88, 88, 88, 88};

// __constant unsigned char HA_X_POS_16x16_G4[8] = {24, 88, 24, 88, 24, 88, 24, 88}; // QT-TV-TV-TH
// __constant unsigned char HA_Y_POS_16x16_G4[8] = {0, 0, 48, 48, 64, 64, 112, 112};

__constant unsigned char HA_ALL_X_POS[18][32] = 
{
  /* 64x32 */    {0, 64, 0,  64}, // QT-TH
  /* 32x64 */    {16, 80, 16, 80}, // QT-TV
  /* 64x16 G1 */ {0, 64, 0,  64, 0,  64, 0,   64}, // QT-BH-TH
  /* 64x16 G2 */ {0,  64, 0,  64}, // QT-TH-TH
  /* 16x64 G1 */ {8, 40, 72, 104, 8,  40, 72, 104}, //QT-BV-TV
  /* 16x64 G2 */ {24, 88, 24, 88}, // QT-TV-TV
  /* 32x32 G1 */ {16, 80, 16, 80, 16, 80, 16, 80}, // QT-TV-BH
  /* 32x32 G2 */ {0,  32, 64, 96, 0,  32, 64, 96}, // QT-TH-BV
  /* 32x16 G1 */ {0, 32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,   32,  64,  96}, // QT-QT-TH
  /* 32x16 G2 */ {0,  32, 64, 96, 0,  32, 64, 96}, // QT-BV-TH-TH
  /* 32x16 G3 */ {16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16,  80}, // QT-TV-BH-BH
  /* 16x32 G1 */ {8, 40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104}, // QT-QT-TV
  /* 16x32 G2 */ {24, 88, 24, 88, 24, 88, 24, 88}, // QT-BH-TV-TV
  /* 16x32 G3 */ {0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112}, // QT-TH-BV-BV
  /* 16x16 G1 */ {0, 16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0, 16, 32, 48, 64, 80, 96, 112, 0, 16, 32, 48,  64, 80, 96, 112}, // QT-QT-BV-TH
  /* 16x16 G2 */ {8, 40, 72, 104, 8, 40, 72, 104, 8,  40, 72, 104, 8, 40, 72, 104, 8, 40, 72, 104, 8, 40, 72, 104, 8, 40, 72, 104, 8,  40, 72, 104}, // QT-QT-BH-TV
  /* 16x16 G3 */ {0, 16, 32, 48, 64, 80, 96, 112, 0, 16, 32, 48, 64, 80, 96, 112}, // QT-TH-TH-TV
  /* 16x16 G4 */ {24, 88, 24, 88, 24, 88, 24, 88, 24, 88, 24, 88, 24, 88, 24, 88} // QT-TV-TV-TH
};

 __constant unsigned char HA_ALL_Y_POS[18][32] = {
  /* 64x32 */    {16, 16,  80, 80},
  /* 32x64 */    {0, 0,  64,  64},
  /* 64x16 G1 */ {8, 8,  40, 40, 72, 72, 104, 104}, 
  /* 64x16 G2 */ {24, 24, 88, 88}, 
  /* 16x64 G1 */ {0, 0,  0,  0,   64, 64, 64, 64}, 
  /* 16x64 G2 */ {0, 0, 64, 64},
  /* 32x32 G1 */ {0,  0,  32, 32, 64, 64, 96, 96},
  /* 32x32 G2 */ {16, 16, 16, 16, 80, 80, 80, 80},
  /* 32x16 G1 */ {8, 8,  8,  8,  40, 40, 40, 40, 72, 72, 72, 72, 104, 104, 104, 104},
  /* 32x16 G2 */ {24, 24, 24, 24, 88, 88, 88, 88},
  /* 32x16 G3 */ {0,  0,  16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 112, 112},
  /* 16x32 G1 */ {0, 0,  0,  0,   32, 32, 32, 32,  64, 64, 64, 64,  96, 96, 96, 96},
  /* 16x32 G2 */ {0,  0,  32, 32, 64, 64, 96, 96},
  /* 16x32 G3 */ {16, 16, 16, 16, 16, 16, 16, 16,  80, 80, 80, 80, 80, 80, 80, 80},
  /* 16x16 G1 */ {8, 8,  8,  8,  8,  8,  8,  8,   40, 40, 40, 40, 40, 40, 40, 40,  72, 72, 72, 72,  72, 72, 72, 72, 104, 104, 104, 104, 104, 104, 104, 104},
  /* 16x16 G2 */ {0, 0,  0,  0,  16, 16, 16, 16,  32, 32, 32, 32,  48, 48, 48, 48,  64, 64, 64, 64, 80, 80, 80, 80, 96,  96,  96,  96,  112, 112, 112, 112},
  /* 16x16 G3 */ {24, 24, 24, 24, 24, 24, 24, 24, 88, 88, 88, 88, 88, 88, 88, 88},
  /* 16x16 G4 */ {0, 0, 16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 112, 112}
};

// Some CU sizes are duplicated because we can generate half-aligned blocks with different sequences of splits
// These different sequences are separated by different groups (G1, G2, G3, and G4) to maintain the number of CUs per CTU a power of 2
__constant unsigned char HA_WIDTH_LIST[18] = 
{
  64,  //64x32 (QT-TH)
  32,  //32x64 (QT-TV)

  64,  //64x16 G1 (QT-BH-TH)
  64,  //64x16 G2 (QT-TH-TH)

  16,  //16x64 G1 (QT-BV-TV)
  16,  //16x64 G2 (QT-TV-TV)

  32, //32x32 G1 (QT-TV-BH)
  32, //32x32 G2 (QT-TH-BV)

  32,  //32x16 G1 (QT-QT-TH)
  32,  //32x16 G2 (QT-BV-TH-TH)
  32,  //32x16 G3 (QT-TV-BH-BH)

  16,  //16x32 G1 (QT-QT-TV)
  16,  //16x32 G2 (QT-BH-TV-TV)
  16,  //16x32 G3 (QT-TH-BV-BV)

  16,  //16x16 G1 (QT-QT-BV-TH)
  16,  //16x16 G2 (QT-QT-BH-TV)
  16,  //16x16 G3 (QT-TH-TH-TV)
  16   //16x16 G4 (QT-TV-TV-TH)
};
// Some CU sizes are duplicated because we can generate half-aligned blocks with different sequences of splits
// These different sequences are separated by different groups (G1, G2, G3, and G4) to maintain the number of CUs per CTU a power of 2
__constant unsigned char HA_HEIGHT_LIST[18] = 
{
  32,  //64x32 (QT-TH)
  64,  //32x64 (QT-TV)

  16,  //64x16 G1 (QT-BH-TH)
  16,  //64x16 G2 (QT-TH-TH)

  64,  //16x64 (QT-BV-TV)
  64,  //16x64 (QT-TV-TV)
  
  32, //32x32 G1 (QT-TV-BH)
  32, //32x32 G2 (QT-TH-BV)

  16,  //32x16 G1 (QT-QT-TH)
  16,  //32x16 G2 (QT-BV-TH-TH)
  16,  //32x16 G3 (QT-TV-BH-BH)

  32,  //16x32 (QT-QT-TV)
  32,  //16x32 (QT-BH-TV-TV)
  32,  //16x32 (QT-TH-BV-BV)

  16,  //16x16 G1 (QT-QT-BV-TH)
  16,  //16x16 G2 (QT-QT-BH-TV)
  16,  //16x16 G3 (QT-TH-TH-TV)
  16   //16x16 G4 (QT-TV-TV-TH)
};

// The number of HALF-ALIGNED CUs inside each CTU, considering different groups of CUs (i.e., different sequences of splits)
__constant unsigned char HA_CUS_PER_CTU[18] = {
  4,  //64x32 (QT-TH)
  4,  //32x64 (QT-TV)

  8,  //64x16 G1 (QT-BH-TH)
  4,  //64x16 G2 (QT-TH-TH)

  8,  //16x64 (QT-BV-TV)
  4,  //16x64 (QT-TV-TV)
  
  8, //32x32 G1 (QT-TV-BH)
  8, //32x32 G2 (QT-TH-BV)

  16, //32x16 G1 (QT-QT-TH)
  8,  //32x16 G2 (QT-BV-TH-TH)
  16, //32x16 G3 (QT-TV-BH-BH)

  16, //16x32 (QT-QT-TV)
  8,  //16x32 (QT-BH-TV-TV)
  16, //16x32 (QT-TH-BV-BV)

  32,  //16x16 G1 (QT-QT-BV-TH)
  32,  //16x16 G2 (QT-QT-BH-TV)
  16,  //16x16 G3 (QT-TH-TH-TV)
  16  //16x16 G4 (QT-TV-TV-TH)
};

// This list is used to help indexing the result (CPMVs, distortion) into the global array at the end of computation
// TODO: It is designed to deal with "aligned blocks" only, i.e., blocks positioned into (x,y) positions that are multiple of its dimensions
__constant int HA_RETURN_STRIDE_LIST[18] = 
{
  0,    // 64x32 -> first position
  4,    // 32x64 -> first position after the FOUR HA 64x32 CUs
  8,    // 64x16 G1 -> first position after the FOUR HA 32x64 CUs
  16,   // 64x16 G2  -> first position after the EIGHT HA 64x16 G1 CUs
  20,   // 16x64 G1  -> first position after the FOUR HA 64x16 G2 CUs
  28,   // 16x64 G2  -> each line is the previous index plus the number of CUs in the previous index
  32,   // 32x32 G1
  40,   // 32x32 G2
  48,   // 32x16 G1
  64,   // 32x16 G2
  72,   // 32x16 G3
  88,   // 16x32 G1
  104,  // 16x32 G2
  112,  // 16x32 G3
  128,  // 16x16 G1
  160,  // 16x16 G2
  192,  // 16x16 G3
  208   // 16x16 G4
};

// TODO: This macro controls some memory access of the kernel. When equal to 1 some memory access are done using vload/vstore. When 0, the access is done indexing the values one by one
// [!] DO NOT CHANGE IT TO ZERO: The indexing inside affine.cl was not corrected to support non vectorized memory access
#define VECTORIZED_MEMORY 1

#define LOW_DELAY_P 1