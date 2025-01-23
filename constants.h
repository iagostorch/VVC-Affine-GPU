/*
    THIS FILE CONTAINS SEVERAL CONSTANTS INHERITED FROM VTM-12.0 OR CREATED SPECIFICALLY FOR THE AFFINE KERNEL
    THEY ARE USED TO IMPROVE THE CLARITY AND AVOID MAGIC NUMBERS IN THE CODE
*/

// #############################################################
// The following enum is used to avoid using magic numbers when
// referring to different CU sizes. They are used to index the
// RETURN_STRIDE_LIST array
// #############################################################

enum PRED_TYPES{
  FULL_2CP = 0,
  FULL_3CP = 1,
  HALF_2CP = 2,
  HALF_3CP = 3,
  N_PREDS  = 4
};

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

enum HA_cuSizeIdx{
  HA_64x32    = 0,
  HA_32x64    = 1,
  HA_64x16_G1 = 2,
  HA_64x16_G2 = 3,
  HA_16x64_G1 = 4,
  HA_16x64_G2 = 5,
  HA_32x32_G1 = 6,
  HA_32x32_G2 = 7,
  HA_32x16_G1 = 8,
  HA_32x16_G2 = 9,
  HA_32x16_G3 = 10,
  HA_16x32_G1 = 11,
  HA_16x32_G2 = 12,
  HA_16x32_G3 = 13,
  HA_16x16_G1 = 14,
  HA_16x16_G2 = 15,
  HA_16x16_G3 = 16,
  HA_16x16_G4 = 17,

  HA_32x32_U1 = 18,
  HA_32x16_U1 = 29,
  HA_32x16_U2 = 20,
  HA_16x32_U1 = 21,
  HA_16x32_U2 = 22,
  HA_16x16_U1 = 23,
  HA_16x16_U2 = 24,
  HA_16x16_U3 = 25
};

#define MAX_REFS 4

// These lambdas are valid when using low delay with a single reference frame. Improve this when using multiple reference frames
const float lambdas[4] = 
{
  17.583905,  // QP 22
  39.474532,  // QP 27
  78.949063,  // QP 32
  140.671239 // QP 37
};

// Used to determine the QP variation based on the position inside the GOP when gopSize=8
const int qpOffset[8] = {1, 7, 6, 7, 6, 7, 6, 7};

// These lambdas are adapted for multiple reference frames. They are selected based on the actual QP shown during the encoding and not the input parameter --QP
const float fullLambdas[60] = 
{
  //                    0           1            2           3             4            5            6            7            8            9
  /*  0 - 9 */    000.000000,  000.000000,  000.000000,  000.000000,  000.000000,  000.000000,  000.000000,  000.000000,  000.000000,  000.000000, 
  /* 10 - 19 */   000.000000,    2.769291,    3.108425,    3.489089,    3.916370,    4.395976,    4.934316,    5.538583,    6.216849,    6.978177, 
  /* 20 - 29 */     7.832739,    8.791952,    9.868633,    11.077166,  12.433698,   13.956355,   15.665478,   17.583905,   19.737266,   22.154332,
  /* 30 - 39 */    24.867397,   27.912709,   31.330957,   35.167810,   39.474532,   44.308664,   49.734793,   55.825418,   62.661913,   70.335619,
  /* 40 - 49 */    78.949063,   88.617327,   99.469587,   111.650836, 125.323826,  140.671239,  157.898127,  177.234655,  198.939174,  223.301672,
  /* 50 - 59 */   250.647653,  281.342477,  315.796254,  354.469310,  397.878347,  446.603345,  501.295305,  562.684955,  631.592507,  708.938619
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
int TOTAL_HALF_ALIGNED_CUS_PER_CTU = 284;

const int HA_NUM_CU_SIZES = 26;

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

// #############################################################
// The following variables keep the XY position of all HALF-ALIGNED CUs
// inside one CTU, in RASTER ORDER. Only the supported block sizes
// are described here
// #############################################################

const int HA_X_POS_64x32[4] = {0, 64, 0,  64}; // QT-TH
const int HA_Y_POS_64x32[4] = {16, 16,  80, 80};

const int HA_X_POS_32x64[4] = {16, 80, 16, 80}; // QT-TV
const int HA_Y_POS_32x64[4] = {0, 0,  64,  64};

const int HA_X_POS_64x16_G1[8] = {0, 64, 0,  64, 0,  64, 0,   64}; // QT-BH-TH
const int HA_Y_POS_64x16_G1[8] = {8, 8,  40, 40, 72, 72, 104, 104}; 

const int HA_X_POS_64x16_G2[4] = {0,  64, 0,  64}; // QT-TH-TH
const int HA_Y_POS_64x16_G2[4] = {24, 24, 88, 88}; 

const int HA_X_POS_16x64_G1[8] = {8, 40, 72, 104, 8,  40, 72, 104}; //QT-BV-TV
const int HA_Y_POS_16x64_G1[8] = {0, 0,  0,  0,   64, 64, 64, 64}; 

const int HA_X_POS_16x64_G2[4] = {24, 88, 24, 88}; // QT-TV-TV
const int HA_Y_POS_16x64_G2[4] = {0, 0, 64, 64};

const int HA_X_POS_32x32_G1[8] = {16, 80, 16, 80, 16, 80, 16, 80}; // QT-TV-BH
const int HA_Y_POS_32x32_G1[8] = {0,  0,  32, 32, 64, 64, 96, 96};

const int HA_X_POS_32x32_G2[8] = {0,  32, 64, 96, 0,  32, 64, 96}; // QT-TH-BV
const int HA_Y_POS_32x32_G2[8] = {16, 16, 16, 16, 80, 80, 80, 80};

const int HA_X_POS_32x16_G1[16] = {0, 32, 64, 96, 0,  32, 64, 96, 0,  32, 64, 96, 0,   32,  64,  96}; // QT-QT-TH
const int HA_Y_POS_32x16_G1[16] = {8, 8,  8,  8,  40, 40, 40, 40, 72, 72, 72, 72, 104, 104, 104, 104};

const int HA_X_POS_32x16_G2[8] = {0,  32, 64, 96, 0,  32, 64, 96}; // QT-BV-TH-TH
const int HA_Y_POS_32x16_G2[8] = {24, 24, 24, 24, 88, 88, 88, 88};

const int HA_X_POS_32x16_G3[16] = {16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16, 80, 16,  80}; // QT-TV-BH-BH
const int HA_Y_POS_32x16_G3[16] = {0,  0,  16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 112, 112};

const int HA_X_POS_16x32_G1[16] = {8, 40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104}; // QT-QT-TV
const int HA_Y_POS_16x32_G1[16] = {0, 0,  0,  0,   32, 32, 32, 32,  64, 64, 64, 64,  96, 96, 96, 96};

const int HA_X_POS_16x32_G2[8] = {24, 88, 24, 88, 24, 88, 24, 88}; // QT-BH-TV-TV
const int HA_Y_POS_16x32_G2[8] = {0,  0,  32, 32, 64, 64, 96, 96};

const int HA_X_POS_16x32_G3[16] = {0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112}; // QT-TH-BV-BV
const int HA_Y_POS_16x32_G3[16] = {16, 16, 16, 16, 16, 16, 16, 16,  80, 80, 80, 80, 80, 80, 80, 80};

const int HA_X_POS_16x16_G1[32] = {0, 16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,  16, 32, 48, 64, 80, 96, 112, 0,   16,  32,  48,  64 , 80,  96,  112}; // QT-QT-BV-TH
const int HA_Y_POS_16x16_G1[32] = {8, 8,  8,  8,  8,  8,  8,  8,   40, 40, 40, 40, 40, 40, 40, 40,  72, 72, 72, 72, 72, 72, 72, 72,  104, 104, 104, 104, 104, 104, 104, 104, };

const int HA_X_POS_16x16_G2[32] = {8, 40, 72, 104, 8, 40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,  40, 72, 104, 8,   40,  72,  104}; // QT-QT-BH-TV
const int HA_Y_POS_16x16_G2[32] = {0, 0,  0,  0,  16, 16, 16, 16,  32, 32, 32, 32,  48, 48, 48, 48,  64, 64, 64, 64,  80, 80, 80, 80,  96, 96, 96, 96,  112, 112, 112, 112};
                                                // First the coordinates from "original G3" then "original G4"
const int HA_X_POS_16x16_G34[16] = {0, 48, 64, 112, 0, 48, 64, 112, 24, 88, 24, 88, 24, 88, 24, 88};
const int HA_Y_POS_16x16_G34[16] = {24, 24, 24, 24, 88, 88, 88, 88, 0, 0, 48, 48, 64, 64, 112, 112};

// const int HA_X_POS_16x16_G3[8] = {0, 48, 64, 112, 0, 48, 64, 112}; // QT-TH-TH-TV
// const int HA_Y_POS_16x16_G3[8] = {24, 24, 24, 24, 88, 88, 88, 88};

// const int HA_X_POS_16x16_G4[8] = {24, 88, 24, 88, 24, 88, 24, 88}; // QT-TV-TV-TH
// const int HA_Y_POS_16x16_G4[8] = {0, 0, 48, 48, 64, 64, 112, 112};

const int HA_ALL_X_POS[26][32] = 
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
  /* 16x16 G4 */ {24, 88, 24, 88, 24, 88, 24, 88, 24, 88, 24, 88, 24, 88, 24, 88}, // QT-TV-TV-TH

  /* 32x32 U1 */ {16, 80, 16, 80}, // QT-TV-TH
  /* 32x16 U1 */ {16, 80, 16, 80, 16, 80, 16, 80}, // QT-TV-BH-TH
  /* 32x16 U2 */ {16, 80, 16, 80}, // QT-TH-TH-TV
  /* 16x32 U1 */ {8, 40, 72, 104, 8, 40, 72, 104}, // QT-TH-BV-TV
  /* 16x32 U2 */ {24, 88, 24, 88}, // QT-TV-TV-TH
  /* 16x16 U1 */ {8, 40, 72, 104, 8, 40, 72, 104, 8, 40, 72, 104, 8, 40, 72, 104}, // QT-BH-BV-TH-TV
  /* 16x16 U2 */ {24, 88, 24, 88, 24, 88, 24, 88}, // QT-BH-TH-TV-TV
  /* 16x16 U3 */ {8, 40, 72, 104, 8, 40, 72, 104 } // QT-BV-TV-TH-TH
};

const int HA_ALL_Y_POS[26][32] = {
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
  /* 16x16 G4 */ {0, 0, 16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 112, 112},

  /* 32x32 U1 */ {16, 16, 80, 80}, 
  /* 32x16 U1 */ {8, 8, 40, 40, 72, 72, 104, 104},
  /* 32x16 U2 */ {24, 24, 88, 88}, 
  /* 16x32 U1 */ {16, 16, 16, 16, 80, 80, 80, 80},
  /* 16x32 U2 */ {16, 16, 80, 80}, 
  /* 16x16 U1 */ {8, 8, 8, 8, 40, 40, 40, 40, 72, 72, 72, 72, 104, 104, 104, 104},
  /* 16x16 U2 */ {8, 8, 40, 40, 72, 72, 104, 104},
  /* 16x16 U3 */ {24, 24, 24, 24, 88, 88, 88, 88}
};

// Some CU sizes are duplicated because we can generate half-aligned blocks with different sequences of splits
// These different sequences are separated by different groups (G1, G2, G3, and G4) to maintain the number of CUs per CTU a power of 2
const int HA_WIDTH_LIST[26] = 
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
  16,  //16x16 G2 (QT-QTBH-TV)
  16,  //16x16 G3 (QT-TH-TH-TV)
  16,   //16x16 G4 (QT-TV-TV-TH)

  32, // 32x32 U1
  32, // 32x16 U1
  32, // 32x16 U2
  16, // 16x32 U1
  16, // 16x32 U2
  16, // 16x16 U1
  16, // 16x16 U2
  16  // 16x16 U3 
};
// Some CU sizes are duplicated because we can generate half-aligned blocks with different sequences of splits
// These different sequences are separated by different groups (G1, G2, G3, and G4) to maintain the number of CUs per CTU a power of 2
int const HA_HEIGHT_LIST[26] = 
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
  16,   //16x16 G4 (QT-TV-TV-TH)

  32, // 32x32 U1
  16, // 32x16 U1
  16, // 32x16 U2
  32, // 16x32 U1
  32, // 16x32 U2
  16, // 16x16 U1
  16, // 16x16 U2
  16  // 16x16 U3
};

// The number of HALF-ALIGNED CUs inside each CTU, considering different groups of CUs (i.e., different sequences of splits)
const int HA_CUS_PER_CTU[26] = {
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
  16,  //16x16 G4 (QT-TV-TV-TH)

  4, // 32x32 U1
  8, // 32x16 U1
  4, // 32x16 U2
  8, // 16x32 U1
  4, // 16x32 U2
  16, // 16x16 U1
  8, // 16x16 U2
  8  // 16x16 U3
};

const int HA_RETURN_STRIDE_LIST[26] = 
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
  208,   // 16x16 G4

  208+16, // 32x32 U1
  208+16+4, // 32x16 U1
  208+16+4+8, // 32x16 U2
  208+16+4+8+4, // 16x32 U1
  208+16+4+8+4+8, // 16x32 U2
  208+16+4+8+4+8+4,  // 16x16 U1
  208+16+4+8+4+8+4+16,  // 16x16 U2
  208+16+4+8+4+8+4+16+8  // 16x16 U3
  // total = 208+16+4+8+4+8+4+16+8+8
};