/*
    THIS FILE CONTAINS SEVERAL CONSTANTS INHERITED FROM VTM-12.0 OR CREATED SPECIFICALLY FOR THE AFFINE KERNEL
    THEY ARE USED TO IMPROVE THE CLARITY AND AVOID MAGIC NUMBERS IN THE CODE
*/

// CONSTANT USED TO DEBUG INSTANCES OF THE KERNEL
__constant int target_gid = 1023;

// CONSTANTS INHERITED FROM VTM-12
__constant int MAX_CU_DEPTH = 7;
__constant int MV_FRACTIONAL_BITS_INTERNAL = 4;
__constant int MAX_CU_WIDTH = 128;
__constant int MAX_CU_HEIGHT = 128;
__constant int IF_FILTER_PREC = 6;
__constant int IF_INTERNAL_PREC = 14; ///< Number of bits for internal precision
__constant int IF_INTERNAL_OFFS = 1 <<(IF_INTERNAL_PREC-1); // (1<<(IF_INTERNAL_PREC-1)) ///< Offset used internally
__constant int CLP_RNG_MAX = 1023;
__constant int CLP_RNG_MIN = 0;
__constant int CLP_RNG_BD = 10;
__constant int NTAPS_LUMA = 8; // Number of taps for luma filter
__constant int MV_PRECISION_INTERNAL = 2 + MV_FRACTIONAL_BITS_INTERNAL;
__constant int MAX_CU_SIZE = 1<<MAX_CU_DEPTH;
// The following are used for AMVR, for cu.imv = 0, 1 and 2
__constant int AFFINE_MV_PRECISION_QUARTER = 4;
__constant int AFFINE_MV_PRECISION_SIXTEENTH = 1;
__constant int AFFINE_MV_PRECISION_INT = 2;

// CONSTANTS CREATED TO AFFINE KERNEL
__constant int SUBBLOCK_SIZE = 4;
__constant int PROF_PADDING = 1;

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
