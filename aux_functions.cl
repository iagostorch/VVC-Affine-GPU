/*
    THIS FILE CONTAINS SEVERAL AUXILIARY FUNCTIONS TO SEPARATE THE KERNELS FROM MINOR OPERATIONS
    SOME OF THESE FUNCTIONS ARE ADAPTED FROM VTM-12.0, AND SOME OF THEM ARE DESIGNED FROM SCRATCH
*/

#include "constants.cl"

#define SIGN(x) ( (x) >= 0 ? 1 : -1 )

int16 roundValue16(int16 orig, const int shift){
    int offset = 1 << (shift-1);
    
    int16 rounded;// = (orig + offset - (orig>=0)) >> shift; // TODO: This assignment was not working as expected. Debug and correct it in the future

    rounded.s0 = (orig.s0 + offset - (orig.s0>=0)) >> shift;
    rounded.s1 = (orig.s1 + offset - (orig.s1>=0)) >> shift;
    rounded.s2 = (orig.s2 + offset - (orig.s2>=0)) >> shift;
    rounded.s3 = (orig.s3 + offset - (orig.s3>=0)) >> shift;
    rounded.s4 = (orig.s4 + offset - (orig.s4>=0)) >> shift;
    rounded.s5 = (orig.s5 + offset - (orig.s5>=0)) >> shift;
    rounded.s6 = (orig.s6 + offset - (orig.s6>=0)) >> shift;
    rounded.s7 = (orig.s7 + offset - (orig.s7>=0)) >> shift;
    rounded.s8 = (orig.s8 + offset - (orig.s8>=0)) >> shift;
    rounded.s9 = (orig.s9 + offset - (orig.s9>=0)) >> shift;
    rounded.sa = (orig.sa + offset - (orig.sa>=0)) >> shift;
    rounded.sb = (orig.sb + offset - (orig.sb>=0)) >> shift;
    rounded.sc = (orig.sc + offset - (orig.sc>=0)) >> shift;
    rounded.sd = (orig.sd + offset - (orig.sd>=0)) >> shift;
    rounded.se = (orig.se + offset - (orig.se>=0)) >> shift;
    rounded.sf = (orig.sf + offset - (orig.sf>=0)) >> shift;
    

    return rounded;
}

// Rounding function such as CommonLib/Mv.cpp/roundAffineMv() in VTM-12.0
int2 roundMv(const int2 origMv, const int shift){
    int2 roundedMv;

    int offset = 1 << (shift-1);

    roundedMv.x = (origMv.x + offset - (origMv.x>=0)) >> shift;
    roundedMv.y = (origMv.y + offset - (origMv.y>=0)) >> shift;

    return roundedMv;
}

// Clip the MV when it spams too much outside the frame
// Adapted from CommoLib/Mv.cpp/clipMvInPic(), which overrides clipMv() in the code
int2 clipMv(int2 origMv, int block_x, int block_y, int blockWidth, int blockHeight, int frameWidth, int frameHeight){
    int mvShift = MV_FRACTIONAL_BITS_INTERNAL; 
    int offset = 8;

    int horMax = (frameWidth + offset - block_x - 1) << mvShift;
    int horMin = (- MAX_CU_WIDTH - offset - block_x + 1) << mvShift;

    int verMax = (frameHeight + offset - block_y - 1) << mvShift;
    int verMin = (- MAX_CU_HEIGHT - offset - block_y + 1) << mvShift;  

    int2 retMv;

    retMv.x = clamp(origMv.x, horMin, horMax);
    retMv.y = clamp(origMv.y, verMin, verMax);

    return retMv;
}

// Round the MV to original precision and clip when it goes to much outside the picture
// Basically combines roundMv() and clipMv() to improve readablity
int2 roundAndClipMv(const int2 origMv, const int pu_x, const int pu_y, const int pu_width, const int pu_height, const int frameWidth, const int frameHeight){
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL;
    
    int2 retMv;

    // Round the MV to original precision
    retMv = roundMv(origMv, shift);
    // Clip MV when it goes too much outside the frame
    retMv = clipMv(retMv, pu_x, pu_y, pu_width, pu_height, frameWidth, frameHeight);
    
    return retMv;
}

// Determines if the MVs of CPs point too far apart from each other, based on document N-0068
// Copied from isSubblockVectorSpreadOverLimit() in VTM-12.0
// TODO: if/else structures are undesired in GPU programming. Verify the possibility to avoid this test (maybe constraining the ME process to avoid distant MVs)
bool isSubblockVectorSpreadOverLimit(const int a, const int b, const int c, const int d, const bool bipred){
  int gid = get_global_id(0);

  int s4 = ( 4 << 11 );
  int filterTap = 6;

  if ( bipred ){
    int refBlkWidth  = max( max( 0, 4 * a + s4 ), max( 4 * c, 4 * a + 4 * c + s4 ) ) - min( min( 0, 4 * a + s4 ), min( 4 * c, 4 * a + 4 * c + s4 ) );
    int refBlkHeight = max( max( 0, 4 * b ), max( 4 * d + s4, 4 * b + 4 * d + s4 ) ) - min( min( 0, 4 * b ), min( 4 * d + s4, 4 * b + 4 * d + s4 ) );
    refBlkWidth  = ( refBlkWidth >> 11 ) + filterTap + 3;
    refBlkHeight = ( refBlkHeight >> 11 ) + filterTap + 3;

    if ( refBlkWidth * refBlkHeight > ( filterTap + 9 ) * ( filterTap + 9 ) ){
      return true;
    }
  }
  else{   
    int refBlkWidth  = max( 0, 4 * a + s4 ) - min( 0, 4 * a + s4 );
    int refBlkHeight = max( 0, 4 * b ) - min( 0, 4 * b );
    refBlkWidth  = ( refBlkWidth >> 11 ) + filterTap + 3;
    refBlkHeight = ( refBlkHeight >> 11 ) + filterTap + 3;
    if ( refBlkWidth * refBlkHeight > ( filterTap + 9 ) * ( filterTap + 5 ) ){
      return true;
    }

    refBlkWidth  = max( 0, 4 * c ) - min( 0, 4 * c );
    refBlkHeight = max( 0, 4 * d + s4 ) - min( 0, 4 * d + s4 );
    refBlkWidth  = ( refBlkWidth >> 11 ) + filterTap + 3;
    refBlkHeight = ( refBlkHeight >> 11 ) + filterTap + 3;
    if ( refBlkWidth * refBlkHeight > ( filterTap + 5 ) * ( filterTap + 9 ) ){
      return true;
    }    
  }

  return false;
}

// Generate the MVs for each 4x4 sub-block inside a PU based based on 2 control point motion vectors, pu dimensions and sub-block position
// The MVs must be rounded and clipped in sequence
// Also returns if the CPMVs are too spread apart (when all sub-blocks will have the same motion)
int3 deriveMv2Cps_and_spread(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL; // = 7 
    
    int center_x = subBlock_corner_x+2;
    int center_y = subBlock_corner_y+2;
    
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvVerX = -iDMvHorY; // If it is 4 params, there is not vertically-neighboring CPs. Then, estimate it based on horizontal neighbors LT and RT
    int iDMvVerY = iDMvHorX;

    int iMvScaleHor = LT_x << shift;
    int iMvScaleVer = LT_y << shift;

    bool isSpread = isSubblockVectorSpreadOverLimit(iDMvHorX, iDMvHorY, iDMvVerX, iDMvVerY, bipred);

    int2 mv_spread; // Compute the MV in case the CPMVs are spread
    mv_spread.x = iMvScaleHor + iDMvHorX * ( pu_width >> 1 ) + iDMvVerX * ( pu_height >> 1 );
    mv_spread.y = iMvScaleVer + iDMvHorY * ( pu_width >> 1 ) + iDMvVerY * ( pu_height >> 1 );

    int2 mv_NOT_spread; // Compute the MV in case the CPMVs are NOT spread
    mv_NOT_spread.x = iMvScaleHor + iDMvHorX * center_x + iDMvVerX * center_y;
    mv_NOT_spread.y = iMvScaleVer + iDMvHorY * center_x + iDMvVerY * center_y;

    // Selects the correct MV based on isSpread
    int2 mv = select(mv_NOT_spread, mv_spread, (int2)isSpread);

    return (int3)(mv.x, mv.y, isSpread);
}

// Generate the MVs for each 4x4 sub-block inside a PU based based on 3 control point motion vectors, pu dimensions and sub-block position
// The MVs must be rounded and clipped in sequence
// Also returns if the CPMVs are too spread apart (when all sub-blocks will have the same motion)
int3 deriveMv3Cps_and_spread(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int LB_x, const int LB_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL; // = 7
    
    int sub_center_x = subBlock_corner_x+2;
    int sub_center_y = subBlock_corner_y+2;
    
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2((float)pu_width))); 
   
    int iDMvVerX = (LB_x - LT_x) << (shift - (int)floor(native_log2((float)pu_height))); 
    int iDMvVerY = (LB_y - LT_y) << (shift - (int)floor(native_log2((float)pu_height))); 
   
    int iMvScaleHor = LT_x << shift;
    int iMvScaleVer = LT_y << shift;

    bool isSpread = isSubblockVectorSpreadOverLimit(iDMvHorX, iDMvHorY, iDMvVerX, iDMvVerY, bipred);

    int2 mv_spread; // Compute the MV in case the CPMVs are spread
    mv_spread.x = iMvScaleHor + iDMvHorX * ( pu_width >> 1 ) + iDMvVerX * ( pu_height >> 1 );
    mv_spread.y = iMvScaleVer + iDMvHorY * ( pu_width >> 1 ) + iDMvVerY * ( pu_height >> 1 );    

    int2 mv_NOT_spread; // Compute the MV in case the CPMVs are NOT spread
    mv_NOT_spread.x = iMvScaleHor + iDMvHorX * sub_center_x + iDMvVerX * sub_center_y;
    mv_NOT_spread.y = iMvScaleVer + iDMvHorY * sub_center_x + iDMvVerY * sub_center_y;    

    // Selects the correct MV based on isSpread
    int2 mv = select(mv_NOT_spread, mv_spread, (int2)isSpread);

    return (int3)(mv.x, mv.y, isSpread);
}

// This function is based on a fragment of InterPrediction::xPredAffineBlk of VTM-12.0
// It reuses some computations from deriveMv2CPs_and_spread and deriveMv3CPs_and_spread to compute the horizontal and vertical deltas used in PROF
// TODO: It may be possible to improve performance by merging the computation of horizontal and vertical computation, and returning the results using shared memory
// TODO: It may be possible to improve performance by using int16 to store the deltas
int16 getHorizontalDeltasPROF2Cps(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
    // This part is EXACTLY THE SAME as deriveMv2/3CPs
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL; // = 7 
        
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvVerX = -iDMvHorY; // If it is 4 params, there is not vertically-neighboring CPs. Then, estimate it based on horizontal neighbors LT and RT
    int iDMvVerY = iDMvHorX;
     
    int dMvH[SUBBLOCK_SIZE * SUBBLOCK_SIZE]; // These are the /deltas from PROF (Section 3.4.4.4 of document T2002)

    int quadHorX = iDMvHorX << 2;
    int quadVerX = iDMvVerX << 2;

    // Novel computation for PROF
    dMvH[0] = ((iDMvHorX + iDMvVerX) << 1) - ((quadHorX + quadVerX) << 1);

    int w=0, h=0;

    for (w = 1; w < SUBBLOCK_SIZE; w++)
    {
      dMvH[h*SUBBLOCK_SIZE+w] = dMvH[h*SUBBLOCK_SIZE+w - 1] + quadHorX;
    }

    for (h = 1; h < SUBBLOCK_SIZE; h++)
    {
      for (w = 0; w < SUBBLOCK_SIZE; w++)
      {
        dMvH[h*SUBBLOCK_SIZE+w] = dMvH[h*SUBBLOCK_SIZE+w - SUBBLOCK_SIZE] + quadVerX;
      }
    }

    const int mvShift  = 8;
    const int dmvLimit = ( 1 << 5 ) - 1;
    
    int16 deltaHor = vload16(0, dMvH);
    
    deltaHor = roundValue16(deltaHor, mvShift);
    deltaHor = clamp(deltaHor, -dmvLimit, dmvLimit);

    return deltaHor;
}

// This function is based on a fragment of InterPrediction::xPredAffineBlk of VTM-12.0
// It reuses some computations from deriveMv2CPs_and_spread and deriveMv3CPs_and_spread to compute the horizontal and vertical deltas used in PROF
// TODO: It may be possible to improve performance by merging the computation of horizontal and vertical computation, and returning the results using shared memory
// TODO: It may be possible to improve performance by using int16 to store the deltas
int16 getVerticalDeltasPROF2Cps(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
    // This part is EXACTLY THE SAME as deriveMv2/3CPs
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL; // = 7 
    
    int center_x = subBlock_corner_x+2;
    int center_y = subBlock_corner_y+2;
    
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvVerX = -iDMvHorY; // If it is 4 params, there is not vertically-neighboring CPs. Then, estimate it based on horizontal neighbors LT and RT
    int iDMvVerY = iDMvHorX;
     
    // Novel computation for PROF
    int dMvV[SUBBLOCK_SIZE * SUBBLOCK_SIZE];

    int quadHorY = iDMvHorY << 2;
    int quadVerY = iDMvVerY << 2;

    dMvV[0] = ((iDMvHorY + iDMvVerY) << 1) - ((quadHorY + quadVerY) << 1);

    int w=0, h=0;

    for (w = 1; w < SUBBLOCK_SIZE; w++)
    {
      dMvV[h*SUBBLOCK_SIZE+w] = dMvV[h*SUBBLOCK_SIZE+w - 1] + quadHorY;
    }

    for (h = 1; h < SUBBLOCK_SIZE; h++)
    {
      for (w = 0; w < SUBBLOCK_SIZE; w++)
      {
        dMvV[h*SUBBLOCK_SIZE+w] = dMvV[h*SUBBLOCK_SIZE+w - SUBBLOCK_SIZE] + quadVerY;
      }
    }

    const int mvShift  = 8;
    const int dmvLimit = ( 1 << 5 ) - 1;

    int16 deltaVer = vload16(0, dMvV);
    
    deltaVer = roundValue16(deltaVer, mvShift);
    deltaVer = clamp(deltaVer, -dmvLimit, dmvLimit);

    return deltaVer;
}

// This function is based on a fragment of InterPrediction::xPredAffineBlk of VTM-12.0
// It reuses some computations from deriveMv2CPs_and_spread and deriveMv3CPs_and_spread to compute the horizontal and vertical deltas used in PROF
// TODO: It may be possible to improve performance by merging the computation of horizontal and vertical computation, and returning the results using shared memory
// TODO: It may be possible to improve performance by using int16 to store the deltas
int16 getHorizontalDeltasPROF3Cps(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int LB_x, const int LB_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
    // This part is EXACTLY THE SAME as deriveMv2/3CPs
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL; // = 7
    
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2((float)pu_width))); 
  
    int iDMvVerX = (LB_x - LT_x) << (shift - (int)floor(native_log2((float)pu_height))); 

    // Novel computation for PROF
    int dMvH[SUBBLOCK_SIZE * SUBBLOCK_SIZE]; // These are de /deltas from PROF (Section 3.4.4.4 of document T2002)

    int quadHorX = iDMvHorX << 2;
    int quadVerX = iDMvVerX << 2;

    dMvH[0] = ((iDMvHorX + iDMvVerX) << 1) - ((quadHorX + quadVerX) << 1);

    int w=0, h=0;

    for (w = 1; w < SUBBLOCK_SIZE; w++)
    {
      dMvH[h*SUBBLOCK_SIZE+w] = dMvH[h*SUBBLOCK_SIZE+w - 1] + quadHorX;
    }

    for (h = 1; h < SUBBLOCK_SIZE; h++)
    {
      for (w = 0; w < SUBBLOCK_SIZE; w++)
      {
        dMvH[h*SUBBLOCK_SIZE+w] = dMvH[h*SUBBLOCK_SIZE+w - SUBBLOCK_SIZE] + quadVerX;
      }
    }

    const int mvShift  = 8;
    const int dmvLimit = ( 1 << 5 ) - 1;

    int16 deltaHor = vload16(0, dMvH);
    
    deltaHor = roundValue16(deltaHor, mvShift);
    deltaHor = clamp(deltaHor, -dmvLimit, dmvLimit);

    return deltaHor;
}

// This function is based on a fragment of InterPrediction::xPredAffineBlk of VTM-12.0
// It reuses some computations from deriveMv2CPs_and_spread and deriveMv3CPs_and_spread to compute the horizontal and vertical deltas used in PROF
// TODO: It may be possible to improve performance by merging the computation of horizontal and vertical computation, and returning the results using shared memory
// TODO: It may be possible to improve performance by using int16 to store the deltas
int16 getVerticalDeltasPROF3Cps(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int LB_x, const int LB_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
    // This part is EXACTLY THE SAME as deriveMv2/3CPs
    int shift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL; // = 7
    
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2((float)pu_width))); 
    int iDMvVerY = (LB_y - LT_y) << (shift - (int)floor(native_log2((float)pu_height))); 
     
    // Novel computation for PROF
    int dMvV[SUBBLOCK_SIZE * SUBBLOCK_SIZE]; // These are de /deltas from PROF (Section 3.4.4.4 of document T2002)

    int quadHorY = iDMvHorY << 2;
    int quadVerY = iDMvVerY << 2;

    dMvV[0] = ((iDMvHorY + iDMvVerY) << 1) - ((quadHorY + quadVerY) << 1);

    int w=0, h=0;

    for (w = 1; w < SUBBLOCK_SIZE; w++)
    {
      dMvV[h*SUBBLOCK_SIZE+w] = dMvV[h*SUBBLOCK_SIZE+w - 1] + quadHorY;
    }

    for (h = 1; h < SUBBLOCK_SIZE; h++)
    {
      for (w = 0; w < SUBBLOCK_SIZE; w++)
      {
        dMvV[h*SUBBLOCK_SIZE+w] = dMvV[h*SUBBLOCK_SIZE+w - SUBBLOCK_SIZE] + quadVerY;
      }
    }

    const int mvShift  = 8;
    const int dmvLimit = ( 1 << 5 ) - 1;
    
    int16 deltaVer = vload16(0, dMvV);
    
    deltaVer = roundValue16(deltaVer, mvShift);
    deltaVer = clamp(deltaVer, -dmvLimit, dmvLimit);

    return deltaVer;
}

// Clip the value of a pixel based on predefined allowed ranges
int clipPel(int value){
    int ret = clamp(value, CLP_RNG_MIN, CLP_RNG_MAX);
    return ret;
}

// Computes the horizontal gradient of a predicted 4x4 block with padding (thus, a 6x6 block). Used during PROF.
int16 computeHorizontalGrad(int predicted_padded[(SUBBLOCK_SIZE+2*PROF_PADDING) * (SUBBLOCK_SIZE+2*PROF_PADDING)]){
    int shift1 = 6; // used to control gradient precision. Value 6 comes from VTM-12.0
    int16 gradX;
 
    int paddedWidth = SUBBLOCK_SIZE + 2*PROF_PADDING;

    gradX.s0 = (predicted_padded[1*paddedWidth+2]>>shift1) - (predicted_padded[1*paddedWidth+0]>>shift1);
    gradX.s1 = (predicted_padded[1*paddedWidth+3]>>shift1) - (predicted_padded[1*paddedWidth+1]>>shift1);
    gradX.s2 = (predicted_padded[1*paddedWidth+4]>>shift1) - (predicted_padded[1*paddedWidth+2]>>shift1);
    gradX.s3 = (predicted_padded[1*paddedWidth+5]>>shift1) - (predicted_padded[1*paddedWidth+3]>>shift1);

    gradX.s4 = (predicted_padded[2*paddedWidth+2]>>shift1) - (predicted_padded[2*paddedWidth+0]>>shift1);
    gradX.s5 = (predicted_padded[2*paddedWidth+3]>>shift1) - (predicted_padded[2*paddedWidth+1]>>shift1);
    gradX.s6 = (predicted_padded[2*paddedWidth+4]>>shift1) - (predicted_padded[2*paddedWidth+2]>>shift1);
    gradX.s7 = (predicted_padded[2*paddedWidth+5]>>shift1) - (predicted_padded[2*paddedWidth+3]>>shift1);

    gradX.s8 = (predicted_padded[3*paddedWidth+2]>>shift1) - (predicted_padded[3*paddedWidth+0]>>shift1);
    gradX.s9 = (predicted_padded[3*paddedWidth+3]>>shift1) - (predicted_padded[3*paddedWidth+1]>>shift1);
    gradX.sa = (predicted_padded[3*paddedWidth+4]>>shift1) - (predicted_padded[3*paddedWidth+2]>>shift1);
    gradX.sb = (predicted_padded[3*paddedWidth+5]>>shift1) - (predicted_padded[3*paddedWidth+3]>>shift1);

    gradX.sc = (predicted_padded[4*paddedWidth+2]>>shift1) - (predicted_padded[4*paddedWidth+0]>>shift1);
    gradX.sd = (predicted_padded[4*paddedWidth+3]>>shift1) - (predicted_padded[4*paddedWidth+1]>>shift1);
    gradX.se = (predicted_padded[4*paddedWidth+4]>>shift1) - (predicted_padded[4*paddedWidth+2]>>shift1);
    gradX.sf = (predicted_padded[4*paddedWidth+5]>>shift1) - (predicted_padded[4*paddedWidth+3]>>shift1);

    return gradX;
}

// Computes the vertical gradient of a predicted 4x4 block with padding (thus, a 6x6 block). Used during PROF.    
int16 computeVerticalGrad(int predicted_padded[(SUBBLOCK_SIZE+2*PROF_PADDING) * (SUBBLOCK_SIZE+2*PROF_PADDING)]){
    int shift1 = 6; // used to control gradient precision. Value 6 comes from VTM-12.0

    int paddedWidth = SUBBLOCK_SIZE + 2*PROF_PADDING;

    int16 gradY;

    gradY.s0 = (predicted_padded[2*paddedWidth+1]>>shift1) - (predicted_padded[0*paddedWidth+1]>>shift1);
    gradY.s1 = (predicted_padded[2*paddedWidth+2]>>shift1) - (predicted_padded[0*paddedWidth+2]>>shift1);
    gradY.s2 = (predicted_padded[2*paddedWidth+3]>>shift1) - (predicted_padded[0*paddedWidth+3]>>shift1);
    gradY.s3 = (predicted_padded[2*paddedWidth+4]>>shift1) - (predicted_padded[0*paddedWidth+4]>>shift1);

    gradY.s4 = (predicted_padded[3*paddedWidth+1]>>shift1) - (predicted_padded[1*paddedWidth+1]>>shift1);
    gradY.s5 = (predicted_padded[3*paddedWidth+2]>>shift1) - (predicted_padded[1*paddedWidth+2]>>shift1);
    gradY.s6 = (predicted_padded[3*paddedWidth+3]>>shift1) - (predicted_padded[1*paddedWidth+3]>>shift1);
    gradY.s7 = (predicted_padded[3*paddedWidth+4]>>shift1) - (predicted_padded[1*paddedWidth+4]>>shift1);

    gradY.s8 = (predicted_padded[4*paddedWidth+1]>>shift1) - (predicted_padded[2*paddedWidth+1]>>shift1);
    gradY.s9 = (predicted_padded[4*paddedWidth+2]>>shift1) - (predicted_padded[2*paddedWidth+2]>>shift1);
    gradY.sa = (predicted_padded[4*paddedWidth+3]>>shift1) - (predicted_padded[2*paddedWidth+3]>>shift1);
    gradY.sb = (predicted_padded[4*paddedWidth+4]>>shift1) - (predicted_padded[2*paddedWidth+4]>>shift1);

    gradY.sc = (predicted_padded[5*paddedWidth+1]>>shift1) - (predicted_padded[3*paddedWidth+1]>>shift1);
    gradY.sd = (predicted_padded[5*paddedWidth+2]>>shift1) - (predicted_padded[3*paddedWidth+2]>>shift1);
    gradY.se = (predicted_padded[5*paddedWidth+3]>>shift1) - (predicted_padded[3*paddedWidth+3]>>shift1);
    gradY.sf = (predicted_padded[5*paddedWidth+4]>>shift1) - (predicted_padded[3*paddedWidth+4]>>shift1);

    return gradY;

}

// TODO: Verify if using int16 for "predicted" can improve performance. The gradient is computed over a 6x6 block, and the vload/vstore operations can counter the performance gains 
// Apply PROF refinement to a block
int16 PROF(int predicted[SUBBLOCK_SIZE*SUBBLOCK_SIZE], int referenceWindow[11*11], int xFrac, int yFrac, int16 deltaHor, int16 deltaVer){
    int16 predicted_vec; // = vload16(0, predicted);
    predicted_vec.s0 = predicted[0];
    predicted_vec.s1 = predicted[1];
    predicted_vec.s2 = predicted[2];
    predicted_vec.s3 = predicted[3];
    predicted_vec.s4 = predicted[4];
    predicted_vec.s5 = predicted[5];
    predicted_vec.s6 = predicted[6];
    predicted_vec.s7 = predicted[7];
    predicted_vec.s8 = predicted[8];
    predicted_vec.s9 = predicted[9];
    predicted_vec.sa = predicted[10];
    predicted_vec.sb = predicted[11];
    predicted_vec.sc = predicted[12];
    predicted_vec.sd = predicted[13];
    predicted_vec.se = predicted[14];
    predicted_vec.sf = predicted[15];

    // Compute gradient of predicted block. The borders of the block are padded with the closest sample from the reference frame (i.e., reference window). It may be the reference sample from the 4x4 sub-block itself or a neighbor, depending on xFrac and yFrac
    // Get most significant part of fraction: either 0 or 1.
    int xOffset = xFrac >> 3;
    int yOffset = yFrac >> 3;

    // Creates auxiliary arrays to store the samples for padding the borders
    // All the assignments for these arrays, the computation of anchor and curr, etc, are based on the VTM-12.0 code. The code segments are inside the last "if(enablePROF)" of InterPrediction::xPredAffineBlk of VTM-12.0
    int padFirstCol[SUBBLOCK_SIZE], padLastCol[SUBBLOCK_SIZE], padFirstRow[SUBBLOCK_SIZE+2*PROF_PADDING], padLastRow[SUBBLOCK_SIZE+2*PROF_PADDING]; // Together, these 4 arrays create a 6x6 border around the predicted block (1 extra column and row in each side)
    int windowWidth = 11;

    int anchor = 3*11+3; // This points to the corner of the reference 4x4 block inside the reference window
    int curr;
    
    curr = anchor + yOffset * 11 + xOffset;

    padFirstCol[0]=referenceWindow[curr + 0*windowWidth - 1]; 
    padFirstCol[1]=referenceWindow[curr + 1*windowWidth - 1]; 
    padFirstCol[2]=referenceWindow[curr + 2*windowWidth - 1];  
    padFirstCol[3]=referenceWindow[curr + 3*windowWidth - 1]; 
    
    padLastCol[0]=referenceWindow[curr + 0*windowWidth + 4];
    padLastCol[1]=referenceWindow[curr + 1*windowWidth + 4];
    padLastCol[2]=referenceWindow[curr + 2*windowWidth + 4];
    padLastCol[3]=referenceWindow[curr + 3*windowWidth + 4];

    curr = anchor - (1-yOffset) * 11 + xOffset - 1;

    padFirstRow[0]=referenceWindow[curr+0];
    padFirstRow[1]=referenceWindow[curr+1];
    padFirstRow[2]=referenceWindow[curr+2];
    padFirstRow[3]=referenceWindow[curr+3];
    padFirstRow[4]=referenceWindow[curr+4];
    padFirstRow[5]=referenceWindow[curr+5];

    padLastRow[0]=referenceWindow[curr + 5*windowWidth + 0];
    padLastRow[1]=referenceWindow[curr + 5*windowWidth + 1];
    padLastRow[2]=referenceWindow[curr + 5*windowWidth + 2];
    padLastRow[3]=referenceWindow[curr + 5*windowWidth + 3];
    padLastRow[4]=referenceWindow[curr + 5*windowWidth + 4];
    padLastRow[5]=referenceWindow[curr + 5*windowWidth + 5];

    // STARTS THE SCALE CORRECTION. Since the reference window was not shifted/offseted/etc such as the predicted samples, it is necessary to scale them to be represented in the same range
    // This scaling is based on the last if(enablePROF) from InterPrediction::xPredAffineBlk in VTM-12.0

    padFirstCol[0] = (padFirstCol[0]<<4) - IF_INTERNAL_OFFS;
    padFirstCol[1] = (padFirstCol[1]<<4) - IF_INTERNAL_OFFS;
    padFirstCol[2] = (padFirstCol[2]<<4) - IF_INTERNAL_OFFS;
    padFirstCol[3] = (padFirstCol[3]<<4) - IF_INTERNAL_OFFS;

    padLastCol[0] = (padLastCol[0]<<4) - IF_INTERNAL_OFFS;
    padLastCol[1] = (padLastCol[1]<<4) - IF_INTERNAL_OFFS;
    padLastCol[2] = (padLastCol[2]<<4) - IF_INTERNAL_OFFS;
    padLastCol[3] = (padLastCol[3]<<4) - IF_INTERNAL_OFFS;

    padFirstRow[0] = (padFirstRow[0]<<4) - IF_INTERNAL_OFFS;
    padFirstRow[1] = (padFirstRow[1]<<4) - IF_INTERNAL_OFFS;
    padFirstRow[2] = (padFirstRow[2]<<4) - IF_INTERNAL_OFFS;
    padFirstRow[3] = (padFirstRow[3]<<4) - IF_INTERNAL_OFFS;
    padFirstRow[4] = (padFirstRow[4]<<4) - IF_INTERNAL_OFFS;
    padFirstRow[5] = (padFirstRow[5]<<4) - IF_INTERNAL_OFFS;

    padLastRow[0] = (padLastRow[0]<<4) - IF_INTERNAL_OFFS;
    padLastRow[1] = (padLastRow[1]<<4) - IF_INTERNAL_OFFS;
    padLastRow[2] = (padLastRow[2]<<4) - IF_INTERNAL_OFFS;
    padLastRow[3] = (padLastRow[3]<<4) - IF_INTERNAL_OFFS;
    padLastRow[4] = (padLastRow[4]<<4) - IF_INTERNAL_OFFS;
    padLastRow[5] = (padLastRow[5]<<4) - IF_INTERNAL_OFFS;


    // Create a padded predicted block. The padding is used to compute the gradient of all 4x4 samples on the sub-block    
    int predicted_padded[(SUBBLOCK_SIZE + 2*PROF_PADDING)*(SUBBLOCK_SIZE + 2*PROF_PADDING)];
    // Fill inner part with actual predicted samples
    for(int i=1; i<5; i++){
        for(int j=1; j<5; j++){
            predicted_padded[i*6+j] = predicted[(i-1)*4+(j-1)];
        }
    }
    // Fill borders with padding values obtained previously
    for(int i=0; i<6; i++){
        predicted_padded[i] = padFirstRow[i];
        predicted_padded[5*6+i] = padLastRow[i];
    }
    for(int i=1; i<5; i++){
        predicted_padded[i*6+0] = padFirstCol[i-1];
        predicted_padded[i*6+5] = padLastCol[i-1];
    }

    // Compute the gradient of the padded block  
    int16 gradX, gradY;

    gradX = computeHorizontalGrad(predicted_padded);
    gradY = computeVerticalGrad(predicted_padded);        

    // Compute the deltaI of PROF based on Section 3.4.4.4 of document T-2002. Multiply gradients and sub-deltas, and clip values in sequence
    int16 deltaI = gradX*deltaHor + gradY*deltaVer;   
    int dILimit = 1 << 13; // 1 << std::max<int>(clpRng.bd + 1, 13);
    deltaI = clamp(deltaI, -dILimit, dILimit-1);
    // Apply the correction on predicted block
    predicted_vec = predicted_vec + deltaI;
    // Correct the scale of samples to be represented in the conventional 10 bits (0 - 1023)
    int shiftNum = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd);
    int prof_offset = (1 << (shiftNum - 1)) + IF_INTERNAL_OFFS;
    predicted_vec = (predicted_vec + prof_offset) >> shiftNum;
    predicted_vec = clamp(predicted_vec, CLP_RNG_MIN, CLP_RNG_MAX);

    // if(get_global_id(0)==target_gid){
    //     printf("AFTER PROF\n");
    //     printf("%d,%d,%d,%d\n", predicted_vec.s0, predicted_vec.s1, predicted_vec.s2, predicted_vec.s3);
    //     printf("%d,%d,%d,%d\n", predicted_vec.s4, predicted_vec.s5, predicted_vec.s6, predicted_vec.s7);
    //     printf("%d,%d,%d,%d\n", predicted_vec.s8, predicted_vec.s9, predicted_vec.sa, predicted_vec.sb);
    //     printf("%d,%d,%d,%d\n", predicted_vec.sc, predicted_vec.sd, predicted_vec.se, predicted_vec.sf);
    // }

    return predicted_vec;
}

// TODO: If we agree to use 3 different functions (only horizontal, only vertical, both), it is not necessary to use the isFirst and isLast parameters in these functions
// This function is based on  InterpolationFilter::filterHor() from VTM-12.0, but simplified to work in Affine ME only
int16 horizontal_filter(__global int *ref_samples, int2 absPosition, int2 intMv, int frameWidth, int block_width, int block_height, int frac, bool isLast){
    int N=NTAPS_LUMA;
    int row, col;
    int coeff[8];
    // Load proper coefficients based on fraction of MV
    coeff[0] = m_lumaFilter4x4[frac][0];
    coeff[1] = m_lumaFilter4x4[frac][1];
    coeff[2] = m_lumaFilter4x4[frac][2];
    coeff[3] = m_lumaFilter4x4[frac][3];
    coeff[4] = m_lumaFilter4x4[frac][4];
    coeff[5] = m_lumaFilter4x4[frac][5];
    coeff[6] = m_lumaFilter4x4[frac][6];
    coeff[7] = m_lumaFilter4x4[frac][7];
    
    // TODO: Add this information as a parameter to the function. It is a parameter of the other functions already
    // Points to the start (top-left) of reference block
    int refPosition = absPosition.y*frameWidth + absPosition.x + intMv.y*frameWidth + intMv.x;

    int isFirst = true; // Horizontal is always first. TODO: Remove this variable to optimeze the code
    
    // Stride of the input (reference frame) and destination (4x4 block)
    int srcStride = frameWidth;
    int dstStride = block_width;    

    // Stride between each filtered sample (horizontal stride is always 1)
    int cStride = 1;
    refPosition -= ( N/2 - 1 ) * cStride; // Points before the reference block to get neighboring samples (3 samples before target filter sample)

    // Positive left slack means we have enough columns to the left. Negative represents the number of columns outside (to the left) the frame
    int leftSlack = absPosition.x + intMv.x - ( N/2 - 1 );
    
    // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
    int rightSpam = absPosition.x + intMv.x + ( N/2 );
    int rightSlack = frameWidth - 1 - rightSpam;

    int referenceWindow[11*4]; // For horizontal filter, the reference window comprehends the colocated block, plus 3 columns to the left and 4 columns to the right
    int windowWidth  = 11;
    int windowHeight = 4;
   
    bool leftCorrect=false, rightCorrect=false; // Used to verify, for each sample, if it lies "outside" the frame
    int currSample; // Used to avoid if/else structures during left/right correction

    // TODO: Unroll the following loop
    for(int row=0; row<windowHeight; row++){
        for(int col=0; col<windowWidth; col++){           
            leftCorrect = select(true,false,leftSlack+col>=0); // moving right (i.e., moving column-wise) increases left slack and decreases right slack
            rightCorrect = select(true,false,rightSlack-col+7>=0); // this +7 corrects the difference between the reference block and the reference window position/width. Slack represents the relation of refBlock+4, whereas refWindow is 3 columns to the left of refBlock, thus the +7 gap between the "anchor slack" and actual slack
            
            if(leftCorrect==true && rightCorrect==1){
                printf("@@@@@\nFATAL ERROR: Left and Right correct in gid=%d\n@@@@@\n",get_local_id(0));
            }

            currSample = ref_samples[refPosition+row*srcStride+col]; // sample before correction
            currSample = select(currSample,ref_samples[refPosition+row*srcStride-leftSlack],leftCorrect==true);
            currSample = select(currSample,ref_samples[refPosition+row*srcStride+7+rightSlack],rightCorrect==true); // This (+7+slack) returns the pointer to the right-edge of the frame (slack is always negative,)

            referenceWindow[row*windowWidth+col] = currSample;
        }
    }
           
    // if(get_global_id(0)==target_gid){
    //     for(int row=0; row<windowHeight; row++){
    //         for(int col=0; col<windowWidth; col++){
    //             printf("%d,",referenceWindow[row*windowWidth+col]);
    //         }
    //         printf("\n");
    //     }
    // }

    int offset;
    int headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    int shift    = IF_FILTER_PREC; // =6
  
    if ( isLast )
    {
        shift += (isFirst) ? 0 : headRoom;
        offset = 1 << (shift - 1);
        offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    }
    else
    {
        shift -= (isFirst) ? headRoom : 0;
        offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    }

    int predicted[16]; // TODO: Unroll the following loop and use int16 from the start to optimize performance
    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;
            // REMINDER: Since there are 8 taps, it is possible to read from the refWindow using vload8(), multiply by coeff using a single opertaion (vec x vec), and sum-up in another operation (using dot product)
            sum  = referenceWindow[ row*windowWidth + col + 0] * coeff[0];
            sum += referenceWindow[ row*windowWidth + col + 1] * coeff[1];
            sum += referenceWindow[ row*windowWidth + col + 2] * coeff[2];
            sum += referenceWindow[ row*windowWidth + col + 3] * coeff[3];
            sum += referenceWindow[ row*windowWidth + col + 4] * coeff[4];
            sum += referenceWindow[ row*windowWidth + col + 5] * coeff[5];
            sum += referenceWindow[ row*windowWidth + col + 6] * coeff[6];
            sum += referenceWindow[ row*windowWidth + col + 7] * coeff[7];

            int val = ( sum + offset ) >> shift;

            if ( isLast )
            {
                val = clipPel( val );
            }
            predicted[row*block_width+col] = val;
        }
    }

    int16 returnPred = vload16(0, predicted);
    
    return returnPred;
}

// TODO: If we agree to use 3 different functions (only horizontal, only vertical, both), it is not necessary to use the isFirst and isLast parameters in these functions
// This function is based on  InterpolationFilter::filterHor() from VTM-12.0, but simplified to work in Affine ME only
int16 vertical_filter(__global int *ref_samples, int2 absPosition, int2 intMv, int frameWidth, int frameHeight, int block_width, int block_height, int frac, bool isFirst, bool isLast){
    int N=NTAPS_LUMA;
    int row, col;
    int coeff[8];
    // Load proper coefficients based on fraction of MV
    coeff[0] = m_lumaFilter4x4[frac][0];
    coeff[1] = m_lumaFilter4x4[frac][1];
    coeff[2] = m_lumaFilter4x4[frac][2];
    coeff[3] = m_lumaFilter4x4[frac][3];
    coeff[4] = m_lumaFilter4x4[frac][4];
    coeff[5] = m_lumaFilter4x4[frac][5];
    coeff[6] = m_lumaFilter4x4[frac][6];
    coeff[7] = m_lumaFilter4x4[frac][7];
    
    // TODO: Add this information as a parameter to the function. It is a parameter of the other functions already
    // Points to the start (top-left) of reference block
    int refPosition = absPosition.y*frameWidth + absPosition.x + intMv.y*frameWidth + intMv.x;

    // Stride of the input (reference frame) and destination (4x4 block)
    int srcStride = frameWidth;
    int dstStride = block_width;    

    // Stride between each filtered sample (vertical stride is 1 row, or 1 frame width)
    int cStride = frameWidth;
    refPosition -= ( N/2 - 1 ) * cStride; // Points before the reference block to get neighboring samples (3 samples before target filter sample)

    // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
    int topSlack = absPosition.y + intMv.y - ( N/2 - 1 );

    // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
    int bottomSpam = absPosition.y + intMv.y + ( N/2 );
    int bottomSlack = frameHeight - 1 - bottomSpam;

    int referenceWindow[4*11]; // For vertical filter, the reference window comprehends the colocated block, plus 3 rows to the top and 4 rows to the bottom
    int windowWidth  = 4;
    int windowHeight = 11;
       
    bool topCorrect=false, bottomCorrect=false; // Used to verify, for each sample, if it lies "outside" the frame
    int currSample; // Used to avoid if/else structures during top/bottom correction

    // TODO: Unroll the following loop and use int16 from the start to optimize performance
    for(int row=0; row<windowHeight; row++){
        for(int col=0; col<windowWidth; col++){           
            topCorrect = select(true,false,topSlack+row>=0); // movineg downwards (i.e., moving row-wise) increses the top slack and decreases the bottom slack
            bottomCorrect = select(true,false, bottomSlack-row+7>=0);
            
            if(topCorrect==true && bottomCorrect==true){
                printf("@@@@@\nFATAL ERROR: Top and Bottom correct in gid=%d\n@@@@@\n",get_local_id(0));
            }

            currSample = ref_samples[refPosition+row*srcStride+col]; // sample before correction
            currSample = select(currSample,ref_samples[refPosition+(-topSlack)*srcStride+col],topCorrect==true);
            currSample = select(currSample,ref_samples[refPosition+(7+bottomSlack)*srcStride+col],bottomCorrect==true);

            referenceWindow[row*windowWidth+col] = currSample;
        }
    }
    
    // if(get_global_id(0)==target_gid){
    //     for(int row=0; row<windowHeight; row++){
    //         for(int col=0; col<windowWidth; col++){
    //             printf("%d,",referenceWindow[row*windowWidth+col]);
    //         }
    //         printf("\n");
    //     }
    // }
    
    int offset;
    int headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    int shift    = IF_FILTER_PREC; // =6
  
    if ( isLast )
    {
        shift += (isFirst) ? 0 : headRoom;
        offset = 1 << (shift - 1);
        offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    }
    else
    {
        shift -= (isFirst) ? headRoom : 0;
        offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    }

    int predicted[16]; // TODO: Unroll the following loop and use int16 from the start to optimize performance
    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;
            // REMINDER: Since there are 8 taps, it is possible to read from the refWindow using vload8(), multiply by coeff using a single opertaion (vec x vec), and sum-up in another operation (using dot product)
            sum  = referenceWindow[ row*windowWidth + col + 0*windowWidth] * coeff[0];
            sum += referenceWindow[ row*windowWidth + col + 1*windowWidth] * coeff[1];
            sum += referenceWindow[ row*windowWidth + col + 2*windowWidth] * coeff[2];
            sum += referenceWindow[ row*windowWidth + col + 3*windowWidth] * coeff[3];
            sum += referenceWindow[ row*windowWidth + col + 4*windowWidth] * coeff[4];
            sum += referenceWindow[ row*windowWidth + col + 5*windowWidth] * coeff[5];
            sum += referenceWindow[ row*windowWidth + col + 6*windowWidth] * coeff[6];
            sum += referenceWindow[ row*windowWidth + col + 7*windowWidth] * coeff[7];

            int val = ( sum + offset ) >> shift;

            if ( isLast )
            {
                val = clipPel( val );
            }
            predicted[row*block_width+col] = val;
        }
    }

    int16 returnPred = vload16(0, predicted);
    
    return returnPred;
}

// TODO: If we agree to use 3 different functions (only horizontal, only vertical, both), it is not necessary to use the isFirst and isLast parameters in these functions
// This function is a combination of InterpolationFilter::filterHor(), InterpolationFilter::filterVer(), and Buffer::applyPROFCore()
// It predicts a sub-block based on its motion vectos and apply PROF when necessary
int16 horizontal_vertical_filter(__global int *ref_samples, int2 absPosition, int2 intMv, int frameWidth, int frameHeight, int block_width, int block_height, int xFrac, int yFrac, const int isSpread, const int16 deltaHor, const int16 deltaVer, int enablePROF){
    
    // TODO: Maybe disable PROF when all CPMVs are the same if this interferes on bitrate
    
    // int applyPROF = !isSpread; // When CPMVs are too spread all sub-blocks have the same motion and PROF is not enabled. When CPMVs are the same (translational motion) PROF makes no difference. Better use it to maintain consistency
    
    int applyPROF = enablePROF && !isSpread; // When CPMVs are too spread all sub-blocks have the same motion and PROF is not enabled. When CPMVs are the same (translational motion) PROF makes no difference. Better use it to maintain consistency
    
    int N=NTAPS_LUMA; // N is the number of taps
    int row, col;
    int coeff[8];
    // Load proper coefficients based on fraction of MV
    coeff[0] = m_lumaFilter4x4[xFrac][0];
    coeff[1] = m_lumaFilter4x4[xFrac][1];
    coeff[2] = m_lumaFilter4x4[xFrac][2];
    coeff[3] = m_lumaFilter4x4[xFrac][3];
    coeff[4] = m_lumaFilter4x4[xFrac][4];
    coeff[5] = m_lumaFilter4x4[xFrac][5];
    coeff[6] = m_lumaFilter4x4[xFrac][6];
    coeff[7] = m_lumaFilter4x4[xFrac][7];

    // TODO: Add this information as a parameter to the function. It is a parameter of the other functions already
    // Points to the start (top-left) of reference block
    int refPosition = absPosition.y*frameWidth + absPosition.x + intMv.y*frameWidth + intMv.x;

    int isFirst = true; // Horizontal is always first. TODO: Remove this variable to optimeze the code
    int isLast = false;

    // Stride of the input (reference frame) and destination (4x4 block)
    int srcStride = frameWidth;
    int dstStride = block_width;

    // height is 11 now
    block_height = block_height + NTAPS_LUMA - 1; // new block size includes 3 lines above the block and 4 lines below the block
    // Ref position now points 3 rows above the reference block
    refPosition = refPosition - ((NTAPS_LUMA >> 1) - 1) * srcStride; // This puts the pointer 3 lines above the referene block: we must make horizontal interpolation of these samples above the block to use them in the vertical interpolation in sequence
    
    // Stride for horizontal filtering is always 1
    int cStride = 1;
    refPosition -= ( N/2 - 1 ) * cStride; // Point 3 columns to the left of reference block (plus 3 rows above, from prev operation) to get neighboring samples

    // Positive left slack means we have enough columns to the left. Negative represents the number of columns outside (to the left) the frame
    int leftSlack = absPosition.x + intMv.x - ( N/2 - 1 );
    // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
    int rightSpam = absPosition.x + intMv.x + ( N/2 );
    int rightSlack = frameWidth - 1 - rightSpam;
    // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
    int topSlack = absPosition.y + intMv.y - ( N/2 - 1 );
    // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
    int bottomSpam = absPosition.y + intMv.y + ( N/2 );
    int bottomSlack = frameHeight - 1 - bottomSpam;

    int referenceWindow[11*11]; // For horizontal+vertical filter, the reference window comprehends the colocated block, plus 3 columns to the left and 4 columns to the right, 3 rows above and 4 rows below
    int windowWidth  = 11;
    int windowHeight = 11;

    // BEGIN Fetch reference window
    bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
    int currSample; // Used to avoid if/else structures during left/right correction
    int properIdx; // This variable is used to update the index of the reference sample until the proper index is found. It is used when the motion vectors point outisde the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
    // TODO: Unroll the following loop
    for(int row=0; row<windowHeight; row++){
        for(int col=0; col<windowWidth; col++){           
            
            // First computes the "individual" left, right, top and left corrections, disconsidering any other correction
            leftCorrect = select(true,false,leftSlack+col>=0); // moving right (i.e., moving column-wise) increases left slack and decreases right slack
            rightCorrect = select(true,false,rightSlack-col+7>=0); // this +7 corrects the difference between the reference block and the reference window
            topCorrect = select(true,false,topSlack+row>=0); // movineg downwards (i.e., moving row-wise) increses the top slack and decreases the bottom slack
            bottomCorrect = select(true,false, bottomSlack-row+7>=0);
            // Then, computes the compound corrections (top-left, top-right ...)
            topLeftCorrect = leftCorrect && topCorrect;
            topRightCorrect = rightCorrect && topCorrect;
            bottomLeftCorrect = leftCorrect && bottomCorrect;
            bottomRightCorrect = rightCorrect && bottomCorrect;
            // Finally, updates individual corrections by disabling them when a compound correction is true (i.e., if both left and topLeft correction are true, then we perform topLeft correction)
            leftCorrect = leftCorrect && !(topLeftCorrect+bottomLeftCorrect);
            rightCorrect = rightCorrect &&  !(topRightCorrect+bottomRightCorrect);
            topCorrect = topCorrect &&  !(topLeftCorrect+topRightCorrect);
            bottomCorrect = bottomCorrect &&  !(bottomLeftCorrect+bottomRightCorrect);

            if(leftCorrect + rightCorrect + topCorrect + bottomCorrect + topLeftCorrect + topRightCorrect + bottomLeftCorrect + bottomRightCorrect > 1){
                printf("@@@@@\nFATAL ERROR: Multiple corrections in gid=%d\n",get_local_id(0));
                printf("L  %d\n", leftCorrect);
                printf("R  %d\n", rightCorrect);
                printf("T  %d\n", topCorrect);
                printf("B  %d\n", bottomCorrect);

                printf("TL %d\n", topLeftCorrect);
                printf("TR %d\n", topRightCorrect);
                printf("BL %d\n", bottomLeftCorrect);
                printf("BR %d\n", bottomRightCorrect);
            }

            properIdx = refPosition+row*srcStride+col; // Position before correction
            // currSample = ref_samples[refPosition+row*srcStride+col]; // sample before correction
            // Tests individual corrections
            properIdx = select(properIdx,refPosition+row*srcStride-leftSlack,leftCorrect==true);
            properIdx = select(properIdx,refPosition+row*srcStride+7+rightSlack,rightCorrect==true); // This (+7+slack) returns the pointer to the right-edge of the frame (slack is always negative,)
            properIdx = select(properIdx,refPosition+(-topSlack)*srcStride+col,topCorrect==true);
            properIdx = select(properIdx,refPosition+(7+bottomSlack)*srcStride+col,bottomCorrect==true);
            // currSample = select(currSample,ref_samples[refPosition+row*srcStride-leftSlack],leftCorrect==true);
            // currSample = select(currSample,ref_samples[refPosition+row*srcStride+7+rightSlack],rightCorrect==true); // This (+7+slack) returns the pointer to the right-edge of the frame (slack is always negative,)
            // currSample = select(currSample,ref_samples[refPosition+(-topSlack)*srcStride+col],topCorrect==true);
            // currSample = select(currSample,ref_samples[refPosition+(7+bottomSlack)*srcStride+col],bottomCorrect==true);
            // Tests compound corrections
            properIdx = select(properIdx,0,topLeftCorrect==true);
            properIdx = select(properIdx,frameWidth-1,topRightCorrect==true);
            properIdx = select(properIdx,(frameHeight-1)*srcStride,bottomLeftCorrect==true);
            properIdx = select(properIdx,frameWidth*frameHeight-1,bottomRightCorrect==true);
            // currSample = select(currSample,ref_samples[0],topLeftCorrect==true);
            // currSample = select(currSample,ref_samples[frameWidth-1],topRightCorrect==true);
            // currSample = select(currSample,ref_samples[(frameHeight-1)*srcStride],bottomLeftCorrect==true);
            // currSample = select(currSample,ref_samples[frameWidth*frameHeight-1],bottomRightCorrect==true);

            currSample = ref_samples[properIdx]; // Only fetch the sample after computing the proper index. This avoids segmentation fault when the original index is negative or larger than the number of samples

            referenceWindow[row*windowWidth+col] = currSample;
        }
    }

    // if(get_global_id(0)==target_gid){
    //     for(int row=0; row<windowHeight; row++){
    //         for(int col=0; col<windowWidth; col++){
    //             printf("%d,",referenceWindow[row*windowWidth+col]);
    //         }
    //         printf("\n");
    //     }
    // }
    // END Fetch reference window

    int offset;
    int headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    int shift    = IF_FILTER_PREC; // =6
  
    if ( isLast ) // TODO: Horizontal is always first, it is not necessary to use this if/else
    {
        shift += (isFirst) ? 0 : headRoom;
        offset = 1 << (shift - 1);
        offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    }
    else
    {
        shift -= (isFirst) ? headRoom : 0;
        offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    }

    int tempBuffer[4*11]; // TODO: Unroll the following loop. It is not possible to use only a single int16 since the window is 4x11, but there may be another useful data structure

    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;

            // REMINDER: Since there are 8 taps, it is possible to read from the refWindow using vload8(), multiply by coeff using a single opertaion (vec x vec), and sum-up in another operation (using dot product)
            sum  = referenceWindow[ row*windowWidth + col + 0 ] * coeff[0];
            sum += referenceWindow[ row*windowWidth + col + 1 ] * coeff[1];
            sum += referenceWindow[ row*windowWidth + col + 2 ] * coeff[2];
            sum += referenceWindow[ row*windowWidth + col + 3 ] * coeff[3];
            sum += referenceWindow[ row*windowWidth + col + 4 ] * coeff[4];
            sum += referenceWindow[ row*windowWidth + col + 5 ] * coeff[5];
            sum += referenceWindow[ row*windowWidth + col + 6 ] * coeff[6];
            sum += referenceWindow[ row*windowWidth + col + 7 ] * coeff[7];

            int val = ( sum + offset ) >> shift;
            if ( isLast ) // TODO: this if is always false in case we agree on using 3 functions (horizontal, vertical, horizontal+vertical filter)
            {
                val = clipPel( val );
            }
            tempBuffer[row*block_width+col] = val;
        }
    }

    // ------    FINISHES HORIZONTAL FILTERGIN AT THIS POINT
    // ------    tempBuffer is a 4x11 block with block filtered in horizontal direction
    // ------    now it is necessary to filter this block in horizontal direction to get the output 4x4 block
    
    isFirst = false;
    isLast = !applyPROF; // When PROF is applied, vertical filtering IS NOT the last. Otherwise, vertical is the last operation

    // The horizontal and vertical filters may have different precision/fraction, we must update the coefficients properly
    coeff[0] = m_lumaFilter4x4[yFrac][0];
    coeff[1] = m_lumaFilter4x4[yFrac][1];
    coeff[2] = m_lumaFilter4x4[yFrac][2];
    coeff[3] = m_lumaFilter4x4[yFrac][3];
    coeff[4] = m_lumaFilter4x4[yFrac][4];
    coeff[5] = m_lumaFilter4x4[yFrac][5];
    coeff[6] = m_lumaFilter4x4[yFrac][6];
    coeff[7] = m_lumaFilter4x4[yFrac][7];

    srcStride = block_width; // for vertical filter, the "source" is a tempBuffer 4x11
    dstStride = block_width;

    cStride = block_width; // for vertical filter, the "source" is a tempBuffer 4x11
    refPosition = 0; // First reference is the first sample at tempBuffer

    headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    shift    = IF_FILTER_PREC; // =6
  
    if ( isLast ) // TODO: read the comment on the assignment "isLast = true". For the same CU, isLast has the same value for all sub-blocks. Depending on the scheduling of the kernel this if/else will not compromise performance
    {
        shift += (isFirst) ? 0 : headRoom; // TODO: Both on this if and on the else statements, isFirst is always false (we are on vertical filtering). Remove this dependency.
        offset = 1 << (shift - 1);
        offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    }
    else
    {
        shift -= (isFirst) ? headRoom : 0;
        offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    }

    int predicted[16]; // TODO: Unroll the following loop and use int16 from the start to optimize performance
    
    block_height = 4; // now, the output will be a 4x4 block again. The input is 4x11 though

    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;

            // REMINDER: since here we do not read 8 values in sequence from memory (the stride is block_width), vload8() does not work
            // If the loop is unrolled, it may be possible to optimize it somehow
            sum  = tempBuffer[ row*block_width + col + 0*block_width] * coeff[0];
            sum += tempBuffer[ row*block_width + col + 1*block_width] * coeff[1];
            sum += tempBuffer[ row*block_width + col + 2*block_width] * coeff[2];
            sum += tempBuffer[ row*block_width + col + 3*block_width] * coeff[3];
            sum += tempBuffer[ row*block_width + col + 4*block_width] * coeff[4];
            sum += tempBuffer[ row*block_width + col + 5*block_width] * coeff[5];
            sum += tempBuffer[ row*block_width + col + 6*block_width] * coeff[6];
            sum += tempBuffer[ row*block_width + col + 7*block_width] * coeff[7];

            int val = ( sum + offset ) >> shift;
            if ( isLast )
            {
                val = clipPel( val );
            }
            predicted[row*block_width+col] = val;
        }
    }
    
    // ------    FINISHES VERTICAL FILTERGIN AT THIS POINT
    // ------    predicted is a 4x4 block with filtered in horizontal and vertical directions
    // ------    now we apply PROF to this block. Depending on the parameters, PROF'd block can be discarded

    // Temp vector used to store the result after vertical filtering. If PROF is undesired, we will recover this result at the end
    int16 predicted_pre_PROF_vec = vload16(0, predicted);
   
    int16 predicted_after_PROF_vec = PROF(predicted, referenceWindow, xFrac, yFrac, deltaHor, deltaVer);
    
    int16 returnPred;
    // Selects the PROF'd or not-PROF'd prediction based on applyPROF
    returnPred = select(predicted_pre_PROF_vec, predicted_after_PROF_vec, (int16)(applyPROF)==1);

    return returnPred;   
}

// TODO: If we agree to use 3 different functions (only horizontal, only vertical, both), it is not necessary to use the isFirst and isLast parameters in these functions
// This function is a combination of InterpolationFilter::filterHor(), InterpolationFilter::filterVer(), and Buffer::applyPROFCore()
// It predicts a sub-block based on its motion vectos and apply PROF when necessary
int16 horizontal_vertical_filter_new(int referenceWindow[11*11], int2 absPosition, int2 intMv, int frameWidth, int frameHeight, int block_width, int block_height, int xFrac, int yFrac, const int isSpread, const int16 deltaHor, const int16 deltaVer, int enablePROF){
    int windowWidth=11, windowHeight=11;
    // printf("a\n");
    // TODO: Maybe disable PROF when all CPMVs are the same if this interferes on bitrate
    
    // int applyPROF = !isSpread; // When CPMVs are too spread all sub-blocks have the same motion and PROF is not enabled. When CPMVs are the same (translational motion) PROF makes no difference. Better use it to maintain consistency
    
    int applyPROF = enablePROF && !isSpread; // When CPMVs are too spread all sub-blocks have the same motion and PROF is not enabled. When CPMVs are the same (translational motion) PROF makes no difference. Better use it to maintain consistency
    
    int N=NTAPS_LUMA; // N is the number of taps
    int row, col;
    int coeff[8];
    // Load proper coefficients based on fraction of MV
    coeff[0] = m_lumaFilter4x4[xFrac][0];
    coeff[1] = m_lumaFilter4x4[xFrac][1];
    coeff[2] = m_lumaFilter4x4[xFrac][2];
    coeff[3] = m_lumaFilter4x4[xFrac][3];
    coeff[4] = m_lumaFilter4x4[xFrac][4];
    coeff[5] = m_lumaFilter4x4[xFrac][5];
    coeff[6] = m_lumaFilter4x4[xFrac][6];
    coeff[7] = m_lumaFilter4x4[xFrac][7];

    int isFirst = true; // Horizontal is always first. TODO: Remove this variable to optimeze the code
    int isLast = false;

    // height is 11 now: original 4x4 plus 3 to the top and 4 to the bottom for filtering
    block_height = 11;
    
    int offset;
    int headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    int shift    = IF_FILTER_PREC; // =6
  
    if ( isLast ) // TODO: Horizontal is always first, it is not necessary to use this if/else
    {
        shift += (isFirst) ? 0 : headRoom;
        offset = 1 << (shift - 1);
        offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    }
    else
    {
        shift -= (isFirst) ? headRoom : 0;
        offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    }

    int tempBuffer[4*11]; // TODO: Unroll the following loop. It is not possible to use only a single int16 since the window is 4x11, but there may be another useful data structure

    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;

            // REMINDER: Since there are 8 taps, it is possible to read from the refWindow using vload8(), multiply by coeff using a single opertaion (vec x vec), and sum-up in another operation (using dot product)
            sum  = referenceWindow[ row*windowWidth + col + 0 ] * coeff[0];
            sum += referenceWindow[ row*windowWidth + col + 1 ] * coeff[1];
            sum += referenceWindow[ row*windowWidth + col + 2 ] * coeff[2];
            sum += referenceWindow[ row*windowWidth + col + 3 ] * coeff[3];
            sum += referenceWindow[ row*windowWidth + col + 4 ] * coeff[4];
            sum += referenceWindow[ row*windowWidth + col + 5 ] * coeff[5];
            sum += referenceWindow[ row*windowWidth + col + 6 ] * coeff[6];
            sum += referenceWindow[ row*windowWidth + col + 7 ] * coeff[7];

            int val = ( sum + offset ) >> shift;
            if ( isLast ) // TODO: this if is always false in case we agree on using 3 functions (horizontal, vertical, horizontal+vertical filter)
            {
                val = clipPel( val );
            }
            tempBuffer[row*block_width+col] = val;
        }
    }

    // ------    FINISHES HORIZONTAL FILTERGIN AT THIS POINT
    // ------    tempBuffer is a 4x11 block with block filtered in horizontal direction
    // ------    now it is necessary to filter this block in horizontal direction to get the output 4x4 block
    
    isFirst = false;
    isLast = !applyPROF; // When PROF is applied, vertical filtering IS NOT the last. Otherwise, vertical is the last operation

    // The horizontal and vertical filters may have different precision/fraction, we must update the coefficients properly
    coeff[0] = m_lumaFilter4x4[yFrac][0];
    coeff[1] = m_lumaFilter4x4[yFrac][1];
    coeff[2] = m_lumaFilter4x4[yFrac][2];
    coeff[3] = m_lumaFilter4x4[yFrac][3];
    coeff[4] = m_lumaFilter4x4[yFrac][4];
    coeff[5] = m_lumaFilter4x4[yFrac][5];
    coeff[6] = m_lumaFilter4x4[yFrac][6];
    coeff[7] = m_lumaFilter4x4[yFrac][7];

    headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    shift    = IF_FILTER_PREC; // =6
  
    if ( isLast ) // TODO: read the comment on the assignment "isLast = true". For the same CU, isLast has the same value for all sub-blocks. Depending on the scheduling of the kernel this if/else will not compromise performance
    {
        shift += (isFirst) ? 0 : headRoom; // TODO: Both on this if and on the else statements, isFirst is always false (we are on vertical filtering). Remove this dependency.
        offset = 1 << (shift - 1);
        offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    }
    else
    {
        shift -= (isFirst) ? headRoom : 0;
        offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    }

    int predicted[16]; // TODO: Unroll the following loop and use int16 from the start to optimize performance
    
    block_height = 4; // now, the output will be a 4x4 block again. The input is 4x11 though

    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;

            // REMINDER: since here we do not read 8 values in sequence from memory (the stride is block_width), vload8() does not work
            // If the loop is unrolled, it may be possible to optimize it somehow
            sum  = tempBuffer[ row*block_width + col + 0*block_width] * coeff[0];
            sum += tempBuffer[ row*block_width + col + 1*block_width] * coeff[1];
            sum += tempBuffer[ row*block_width + col + 2*block_width] * coeff[2];
            sum += tempBuffer[ row*block_width + col + 3*block_width] * coeff[3];
            sum += tempBuffer[ row*block_width + col + 4*block_width] * coeff[4];
            sum += tempBuffer[ row*block_width + col + 5*block_width] * coeff[5];
            sum += tempBuffer[ row*block_width + col + 6*block_width] * coeff[6];
            sum += tempBuffer[ row*block_width + col + 7*block_width] * coeff[7];

            int val = ( sum + offset ) >> shift;
            if ( isLast )
            {
                val = clipPel( val );
            }
            predicted[row*block_width+col] = val;
        }
    }
    
    // ------    FINISHES VERTICAL FILTERGIN AT THIS POINT
    // ------    predicted is a 4x4 block with filtered in horizontal and vertical directions
    // ------    now we apply PROF to this block. Depending on the parameters, PROF'd block can be discarded

    // Temp vector used to store the result after vertical filtering. If PROF is undesired, we will recover this result at the end
    int16 predicted_pre_PROF_vec = vload16(0, predicted);
   
    int16 predicted_after_PROF_vec = PROF(predicted, referenceWindow, xFrac, yFrac, deltaHor, deltaVer);
    
    int16 returnPred;
    // Selects the PROF'd or not-PROF'd prediction based on applyPROF
    returnPred = select(predicted_pre_PROF_vec, predicted_after_PROF_vec, (int16)(applyPROF)==1);

    return returnPred;   
}

// This function is a combination of InterpolationFilter::filterHor(), InterpolationFilter::filterVer(), and Buffer::applyPROFCore()
// It predicts a sub-block based on its motion vectos and apply PROF when necessary
// TODO: This function was unrolled to improve performance, but it seems that it is slowing down the program. Verify in the future
int16 horizontal_vertical_filter_new_unrolled(__private int referenceWindow[11*11], int2 absPosition, int2 intMv, int frameWidth, int frameHeight, int block_width, int block_height, int xFrac, int yFrac, const int isSpread, const int16 deltaHor, const int16 deltaVer, int enablePROF){
    int windowWidth=11, windowHeight=11;
    // printf("a\n");
    // TODO: Maybe disable PROF when all CPMVs are the same if this interferes on bitrate
    
    // int applyPROF = !isSpread; // When CPMVs are too spread all sub-blocks have the same motion and PROF is not enabled. When CPMVs are the same (translational motion) PROF makes no difference. Better use it to maintain consistency
    
    int applyPROF = enablePROF && !isSpread; // When CPMVs are too spread all sub-blocks have the same motion and PROF is not enabled. When CPMVs are the same (translational motion) PROF makes no difference. Better use it to maintain consistency
    
    int N=NTAPS_LUMA; // N is the number of taps
    int row, col;
    int coeff[8];
    // Load proper coefficients based on fraction of MV
    coeff[0] = m_lumaFilter4x4[xFrac][0];
    coeff[1] = m_lumaFilter4x4[xFrac][1];
    coeff[2] = m_lumaFilter4x4[xFrac][2];
    coeff[3] = m_lumaFilter4x4[xFrac][3];
    coeff[4] = m_lumaFilter4x4[xFrac][4];
    coeff[5] = m_lumaFilter4x4[xFrac][5];
    coeff[6] = m_lumaFilter4x4[xFrac][6];
    coeff[7] = m_lumaFilter4x4[xFrac][7];

    int8 coeff_vec = vload8(0,coeff);
    int8 taps_samples, partial_sum;

    int isFirst = true; // Horizontal is always first. TODO: Remove this variable to optimeze the code
    int isLast = false;

    // height is 11 now: original 4x4 plus 3 to the top and 4 to the bottom for filtering
    block_height = 11;
    
    int offset;
    int headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    int shift    = IF_FILTER_PREC; // =6
  
    // if ( isLast ) // TODO: Horizontal is always first, it is not necessary to use this if/else
    // {
    //     shift += (isFirst) ? 0 : headRoom;
    //     offset = 1 << (shift - 1);
    //     offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    // }
    // else
    // {
    //     shift -= (isFirst) ? headRoom : 0;
    //     offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    // }

    shift -= headRoom;
    offset = -IF_INTERNAL_OFFS << shift;

    int tempBuffer[4*11]; // TODO: Unroll the following loop. It is not possible to use only a single int16 since the window is 4x11, but there may be another useful data structure
    int sum2 = 0;
    // for (row = 0; row < block_height; row++){
        // for (col = 0; col < block_width; col++){
            int sum;
            taps_samples = vload8((row*windowWidth+col)/(float)8, referenceWindow);
            partial_sum = taps_samples * coeff_vec;
            sum2 = partial_sum.s0 + partial_sum.s1 + partial_sum.s2 + partial_sum.s3 + partial_sum.s4 + partial_sum.s5 + partial_sum.s6 + partial_sum.s7;

            // taps_samples = vload8(0, referenceWindow);
            // REMINDER: Since there are 8 taps, it is possible to read from the refWindow using vload8(), multiply by coeff using a single opertaion (vec x vec), and sum-up in another operation (using dot product)
            // sum  = referenceWindow[ row*windowWidth + col + 0 ] * coeff[0];
            // sum += referenceWindow[ row*windowWidth + col + 1 ] * coeff[1];
            // sum += referenceWindow[ row*windowWidth + col + 2 ] * coeff[2];
            // sum += referenceWindow[ row*windowWidth + col + 3 ] * coeff[3];
            // sum += referenceWindow[ row*windowWidth + col + 4 ] * coeff[4];
            // sum += referenceWindow[ row*windowWidth + col + 5 ] * coeff[5];
            // sum += referenceWindow[ row*windowWidth + col + 6 ] * coeff[6];
            // sum += referenceWindow[ row*windowWidth + col + 7 ] * coeff[7];

            // int val = ( sum + offset ) >> shift;
            // // if ( isLast ) // TODO: this if is always false in case we agree on using 3 functions (horizontal, vertical, horizontal+vertical filter)
            // // {
            // //     val = clipPel( val );
            // // }
            // tempBuffer[row*block_width+col] = val;


            // sum  = referenceWindow[ row*windowWidth + 0 + 0 ] * coeff[0];
            // sum += referenceWindow[ row*windowWidth + 0 + 1 ] * coeff[1];
            // sum += referenceWindow[ row*windowWidth + 0 + 2 ] * coeff[2];
            // sum += referenceWindow[ row*windowWidth + 0 + 3 ] * coeff[3];
            // sum += referenceWindow[ row*windowWidth + 0 + 4 ] * coeff[4];
            // sum += referenceWindow[ row*windowWidth + 0 + 5 ] * coeff[5];
            // sum += referenceWindow[ row*windowWidth + 0 + 6 ] * coeff[6];
            // sum += referenceWindow[ row*windowWidth + 0 + 7 ] * coeff[7];
            // int val = ( sum + offset ) >> shift;
            // tempBuffer[row*block_width+0] = val;

            // sum  = referenceWindow[ row*windowWidth + 1 + 0 ] * coeff[0];
            // sum += referenceWindow[ row*windowWidth + 1 + 1 ] * coeff[1];
            // sum += referenceWindow[ row*windowWidth + 1 + 2 ] * coeff[2];
            // sum += referenceWindow[ row*windowWidth + 1 + 3 ] * coeff[3];
            // sum += referenceWindow[ row*windowWidth + 1 + 4 ] * coeff[4];
            // sum += referenceWindow[ row*windowWidth + 1 + 5 ] * coeff[5];
            // sum += referenceWindow[ row*windowWidth + 1 + 6 ] * coeff[6];
            // sum += referenceWindow[ row*windowWidth + 1 + 7 ] * coeff[7];
            // val = ( sum + offset ) >> shift;
            // tempBuffer[row*block_width+1] = val;      

            // sum  = referenceWindow[ row*windowWidth + 2 + 0 ] * coeff[0];
            // sum += referenceWindow[ row*windowWidth + 2 + 1 ] * coeff[1];
            // sum += referenceWindow[ row*windowWidth + 2 + 2 ] * coeff[2];
            // sum += referenceWindow[ row*windowWidth + 2 + 3 ] * coeff[3];
            // sum += referenceWindow[ row*windowWidth + 2 + 4 ] * coeff[4];
            // sum += referenceWindow[ row*windowWidth + 2 + 5 ] * coeff[5];
            // sum += referenceWindow[ row*windowWidth + 2 + 6 ] * coeff[6];
            // sum += referenceWindow[ row*windowWidth + 2 + 7 ] * coeff[7];
            // val = ( sum + offset ) >> shift;
            // tempBuffer[row*block_width+2] = val;

            // sum  = referenceWindow[ row*windowWidth + 3 + 0 ] * coeff[0];
            // sum += referenceWindow[ row*windowWidth + 3 + 1 ] * coeff[1];
            // sum += referenceWindow[ row*windowWidth + 3 + 2 ] * coeff[2];
            // sum += referenceWindow[ row*windowWidth + 3 + 3 ] * coeff[3];
            // sum += referenceWindow[ row*windowWidth + 3 + 4 ] * coeff[4];
            // sum += referenceWindow[ row*windowWidth + 3 + 5 ] * coeff[5];
            // sum += referenceWindow[ row*windowWidth + 3 + 6 ] * coeff[6];
            // sum += referenceWindow[ row*windowWidth + 3 + 7 ] * coeff[7];
            // val = ( sum + offset ) >> shift;
            // tempBuffer[row*block_width+3] = val;  
			
            // fully unrolled
            sum  = referenceWindow[ 0*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 0*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 0*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 0*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 0*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 0*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 0*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 0*windowWidth + 0 + 7 ] * coeff[7];
            int val = ( sum + offset ) >> shift;
            tempBuffer[0*block_width+0] = val;

            sum  = referenceWindow[ 0*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 0*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 0*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 0*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 0*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 0*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 0*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 0*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[0*block_width+1] = val;      

            sum  = referenceWindow[ 0*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 0*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 0*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 0*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 0*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 0*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 0*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 0*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[0*block_width+2] = val;

            sum  = referenceWindow[ 0*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 0*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 0*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 0*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 0*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 0*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 0*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 0*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[0*block_width+3] = val;  

			sum  = referenceWindow[ 1*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 1*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 1*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 1*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 1*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 1*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 1*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 1*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[1*block_width+0] = val;

            sum  = referenceWindow[ 1*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 1*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 1*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 1*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 1*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 1*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 1*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 1*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[1*block_width+1] = val;      

            sum  = referenceWindow[ 1*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 1*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 1*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 1*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 1*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 1*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 1*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 1*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[1*block_width+2] = val;

            sum  = referenceWindow[ 1*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 1*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 1*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 1*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 1*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 1*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 1*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 1*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[1*block_width+3] = val; 

			sum  = referenceWindow[ 2*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 2*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 2*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 2*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 2*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 2*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 2*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 2*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[2*block_width+0] = val;

            sum  = referenceWindow[ 2*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 2*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 2*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 2*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 2*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 2*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 2*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 2*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[2*block_width+1] = val;      

            sum  = referenceWindow[ 2*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 2*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 2*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 2*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 2*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 2*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 2*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 2*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[2*block_width+2] = val;

            sum  = referenceWindow[ 2*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 2*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 2*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 2*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 2*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 2*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 2*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 2*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[2*block_width+3] = val; 

	        sum  = referenceWindow[ 3*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 3*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 3*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 3*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 3*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 3*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 3*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 3*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[3*block_width+0] = val;

            sum  = referenceWindow[ 3*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 3*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 3*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 3*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 3*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 3*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 3*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 3*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[3*block_width+1] = val;      

            sum  = referenceWindow[ 3*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 3*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 3*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 3*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 3*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 3*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 3*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 3*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[3*block_width+2] = val;

            sum  = referenceWindow[ 3*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 3*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 3*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 3*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 3*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 3*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 3*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 3*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[3*block_width+3] = val; 

			sum  = referenceWindow[ 4*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 4*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 4*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 4*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 4*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 4*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 4*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 4*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[4*block_width+0] = val;

            sum  = referenceWindow[ 4*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 4*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 4*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 4*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 4*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 4*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 4*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 4*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[4*block_width+1] = val;      

            sum  = referenceWindow[ 4*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 4*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 4*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 4*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 4*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 4*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 4*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 4*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[4*block_width+2] = val;

            sum  = referenceWindow[ 4*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 4*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 4*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 4*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 4*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 4*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 4*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 4*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[4*block_width+3] = val; 

			sum  = referenceWindow[ 5*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 5*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 5*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 5*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 5*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 5*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 5*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 5*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[5*block_width+0] = val;

            sum  = referenceWindow[ 5*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 5*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 5*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 5*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 5*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 5*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 5*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 5*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[5*block_width+1] = val;      

            sum  = referenceWindow[ 5*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 5*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 5*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 5*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 5*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 5*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 5*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 5*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[5*block_width+2] = val;

            sum  = referenceWindow[ 5*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 5*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 5*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 5*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 5*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 5*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 5*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 5*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[5*block_width+3] = val; 

			sum  = referenceWindow[ 6*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 6*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 6*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 6*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 6*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 6*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 6*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 6*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[6*block_width+0] = val;

            sum  = referenceWindow[ 6*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 6*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 6*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 6*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 6*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 6*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 6*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 6*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[6*block_width+1] = val;      

            sum  = referenceWindow[ 6*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 6*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 6*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 6*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 6*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 6*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 6*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 6*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[6*block_width+2] = val;

            sum  = referenceWindow[ 6*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 6*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 6*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 6*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 6*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 6*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 6*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 6*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[6*block_width+3] = val; 

			sum  = referenceWindow[ 7*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 7*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 7*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 7*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 7*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 7*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 7*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 7*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[7*block_width+0] = val;

            sum  = referenceWindow[ 7*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 7*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 7*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 7*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 7*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 7*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 7*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 7*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[7*block_width+1] = val;      

            sum  = referenceWindow[ 7*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 7*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 7*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 7*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 7*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 7*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 7*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 7*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[7*block_width+2] = val;

            sum  = referenceWindow[ 7*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 7*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 7*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 7*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 7*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 7*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 7*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 7*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[7*block_width+3] = val; 

			sum  = referenceWindow[ 8*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 8*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 8*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 8*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 8*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 8*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 8*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 8*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[8*block_width+0] = val;

            sum  = referenceWindow[ 8*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 8*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 8*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 8*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 8*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 8*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 8*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 8*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[8*block_width+1] = val;      

            sum  = referenceWindow[ 8*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 8*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 8*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 8*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 8*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 8*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 8*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 8*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[8*block_width+2] = val;

            sum  = referenceWindow[ 8*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 8*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 8*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 8*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 8*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 8*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 8*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 8*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[8*block_width+3] = val; 

			sum  = referenceWindow[ 9*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 9*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 9*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 9*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 9*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 9*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 9*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 9*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[9*block_width+0] = val;

            sum  = referenceWindow[ 9*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 9*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 9*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 9*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 9*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 9*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 9*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 9*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[9*block_width+1] = val;      

            sum  = referenceWindow[ 9*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 9*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 9*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 9*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 9*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 9*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 9*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 9*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[9*block_width+2] = val;

            sum  = referenceWindow[ 9*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 9*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 9*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 9*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 9*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 9*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 9*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 9*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[9*block_width+3] = val; 

			sum  = referenceWindow[ 10*windowWidth + 0 + 0 ] * coeff[0];
            sum += referenceWindow[ 10*windowWidth + 0 + 1 ] * coeff[1];
            sum += referenceWindow[ 10*windowWidth + 0 + 2 ] * coeff[2];
            sum += referenceWindow[ 10*windowWidth + 0 + 3 ] * coeff[3];
            sum += referenceWindow[ 10*windowWidth + 0 + 4 ] * coeff[4];
            sum += referenceWindow[ 10*windowWidth + 0 + 5 ] * coeff[5];
            sum += referenceWindow[ 10*windowWidth + 0 + 6 ] * coeff[6];
            sum += referenceWindow[ 10*windowWidth + 0 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[10*block_width+0] = val;

            sum  = referenceWindow[ 10*windowWidth + 1 + 0 ] * coeff[0];
            sum += referenceWindow[ 10*windowWidth + 1 + 1 ] * coeff[1];
            sum += referenceWindow[ 10*windowWidth + 1 + 2 ] * coeff[2];
            sum += referenceWindow[ 10*windowWidth + 1 + 3 ] * coeff[3];
            sum += referenceWindow[ 10*windowWidth + 1 + 4 ] * coeff[4];
            sum += referenceWindow[ 10*windowWidth + 1 + 5 ] * coeff[5];
            sum += referenceWindow[ 10*windowWidth + 1 + 6 ] * coeff[6];
            sum += referenceWindow[ 10*windowWidth + 1 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[10*block_width+1] = val;      

            sum  = referenceWindow[ 10*windowWidth + 2 + 0 ] * coeff[0];
            sum += referenceWindow[ 10*windowWidth + 2 + 1 ] * coeff[1];
            sum += referenceWindow[ 10*windowWidth + 2 + 2 ] * coeff[2];
            sum += referenceWindow[ 10*windowWidth + 2 + 3 ] * coeff[3];
            sum += referenceWindow[ 10*windowWidth + 2 + 4 ] * coeff[4];
            sum += referenceWindow[ 10*windowWidth + 2 + 5 ] * coeff[5];
            sum += referenceWindow[ 10*windowWidth + 2 + 6 ] * coeff[6];
            sum += referenceWindow[ 10*windowWidth + 2 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[10*block_width+2] = val;

            sum  = referenceWindow[ 10*windowWidth + 3 + 0 ] * coeff[0];
            sum += referenceWindow[ 10*windowWidth + 3 + 1 ] * coeff[1];
            sum += referenceWindow[ 10*windowWidth + 3 + 2 ] * coeff[2];
            sum += referenceWindow[ 10*windowWidth + 3 + 3 ] * coeff[3];
            sum += referenceWindow[ 10*windowWidth + 3 + 4 ] * coeff[4];
            sum += referenceWindow[ 10*windowWidth + 3 + 5 ] * coeff[5];
            sum += referenceWindow[ 10*windowWidth + 3 + 6 ] * coeff[6];
            sum += referenceWindow[ 10*windowWidth + 3 + 7 ] * coeff[7];
            val = ( sum + offset ) >> shift;
            tempBuffer[10*block_width+3] = val;                                                             


        // }
    // }

    // ------    FINISHES HORIZONTAL FILTERGIN AT THIS POINT
    // ------    tempBuffer is a 4x11 block with block filtered in horizontal direction
    // ------    now it is necessary to filter this block in horizontal direction to get the output 4x4 block
    
    isFirst = false;
    isLast = !applyPROF; // When PROF is applied, vertical filtering IS NOT the last. Otherwise, vertical is the last operation

    // The horizontal and vertical filters may have different precision/fraction, we must update the coefficients properly
    coeff[0] = m_lumaFilter4x4[yFrac][0];
    coeff[1] = m_lumaFilter4x4[yFrac][1];
    coeff[2] = m_lumaFilter4x4[yFrac][2];
    coeff[3] = m_lumaFilter4x4[yFrac][3];
    coeff[4] = m_lumaFilter4x4[yFrac][4];
    coeff[5] = m_lumaFilter4x4[yFrac][5];
    coeff[6] = m_lumaFilter4x4[yFrac][6];
    coeff[7] = m_lumaFilter4x4[yFrac][7];

    headRoom = 4; //IF_INTERNAL_FRAC_BITS(clpRng.bd); // =4
    shift    = IF_FILTER_PREC; // =6
  
    // if ( isLast ) // TODO: read the comment on the assignment "isLast = true". For the same CU, isLast has the same value for all sub-blocks. Depending on the scheduling of the kernel this if/else will not compromise performance
    // {
    //     shift += (isFirst) ? 0 : headRoom; // TODO: Both on this if and on the else statements, isFirst is always false (we are on vertical filtering). Remove this dependency.
    //     offset = 1 << (shift - 1);
    //     offset += (isFirst) ? 0 : IF_INTERNAL_OFFS << IF_FILTER_PREC;
    // }
    // else
    // {
    //     shift -= (isFirst) ? headRoom : 0;
    //     offset = (isFirst) ? -IF_INTERNAL_OFFS << shift : 0;
    // }


    shift = select(shift,shift+headRoom,isLast);
    offset = 1 << (shift - 1);
    offset += IF_INTERNAL_OFFS << IF_FILTER_PREC;
    offset = select(0,offset,isLast);

    int predicted[16]; // TODO: Unroll the following loop and use int16 from the start to optimize performance
    
    block_height = 4; // now, the output will be a 4x4 block again. The input is 4x11 though

    for (row = 0; row < block_height; row++){
        for (col = 0; col < block_width; col++){
            int sum;

            // REMINDER: since here we do not read 8 values in sequence from memory (the stride is block_width), vload8() does not work
            // If the loop is unrolled, it may be possible to optimize it somehow
            sum  = tempBuffer[ row*block_width + col + 0*block_width] * coeff[0];
            sum += tempBuffer[ row*block_width + col + 1*block_width] * coeff[1];
            sum += tempBuffer[ row*block_width + col + 2*block_width] * coeff[2];
            sum += tempBuffer[ row*block_width + col + 3*block_width] * coeff[3];
            sum += tempBuffer[ row*block_width + col + 4*block_width] * coeff[4];
            sum += tempBuffer[ row*block_width + col + 5*block_width] * coeff[5];
            sum += tempBuffer[ row*block_width + col + 6*block_width] * coeff[6];
            sum += tempBuffer[ row*block_width + col + 7*block_width] * coeff[7];

            int val = ( sum + offset ) >> shift;
            val = select(val,clipPel( val ),isLast);
            // if ( isLast )
            // {
            //     val = clipPel( val );
            // }
            predicted[row*block_width+col] = val;
        }
    }
    
    // ------    FINISHES VERTICAL FILTERGIN AT THIS POINT
    // ------    predicted is a 4x4 block with filtered in horizontal and vertical directions
    // ------    now we apply PROF to this block. Depending on the parameters, PROF'd block can be discarded

    // Temp vector used to store the result after vertical filtering. If PROF is undesired, we will recover this result at the end
    int16 predicted_pre_PROF_vec = vload16(0, predicted);
   
    int16 predicted_after_PROF_vec = PROF(predicted, referenceWindow, xFrac, yFrac, deltaHor, deltaVer);
    
    int16 returnPred;
    // Selects the PROF'd or not-PROF'd prediction based on applyPROF
    returnPred = select(predicted_pre_PROF_vec, predicted_after_PROF_vec, (int16)(applyPROF)==1);

    return returnPred;   
}


// This function is inherited from VTM-12.0: RdCost::xCalcHADs4x4
int satd_4x4(int16 original_samples, int16 filtered_samples){
    int k;
    int satd = 0;
    int diff[16], m[16], d[16];
    
    int16 difference_samples = original_samples - filtered_samples;
    diff[0] = difference_samples.s0;
    diff[1] = difference_samples.s1;
    diff[2] = difference_samples.s2;
    diff[3] = difference_samples.s3;
    diff[4] = difference_samples.s4;
    diff[5] = difference_samples.s5;
    diff[6] = difference_samples.s6;
    diff[7] = difference_samples.s7;
    diff[8] = difference_samples.s8;
    diff[9] = difference_samples.s9;
    diff[10] = difference_samples.sa;
    diff[11] = difference_samples.sb;
    diff[12] = difference_samples.sc;
    diff[13] = difference_samples.sd;
    diff[14] = difference_samples.se;
    diff[15] = difference_samples.sf;

    /*===== hadamard transform =====*/
    m[ 0] = diff[ 0] + diff[12];
    m[ 1] = diff[ 1] + diff[13];
    m[ 2] = diff[ 2] + diff[14];
    m[ 3] = diff[ 3] + diff[15];
    m[ 4] = diff[ 4] + diff[ 8];
    m[ 5] = diff[ 5] + diff[ 9];
    m[ 6] = diff[ 6] + diff[10];
    m[ 7] = diff[ 7] + diff[11];
    m[ 8] = diff[ 4] - diff[ 8];
    m[ 9] = diff[ 5] - diff[ 9];
    m[10] = diff[ 6] - diff[10];
    m[11] = diff[ 7] - diff[11];
    m[12] = diff[ 0] - diff[12];
    m[13] = diff[ 1] - diff[13];
    m[14] = diff[ 2] - diff[14];
    m[15] = diff[ 3] - diff[15];

    d[ 0] = m[ 0] + m[ 4];
    d[ 1] = m[ 1] + m[ 5];
    d[ 2] = m[ 2] + m[ 6];
    d[ 3] = m[ 3] + m[ 7];
    d[ 4] = m[ 8] + m[12];
    d[ 5] = m[ 9] + m[13];
    d[ 6] = m[10] + m[14];
    d[ 7] = m[11] + m[15];
    d[ 8] = m[ 0] - m[ 4];
    d[ 9] = m[ 1] - m[ 5];
    d[10] = m[ 2] - m[ 6];
    d[11] = m[ 3] - m[ 7];
    d[12] = m[12] - m[ 8];
    d[13] = m[13] - m[ 9];
    d[14] = m[14] - m[10];
    d[15] = m[15] - m[11];

    m[ 0] = d[ 0] + d[ 3];
    m[ 1] = d[ 1] + d[ 2];
    m[ 2] = d[ 1] - d[ 2];
    m[ 3] = d[ 0] - d[ 3];
    m[ 4] = d[ 4] + d[ 7];
    m[ 5] = d[ 5] + d[ 6];
    m[ 6] = d[ 5] - d[ 6];
    m[ 7] = d[ 4] - d[ 7];
    m[ 8] = d[ 8] + d[11];
    m[ 9] = d[ 9] + d[10];
    m[10] = d[ 9] - d[10];
    m[11] = d[ 8] - d[11];
    m[12] = d[12] + d[15];
    m[13] = d[13] + d[14];
    m[14] = d[13] - d[14];
    m[15] = d[12] - d[15];

    d[ 0] = m[ 0] + m[ 1];
    d[ 1] = m[ 0] - m[ 1];
    d[ 2] = m[ 2] + m[ 3];
    d[ 3] = m[ 3] - m[ 2];
    d[ 4] = m[ 4] + m[ 5];
    d[ 5] = m[ 4] - m[ 5];
    d[ 6] = m[ 6] + m[ 7];
    d[ 7] = m[ 7] - m[ 6];
    d[ 8] = m[ 8] + m[ 9];
    d[ 9] = m[ 8] - m[ 9];
    d[10] = m[10] + m[11];
    d[11] = m[11] - m[10];
    d[12] = m[12] + m[13];
    d[13] = m[12] - m[13];
    d[14] = m[14] + m[15];
    d[15] = m[15] - m[14];
 
    for (k=0; k<16; ++k)
    {
        satd += abs(d[k]);
    }

    //JVET_R0164_MEAN_SCALED_SATD // This is true on VTM
    satd -= abs(d[0]);
    satd += abs(d[0]) >> 2;
    satd = ((satd+1)>>1);
    
    return satd;
}

// This function computes the SAD of a 4x4 block
int sad_4x4(int16 original_samples, int16 filtered_samples){
    int sad = 0;
    
    uint16 diff = abs_diff(original_samples, filtered_samples);
    
    sad = diff.s0 + diff.s1 + diff.s2 + diff.s3 + diff.s4 + diff.s5 + diff.s6 + diff.s7 + diff.s8 + diff.s9 + diff.sa + diff.sb + diff.sc + diff.sd + diff.se + diff.sf;

    return sad;
}

// This function is inherited from changePrecision() in VTM-12.0. It is used in bitrate estimation
int2 changeAffinePrecInternal2Amvr(int2 MV, int MV_PRECISION){
    int src = MV_PRECISION_INTERNAL;
    int dst = MV_PRECISION;

    // TODO: Improve this to avoid using if/else
    int shift = dst - src;
    if(shift >=0){ // Shift the MVs LEFT (increase value)
        MV.x = MV.x << shift;
        MV.y = MV.y << shift;
    }
    else{
        int rightShift = -shift;
        int nOffset = 1 << (rightShift - 1);
                        
        MV.x = select((MV.x + nOffset) >> rightShift, (MV.x + nOffset - 1) >> rightShift, MV.x>=0);
        MV.y = select((MV.y + nOffset) >> rightShift, (MV.y + nOffset - 1) >> rightShift, MV.y>=0);
    }
    return MV;
}

// This function is an adaptation of xGetExpGolombNumberOfBits from VTM-12.0. It is used in bitrate estimation
int xGetExpGolombNumberOfBits(int value){
    unsigned int uiLength2 = 1;
    unsigned int uiTemp2 = select((unsigned int)( value << 1 ), ( (unsigned int)( -value ) << 1 ) + 1, value <= 0);
    
    
    while( uiTemp2 > MAX_CU_SIZE )
    {
      uiLength2 += ( MAX_CU_DEPTH << 1 );
      uiTemp2  >>=   MAX_CU_DEPTH;
    }

    return uiLength2 + (((int)floor(native_log2((float)uiTemp2))) << 1);
}

// This function is an adaptation of getBitsOfVectorWithPredictor from VTM-12.0. It is used in bitrate estimation
int getBitsOfVectorWithPredictor(int2 predictor_MV, int2 selected_MV, int cost_scale, int imvShift){
    int hor_bits = xGetExpGolombNumberOfBits(((selected_MV.x << cost_scale) - predictor_MV.x)>>imvShift);
    int ver_bits = xGetExpGolombNumberOfBits(((selected_MV.y << cost_scale) - predictor_MV.y)>>imvShift);

    return (hor_bits+ver_bits);
}

// This function is a combination of InterSearch::xCalcAffineMVBits and its child functions from VTM-12.0. It is used in bitrate estimation
int calc_affine_bits(int MV_PRECISION, int nCP, int LT_x, int LT_y, int RT_x, int RT_y, int LB_x, int LB_y, int pred_LT_x, int pred_LT_y, int pred_RT_x, int pred_RT_y, int pred_LB_x, int pred_LB_y){
    int mvNum  = nCP; 
    int cost_scale = 0; // m_pcRdCost->setCostScale( 0 );
    int bitsTemp = 0;

    int2 tempMV;

    int2 predictor, selected;

    // #######################################
    // first MV
    tempMV.x = pred_LT_x;
    tempMV.y = pred_LT_y;
    predictor = changeAffinePrecInternal2Amvr(tempMV, MV_PRECISION);

    tempMV.x = LT_x;
    tempMV.y = LT_y;
    selected = changeAffinePrecInternal2Amvr(tempMV, MV_PRECISION);

    bitsTemp += getBitsOfVectorWithPredictor(predictor, selected, cost_scale, 0);

    // #######################################
    // second MV
    tempMV.x = pred_RT_x + LT_x - pred_LT_x;
    tempMV.y = pred_RT_y + LT_y - pred_LT_y;
    predictor = changeAffinePrecInternal2Amvr(tempMV, MV_PRECISION);

    tempMV.x = RT_x;
    tempMV.y = RT_y;
    selected = changeAffinePrecInternal2Amvr(tempMV, MV_PRECISION);

    bitsTemp += getBitsOfVectorWithPredictor(predictor, selected, cost_scale, 0);

    // #######################################
    // third MV -> this may or may not be added depending on parameter nCP
    tempMV.x = pred_LB_x + LT_x - pred_LT_x;
    tempMV.y = pred_LB_y + LT_y - pred_LT_y;
    predictor = changeAffinePrecInternal2Amvr(tempMV, MV_PRECISION);

    tempMV.x = LB_x;
    tempMV.y = LB_y;
    selected = changeAffinePrecInternal2Amvr(tempMV, MV_PRECISION);

    int extraBits = getBitsOfVectorWithPredictor(predictor, selected, cost_scale, 0);

    // Check if we are using 3 CPs to add the extraBits
    bitsTemp = select(bitsTemp,bitsTemp+extraBits,nCP==3);

    return bitsTemp;
}

// This function takes as input the affine parameters obtained by solving the system of linear equations and transforms it into deltaMVs
// It is adapted from a part of InterSearch::xAffineMotionEstimation() in VTM-12.0
// It was simplified assuming that AMVR is disabled
int8 scaleDeltaMvs(double8 dDeltaMv, int nCP, int cuWidth, int cuHeight){
    int8 deltaMvs;
    
    int normShiftTab = AFFINE_MV_PRECISION_QUARTER - AFFINE_MV_PRECISION_INT;
    int stepShiftTab =  MV_PRECISION_INTERNAL - AFFINE_MV_PRECISION_QUARTER;
    int multiShift = 1<<normShiftTab;
    int mvShift = stepShiftTab;

    // deltaMvs[0] = Mv( ( int ) ( dDeltaMv[0] * multiShift + SIGN( dDeltaMv[0] ) * 0.5 ) << mvShift, ( int ) ( dDeltaMv[2] * multiShift + SIGN( dDeltaMv[2] ) * 0.5 ) << mvShift );
    deltaMvs.s0 = ( int ) ( dDeltaMv.s0 * multiShift + SIGN( dDeltaMv.s0 ) * 0.5 ) << mvShift;
    deltaMvs.s1 = ( int ) ( dDeltaMv.s2 * multiShift + SIGN( dDeltaMv.s2 ) * 0.5 ) << mvShift;
    // deltaMvs[1] = Mv( ( int ) ( dDeltaMv[1] * multiShift + SIGN( dDeltaMv[1] ) * 0.5 ) << mvShift, ( int ) ( dDeltaMv[3] * multiShift + SIGN( dDeltaMv[3] ) * 0.5 ) << mvShift );
    deltaMvs.s2 = ( int ) ( dDeltaMv.s1 * multiShift + SIGN( dDeltaMv.s1 ) * 0.5 ) << mvShift;
    deltaMvs.s3 = ( int ) ( dDeltaMv.s3 * multiShift + SIGN( dDeltaMv.s3 ) * 0.5 ) << mvShift;
    // if(nCP==3){ // This if is not required: when using 2 CPs, the last values are ignored
    deltaMvs.s4 = ( int ) ( dDeltaMv.s4 * multiShift + SIGN( dDeltaMv.s4 ) * 0.5 ) << mvShift;
    deltaMvs.s5 = ( int ) ( dDeltaMv.s5 * multiShift + SIGN( dDeltaMv.s5 ) * 0.5 ) << mvShift;
    // }

    return deltaMvs;

}

// Gets a bitrate in terms of number of bits, and scales it by a lambda to return a bitrate in terms of distoriton
// TODO: This lambda is specific to the POC=1 of LowDelay with QP 32. It must be improved to support multiple frames and QPS
int getCost(int bitrate){
    return floor(78.949063*bitrate);
}