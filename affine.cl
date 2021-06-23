#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#endif

// CONSTANT USED TO DEBUG INSTANCES OF THE KERNEL
__constant int target_gid = 1023;

// CONSTANTS INHERITED FROM VTM-12 TO IMPROVE CLARITY AND AVOID MAGIC NUMBERS
__constant int MAX_CU_DEPTH = 7;
__constant int MV_FRACTIONAL_BITS_INTERNAL = 4;
__constant int MAX_CU_WIDTH = 128;
__constant int MAX_CU_HEIGHT = 128;
__constant int IF_FILTER_PREC = 6;
__constant int IF_INTERNAL_OFFS = 32; // (1<<(IF_INTERNAL_PREC-1)) ///< Offset used internally
__constant int CLP_RNG_MAX = 1023;
__constant int CLP_RNG_MIN = 0;
__constant int CLP_RNG_BD = 10;
__constant int NTAPS_LUMA = 8; // Number of taps for luma filter

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

// Rounding function such as CommonLib/Mv.cpp/roundAffineMv() in VTM-12.0
int2 roundMv(const int2 origMv, const int shift){
    int2 roundedMv;

    int offset = 1 << (shift-1);

    roundedMv.x = (origMv.x + offset - (origMv.x>=0)) >> shift;
    roundedMv.y = (origMv.y + offset - (origMv.y>=0)) >> shift;

    return roundedMv;
}

// TODO: Substitute the max and min operations by the "clamp" OpenCL function
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

    retMv.x = min(horMax, max(horMin, origMv.x));
    retMv.y = min(verMax, max(verMin, origMv.y));

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
int2 deriveMv2Cps(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
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

    int2 mv;

    // TODO: If/else structures are undesired in GPU programming. Verify the possibility to avoid this test (maybe constraining the ME process)
    if( ! isSpread){ // MVs ARE NOT too spread out
        mv.x = iMvScaleHor + iDMvHorX * center_x + iDMvVerX * center_y;
        mv.y = iMvScaleVer + iDMvHorY * center_x + iDMvVerY * center_y;
        // mv.x = (local_RT_x-local_LT_x)*center_x/pu_width - (local_RT_y-local_LT_y)*center_y/pu_width + local_LT_x;
        // mv.y = (local_RT_y-local_LT_y)*center_x/pu_width + (local_RT_x-local_LT_x)*center_y/pu_width + local_LT_y;
    }
    else{ // MVs ARE too spread out      
        mv.x = iMvScaleHor + iDMvHorX * ( pu_width >> 1 ) + iDMvVerX * ( pu_height >> 1 );
        mv.y = iMvScaleVer + iDMvHorY * ( pu_width >> 1 ) + iDMvVerY * ( pu_height >> 1 );
        // mv.x = (local_RT_x-local_LT_x)*(pu_width/2)/pu_width + (local_RT_y-local_LT_y)*(pu_width/2)/pu_width + local_LT_x;
        // mv.y = (local_RT_y-local_LT_y)*(pu_width/2)/pu_width + (local_RT_x-local_LT_x)*(pu_width/2)/pu_width + local_LT_y;
    }

    return mv;
}

// Generate the MVs for each 4x4 sub-block inside a PU based based on 3 control point motion vectors, pu dimensions and sub-block position
// The MVs must be rounded and clipped in sequence
int2 deriveMv3Cps(const int LT_x, const int LT_y, const int RT_x, const int RT_y, const int LB_x, const int LB_y, const int pu_width, const int pu_height, const int subBlock_corner_x, const int subBlock_corner_y, const bool bipred){
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

    int2 mv;
    // TODO: if/else structures are undesired in GPU programming. Verify the possibility to avoid this test (maybe constraining the ME process to avoid distant MVs)
    if( ! isSpread){ // MVs ARE NOT too spread out
        mv.x = iMvScaleHor + iDMvHorX * sub_center_x + iDMvVerX * sub_center_y;
        mv.y = iMvScaleVer + iDMvHorY * sub_center_x + iDMvVerY * sub_center_y;    
    //    mv.x = (local_RT_x-local_LT_x)*center_x/pu_width + (local_LB_x-local_LT_x)*center_y/pu_height + local_LT_x;
    //    mv.y = (local_RT_y-local_LT_y)*center_x/pu_width + (local_LB_y-local_LT_y)*center_y/pu_height + local_LT_y;
    }
    else{ // MVs ARE too spread out
        mv.x = iMvScaleHor + iDMvHorX * ( pu_width >> 1 ) + iDMvVerX * ( pu_height >> 1 );
        mv.y = iMvScaleVer + iDMvHorY * ( pu_width >> 1 ) + iDMvVerY * ( pu_height >> 1 );        
        // mv.x = (local_RT_x-local_LT_x)*(pu_width >> 1)/pu_width + (local_LB_x-local_LT_x)*(pu_height >> 1)/pu_height + local_LT_x;
        // mv.y = (local_RT_y-local_LT_y)*(pu_width >> 1)/pu_width + (local_LB_y-local_LT_y)*(pu_height >> 1)/pu_height + local_LT_y;
    }

    return mv;
}

// Clip the value of a pixel based on predefined allowed ranges
int clipPel(int value){
    int ret = min(max(value, CLP_RNG_MIN), CLP_RNG_MAX); // TODO Acho que o OpenCL tem uma função própria pra isso (clip entre min e max)
    return ret;
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
// This function is a combination of InterpolationFilter::filterHor() and InterpolationFilter::filterVer()
int16 horizontal_vertical_filter(__global int *ref_samples, int2 absPosition, int2 intMv, int frameWidth, int frameHeight, int block_width, int block_height, int xFrac, int yFrac){
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

            currSample = ref_samples[refPosition+row*srcStride+col]; // sample before correction
            // Tests individual corrections
            currSample = select(currSample,ref_samples[refPosition+row*srcStride-leftSlack],leftCorrect==true);
            currSample = select(currSample,ref_samples[refPosition+row*srcStride+7+rightSlack],rightCorrect==true); // This (+7+slack) returns the pointer to the right-edge of the frame (slack is always negative,)
            currSample = select(currSample,ref_samples[refPosition+(-topSlack)*srcStride+col],topCorrect==true);
            currSample = select(currSample,ref_samples[refPosition+(7+bottomSlack)*srcStride+col],bottomCorrect==true);
            // Tests compound corrections
            currSample = select(currSample,ref_samples[0],topLeftCorrect==true);
            currSample = select(currSample,ref_samples[frameWidth-1],topRightCorrect==true);
            currSample = select(currSample,ref_samples[(frameHeight-1)*srcStride],bottomLeftCorrect==true);
            currSample = select(currSample,ref_samples[frameWidth*frameHeight-1],bottomRightCorrect==true);
            
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
    isLast = true;

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
    
    int16 returnPred = vload16(0, predicted);//(int16)(predicted[0],predicted[1],predicted[2],predicted[3],predicted[4],predicted[5],predicted[6],predicted[7],predicted[8],predicted[9],predicted[10],predicted[11],predicted[12],predicted[13],predicted[14],predicted[15]);
    
    return returnPred;   
}


__kernel void affine(__global int *subMVs_x, __global int *subMVs_y, __global int *LT_x, __global int *LT_y, __global int *RT_x, __global int *RT_y, __global int *LB_x, __global int *LB_y, __global int *pu_x, __global int *pu_y, __global int *pu_width, __global int *pu_height, __global int *subBlock_x, __global int *subBlock_y, __global bool *bipred, __global int *nCP, const int frameWidth, const int frameHeight, __global int *ref_samples, __global int *curr_samples, __global int *filtered_samples){
    int gid = get_global_id(0);

    int2 subMv;
    
    // Derive the MV for the sub-block based on CPMVs
    // TODO: This if/else is undesired to improve performance. This will be solved when we consider "real inputs" instead of a chunck of mixed input data
    if(nCP[gid]==2){
        subMv = deriveMv2Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
    }
    else{
        subMv = deriveMv3Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], LB_x[gid], LB_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
    }

    subMv = roundAndClipMv(subMv, pu_x[gid], pu_y[gid],  pu_width[gid], pu_height[gid], frameWidth, frameHeight);

    // Store sub-mvs in memory
    // TODO: This may not be necessary depending on our final framework: sub-mvs will be tested to get the RD, and only the best will be stored
    subMVs_x[gid] = subMv.x;
    subMVs_y[gid] = subMv.y;
    
    // Split the int and fractional part of the MVs to eprform prediction/filtering
    // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
    int2 subMv_int, subMv_frac;

    subMv_int.x  = subMv.x >> 4;
    subMv_frac.x = subMv.x & 15;
    subMv_int.y  = subMv.y >> 4;
    subMv_frac.y = subMv.y & 15;

    int currPuPosition =   pu_y[gid]*frameWidth + pu_x[gid];
    // This is the position of the block pointed by the integer MV. It will be filtered later to get the fractional movement
    int refBlockPosition = currPuPosition + (subMv_int.y + subBlock_y[gid])*frameWidth + subMv_int.x + subBlock_x[gid];
    int16 predBlock; // Represents the current 4x4 block, containinig 16 samples
    bool isLast, isFirst;
    // TODO: It is possible to avoid this if/else structure: when frac=0, the filter coefficients are {0,0,0,64,0,0,0,0}
    if (subMv_frac.y == 0) // If VERTICAL component IS NOT fractional, then it is not necessary to interpolate vertically. Perform horizontal interpolarion only.
    {
        isLast = true;
        // horizontal is both first and last pass (the only pass)
        predBlock = horizontal_filter(ref_samples, (int2)(pu_x[gid]+subBlock_x[gid],pu_y[gid]+subBlock_y[gid]), subMv_int, frameWidth, 4, 4, subMv_frac.x, isLast);
    }
    else if (subMv_frac.x == 0)  // If HORIZONTAL component IS NOT fractional and vertical component is fractional (yFrac!=0), then perform vertical interpolation only
    {
        isLast = true;
        isFirst = true;
        // vertical is both first and last pass (the only pass)
        predBlock = vertical_filter(ref_samples, (int2)(pu_x[gid]+subBlock_x[gid],pu_y[gid]+subBlock_y[gid]), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.y, isFirst, isLast);
    }
    else // Both vertical and horizontal components are fractional, then perform interpolation in both directions
    {
        // horizontal is first pass, vertical is last pass
        // These assignments are performed inside the function
        predBlock = horizontal_vertical_filter(ref_samples, (int2)(pu_x[gid]+subBlock_x[gid],pu_y[gid]+subBlock_y[gid]), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y);     
    }

    //  The following code is used to debug the filtering operations
    /*
        if(gid==target_gid){   
            printf("REF SAMPLES\n");
        
            printf("%d, %d, %d, %d\n", ref_samples[refBlockPosition], ref_samples[refBlockPosition+1], ref_samples[refBlockPosition+2], ref_samples[refBlockPosition+3]);
            printf("%d, %d, %d, %d\n", ref_samples[refBlockPosition+1*frameWidth], ref_samples[refBlockPosition+1*frameWidth+1], ref_samples[refBlockPosition+1*frameWidth+2], ref_samples[refBlockPosition+1*frameWidth+3]);
            printf("%d, %d, %d, %d\n", ref_samples[refBlockPosition+2*frameWidth], ref_samples[refBlockPosition+2*frameWidth+1], ref_samples[refBlockPosition+2*frameWidth+2], ref_samples[refBlockPosition+2*frameWidth+3]);
            printf("%d, %d, %d, %d\n", ref_samples[refBlockPosition+3*frameWidth], ref_samples[refBlockPosition+3*frameWidth+1], ref_samples[refBlockPosition+3*frameWidth+2], ref_samples[refBlockPosition+3*frameWidth+3]);
        }

        if(gid==target_gid){
            printf("FILTERED SAMPLES\n");

            printf("%d, %d, %d, %d\n", predBlock.s0, predBlock.s1, predBlock.s2, predBlock.s3);
            printf("%d, %d, %d, %d\n", predBlock.s4, predBlock.s5, predBlock.s6, predBlock.s7);
            printf("%d, %d, %d, %d\n", predBlock.s8, predBlock.s9, predBlock.sa, predBlock.sb);
            printf("%d, %d, %d, %d\n", predBlock.sc, predBlock.sd, predBlock.se, predBlock.sf);
        }
    //*/

}

__kernel void FS_ME(__global const int *reference, __global const int *samples, __global int *sad, int n_samples, int srWidth, int srHeight, int frameWidth, int frameHeight, int blockWidth, int blockHeight, int blockTop, int blockLeft) {
 
    int local_sad = 0;

    // Get the index of the current element to be processed, and derive X and Y MV values
    int gid = get_global_id(0);
    int srY = gid/(2*srWidth)-64;
    int srX = gid%(2*srHeight)-64;

    int lid = get_local_id(0);
    int wgSize = get_local_size(0);
    int wgIdx = get_group_id(0);
    int calc = wgSize*wgIdx+lid;


    // printf("Gid: %.10d\t%d*%d+%d=%d\n", gid, wgSize, wgIdx, lid, calc);


    int row, col, topLeftPos;
    topLeftPos = (blockTop+srY)*frameWidth + blockLeft+srX; // top-left corner of reference block, 1st sample of block

    bool extraWidth, extraHeight; // these identify when the reference block lies outside of reference frame
    extraWidth = (blockLeft + srX < 0) || (blockLeft + blockWidth-1 + srX >= frameWidth);
    extraHeight = (blockTop + srY < 0) || (blockTop + blockHeight-1 + srY >= frameHeight);

    if(extraWidth || extraHeight){ // part of candidate block lies outside of reference frame, assigned MAX_INT;
        // printf("srY srX: %d %d\n", srY, srX);
        // sad[topLeftPos] = -1;
        sad[topLeftPos] = INT_MAX;
    }
    else{
        // cout << "else" << endl;
        for(row=0; row<blockHeight; row++){
            for(col=0; col<blockWidth; col++){
                local_sad += abs(reference[topLeftPos+row*frameWidth+col]-samples[row*blockWidth+col]);
            }
        }
        sad[topLeftPos] = local_sad;
    }

    // printf("MV y x: %dx%d \t TopLeft %d \t top x left %d %d \t SAD %d\n", srY, srX, topLeftPos, topLeftPos/frameWidth, topLeftPos%frameWidth, sad[topLeftPos]);
    
}