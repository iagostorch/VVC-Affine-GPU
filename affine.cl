#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#endif

// CONSTANTS INHERITED FROM VTM-12 TO IMPROVE CLARITY AND AVOID MAGIC NUMBERS
__constant int MAX_CU_DEPTH = 7;
__constant int MV_FRACTIONAL_BITS_INTERNAL = 4;
__constant int MAX_CU_WIDTH = 128;
__constant int MAX_CU_HEIGHT = 128;

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
    
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2(pu_width))); 
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2(pu_width))); 
    int iDMvVerX = -iDMvHorY; // If it is 4 params, there is not vertically-neighboring CPs. Then, estimate it based on horizontal neighbors LT and RT
    int iDMvVerY = iDMvHorX;

    int iMvScaleHor = LT_x << shift;
    int iMvScaleVer = LT_y << shift;

    bool isSpread = isSubblockVectorSpreadOverLimit(iDMvHorX, iDMvHorY, iDMvVerX, iDMvVerY, bipred);

    int2 mv;

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
    
    int iDMvHorX = (RT_x - LT_x) << (shift - (int)floor(native_log2(pu_width))); 
    int iDMvHorY = (RT_y - LT_y) << (shift - (int)floor(native_log2(pu_width))); 
   
    int iDMvVerX = (LB_x - LT_x) << (shift - (int)floor(native_log2(pu_height))); 
    int iDMvVerY = (LB_y - LT_y) << (shift - (int)floor(native_log2(pu_height))); 
   
    int iMvScaleHor = LT_x << shift;
    int iMvScaleVer = LT_y << shift;

    bool isSpread = isSubblockVectorSpreadOverLimit(iDMvHorX, iDMvHorY, iDMvVerX, iDMvVerY, bipred);

    int2 mv;
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


__kernel void affine(__global int *subMVs_x, __global int *subMVs_y, __global int *LT_x, __global int *LT_y, __global int *RT_x, __global int *RT_y, __global int *LB_x, __global int *LB_y, __global int *pu_x, __global int *pu_y, __global int *pu_width, __global int *pu_height, __global int *subBlock_x, __global int *subBlock_y, __global bool *bipred, __global int *nCP, const int frameWidth, const int frameHeight){
    int gid = get_global_id(0);

    int2 subMv;
    // TODO: This if/else is undesired to improve performance. This will be solved when we consider "real inputs" instead of a chunck of mixed input data
    if(nCP[gid]==2){
        subMv = deriveMv2Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
    }
    else{
        subMv = deriveMv3Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], LB_x[gid], LB_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
    }

    subMv = roundAndClipMv(subMv, pu_x[gid], pu_y[gid],  pu_width[gid], pu_height[gid], frameWidth, frameHeight);
    
    subMVs_x[gid] = subMv.x;
    subMVs_y[gid] = subMv.y;

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