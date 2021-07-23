#include "aux_functions.cl"

#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#endif

__kernel void affine(__global int *subMVs_x, __global int *subMVs_y, __global int *LT_x, __global int *LT_y, __global int *RT_x, __global int *RT_y, __global int *LB_x, __global int *LB_y, __global int *pu_x, __global int *pu_y, __global int *pu_width, __global int *pu_height, __global int *subBlock_x, __global int *subBlock_y, __global bool *bipred, __global int *nCP, const int frameWidth, const int frameHeight, __global int *ref_samples, __global int *curr_samples, __global int *filtered_samples){
    int gid = get_global_id(0);

    int3 subMv_and_spread;
    int2 subMv;
    int isSpread;
    
    // Derive the MV for the sub-block based on CPMVs
    int16 deltaHorVec, deltaVerVec;

    if(nCP[gid]==2){
        subMv_and_spread = deriveMv2Cps_and_spread(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
        subMv = subMv_and_spread.xy;
        isSpread = subMv_and_spread.z;
        
        deltaHorVec = getHorizontalDeltasPROF2Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
        deltaVerVec = getVerticalDeltasPROF2Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
    }
    else{
        subMv_and_spread = deriveMv3Cps_and_spread(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], LB_x[gid], LB_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
        subMv = subMv_and_spread.xy;
        isSpread = subMv_and_spread.z;        
        
        deltaHorVec = getHorizontalDeltasPROF3Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], LB_x[gid], LB_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
        deltaVerVec = getVerticalDeltasPROF3Cps(LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], LB_x[gid], LB_y[gid], pu_width[gid], pu_height[gid], subBlock_x[gid], subBlock_y[gid], bipred[gid]);
    }

    subMv = roundAndClipMv(subMv, pu_x[gid], pu_y[gid],  pu_width[gid], pu_height[gid], frameWidth, frameHeight);

    // Store sub-mvs in memory
    // TODO: This may not be necessary depending on our final framework: sub-mvs will be tested to get the RD, and only the best will be stored
    subMVs_x[gid] = subMv.x;
    subMVs_y[gid] = subMv.y;
    
    // Split the int and fractional part of the MVs to perform prediction/filtering
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

    // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
    predBlock = horizontal_vertical_filter(ref_samples, (int2)(pu_x[gid]+subBlock_x[gid],pu_y[gid]+subBlock_y[gid]), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec);     
    
    // Write predicted samples into memory object
    int sub_x = subBlock_x[gid];
    int sub_y = subBlock_y[gid];
    int stride = pu_width[gid];

    int4 tmp;
    int offset;

    // TODO: Maybe it's not necessary to write the filtered samples here since they are not the final prediction, and the SATD is computed on a 4x4 basis
    tmp = predBlock.lo.lo;
    offset = (sub_y*stride + sub_x);
    vstore4(tmp,offset/4,filtered_samples); // Offset divided by 4 since we are writing int4

    tmp = predBlock.lo.hi;
    offset = ((sub_y+1)*stride + sub_x);
    vstore4(tmp,offset/4,filtered_samples);

    tmp = predBlock.hi.lo;
    offset = ((sub_y+2)*stride + sub_x);
    vstore4(tmp,offset/4,filtered_samples);

    tmp = predBlock.hi.hi;
    offset = ((sub_y+3)*stride + sub_x);
    vstore4(tmp,offset/4,filtered_samples);

    int16 original_block;

    // TODO: This can be fetched only once, and used to compute the distortion of all candidate blocks
    // Fetch original blocks from memory to compute distortion
    // Point to the top-left corner of the sub-block
    offset = (pu_y[gid] + sub_y)*frameWidth + pu_x[gid] + sub_x;
    original_block.lo.lo = vload4(offset/4, curr_samples);
    offset += frameWidth;
    original_block.lo.hi = vload4(offset/4, curr_samples);
    offset += frameWidth;
    original_block.hi.lo = vload4(offset/4, curr_samples);
    offset += frameWidth;
    original_block.hi.hi = vload4(offset/4, curr_samples);

    // Compute the SATD 4x4 for the current sub-block
    // TODO: This SATD must be stored in a shared matrix to compose the SATD of the complete CU/PU
    int satd = satd_4x4(original_block, predBlock);

    // If using AMVR, the AFFINE_MV_PRECISION_QUARTER constant should be updated with the proper AMVR index
    // TODO: these predicted (0,0) MVs should be substituted by the AMVP result in the future. The bitrate computation was verified with zero and non-zero predicted MVs
    // TODO: in a real scenario, the bitrate is computed only once for the entire PU/CU. This computation must be moved to an upper level in the call hierarchy
    int pred_LT_x = 0, pred_LT_y = 0, pred_RT_x = 0, pred_RT_y = 0, pred_LB_x = 0, pred_LB_y = 0;
    int affineMvBits = calc_affine_bits(AFFINE_MV_PRECISION_QUARTER, nCP[gid], LT_x[gid], LT_y[gid], RT_x[gid], RT_y[gid], LB_x[gid], LB_y[gid], pred_LT_x, pred_LT_y, pred_RT_x, pred_RT_y, pred_LB_x, pred_LB_y);
   

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