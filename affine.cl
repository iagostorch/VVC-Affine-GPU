#include "aux_functions.cl"

#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#endif

// This is a naive implementation of affine for CUs 128x128 with 2 CPs
// A set of 16 WGs will process a single CU, and each WG contains 256 work-items
// Each work-item will perform the entire prediciton (all sub-blocks) for a subset of the motion vectors
// MVs are in the range [-8,+7] considering 1/4 precision (between -2 and +1,75)
__kernel void naive_affine_2CPs_CU_128x128(__global int *referenceFrameSamples, __global int *currentFrameSamples,const int frameWidth, const int frameHeight, __global int *wgSATDs, __global int *debug, __global int *retCU, __global int *wg_LT_X, __global int *wg_LT_Y, __global int *wg_RT_X, __global int *wg_RT_Y){
    // TODO REMINDER: These __local arrays were kernel parameters but they cause MASSIVE SLOWDOWN on the execution. Declaring them inside the kernel provides gigantic speedups
    // The lenght of 256 is the maximum workgroup size of the GPU
    __local int wgSATD[256];
    __local int wgLtX[256];
    __local int wgLtY[256];
    __local int wgRtX[256];
    __local int wgRtY[256];

    // Variables for indeXing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    int blockWidth = 128;
    int blockHeight = 128;

    // TODO: Currently, encoding only the first CU
    // Variables for indexing the current CU inside the frame
    int cuX, cuY;
    cuX = 0;
    cuY = 0;
    int cuCorner = cuX + cuY*frameWidth; // Corner of the CU inside the reference frame

    // Variables to store the subMv, variables for PROF, predicted sub-block and partial SATD
    int16 predictedBlock, original_block;
    int cumulativeSATD = 0, isSpread;
    int3 subMv_and_spread;
    int2 subMv;
    int16 deltaHorVec, deltaVerVec;
    int enablePROF=1;
    int bipred = 0;
    int offset;
    int satd;
    int bestSATD = MAX_INT;
    int bestLT_X, bestLT_Y, bestRT_X, bestRT_Y;

    // // This can be improved by saving the samples in zig-zag order, so all 16 samples are in sequence
    // TODO: Use currentCU in __local memory and fill it using all workitems concurrently
    int currentCU[128*128];
    for(int i=0; i<128; i++){
        for(int j=0; j<128; j++){
            currentCU[i*128+j] = currentFrameSamples[(cuY+i)*frameWidth + cuX + j];
        }
    }

    // Each workitem uses a fixed value for 3 parameters based on their gid, and tests multiple variations of a 4th parameter
    // TODO: This and/shift structure works for the current scenario testing MVs [-8,7], that require 4 bits each
    int first_LT_Y = (gid & 15) - 8; // least 3,2,1,0
    int last_LT_Y  = first_LT_Y;
    int first_LT_X = ((gid & 240)>>4) - 8; // bits 7,6,5,4
    int last_LT_X  = first_LT_X;
    int first_RT_Y = ((gid & 3840)>>8) - 8; // bits 11,10,9,8
    int last_RT_Y  = first_RT_Y;
    int first_RT_X = -8;
    int last_RT_X  = 7;
        
    int LT_X, LT_Y, RT_X, RT_Y;

    int16 predBlock; // Represents the current 4x4 block, containinig 16 samples
    
    int referenceWindow[11*11]; // For horizontal+vertical filter, the reference window comprehends the colocated block, plus 3 columns to the left and 4 columns to the right, 3 rows above and 4 rows below
    int windowWidth  = 11;
    int windowHeight = 11;

    // Used to export the predicted CU back to host
    // int predicted_CU[128*128];

    // This loop tests 16 possibilities for each MV, [-8,+7]
    // The inside multiply this number by 4 to get the actual MV using 1/16 precision. -8 actually represents the MV (-8 << 2) / 16 = -2.  7 represents (7 << 2) / 16 = 1.75
    for(int int_LT_Y=first_LT_Y; int_LT_Y<=last_LT_Y; int_LT_Y+=1){
        for(int int_LT_X=first_LT_X; int_LT_X<=last_LT_X; int_LT_X+=1){            
            for(int int_RT_Y=first_RT_Y; int_RT_Y<=last_RT_Y; int_RT_Y+=1){
                for(int int_RT_X=first_RT_X; int_RT_X<=last_RT_X; int_RT_X+=1){   
                    // These are the MVs in 1/16 precision
                    LT_X = int_LT_X << 2;
                    LT_Y = int_LT_Y << 2;
                    RT_X = int_RT_X << 2;
                    RT_Y = int_RT_Y << 2;   
                                     
                    // Filter each sub-block and compute the SATD
                    for(int sub_Y=0; sub_Y<128; sub_Y+=4){
                        for(int sub_X=0; sub_X<128; sub_X+=4){                           
                            subMv_and_spread = deriveMv2Cps_and_spread(LT_X, LT_Y, RT_X, RT_Y, blockWidth, blockHeight, sub_X, sub_Y, bipred);
                            subMv = subMv_and_spread.xy;
                            isSpread = subMv_and_spread.z;
                            
                            deltaHorVec = getHorizontalDeltasPROF2Cps(LT_X, LT_Y, RT_X, RT_Y, blockWidth, blockHeight, sub_X, sub_Y, bipred);
                            deltaVerVec = getVerticalDeltasPROF2Cps(LT_X, LT_Y, RT_X, RT_Y, blockWidth, blockHeight, sub_X, sub_Y, bipred);

                            subMv = roundAndClipMv(subMv, cuX, cuY,  blockWidth, blockHeight, frameWidth, frameHeight);

                            // Split the int and fractional part of the MVs to perform prediction/filtering
                            // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
                            int2 subMv_int, subMv_frac;

                            subMv_int.x  = subMv.x >> 4;
                            subMv_frac.x = subMv.x & 15;
                            subMv_int.y  = subMv.y >> 4;
                            subMv_frac.y = subMv.y & 15;

                            int refPosition = (cuY+sub_Y)*frameWidth + cuX+sub_X + subMv_int.y*frameWidth + subMv_int.x;

                            // Stride of the input (reference frame) and destination (4x4 block)
                            int srcStride = frameWidth;
                            int dstStride = 4; // Sub-block width

                            int N=NTAPS_LUMA; // N is the number of taps 8

                            // Ref position now points 3 rows above the reference block
                            refPosition = refPosition - ((NTAPS_LUMA >> 1) - 1) * srcStride; // This puts the pointer 3 lines above the referene block: we must make horizontal interpolation of these samples above the block to use them in the vertical interpolation in sequence
                            
                            // Stride for horizontal filtering is always 1
                            int cStride = 1;
                            refPosition -= ( N/2 - 1 ) * cStride; // Point 3 columns to the left of reference block (plus 3 rows above, from prev operation) to get neighboring samples
                            
                            // These slack and corrections are used to fetch the correct reference window when the MVs point outside the frame
                            // Positive left slack means we have enough columns to the left. Negative represents the number of columns outside (to the left) the frame
                            int leftSlack = cuX+sub_X + subMv_int.x - ( N/2 - 1 );
                            // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
                            int rightSpam = cuX+sub_X + subMv_int.x + ( N/2 );
                            int rightSlack = frameWidth - 1 - rightSpam;
                            // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
                            int topSlack = cuY+sub_Y + subMv_int.y - ( N/2 - 1 );
                            // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
                            int bottomSpam = cuY+sub_Y + subMv_int.y + ( N/2 );
                            int bottomSlack = frameHeight - 1 - bottomSpam;

                            // BEGIN Fetch reference window
                            bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
                            int currSample; // Used to avoid if/else structures during left/right correction
                            int properIdx; // This variable is used to update the index of the reference sample until the proper index is found. It is used when the motion vectors point outisde the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
                            // TODO: Unroll the following loop (or not?)
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

                                    // This if causes a massive slowdown even when all threads evaluate false.
                                    // Found out that the print statements require a lot of resources from the GPU in https://stackoverflow.com/questions/68744614/opencl-non-diverging-if-followed-by-printf-causing-massive-slowdown-and-kern
                                    // if(leftCorrect + rightCorrect + topCorrect + bottomCorrect + topLeftCorrect + topRightCorrect + bottomLeftCorrect + bottomRightCorrect > 1){
                                    //     printf("@@@@@\nFATAL ERROR: Multiple corrections in gid=%d\n",get_local_id(0));
                                    //     printf("L  %d\n", leftCorrect);
                                    //     printf("R  %d\n", rightCorrect);
                                    //     printf("T  %d\n", topCorrect);
                                    //     printf("B  %d\n", bottomCorrect);

                                    //     printf("TL %d\n", topLeftCorrect);
                                    //     printf("TR %d\n", topRightCorrect);
                                    //     printf("BL %d\n", bottomLeftCorrect);
                                    //     printf("BR %d\n", bottomRightCorrect);
                                    // }
                                    
                                    // Index of reference sample in case there is no correction
                                    properIdx = refPosition+row*srcStride+col; // Position before correction
                                    
                                    // Tests individual corrections
                                    properIdx = select(properIdx,refPosition+row*srcStride-leftSlack,leftCorrect==true);
                                    properIdx = select(properIdx,refPosition+row*srcStride+7+rightSlack,rightCorrect==true); // This (+7+slack) returns the pointer to the right-edge of the frame (slack is always negative,)
                                    properIdx = select(properIdx,refPosition+(-topSlack)*srcStride+col,topCorrect==true);
                                    properIdx = select(properIdx,refPosition+(7+bottomSlack)*srcStride+col,bottomCorrect==true);
                                    
                                    // Tests compound corrections
                                    properIdx = select(properIdx,0,topLeftCorrect==true);
                                    properIdx = select(properIdx,frameWidth-1,topRightCorrect==true);
                                    properIdx = select(properIdx,(frameHeight-1)*srcStride,bottomLeftCorrect==true);
                                    properIdx = select(properIdx,frameWidth*frameHeight-1,bottomRightCorrect==true);
               
                                    // TODO: Improve this to avoid several global memory access
                                    currSample = referenceFrameSamples[properIdx];
               
                                    referenceWindow[row*windowWidth+col] = currSample;
                                }
                            }

                            // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
                            // predBlock = horizontal_vertical_filter(referenceFrameSamples, (int2)(cuX+sub_X,cuY+sub_Y), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     
                            predBlock = horizontal_vertical_filter_new(referenceWindow, (int2)(cuX+sub_X,cuY+sub_Y), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     

                            // Fetch samples of original sub-block to compute distortion
                            original_block.s0 = currentCU[(sub_Y+0)*128+sub_X+0];
                            original_block.s1 = currentCU[(sub_Y+0)*128+sub_X+1];
                            original_block.s2 = currentCU[(sub_Y+0)*128+sub_X+2];
                            original_block.s3 = currentCU[(sub_Y+0)*128+sub_X+3];
                            original_block.s4 = currentCU[(sub_Y+1)*128+sub_X+0];
                            original_block.s5 = currentCU[(sub_Y+1)*128+sub_X+1];
                            original_block.s6 = currentCU[(sub_Y+1)*128+sub_X+2];
                            original_block.s7 = currentCU[(sub_Y+1)*128+sub_X+3];
                            original_block.s8 = currentCU[(sub_Y+2)*128+sub_X+0];
                            original_block.s9 = currentCU[(sub_Y+2)*128+sub_X+1];
                            original_block.sa = currentCU[(sub_Y+2)*128+sub_X+2];
                            original_block.sb = currentCU[(sub_Y+2)*128+sub_X+3];
                            original_block.sc = currentCU[(sub_Y+3)*128+sub_X+0];
                            original_block.sd = currentCU[(sub_Y+3)*128+sub_X+1];
                            original_block.se = currentCU[(sub_Y+3)*128+sub_X+2];
                            original_block.sf = currentCU[(sub_Y+3)*128+sub_X+3];
                            
                            /* Used to export predicted CU back to host
                            predicted_CU[(sub_Y+0)*128+sub_X+0] = predBlock.s0;
                            predicted_CU[(sub_Y+0)*128+sub_X+1] = predBlock.s1;
                            predicted_CU[(sub_Y+0)*128+sub_X+2] = predBlock.s2;
                            predicted_CU[(sub_Y+0)*128+sub_X+3] = predBlock.s3;
                            predicted_CU[(sub_Y+1)*128+sub_X+0] = predBlock.s4;
                            predicted_CU[(sub_Y+1)*128+sub_X+1] = predBlock.s5;
                            predicted_CU[(sub_Y+1)*128+sub_X+2] = predBlock.s6;
                            predicted_CU[(sub_Y+1)*128+sub_X+3] = predBlock.s7;
                            predicted_CU[(sub_Y+2)*128+sub_X+0] = predBlock.s8;
                            predicted_CU[(sub_Y+2)*128+sub_X+1] = predBlock.s9;
                            predicted_CU[(sub_Y+2)*128+sub_X+2] = predBlock.sa;
                            predicted_CU[(sub_Y+2)*128+sub_X+3] = predBlock.sb;
                            predicted_CU[(sub_Y+3)*128+sub_X+0] = predBlock.sc;
                            predicted_CU[(sub_Y+3)*128+sub_X+1] = predBlock.sd;
                            predicted_CU[(sub_Y+3)*128+sub_X+2] = predBlock.se;
                            predicted_CU[(sub_Y+3)*128+sub_X+3] = predBlock.sf;
                            */
                            
                            // Compute the SATD 4x4 for the current sub-block
                            satd = satd_4x4(original_block, predBlock);
                            cumulativeSATD += satd;
                        }
                    }
                    // Update best CPMV and SATD
                    bestLT_X = select(bestLT_X, LT_X, cumulativeSATD<bestSATD);
                    bestLT_Y = select(bestLT_Y, LT_Y, cumulativeSATD<bestSATD);
                    bestRT_X = select(bestRT_X, RT_X, cumulativeSATD<bestSATD);
                    bestRT_Y = select(bestRT_Y, RT_Y, cumulativeSATD<bestSATD);
                    bestSATD = select(bestSATD, cumulativeSATD, cumulativeSATD<bestSATD);
                    cumulativeSATD = 0;
                }
            }
        }
    }
    
    // Write best SATD and CPMV of this work-item to local buffer
    wgSATD[lid] = bestSATD;
    wgLtX[lid] = bestLT_X;
    wgLtY[lid] = bestLT_Y;
    wgRtX[lid] = bestRT_X;
    wgRtY[lid] = bestRT_Y;

    // Waint until all work-items have finished
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Create temp variables to hold the "current" best while scanning the array
    int final_satd = bestSATD;
    int final_LT_X = bestLT_X;
    int final_LT_Y = bestLT_Y;
    int final_RT_X = bestRT_X;
    int final_RT_Y = bestRT_Y;

    // TODO: Look for a way to improve this search
    // Only one workitem scans the WG array to find the best CPMV
    if(lid==0){
        for(int i=0; i<wgSize; i++){
            final_satd = select(final_satd,wgSATD[i], wgSATD[i]<final_satd);
            
            final_LT_X = select(final_LT_X,wgLtX[i], wgSATD[i]<final_satd);
            final_LT_Y = select(final_LT_Y,wgLtY[i], wgSATD[i]<final_satd);
            final_RT_X = select(final_RT_X,wgRtX[i], wgSATD[i]<final_satd);
            final_RT_Y = select(final_RT_Y,wgRtY[i], wgSATD[i]<final_satd);
        }
        // After finding the "real" best for this workgroup, write it to global array and return to host
        wgSATDs[wg] = final_satd;
        wg_LT_X[wg] = final_LT_X;
        wg_LT_Y[wg] = final_LT_Y;
        wg_RT_X[wg] = final_RT_X;
        wg_RT_Y[wg] = final_RT_Y;
    }  
}


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
    predBlock = horizontal_vertical_filter(ref_samples, (int2)(pu_x[gid]+subBlock_x[gid],pu_y[gid]+subBlock_y[gid]), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec,1);     
    
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