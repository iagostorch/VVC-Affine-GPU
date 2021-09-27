#include "aux_functions.cl"

#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#endif

__kernel void affine_gradient_128x128(__global int *referenceFrameSamples, __global int *currentFrameSamples,const int frameWidth, const int frameHeight, __global short *horizontalGrad, __global short *verticalGrad, __global long *global_pEqualCoeff, __global long *gBestCost, __global int *gBestLT_X, __global int *gBestLT_Y, __global int *gBestRT_X, __global int *gBestRT_Y, __global int *gBestLB_X, __global int *gBestLB_Y,  __global long *debug, __global short *retCU){
    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    // This is fixed for this function
    int cuWidth = 128, cuHeight = 128;

    // TODO: Improve this usage when porting to 3 CPs
    int nCP = 2;

    // Variables to keep track of the current and best MVs/costs
    __local int LT_Y, LT_X, RT_Y, RT_X, LB_Y, LB_X;
    __local int best_LT_X, best_LT_Y, best_RT_X, best_RT_Y, best_LB_X, best_LB_Y;
    __local long currCost, bestCost, bestDist;
    
    // TODO: Design a motion vector predicton algorithm. Currently MPV is always zero
    int pred_LT_X = 0;
    int pred_LT_Y = 0;
    int pred_RT_X = 0;
    int pred_RT_Y = 0;
    int pred_LB_X = 0;
    int pred_LB_Y = 0;

    // Hold a fraction of the total distortion of current CPMVs (each workitem uses one position)
    __local long local_cumulativeSATD[256];

    // Derive position of current CU based on WG
    int cusPerRow = ceil((float) frameWidth/cuWidth);
    int cuX = (wg%cusPerRow) * cuWidth;
    int cuY = (wg/cusPerRow) * cuHeight;

    // TODO: All these VECTORIZED_MEMORY if/elses are used to control memory access: either using vload/vstore operations, or indexing the values one by one
    // Fetch current CU for private memory. Only the samples used by this workitem are fetched to decrease memory bandwidth
    int16 currentCU_subBlock[4];
    int currentCU[128*128];
    if(VECTORIZED_MEMORY){
        for(int pass=0; pass<4; pass++){   
            int index = pass*256 + lid; // absolute index of sub-block inside the CU
            int sub_X, sub_Y;
            sub_Y = (index/32)<<2;
            sub_X = (index%32)<<2;
            int offset = (cuY + sub_Y)*frameWidth + cuX + sub_X;
            currentCU_subBlock[pass].lo.lo = vload4(offset/4, currentFrameSamples);
            offset += frameWidth;
            currentCU_subBlock[pass].lo.hi = vload4(offset/4, currentFrameSamples);
            offset += frameWidth;
            currentCU_subBlock[pass].hi.lo = vload4(offset/4, currentFrameSamples);
            offset += frameWidth;
            currentCU_subBlock[pass].hi.hi = vload4(offset/4, currentFrameSamples);
        }
    }
    else{
        for(int pass=0; pass<4; pass++){
            int index = pass*256 + lid; // absolute index of sub-block inside the CU
            int sub_X, sub_Y;
            sub_Y = (index/32)<<2;
            sub_X = (index%32)<<2;
            for(int i=sub_Y; i<sub_Y+4; i++){
                for(int j=sub_X; j<sub_X+4; j++){
                    currentCU[i*128+j] = currentFrameSamples[(cuY+i)*frameWidth + cuX+j];
                }
            }
        }
    }
    
    // At first, this holds the predicted samples for the entire CU
    // Then, the prediction error is stored here to accelerate building the system of equations of gradient-ME
    // Due to memory limitations it is not possible to declare two arrays with these dimensions
    __local short predCU_then_error[128*128];

    int3 subMv_and_spread;
    int2 subMv;
    int isSpread;
    int16 deltaHorVec, deltaVerVec, predBlock, original_block;
    int bipred=0, windowWidth=11, windowHeight=11, referenceWindow[11*11], satd;
    int enablePROF=0; // Enable or disable PROF after filtering (similar to --PROF=0/1)
    long cumulativeSATD;
    int bitrate = MAX_INT;
    int numGradientIter = 5; // Number of iteration in gradient ME search (i.e., number of CPMV updates after predicted MV)


    // TODO: Maybe it is faster to make all workitems write to the same local variable than using if()+barrier
    // Initial MVs from AMVP and initialize rd-cost
    if(lid==0){
        LT_X = pred_LT_X;
        LT_Y = pred_LT_Y;
        RT_X = pred_RT_X;
        RT_Y = pred_RT_Y;
        LB_X = pred_LB_X;
        LB_Y = pred_LB_Y;
        bestCost = MAX_LONG;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // sync all workitems after initializing the MV
    
    // Starts the motion estimation based on gradient and optical flow
    for(int iter=0; iter<numGradientIter+1; iter++){ // +1 because we need to conduct the initial prediction (with AMVP) in addition to the gradient-ME
        // ###############################################################################
        // ###### HERE IT STARTS THE PREDICTION OF THE BLOCK AND COMPUTES THE COSTS ######
        // ###############################################################################
  
        // Reset SATD for current iteration
        cumulativeSATD = 0;
        local_cumulativeSATD[lid] = 0;

        // Each workitem will predict 4 sub-blocks (1024 sub-blocks, 256 workitems)
        for(int pass=0; pass<4; pass++){   
            int index = pass*256 + lid; // absolute index of sub-block inside the CU
            int sub_X, sub_Y;
            sub_Y = (index/32)<<2;
            sub_X = (index%32)<<2;

            // Derive sub-MVs and determine if they are too spread (when spread, all sub-blocks have the same MV)
            subMv_and_spread = deriveMv2Cps_and_spread(LT_X, LT_Y, RT_X, RT_Y, cuWidth, cuHeight, sub_X, sub_Y, bipred);
            subMv = subMv_and_spread.xy;
            isSpread = subMv_and_spread.z;

            // These deltas are used during PROF
            deltaHorVec = getHorizontalDeltasPROF2Cps(LT_X, LT_Y, RT_X, RT_Y, cuWidth, cuHeight, sub_X, sub_Y, bipred);
            deltaVerVec = getVerticalDeltasPROF2Cps(LT_X, LT_Y, RT_X, RT_Y, cuWidth, cuHeight, sub_X, sub_Y, bipred);

            subMv = roundAndClipMv(subMv, cuX, cuY,  cuWidth, cuHeight, frameWidth, frameHeight);

            // Split the int and fractional part of the MVs to perform prediction/filtering
            // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
            int2 subMv_int, subMv_frac;
            subMv_int.x  = subMv.x >> 4;
            subMv_frac.x = subMv.x & 15;
            subMv_int.y  = subMv.y >> 4;
            subMv_frac.y = subMv.y & 15;

            // Top-left corner of the reference block in integer pixels
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

            // Fetch reference window (i.e., samples that may be used for filtering)
            bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
            int currSample; // Used to avoid if/else structures during left/right correction
            int properIdx; // This variable is used to update the index of the reference sample without accessing global memory until the proper index is found. It avoid segmentation fault when the motion vectors point outside the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
            
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
                    
                    // Index of reference sample in case there is no correction
                    properIdx = refPosition+row*srcStride+col;
                    
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
                    // Fetch the correct sample and store it on the referenceWindow
                    currSample = referenceFrameSamples[properIdx];
                    referenceWindow[row*windowWidth+col] = currSample;
                }
            }
            // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
            predBlock = horizontal_vertical_filter_new(referenceWindow, (int2)(cuX+sub_X,cuY+sub_Y), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     

            // Fetch samples of original sub-block to compute distortion
            if(!VECTORIZED_MEMORY){
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
            }
            

            // Store sub-block on __local array of entire CU
            predCU_then_error[(sub_Y+0)*128+sub_X+0] = predBlock.s0;
            predCU_then_error[(sub_Y+0)*128+sub_X+1] = predBlock.s1;
            predCU_then_error[(sub_Y+0)*128+sub_X+2] = predBlock.s2;
            predCU_then_error[(sub_Y+0)*128+sub_X+3] = predBlock.s3;
            predCU_then_error[(sub_Y+1)*128+sub_X+0] = predBlock.s4;
            predCU_then_error[(sub_Y+1)*128+sub_X+1] = predBlock.s5;
            predCU_then_error[(sub_Y+1)*128+sub_X+2] = predBlock.s6;
            predCU_then_error[(sub_Y+1)*128+sub_X+3] = predBlock.s7;
            predCU_then_error[(sub_Y+2)*128+sub_X+0] = predBlock.s8;
            predCU_then_error[(sub_Y+2)*128+sub_X+1] = predBlock.s9;
            predCU_then_error[(sub_Y+2)*128+sub_X+2] = predBlock.sa;
            predCU_then_error[(sub_Y+2)*128+sub_X+3] = predBlock.sb;
            predCU_then_error[(sub_Y+3)*128+sub_X+0] = predBlock.sc;
            predCU_then_error[(sub_Y+3)*128+sub_X+1] = predBlock.sd;
            predCU_then_error[(sub_Y+3)*128+sub_X+2] = predBlock.se;
            predCU_then_error[(sub_Y+3)*128+sub_X+3] = predBlock.sf;
            
            // Compute the SATD 4x4 for the current sub-block and accumulate on the private accumulator
            if(VECTORIZED_MEMORY){
                satd = satd_4x4(currentCU_subBlock[pass], predBlock);
            }
            else{
                satd = satd_4x4(original_block, predBlock);
            }
            cumulativeSATD += (long) satd;
        }
        // Each position of local_cumulativeSATD will hold the cumulativeSATD of the sub-block of current workitem
        // Wait until all workitems compute their cumulativeSATD and reduce (i.e., sum all SATDs) to the first position of local_cumulativeSATD
        local_cumulativeSATD[lid] = cumulativeSATD;
        barrier(CLK_LOCAL_MEM_FENCE);

        // TODO: This reduction may be accelerated using multiple workitems
        // Reduce the smaller SATDs, compute the cost and update CPMVs
        if(lid==0){
            for(int i=1; i<256; i++){
                local_cumulativeSATD[0] += local_cumulativeSATD[i];
            }

            // TODO: Review the implementation of calc_affine_bits(). Verify how it behaves when current MV is equal to predicted MV, and it is necessary to add an offset (ruiCost) to the number of bits when computing the cost
            bitrate = calc_affine_bits(AFFINE_MV_PRECISION_QUARTER, nCP, LT_X, LT_Y, RT_X, RT_Y, LB_X, LB_Y, pred_LT_X, pred_LT_Y, pred_RT_X, pred_RT_Y, pred_LB_X, pred_LB_Y);

            // TODO: These lambdas are valid when using low delay with a single reference frame. Improve this when using multiple reference frames
            float lambda_QP22 = 17.583905;
            float lambda_QP27 = 39.474532;
            float lambda_QP32 = 78.949063;
            float lambda_QP37 = 140.671239;

            // TODO: This "+4" represents the ruiBits of the VTM-12.0 encoder, and it is the base-bitrate for using affine. The "+4" when using low delay with a single reference frame. Improve this when using multiple reference frames
            currCost = local_cumulativeSATD[0] + (long) getCost(bitrate+4, lambda_QP37);

            // If the current CPMVs are not better than the previous (rd-cost wise), the best CPMVs are not updated but the next iteration continues from the current CPMVs
            if(currCost < bestCost){
                bestCost = currCost;
                bestDist = local_cumulativeSATD[0];
                best_LT_X = LT_X;
                best_LT_Y = LT_Y;
                best_RT_X = RT_X;
                best_RT_Y = RT_Y;
                best_LB_X = LB_X;
                best_LB_Y = LB_Y;
            }
        }

        // #########################################################################################################
        // ###### AT THIS POINT WE HAVE COMPUTED THE COST OF THE CURRENT CPMVs. IF IT IS THE LAST ITERATION   ######
        // ###### THERE IS NO NEED TO PERFORM THE GRADIENT REFINEMENT SINCE THE UPDATED MV WILL NOT BE TESTED ######
        // #########################################################################################################

        if(iter == numGradientIter)
            break;

        // #########################################################################################
        // ###### HERE IT STARTS COMPUTING THE GRADIENTS AND BUILDING THE SYSTEM OF EQUATIONS ######
        // #########################################################################################

        int centerSample;
        // Compute general case of gradient. It will be necessary to refill the border and corner values
        // Each workitem will compute the gradient on 64 positions (128x128 samples, 256 workitems)
        for(int pass=0; pass<64; pass++){
            centerSample = pass*256 + lid;
            // Border values are not valid
            int isValid = !((centerSample%cuWidth==0) || (centerSample%cuWidth==cuWidth-1) || (centerSample/cuWidth==0) || (centerSample/cuWidth==cuHeight-1));
            if(isValid){
                // Stride memory to location of current wg (1 CU per WG)
                // TODO: Verify if this stride works for smaller CU sizes
                horizontalGrad[wg*cuWidth*cuHeight + centerSample] = predCU_then_error[centerSample-cuWidth+1]-predCU_then_error[centerSample-cuWidth-1] + 2*predCU_then_error[centerSample+1]-2*predCU_then_error[centerSample-1] + predCU_then_error[centerSample+cuWidth+1]-predCU_then_error[centerSample+cuWidth-1];    
                verticalGrad[wg*cuWidth*cuHeight + centerSample] = predCU_then_error[centerSample+cuWidth-1]-predCU_then_error[centerSample-cuWidth-1] + 2*predCU_then_error[centerSample+cuWidth]-2*predCU_then_error[centerSample-cuWidth] + predCU_then_error[centerSample+cuWidth+1]-predCU_then_error[centerSample-cuWidth+1];
            }
            else{
                horizontalGrad[wg*cuWidth*cuHeight + centerSample] = 0;
                verticalGrad[wg*cuWidth*cuHeight + centerSample] = 0;
            }
        }
        
        // Wait until all workitems have computed their gradients
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Only necessary when we are exporting the predicted CU
        if(lid==0 && iter==1){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                }
            }
            
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        // TODO: use local IDs to accelerate the following computations as well
        if(lid==0){
            // Fills the first and last rows of gradient with the correct values (upper/lower row values)
            for(int col=0; col<cuWidth; col++){
                horizontalGrad[wg*cuWidth*cuHeight + col] = horizontalGrad[wg*cuWidth*cuHeight + col+cuWidth];
                horizontalGrad[wg*cuWidth*cuHeight + col+(cuHeight-1)*cuWidth] = horizontalGrad[wg*cuWidth*cuHeight + col+(cuHeight-2)*cuWidth];

                verticalGrad[wg*cuWidth*cuHeight + col] = verticalGrad[wg*cuWidth*cuHeight + col+cuWidth];
                verticalGrad[wg*cuWidth*cuHeight + col+(cuHeight-1)*cuWidth] = verticalGrad[wg*cuWidth*cuHeight + col+(cuHeight-2)*cuWidth];
            }
            // Fills the first and last columns of the gradient with the correct values (left/right column values)
            for(int row=0; row<cuHeight; row++){
                horizontalGrad[wg*cuWidth*cuHeight + row*cuWidth] = horizontalGrad[wg*cuWidth*cuHeight + row*cuWidth+1];
                horizontalGrad[wg*cuWidth*cuHeight + row*cuWidth+cuWidth-1] = horizontalGrad[wg*cuWidth*cuHeight + row*cuWidth+cuWidth-2];

                verticalGrad[wg*cuWidth*cuHeight + row*cuWidth] = verticalGrad[wg*cuWidth*cuHeight + row*cuWidth+1];
                verticalGrad[wg*cuWidth*cuHeight + row*cuWidth+cuWidth-1] = verticalGrad[wg*cuWidth*cuHeight + row*cuWidth+cuWidth-2];
            }
            // Fills the four corners
            horizontalGrad[wg*cuWidth*cuHeight + 0] = horizontalGrad[wg*cuWidth*cuHeight + cuWidth+1];
            verticalGrad[wg*cuWidth*cuHeight + 0] = verticalGrad[wg*cuWidth*cuHeight + cuWidth+1];
            horizontalGrad[wg*cuWidth*cuHeight + cuWidth-1] = horizontalGrad[wg*cuWidth*cuHeight + cuWidth+cuWidth-2];
            verticalGrad[wg*cuWidth*cuHeight + cuWidth-1] = verticalGrad[wg*cuWidth*cuHeight + cuWidth+cuWidth-2];
            horizontalGrad[wg*cuWidth*cuHeight + (cuHeight-1)*cuWidth] = horizontalGrad[wg*cuWidth*cuHeight + (cuHeight-2)*cuWidth+1];
            verticalGrad[wg*cuWidth*cuHeight + (cuHeight-1)*cuWidth] = verticalGrad[wg*cuWidth*cuHeight + (cuHeight-1)*cuWidth+1];
            horizontalGrad[wg*cuWidth*cuHeight + (cuHeight-1)*cuWidth+cuWidth-1] = horizontalGrad[wg*cuWidth*cuHeight + (cuHeight-2)*cuWidth+cuWidth-2];
            verticalGrad[wg*cuWidth*cuHeight + (cuHeight-1)*cuWidth+cuWidth-1] = verticalGrad[wg*cuWidth*cuHeight + (cuHeight-2)*cuWidth+cuWidth-2];        
        }
        
        // Wait until the border values are filled. They are required in the next stage
        barrier(CLK_LOCAL_MEM_FENCE);
                
        // Since the prediction will not be used again in the same iteration, it is possible to reuse the __local array to store the error (pred = pred-orig). The error will be used to build the system of equations
        // Each workitem computes the error of a subset of sub-blocks (the same sub-blocks it predicted earlier)
        if(VECTORIZED_MEMORY){
            for(int pass=0; pass<4; pass++){   
                int index = pass*256 + lid; // absolute index of sub-block inside the CU
                int sub_X, sub_Y;
                sub_Y = (index/32)<<2;
                sub_X = (index%32)<<2;

                int cuOffset = sub_Y*cuWidth + sub_X;
                short4 linePred, lineDiff;
                
                // Read one line from predicted sub-block, subtract from original CU, overwrite predCU_then_error
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
                
                cuOffset += cuWidth;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += cuWidth;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += cuWidth;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
            }   
        }
        else{
            for(int pass=0; pass<4; pass++){
                int index = pass*256 + lid; // absolute index of sub-block inside the CU
                int sub_X, sub_Y;
                sub_Y = (index/32)<<2;
                sub_X = (index%32)<<2;
                for(int i=0; i<4; i++){
                    for(int j=0; j<4; j++){
                        predCU_then_error[(sub_Y+i)*128 + sub_X+j] = currentCU[(sub_Y+i)*128 + sub_X+j] - predCU_then_error[(sub_Y+i)*128 + sub_X+j];
                    }
                }
            }
        } 
        
        // Holds the "complete" system of equations to be solved during gradient-ME. For 2 CPs, only the first [5][4] indices are used.
        __local long local_pEqualCoeff[7][6]; 
        // The long variable is used when building the system of equations (only integers). The double variable is used for solving the system
        __private long private_pEqualCoeff[7][6];
        __private double private_dEqualCoeff[7][6];
        // The double variable is purely based on the affine parameters, the int variable is used after rounding the double to the proper precision
        double8 dDeltaMv = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int8 intDeltaMv;

        // TODO: Improve this to be adaptive to number of CPs
        // Initialize the system of equations
        for(int i=0; i<7; i++){
            for(int j=0; j<6; j++){
                private_pEqualCoeff[i][j] = 0;
            }
        }

        // Build the system of equations
        long tmp_atomic;
        // Computes a fraction of the entire system on private memory for the current workitem
        // It is necessary to go through the 128x128 samples, therefore each workitem processes 64 positions (128x128 divided by 256 workitems)
        for(int pass=0; pass<64; pass++){           
            int iC[6], j, k, cx, cy;
            int idx = pass*256 + lid; // Absolute position of current sample inside the CU
            j = idx/128;
            k = idx%128;
            cy = ((j >> 2) << 2) + 2; // cy represents the center of the sub-block (4x4) of current sample. If sample is in y=11, cy=10 (the center of the third block: 2, 6, 10, 14, ...)
            cx = ((k >> 2) << 2) + 2; // cx represents the center of the sub-block (4x4) of current sample. If sample is in x=4, cx=6 (the center of the second block: 2, 6, 10, 14, ...)
            
            // TODO: Improve this to avoid using if/else (maybe not necessary since all workitems will be true or false)
            if(nCP==2){
                iC[0] = horizontalGrad[wg*cuWidth*cuHeight + idx];
                iC[1] = cx * horizontalGrad[wg*cuWidth*cuHeight + idx] + cy * verticalGrad[wg*cuWidth*cuHeight + idx];
                iC[2] = verticalGrad[wg*cuWidth*cuHeight + idx];
                iC[3] = cy * horizontalGrad[wg*cuWidth*cuHeight + idx] - cx * verticalGrad[wg*cuWidth*cuHeight + idx];
            }
            else{
                iC[0] = horizontalGrad[wg*cuWidth*cuHeight + idx];
                iC[1] = cx * horizontalGrad[wg*cuWidth*cuHeight + idx];
                iC[2] = verticalGrad[wg*cuWidth*cuHeight + idx];
                iC[3] = cx * verticalGrad[wg*cuWidth*cuHeight + idx];
                iC[4] = cy * horizontalGrad[wg*cuWidth*cuHeight + idx];
                iC[5] = cy * verticalGrad[wg*cuWidth*cuHeight + idx];;
            }
            
            // TODO: Test if using atomic operations (adding the sub-systems) directly over global memory improves performance
            for(int col=0; col<2*nCP; col++){
                for(int row=0; row<2*nCP; row++){
                    tmp_atomic = (long)iC[col] * (long)iC[row];
                    private_pEqualCoeff[col + 1][row] += tmp_atomic;
                    // pEqualCoeff[col + 1][row] += (int64_t)iC[col] * iC[row]; 
                }
                tmp_atomic = ((long)iC[col] * (long)predCU_then_error[idx]) << 3;
                private_pEqualCoeff[col + 1][2*nCP] += tmp_atomic;
                // pEqualCoeff[col + 1][affineParamNum] += ((int64_t)iC[col] * pResidue[idx]) << 3;
            }
        }

        // TODO: Improve this to be adaptive to number of CPs
        // Copy the private fraction of the system to the global memory
        for(int col=0; col<7; col++){
            for(int row=0; row<6; row++){
                // Strides to the memory location of this workgroup by wg*256*7*6 and to the location of this workitem by id*7*6
                global_pEqualCoeff[wg*256*7*6 + lid*7*6 + col*6 + row] = private_pEqualCoeff[col][row];
            }
        }

        // Wait until all workitems have copied their systems to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // #########################################################################
        // ###### HERE IT SOLVES THE SYSTEM OF EQUATIONS AND UPDATE THE CPMVs ######
        // #########################################################################

        // TODO: Try to distribute this computation over multiple workitems
        // Reduce all partial sums of global memory into local memory (maybe try using atomic operations previously?), solve the system of equations, compute the deltaMV
        // All this computation is handled by a single workitem
        if(lid==0){
            // Copy values from first workitem
            // TODO: Make these for loops adaptive to number of CPs
            for(int col=0; col<7; col++){
                for(int row=0; row<6; row++){
                    // Stride memory to the local of this WG
                    local_pEqualCoeff[col][row] = global_pEqualCoeff[wg*256*7*6 + col*6 + row];
                }
            }
            // Reduce remaining workitems by adding over the first values
            for(int item=1; item<256; item++){
                // TODO: Make these for loops adaptive to number of CPs
                for(int col=0; col<7; col++){
                    for(int row=0; row<6; row++){
                        local_pEqualCoeff[col][row] += global_pEqualCoeff[wg*256*7*6 + item*7*6 + col*6 + row];
                    }
                }
            }
            // TODO: Make these for loops adaptive to number of CPs
            // Copy the system to private memory using double type to solve the system in sequence
            for(int col=0; col<7; col++){
                for(int row=0; row<6; row++){
                    private_dEqualCoeff[col][row] = (double) local_pEqualCoeff[col][row];
                }
            }

            // BEGIN solving the system of equations
            int iOrder = 2*nCP;
            double dAffinePara[6];

            // Initialize parameters
            for ( int k = 0; k < iOrder; k++ )
            {
                dAffinePara[k] = 0.;
            }
            // The following code is directly copied from the function solveEqual() in VTM-12.0. Some parts of the function were discarded (commented) because they would early-terminate the ME
            // TODO: How could we early terminate the kernel without losing the desired results? A "return" statement on the following code would represent that the system is already solved and the delta is zero
            // row echelon
            for ( int i = 1; i < iOrder; i++ )
            {
                // find column max
                double temp = fabs(private_dEqualCoeff[i][i-1]);
                int tempIdx = i;
                for ( int j = i+1; j < iOrder+1; j++ )
                {
                    if ( fabs(private_dEqualCoeff[j][i-1]) > temp )
                    {
                        temp = fabs(private_dEqualCoeff[j][i-1]);
                        tempIdx = j;
                    }
                }

                // swap line
                if ( tempIdx != i )
                {
                    for ( int j = 0; j < iOrder+1; j++ )
                    {
                        private_dEqualCoeff[0][j] = private_dEqualCoeff[i][j];
                        private_dEqualCoeff[i][j] = private_dEqualCoeff[tempIdx][j];
                        private_dEqualCoeff[tempIdx][j] = private_dEqualCoeff[0][j];
                    }
                }

                // // elimination first column
                // if ( private_dEqualCoeff[i][i - 1] == 0. )
                // {
                //     return;
                // }

                for ( int j = i+1; j < iOrder+1; j++ )
                {
                    for ( int k = i; k < iOrder+1; k++ )
                    {
                        private_dEqualCoeff[j][k] = private_dEqualCoeff[j][k] - private_dEqualCoeff[i][k] * private_dEqualCoeff[j][i-1] / private_dEqualCoeff[i][i-1];
                    }
                }
            }

            // if ( private_dEqualCoeff[iOrder][iOrder - 1] == 0. )
            // {
            //     return;
            // }

            dAffinePara[iOrder-1] = private_dEqualCoeff[iOrder][iOrder] / private_dEqualCoeff[iOrder][iOrder-1];
            for ( int i = iOrder-2; i >= 0; i-- )
            {
                if ( private_dEqualCoeff[i + 1][i] == 0. )
                {
                    for ( int k = 0; k < iOrder; k++ )
                    {
                        dAffinePara[k] = 0.;
                    }
                    // return;
                }
                double temp = 0;
                for ( int j = i+1; j < iOrder; j++ )
                {
                    temp += private_dEqualCoeff[i+1][j] * dAffinePara[j];
                }
                dAffinePara[i] = ( private_dEqualCoeff[i+1][iOrder] - temp ) / private_dEqualCoeff[i+1][i];
            }
            
            // Copy the affine parameters derived from the system to the deltaMVs
            dDeltaMv.s0 = dAffinePara[0];
            dDeltaMv.s2 = dAffinePara[2];        
            if(nCP == 2){
                dDeltaMv.s1 = dAffinePara[1] * cuWidth + dAffinePara[0];
                dDeltaMv.s3 = -dAffinePara[3] * cuWidth + dAffinePara[2]; 
            }
            else{
                dDeltaMv.s1 = dAffinePara[1] * cuWidth + dAffinePara[0];
                dDeltaMv.s3 = dAffinePara[3] * cuWidth + dAffinePara[2];
                dDeltaMv.s4 = dAffinePara[4] * cuHeight + dAffinePara[0];
                dDeltaMv.s5 = dAffinePara[5] * cuHeight + dAffinePara[2];            
            }

            // Scale the fractional delta MVs (double type) to integer type
            intDeltaMv = scaleDeltaMvs(dDeltaMv, nCP, cuWidth, cuHeight);

            // Update the current MVs and clip to allowed values
            LT_X += intDeltaMv.s0;
            LT_Y += intDeltaMv.s1;
            RT_X += intDeltaMv.s2;
            RT_Y += intDeltaMv.s3;
            LB_X += intDeltaMv.s4;
            LB_Y += intDeltaMv.s5;

            LT_X = clamp(LT_X, MV_MIN, MV_MAX);
            LT_Y = clamp(LT_Y, MV_MIN, MV_MAX);
            RT_X = clamp(RT_X, MV_MIN, MV_MAX);
            RT_Y = clamp(RT_Y, MV_MIN, MV_MAX);
            LB_X = clamp(LB_X, MV_MIN, MV_MAX);
            LB_Y = clamp(LB_Y, MV_MIN, MV_MAX);
            
            int2 tmp_mv;

            tmp_mv = clipMv((int2)(LT_X,LT_Y),cuX, cuY, cuWidth, cuHeight, frameWidth, frameHeight);
            LT_X = tmp_mv.s0;
            LT_Y = tmp_mv.s1;
            tmp_mv = clipMv((int2)(RT_X,RT_Y),cuX, cuY, cuWidth, cuHeight, frameWidth, frameHeight);
            RT_X = tmp_mv.s0;
            RT_Y = tmp_mv.s1;
            tmp_mv = clipMv((int2)(LB_X,LB_Y),cuX, cuY, cuWidth, cuHeight, frameWidth, frameHeight);
            LB_X = tmp_mv.s0;
            LB_Y = tmp_mv.s1;      

            // The next iteration will perform prediction with the new CPMVs and verify if this improves performance

        } // if(lid==0) for solving the system and updating the MVs
        
        // All workitems must wait until the CPMVs are updated before starting the next prediction
        barrier(CLK_LOCAL_MEM_FENCE);

    } // end for the five iterations of gradient
    
    // Write the best CPMVs and corresponding costs to global memory and return to host
    if(lid==0){
        // printf("BEST MVs WG: %d \t Pos %dx%d\n  SATD %ld\n  Cost %ld\n  LT %dx%d\n  RT %dx%d\n  LB %dx%d\n\n", wg, cuX, cuY, bestDist, bestCost, best_LT_X, best_LT_Y, best_RT_X, best_RT_Y, LB_X, best_LB_Y);
        gBestCost[wg] = bestCost;
        gBestLT_X[wg] = best_LT_X;
        gBestLT_Y[wg] = best_LT_Y;
        gBestRT_X[wg] = best_RT_X;
        gBestRT_Y[wg] = best_RT_Y;
        gBestLB_X[wg] = best_LB_X;
        gBestLB_Y[wg] = best_LB_Y;
    }
       
}

// This is a naive implementation of affine for CUs 128x128 with 2 CPs
// A set of 16 WGs will process a single CU, and each WG contains 256 work-items
// Each work-item will perform the entire prediciton (all sub-blocks) for a subset of the motion vectors
// MVs are in the range [-8,+7] considering 1/4 precision (between -2 and +1,75)
__kernel void naive_affine_2CPs_CU_128x128(__global int *referenceFrameSamples, __global int *currentFrameSamples,const int frameWidth, const int frameHeight, __global int *wgSATDs, __global int *debug, __global int *retCU, __global short *wg_LT_X, __global short *wg_LT_Y, __global short *wg_RT_X, __global short *wg_RT_Y){
    // TODO REMINDER: These __local arrays were kernel parameters but they cause MASSIVE SLOWDOWN on the execution. Declaring them inside the kernel provides gigantic speedups
    // The lenght of 256 is the maximum workgroup size of the GPU
    __local int wgSATD[256];
    __local short wgLtX[256];
    __local short wgLtY[256];
    __local short wgRtX[256];
    __local short wgRtY[256];
    __local short currentCU[128*128];

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
    short bestLT_X, bestLT_Y, bestRT_X, bestRT_Y;

    // // This can be improved by saving the samples in zig-zag order, so all 16 samples are in sequence
    for(int i=0; i<64; i++){ // 128x128 samples being fetched by 256 workitems requires 64 cycles. Each cycle fetches 2 lines of samples
            currentCU[i*256+lid] = currentFrameSamples[(cuY+i*2+lid/128)*frameWidth + cuX + lid%128];
    }
    /*
    for(int i=0; i<128; i++){
        for(int j=0; j<128; j++){
            retCU[i*128+j]=currentCU[i*128+j];
        }
    }
    */
    
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
        
    short LT_X, LT_Y, RT_X, RT_Y;

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
                            // predBlock = horizontal_vertical_filter_new_unrolled(referenceWindow, (int2)(cuX+sub_X,cuY+sub_Y), subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     

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
                    bestLT_X = select(bestLT_X, LT_X, (short)(cumulativeSATD<bestSATD));
                    bestLT_Y = select(bestLT_Y, LT_Y, (short)(cumulativeSATD<bestSATD));
                    bestRT_X = select(bestRT_X, RT_X, (short)(cumulativeSATD<bestSATD));
                    bestRT_Y = select(bestRT_Y, RT_Y, (short)(cumulativeSATD<bestSATD));
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
    short final_LT_X = bestLT_X;
    short final_LT_Y = bestLT_Y;
    short final_RT_X = bestRT_X;
    short final_RT_Y = bestRT_Y;

    // TODO: Look for a way to improve this search
    // Only one workitem scans the WG array to find the best CPMV
    if(lid==0){
        for(int i=0; i<wgSize; i++){
            final_satd = select(final_satd,wgSATD[i], wgSATD[i]<final_satd);
            
            final_LT_X = select(final_LT_X,wgLtX[i], (short)(wgSATD[i]<final_satd));
            final_LT_Y = select(final_LT_Y,wgLtY[i], (short)(wgSATD[i]<final_satd));
            final_RT_X = select(final_RT_X,wgRtX[i], (short)(wgSATD[i]<final_satd));
            final_RT_Y = select(final_RT_Y,wgRtY[i], (short)(wgSATD[i]<final_satd));
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