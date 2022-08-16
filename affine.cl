#include "aux_functions.cl"

#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

#endif

__kernel void affine_gradient_mult_sizes_2CPs(__global short *referenceFrameSamples, __global short *currentFrameSamples,const int frameWidth, const int frameHeight, const float lambda, __global short *horizontalGrad, __global short *verticalGrad, __global long *global_pEqualCoeff, __global long *gBestCost, __global Cpmvs *gBestCpmvs, __global long *debug, __global short *retCU){
    // TODO: Improve this to use 2 and 3 CPs
    int nCP = 2;
    // Used to debug the information of specific workitems and encoding stages
    int targetIter = 0;
    int targetWg = 134;
    int targetCuIdx = 1;
    int targetLid = 224;

    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    // Based on the WG index, the correct block size is selected
    // This assumes that each WG will process a single CU size inside a single CTU, considering ALIGNED CUs only
    int cuWidth = WIDTH_LIST[wg%NUM_CU_SIZES];
    int cuHeight = HEIGHT_LIST[wg%NUM_CU_SIZES];
    int cusPerCtu = (128*128)/(cuWidth*cuHeight);
    int itemsPerCu = wgSize/cusPerCtu;
    int subBlocksPerCu = (cuWidth/4)*(cuHeight/4);
    int subBlockColumnsPerCu = cuWidth/4;
    int subBlockRowsPerCu = cuHeight/4;
    int cuIdx = lid/itemsPerCu;

    
    // Index of current CTU. Every NUM_CU_SIZES workgroups share the same ctuIdx: one for each CU size
    int ctuIdx = wg/NUM_CU_SIZES;
    // Derive position of current CTU based on WG, and position of current CU based on lid
    int ctusPerRow = ceil((float) frameWidth/CTU_WIDTH);
    int ctuX = (ctuIdx%ctusPerRow) * CTU_WIDTH;  // CTU position inside the frame
    int ctuY = (ctuIdx/ctusPerRow) * CTU_HEIGHT; 
    int cuColumnsPerCtu = CTU_WIDTH/cuWidth;

    int cuX = (cuIdx%cuColumnsPerCtu) * cuWidth; // CU position inside the CTU [0,128]
    int cuY = (cuIdx/cuColumnsPerCtu) * cuHeight;

    // Variables to keep track of the current and best MVs/costs
    // TODO: Even if the current CTU holds a single 128x128 CU, we are allocating MAX_CUS_PER_CTU positions in the array
    __local Cpmvs lBestCpmvs[MAX_CUS_PER_CTU], lCurrCpmvs[MAX_CUS_PER_CTU];
    __local long currCost[MAX_CUS_PER_CTU], bestCost[MAX_CUS_PER_CTU], bestDist[MAX_CUS_PER_CTU];
    
    // TODO: Design a motion vector predicton algorithm. Currently MPV is always zero
    Cpmvs predCpmvs;
    predCpmvs.LT.x = 0;
    predCpmvs.LT.y = 0;
    predCpmvs.RT.x = 0;
    predCpmvs.RT.y = 0;
    predCpmvs.LB.x = 0;
    predCpmvs.LB.y = 0;

    // Hold a fraction of the total distortion of current CPMVs (each workitem uses one position)
    // TODO: Modify this 256 to use a MACRO value. Different architectures may support fewer workitems per WG
    __local long local_cumulativeSATD[256];

    // TODO: All these VECTORIZED_MEMORY if/elses are used to control memory access: either using vload/vstore operations, or indexing the values one by one
    // Fetch current CU for private memory. Only the samples used by this workitem are fetched to decrease memory bandwidth
    int16 currentCU_subBlock[4];
    int currentCU[CTU_WIDTH*CTU_HEIGHT];
    if(VECTORIZED_MEMORY){
        int stridePerPass = (cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
        for(int pass=0; pass<4; pass++){   
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu)<<2;
            sub_X = (index%subBlockColumnsPerCu)<<2;
            int offset = (ctuY + cuY + sub_Y)*frameWidth + ctuX + cuX + sub_X;
            currentCU_subBlock[pass].lo.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].lo.hi = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.hi = convert_int4(vload4(offset/4, currentFrameSamples));
        }
    }
    else{
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // THIS WILL NOT WORK BECAUSE THE INDEXING WAS NOT CORRECTED TO SUPPORT MORE BLOCK SIZES
        // cuX and cuY were modified and must be combined with ctuX and ctuY
        // correct this code if VECTORIZED_MEMORY=0

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
    

    // At first, this holds the predicted samples for the entire CTU
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
    int numGradientIter = select(4, 5, nCP==2); // Number of iteration in gradient ME search (i.e., number of CPMV updates after predicted MV)
    // TODO: Maybe it is faster to make all workitems write to the same local variable than using if()+barrier
    // Initial MVs from AMVP and initialize rd-cost
    if(lid%itemsPerCu==0){ // The first id of each sub-group initializes the CPMVs of its CU
        bestCost[cuIdx] = MAX_LONG;
        lCurrCpmvs[cuIdx] = predCpmvs;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // sync all workitems after initializing the MV
    
    // Starts the motion estimation based on gradient and optical flow
    __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 

    // Here each workitem verify if its CU lies completely within the frame
    // This is used to avoid performing the prediction of CUs "outside the frame" when the last CTU row is composed of partil CUs (i.e., vertical resolution not multiple of 128)
    int isWithinFrame =   (ctuX + cuX + cuWidth <= frameWidth)
                        && (ctuY + cuY + cuHeight <= frameHeight);

     for(int iter=0; iter<numGradientIter+1; iter++){ // +1 because we need to conduct the initial prediction (with AMVP) in addition to the gradient-ME
        
        // ###############################################################################
        // ###### HERE IT STARTS THE PREDICTION OF THE BLOCK AND COMPUTES THE COSTS ######
        // ###############################################################################
  
        // Reset SATD for current iteration
        cumulativeSATD = 0;
        local_cumulativeSATD[lid] = 0;

        // TODO: These 4 passes are valid for aligned blocks. When considering unaligned blocks this value may change
        // Each workitem will predict 4 sub-blocks (1024 sub-blocks per CTU, 256 workitems)
        int stridePerPass = (cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
        for(int pass=0; pass<isWithinFrame*4; pass++){   // When isWithinFrame==0 we do not conduct any passes and avoid performing the prediction and distortion
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu)<<2;
            sub_X = (index%subBlockColumnsPerCu)<<2;
            
            // Derive sub-MVs and determine if they are too spread (when spread, all sub-blocks have the same MV)
            // TODO: Create a wrapper function that receives the nCPs and calls the proper sub-MV derivation
            if(nCP==2){
                subMv_and_spread = deriveMv2Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                subMv_and_spread = deriveMv3Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }

            subMv = subMv_and_spread.xy;
            isSpread = subMv_and_spread.z;

            // These deltas are used during PROF
            // TODO: Create a wrapper function that receives the nCPs and calls the proper function for each CP
            // While enablePROF=0 this part of the code makes no difference
            ///*
            if(nCP==2){
                deltaHorVec = getHorizontalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                deltaHorVec = getHorizontalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            //*/

            subMv = roundAndClipMv(subMv, ctuX+cuX, ctuY+cuY,  cuWidth, cuHeight, frameWidth, frameHeight);

            // Export the sub-MV for current sub-block
            /*
            if(wg==targetWg && lid==targetLid){
                printf("iter %d, WG %d, lid %d, LT %dx%d, RT %dx%d\n", iter, wg, lid, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);
                printf("  subXY %dx%d -->> %dx%d\n", sub_X, sub_Y, subMv.x, subMv.y);               
            }
            //*/

            // Split the int and fractional part of the MVs to perform prediction/filtering
            // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
            int2 subMv_int, subMv_frac;
            subMv_int.x  = subMv.x >> 4;
            subMv_frac.x = subMv.x & 15;
            subMv_int.y  = subMv.y >> 4;
            subMv_frac.y = subMv.y & 15;

            // Top-left corner of the reference block in integer pixels
            int refPosition = (ctuY+cuY+sub_Y)*frameWidth + ctuX+cuX+sub_X + subMv_int.y*frameWidth + subMv_int.x;

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
            int leftSlack = ctuX+cuX+sub_X + subMv_int.x - ( N/2 - 1 );
            // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
            int rightSpam = ctuX+cuX+sub_X + subMv_int.x + ( N/2 );
            int rightSlack = frameWidth - 1 - rightSpam;
            // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
            int topSlack = ctuY+cuY+sub_Y + subMv_int.y - ( N/2 - 1 );
            // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
            int bottomSpam = ctuY+cuY+sub_Y + subMv_int.y + ( N/2 );
            int bottomSlack = frameHeight - 1 - bottomSpam;

            // Fetch reference window (i.e., samples that may be used for filtering)
            bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
            int currSample; // Used to avoid if/else structures during left/right correction
            int properIdx; // This variable is used to update the index of the reference sample without accessing global memory until the proper index is found. It avoid segmentation fault when the motion vectors point outside the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
            
            // TODO: Unroll the following loop (or not?)
            for(int row=0; row<windowHeight; row++){
                for(int col=0; col<windowWidth; col++){           
                    // First computes the "individual" left, right, top and bottom corrections, disconsidering any other correction
                    leftCorrect = select(true,false,leftSlack+col>=0); // moving right (i.e., moving column-wise) increases left slack and decreases right slack
                    rightCorrect = select(true,false,rightSlack-col+7>=0); // this +7 corrects the difference between the reference block and the reference window
                    topCorrect = select(true,false,topSlack+row>=0); // moving downwards (i.e., moving row-wise) increses the top slack and decreases the bottom slack
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
                    currSample = (short) referenceFrameSamples[properIdx];
                    referenceWindow[row*windowWidth+col] = currSample;
                }
            }

            // This is used to export the reference window
            /*
            if(wg==targetWg && iter==targetIter && sub_X==0 && sub_Y==0 && lid==targetLid){
                printf("REFERENCE WINDOW\n");
                printf("  Iter: %d WG: %d subX: %d subY: %d cuIdx: %d LT: %dx%d RT: %dx%d\n", iter, wg, sub_X, sub_Y, cuIdx, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);

                for(int i=0; 1<windowHeight; i++){
                    for(int j=0; j<windowWidth; j++){
                        printf("%d,", referenceWindow[i*windowWidth+j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/

            // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
            predBlock = horizontal_vertical_filter_new(referenceWindow, subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     

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
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s0;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s1;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s2;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s3;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s4;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s5;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s6;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s7;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s8;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s9;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+2] = predBlock.sa;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sb;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+0] = predBlock.sc;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+1] = predBlock.sd;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+2] = predBlock.se;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sf;
           
            // Compute the SATD 4x4 for the current sub-block and accumulate on the private accumulator
            if(VECTORIZED_MEMORY){
                satd = satd_4x4(currentCU_subBlock[pass], predBlock);
            }
            else{
                satd = satd_4x4(original_block, predBlock);
            }
            cumulativeSATD += (long) satd;
        }
        
        // Each position of local_cumulativeSATD will hold the cumulativeSATD of the sub-blocks of current workitem
        // Wait until all workitems compute their cumulativeSATD and reduce (i.e., sum all SATDs) to the first position of local_cumulativeSATD
        local_cumulativeSATD[lid] = cumulativeSATD;
        barrier(CLK_LOCAL_MEM_FENCE);


        /* Only necessary when we are exporting the predicted CU
        if(wg==targetWg && lid==targetLid && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                    // printf("%d,", predCU_then_error[i*128+j]);
                }
                // printf("\n");
            }
            
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        // TODO: Only the first item of each sub-group is used to reduce the SATDs. It may be posible to accelerarte using multiple ids
        // Reduce the smaller SATDs, compute the cost and update CPMVs
        if(lid%itemsPerCu==0){
            for(int i=1; i<itemsPerCu; i++){
                local_cumulativeSATD[lid] += local_cumulativeSATD[lid+i];
            }

            // TODO: Review the implementation of calc_affine_bits(). Verify how it behaves when current MV is equal to predicted MV, and it is necessary to add an offset (ruiCost) to the number of bits when computing the cost with different reference frames
            bitrate = calc_affine_bits(AFFINE_MV_PRECISION_QUARTER, nCP, lCurrCpmvs[cuIdx], predCpmvs);

            // TODO: This "+4" represents the ruiBits of the VTM-12.0 encoder, and it is the base-bitrate for using affine. The "+4" when using low delay with a single reference frame. Improve this when using multiple reference frames
            currCost[cuIdx] = local_cumulativeSATD[lid] + (long) getCost(bitrate + 4, lambda);

            // If the current CPMVs are not better than the previous (rd-cost wise), the best CPMVs are not updated but the next iteration continues from the current CPMVs
            if(currCost[cuIdx] < bestCost[cuIdx]){
                bestCost[cuIdx] = currCost[cuIdx];
                bestDist[cuIdx] = local_cumulativeSATD[lid];

                lBestCpmvs[cuIdx] = lCurrCpmvs[cuIdx];
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
        // TODO: These 64 passes are valid for aligned CUs only. For unaligned CUs, it is possible to have fewer samples per CTU
        // Each workitem will compute the gradient on 64 positions (128x128 samples, 256 workitems)
        // This part makes no distinction between subgroups. All data (gradient and predicted samples) are stored on __local or __global memory. This is better to optimize memory transactions
        for(int pass=0; pass<64; pass++){
            centerSample = pass*256 + lid;

            // Values at the border of CUs and CTUs are not valid. However, we can compute the values at the border of CUs to maintain regularity in the kernel and overwrite these values in sequence. Valeus at the borders of a CTU CANNOT be computed since we do not have the neighboring samples
            int isValid = !((centerSample%CTU_WIDTH==0) || (centerSample%CTU_WIDTH==CTU_WIDTH-1) || (centerSample/CTU_WIDTH==0) || (centerSample/CTU_WIDTH==CTU_HEIGHT-1));

            // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
            if(isValid){
                // Stride memory to location of current CTU (1 wg processes all CUs with a given size for one CTU
                // TODO: Verify this behavior when processing unaligned CUs
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample-CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+1]-2*predCU_then_error[centerSample-1] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample+CTU_WIDTH-1];    
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample+CTU_WIDTH-1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+CTU_WIDTH]-2*predCU_then_error[centerSample-CTU_WIDTH] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH+1];
            }
            else{
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
            }
        }
        
        // Wait until all workitems have computed their gradients
        barrier(CLK_LOCAL_MEM_FENCE);

        // TODO: It may be possible to use workitems of the same warp to fill the border of different CUs, but it would require computing a different value for cuY and cuX
        // if(lid%itemsPerCu==0){
        // @ OPTIMIZATION 1
        // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
        // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
        // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
        // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the borders of each CU
            int virtual_cuIdx = lid;
            int virtual_cuX = (virtual_cuIdx%cuColumnsPerCtu) * cuWidth; // CU position inside the CTU [0,128]
            int virtual_cuY = (virtual_cuIdx/cuColumnsPerCtu) * cuHeight;

            // Fills the first and last rows of gradient with the correct values (upper/lower row values)
            for(int col=0; col<cuWidth; col++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];
            }
            // Fills the first and last columns of the gradient with the correct values (left/right column values)
            for(int row=0; row<cuHeight; row++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];
            }
            // Fills the four corners
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];        
        }
        
        // Wait until the border values are filled. They are required in the next stage
        barrier(CLK_LOCAL_MEM_FENCE);
                
        // Since the prediction will not be used again in the same iteration, it is possible to reuse the __local array to store the error (pred = pred-orig). The error will be used to build the system of equations
        // Each workitem computes the error of a subset of sub-blocks (the same sub-blocks it predicted earlier)
        if(VECTORIZED_MEMORY){
            // TODO: Maybe this stridePerPass can be computed exactly as "idx" when computing the system of equations, using itemsPerCu
            // TODO: Verify if this works when processing unaligned CUs
            stridePerPass = (cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
            for(int pass=0; pass<4; pass++){   
                int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
                int sub_X, sub_Y;
                sub_Y = (index/subBlockColumnsPerCu)<<2;
                sub_X = (index%subBlockColumnsPerCu)<<2;

                int cuOffset = (cuY+sub_Y)*CTU_WIDTH + cuX+sub_X;
                short4 linePred, lineDiff;
                
                // Read one line from predicted sub-block, subtract from original CU, overwrite predCU_then_error
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
                
                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
            }   
        }
        else{
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // This error computation is not correct when encoding CUs smaller than 128x128. 
            // It is necessary to use cuIdx, cuX and cuY to correct the indexing
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        // Wait until all woritems have conducted their work over the predCU_then_error buffer
        barrier(CLK_LOCAL_MEM_FENCE); 

        /* Only necessary when we are exporting the prediction error
        barrier(CLK_LOCAL_MEM_FENCE);
        if(wg==targetWg && lid==0 && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); //
        //*/ 
        

        // ######################################################################
        // ###### HERE IT STARTS BUILDING THE PARTIAL SYSTEMS OF EQUATIONS ######
        // ######################################################################

        // Holds the "complete" system of equations to be solved during gradient-ME. For 2 CPs, only the first [5][4] indices are used.
        // __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 
        // The long variable is used when building the system of equations (only integers). The double variable is used for solving the system
        __private long private_pEqualCoeff[7][7];
        __private double private_dEqualCoeff[7][7];
        // The double variable is purely based on the affine parameters, the int variable is used after rounding the double to the proper precision
        double8 dDeltaMv = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int8 intDeltaMv;

        // TODO: Use these "iParaNum" and "affineParaNum" in the place of all these 6's and 7's
        int iParaNum = select(5, 7, nCP==3);
        int affineParaNum = iParaNum - 1;

        // TODO: Improve this to be adaptive to number of CPs
        // Initialize the system of equations
        for(int i=0; i<7; i++){
            for(int j=0; j<7; j++){
                private_pEqualCoeff[i][j] = 0;
            }
        }

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        // TODO:
        // The partial system only depends on the prediction error and gradient at a single position inside the CU
        // The "idx" variable assumes values in the range [0,n], where each CU has n samples
        // cuX and cuY are combined to derive the position of the error/gradient inside the CTU

        // Currently, each workitem only processes the samples inside its own CU. This makes the logic easier, but may cause performance degradation
        // Whenever there is a vertical partition, two sequential workitems may point to non-sequential memory segments: since the next sample is inside another CU, the workitem must go ot the next row of samples to stay inside its CU

        // CONSIDER THE FOLLOWING IDEA:
        // Use memory access similar to gradient computation, where sequantial workitems always evaluate sequential samples and make coalesced accesses. With this approach, one workitem may compute the partial system of a sample inside another CU
        // This causes some synchronization issues, given that the workitems will need to access the global/local memory to store the partial systems (it is not possible to sum them in private memory since they possibly belong to different CUs)
        
        // Maybe the performance is not that affected by the current approach:
        // In the worst case which is a 16x16 CU, we still have 16 sequential and aligned samples (the same size as a half warp) that must be fetched from global memory, whici sums 64 bytes of data
        // For CUs with width=32 we have an entire warp fetching 32 sequential and aligned gradient values, totalizing 128 bytes of data (maximum data obtained in single transaction for NVIDIA)
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        // Build the system of equations
        long tmp_atomic;
        // Computes a fraction of the entire system on private memory for the current workitem
        // It is necessary to go through the 128x128 samples, therefore each workitem processes 64 positions (128x128 divided by 256 workitems)
        // TODO: The number of passes may change when processing aligned and unaligned CUs. This must be verified
        // TODO: I believe the upper limit of "pass"A is always 64
        for(int pass=0; pass<(cuWidth*cuHeight)/itemsPerCu; pass++){           
            int iC[6], j, k, cx, cy;
            int idx = pass*itemsPerCu + lid%itemsPerCu; // lid%itemsPerCu represents the index of this item inside the sub-group
            
            // j and k -> Position of the sample inside the CTU [0,128]
            j = cuY + idx/cuWidth; 
            k = cuX + idx%cuWidth;       

            // cy and cx -> Center of the sub-block inside the CU
            cy = (((idx/cuWidth) >> 2) << 2) + 2; // cy represents the center of the sub-block (4x4) of current sample. If sample is in y=11, cy=10 (the center of the third block: 2, 6, 10, 14, ...)
            cx = (((idx%cuWidth) >> 2) << 2) + 2; // cx represents the center of the sub-block (4x4) of current sample. If sample is in x=4, cx=6 (the center of the second block: 2, 6, 10, 14, ...)

            // TODO: Improve this to avoid using if/else (maybe the if/else will no interfere in performance since all workitems will be true or false)
            if(nCP==2){
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] + cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] - cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }
            else{
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[4] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[5] = cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }

            // TODO: Test if using atomic operations (adding the sub-systems) directly over global memory improves performance
            for(int col=0; col<2*nCP; col++){
                for(int row=0; row<2*nCP; row++){
                    tmp_atomic = (long)iC[col] * (long)iC[row];
                    private_pEqualCoeff[col + 1][row] += tmp_atomic;
                }

                tmp_atomic = ((long)iC[col] * (long)predCU_then_error[j*CTU_WIDTH + k]) << 3;
                private_pEqualCoeff[col + 1][2*nCP] += tmp_atomic;
                // pEqualCoeff[col + 1][affineParamNum] += ((int64_t)iC[col] * pResidue[idx]) << 3;
            }
        }

        // TODO: Improve this to be adaptive to number of CPs
        // Copy the private fraction of the system to the global memory
        for(int col=0; col<7; col++){
            for(int row=0; row<7; row++){
                // Strides to the memory location of this workgroup by wg*256*7*7 and to the location of this workitem by id*7*7
                global_pEqualCoeff[wg*256*7*7 + lid*7*7 + col*7 + row] = private_pEqualCoeff[col][row];
            }
        }

        // Wait until all workitems have copied their systems to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // #########################################################################
        // ###### HERE IT SOLVES THE SYSTEM OF EQUATIONS AND UPDATE THE CPMVs ######
        // #########################################################################
        
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the global memory
            int virtual_cuIdx = lid;
            int virtual_cuX = (virtual_cuIdx%cuColumnsPerCtu) * cuWidth; // CU position inside the CTU [0,128]
            int virtual_cuY = (virtual_cuIdx/cuColumnsPerCtu) * cuHeight;

            // Copy values from first workitem
            // TODO: Make these for loops adaptive to number of CPs
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<7; row++){
                    // Stride memory to the local of this WG
                    private_pEqualCoeff[col][row] = global_pEqualCoeff[wg*256*7*7 + virtual_lid*7*7 + col*7 + row];
                }
            }

            // Reduce remaining workitems by adding over the first values
            for(int item=1; item<itemsPerCu; item++){
                // TODO: Make these for loops adaptive to number of CPs
                // for(int col=0; col<7; col++){
                for(int col=1; col<2*nCP+1; col++){
                    for(int row=0; row<7; row++){
                        private_pEqualCoeff[col][row] += global_pEqualCoeff[wg*256*7*7 + (virtual_lid+item)*7*7 + col*7 + row];
                    }
                }

            }

            // At this point, private_pEqualCoeff holds the final system of equations for the current CU (multiple partial systems were reduced in the previous for loop)

            // Copy the system to private memory using double type to solve the system in sequence
            // TODO: Make these for loops adaptive to number of CPs
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<7; row++){
                    private_dEqualCoeff[col][row] = (double) private_pEqualCoeff[col][row];
                }
            }

            // Export the final system of equations for one CU
            /*
            if(wg==targetWg && virtual_cuIdx==targetCuIdx){
                //printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, cuIdx);
                printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, lid); //lid equals virtual_cuIdx
                for(int col=0; col<iParaNum; col++){
                    for(int row=0; row<iParaNum; row++){
                        printf("%ld,", private_pEqualCoeff[col][row]);
                        // printf("%f,", private_dEqualCoeff[col][row]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/


            // BEGIN solving the system of equations
            int iOrder = affineParaNum;
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
                    break;
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

            // Export progress for a given WG
            /*
            // if(wg==targetWg && lid%itemsPerCu==0 && cuIdx==targetCuIdx){
            if(wg==targetWg && virtual_lid==targetCuIdx){
                printf("Iter: %d WG: %d Width: %d Height: %d X: %d Y: %d cuIdx: %d \n", iter, wg, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY, virtual_cuIdx);
                printf("\tCurrent CPMVs.       LT: %dx%d        RT: %dx%d    LB: %dx%d    isSpread: %d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y, isSpread);
            }
            //*/

            // Update the current MVs and clip to allowed values
            lCurrCpmvs[virtual_cuIdx].LT.x += intDeltaMv.s0;
            lCurrCpmvs[virtual_cuIdx].LT.y += intDeltaMv.s1;
            lCurrCpmvs[virtual_cuIdx].RT.x += intDeltaMv.s2;
            lCurrCpmvs[virtual_cuIdx].RT.y += intDeltaMv.s3;
            lCurrCpmvs[virtual_cuIdx].LB.x += intDeltaMv.s4;
            lCurrCpmvs[virtual_cuIdx].LB.y += intDeltaMv.s5;

            lCurrCpmvs[virtual_cuIdx] = clampCpmvs(lCurrCpmvs[virtual_cuIdx], MV_MIN, MV_MAX);
            
            lCurrCpmvs[virtual_cuIdx] = clipCpmvs(lCurrCpmvs[virtual_cuIdx], ctuX+virtual_cuX, ctuY+virtual_cuY, cuWidth, cuHeight, frameWidth, frameHeight);


            // Export progress for a given WG
            /*
            if(wg==targetWg && lid<cusPerCtu && virtual_cuIdx==targetCuIdx){
                printf("\tIteration %d  WxH %dx%d @ XY %dx%d\n", iter, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY);
                printf("\tCurrent deltas. deltaLT: %dx%d   deltaRT: %dx%d   deltaLB: %dx%d\n", intDeltaMv.s0, intDeltaMv.s1, intDeltaMv.s2, intDeltaMv.s3, intDeltaMv.s4, intDeltaMv.s5);
                printf("\tUpdated CPMVs.       LT: %dx%d        RT: %dx%d        LB: %dx%d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y);
                printf("\tCurrent COST: %ld\n", currCost[virtual_cuIdx]);
                printf("\n-----\n");
            }
            //*/

            // The next iteration will perform prediction with the new CPMVs and verify if this improves performance

        } // if(lid==0) for solving the system and updating the MVs
        
        // All workitems must wait until the CPMVs are updated before starting the next prediction
        barrier(CLK_LOCAL_MEM_FENCE);

    } // end for the five iterations of gradient


    // Write the best CPMVs and corresponding costs to global memory and return to host
    // @ OPTIMIZATION 3
    // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
    // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
    // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
    // if(lid%itemsPerCu==0){
    //     int use_opt3 = 0;
    if(lid < cusPerCtu){
        int use_opt3 = 1;

        int cuDimensionIdx = wg%NUM_CU_SIZES;
        // Stride to the position of this CTU, then for the position of this CU size, and finally for the current CU index
        int virtual_cuIdx = lid;
        int returnArrayIdx;
        if(use_opt3)
            returnArrayIdx = ctuIdx*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuDimensionIdx] + virtual_cuIdx;
        else
            returnArrayIdx = ctuIdx*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuDimensionIdx] + cuIdx;

        // Export the updated cost/CPMV information for a given CU at the end of each iteration
        /*
        if(ctuIdx==49 && cuWidth==64 && cuHeight==64){           
            // printf("\nCTU %d - CU %d\n  LT_X %d\n  LT_Y %d\n  RT_X %d\n  RT_Y %d\n  returnArrayIdx %d\n\n", ctuIdx, cuIdx, lBestCpmvs[cuIdx].LT.x, lBestCpmvs[cuIdx].LT.y, lBestCpmvs[cuIdx].RT.x, lBestCpmvs[cuIdx].RT.y, returnArrayIdx);
            printf("\nCTU %d - CU %d\n  Cost %ld\n  Index %d\n", ctuIdx, virtual_cuIdx, bestCost[virtual_cuIdx], returnArrayIdx);

        }
        //*/
        if(use_opt3){
            gBestCost[returnArrayIdx] = bestCost[virtual_cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[virtual_cuIdx];
        }
        else{
            gBestCost[returnArrayIdx] = bestCost[cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[cuIdx];
        }
        
    }
}

__kernel void affine_gradient_mult_sizes_3CPs(__global short *referenceFrameSamples, __global short *currentFrameSamples,const int frameWidth, const int frameHeight, const float lambda, __global short *horizontalGrad, __global short *verticalGrad, __global long *global_pEqualCoeff, __global long *gBestCost, __global Cpmvs *gBestCpmvs, __global long *debug, __global short *retCU){
    // TODO: Improve this to use 2 and 3 CPs
    int nCP = 3;
    // Used to debug the information of specific workitems and encoding stages
    int targetIter = 0;
    int targetWg = 134;
    int targetCuIdx = 1;
    int targetLid = 224;

    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    // Based on the WG index, the correct block size is selected
    // This assumes that each WG will process a single CU size inside a single CTU, considering ALIGNED CUs only
    int cuWidth = WIDTH_LIST[wg%NUM_CU_SIZES];
    int cuHeight = HEIGHT_LIST[wg%NUM_CU_SIZES];
    int cusPerCtu = (128*128)/(cuWidth*cuHeight);
    int itemsPerCu = wgSize/cusPerCtu;
    int subBlocksPerCu = (cuWidth/4)*(cuHeight/4);
    int subBlockColumnsPerCu = cuWidth/4;
    int subBlockRowsPerCu = cuHeight/4;
    int cuIdx = lid/itemsPerCu;

    
    // Index of current CTU. Every NUM_CU_SIZES workgroups share the same ctuIdx: one for each CU size
    int ctuIdx = wg/NUM_CU_SIZES;

    // Derive position of current CTU based on WG, and position of current CU based on lid
    int ctusPerRow = ceil((float) frameWidth/CTU_WIDTH);
    int ctuX = (ctuIdx%ctusPerRow) * CTU_WIDTH;  // CTU position inside the frame
    int ctuY = (ctuIdx/ctusPerRow) * CTU_HEIGHT; 
    int cuColumnsPerCtu = CTU_WIDTH/cuWidth;

    int cuX = (cuIdx%cuColumnsPerCtu) * cuWidth; // CU position inside the CTU [0,128]
    int cuY = (cuIdx/cuColumnsPerCtu) * cuHeight;

    // Variables to keep track of the current and best MVs/costs
    // TODO: Even if the current CTU holds a single 128x128 CU, we are allocating MAX_CUS_PER_CTU positions in the array
    __local Cpmvs lBestCpmvs[MAX_CUS_PER_CTU], lCurrCpmvs[MAX_CUS_PER_CTU];
    __local long currCost[MAX_CUS_PER_CTU], bestCost[MAX_CUS_PER_CTU], bestDist[MAX_CUS_PER_CTU];

    int inheritCpmvs = 1;
    Cpmvs predCpmvs;
    predCpmvs.LT.x = 0;
    predCpmvs.LT.y = 0;
    predCpmvs.RT.x = 0;
    predCpmvs.RT.y = 0;
    predCpmvs.LB.x = 0;
    predCpmvs.LB.y = 0;

    if(inheritCpmvs){
        int use_opt3 = 1;

        int cuDimensionIdx = wg%NUM_CU_SIZES;
        // Stride to the position of this CTU, then for the position of this CU size, and finally for the current CU index
        int returnArrayIdx;

        returnArrayIdx = ctuIdx*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuDimensionIdx] + cuIdx;

        // Export the updated cost/CPMV information for a given CU at the end of each iteration
        /*
        if(ctuIdx==49 && cuWidth==64 && cuHeight==64){           
            // printf("\nCTU %d - CU %d\n  LT_X %d\n  LT_Y %d\n  RT_X %d\n  RT_Y %d\n  returnArrayIdx %d\n\n", ctuIdx, cuIdx, lBestCpmvs[cuIdx].LT.x, lBestCpmvs[cuIdx].LT.y, lBestCpmvs[cuIdx].RT.x, lBestCpmvs[cuIdx].RT.y, returnArrayIdx);
            printf("\nCTU %d - CU %d\n  Cost %ld\n  Index %d\n", ctuIdx, virtual_cuIdx, bestCost[virtual_cuIdx], returnArrayIdx);

        }
        //*/

        // Read the result for affine with 2 CPs in the same CU
        predCpmvs = gBestCpmvs[returnArrayIdx];   

        int shift = MAX_CU_DEPTH;

        int vx2 = (predCpmvs.LT.x << shift) - ((predCpmvs.RT.y - predCpmvs.LT.y) << (shift + (int)log2((float)cuHeight) - (int)log2((float)cuWidth)));        
        int vy2 = (predCpmvs.LT.y << shift) + ((predCpmvs.RT.x - predCpmvs.LT.x) << (shift + (int)log2((float)cuHeight) - (int)log2((float)cuWidth)));

        int offset = (1 << (shift - 1));
        vx2 = (vx2 + offset - (vx2 >= 0)) >> shift;
        vy2 = (vy2 + offset - (vy2 >= 0)) >> shift;

        predCpmvs.LB.x = vx2;
        predCpmvs.LB.y = vy2;

        // Adjust the precision of derived CPMVs
        predCpmvs.LB.x = clamp(vx2, -(1 << 17), (1 << 17) - 1);
        predCpmvs.LB.y = clamp(vy2, -(1 << 17), (1 << 17) - 1);
        
        int2 mv2;
        mv2.x = predCpmvs.LB.x;
        mv2.y = predCpmvs.LB.y;
        mv2 = roundAffinePrecInternal2Amvr(mv2, 4);
        mv2 = clipMv(mv2, ctuX+cuX, ctuY+cuY, cuWidth, cuHeight, frameWidth, frameHeight);
        predCpmvs.LB.x = mv2.x;
        predCpmvs.LB.y = mv2.y;
    }

    // Hold a fraction of the total distortion of current CPMVs (each workitem uses one position)
    // TODO: Modify this 256 to use a MACRO value. Different architectures may support fewer workitems per WG
    __local long local_cumulativeSATD[256];

    // TODO: All these VECTORIZED_MEMORY if/elses are used to control memory access: either using vload/vstore operations, or indexing the values one by one
    // Fetch current CU for private memory. Only the samples used by this workitem are fetched to decrease memory bandwidth
    int16 currentCU_subBlock[4];
    int currentCU[CTU_WIDTH*CTU_HEIGHT];
    if(VECTORIZED_MEMORY){
        int stridePerPass = (cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
        for(int pass=0; pass<4; pass++){   
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu)<<2;
            sub_X = (index%subBlockColumnsPerCu)<<2;
            int offset = (ctuY + cuY + sub_Y)*frameWidth + ctuX + cuX + sub_X;
            currentCU_subBlock[pass].lo.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].lo.hi = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.hi = convert_int4(vload4(offset/4, currentFrameSamples));
        }
    }
    else{
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // THIS WILL NOT WORK BECAUSE THE INDEXING WAS NOT CORRECTED TO SUPPORT MORE BLOCK SIZES
        // cuX and cuY were modified and must be combined with ctuX and ctuY
        // correct this code if VECTORIZED_MEMORY=0

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
    

    // At first, this holds the predicted samples for the entire CTU
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
    int numGradientIter = select(4, 5, nCP==2); // Number of iteration in gradient ME search (i.e., number of CPMV updates after predicted MV)
    // TODO: Maybe it is faster to make all workitems write to the same local variable than using if()+barrier
    // Initial MVs from AMVP and initialize rd-cost
    if(lid%itemsPerCu==0){ // The first id of each sub-group initializes the CPMVs of its CU
        bestCost[cuIdx] = MAX_LONG;
        lCurrCpmvs[cuIdx] = predCpmvs;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // sync all workitems after initializing the MV
    
    // Starts the motion estimation based on gradient and optical flow
    __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 

    // Here each workitem verify if its CU lies completely within the frame
    // This is used to avoid performing the prediction of CUs "outside the frame" when the last CTU row is composed of partil CUs (i.e., vertical resolution not multiple of 128)
    int isWithinFrame =   (ctuX + cuX + cuWidth <= frameWidth)
                        && (ctuY + cuY + cuHeight <= frameHeight);

    for(int iter=0; iter<numGradientIter+1; iter++){ // +1 because we need to conduct the initial prediction (with AMVP) in addition to the gradient-ME
        
        // ###############################################################################
        // ###### HERE IT STARTS THE PREDICTION OF THE BLOCK AND COMPUTES THE COSTS ######
        // ###############################################################################
  
        // Reset SATD for current iteration
        cumulativeSATD = 0;
        local_cumulativeSATD[lid] = 0;

        // TODO: These 4 passes are valid for aligned blocks. When considering unaligned blocks this value may change
        // Each workitem will predict 4 sub-blocks (1024 sub-blocks per CTU, 256 workitems)
        int stridePerPass = (cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
        for(int pass=0; pass<isWithinFrame*4; pass++){   // When isWithinFrame==0 we do not conduct any passes and avoid performing the prediction and distortion
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu)<<2;
            sub_X = (index%subBlockColumnsPerCu)<<2;
            
            // Derive sub-MVs and determine if they are too spread (when spread, all sub-blocks have the same MV)
            // TODO: Create a wrapper function that receives the nCPs and calls the proper sub-MV derivation
            if(nCP==2){
                subMv_and_spread = deriveMv2Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                subMv_and_spread = deriveMv3Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }

            subMv = subMv_and_spread.xy;
            isSpread = subMv_and_spread.z;

            // These deltas are used during PROF
            // TODO: Create a wrapper function that receives the nCPs and calls the proper function for each CP
            // While enablePROF=0 this part of the code makes no difference
            ///*
            if(nCP==2){
                deltaHorVec = getHorizontalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                deltaHorVec = getHorizontalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            //*/

            subMv = roundAndClipMv(subMv, ctuX+cuX, ctuY+cuY,  cuWidth, cuHeight, frameWidth, frameHeight);

            // Export the sub-MV for current sub-block
            /*
            if(wg==targetWg && lid==targetLid){
                printf("iter %d, WG %d, lid %d, LT %dx%d, RT %dx%d\n", iter, wg, lid, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);
                printf("  subXY %dx%d -->> %dx%d\n", sub_X, sub_Y, subMv.x, subMv.y);               
            }
            //*/

            // Split the int and fractional part of the MVs to perform prediction/filtering
            // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
            int2 subMv_int, subMv_frac;
            subMv_int.x  = subMv.x >> 4;
            subMv_frac.x = subMv.x & 15;
            subMv_int.y  = subMv.y >> 4;
            subMv_frac.y = subMv.y & 15;

            // Top-left corner of the reference block in integer pixels
            int refPosition = (ctuY+cuY+sub_Y)*frameWidth + ctuX+cuX+sub_X + subMv_int.y*frameWidth + subMv_int.x;

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
            int leftSlack = ctuX+cuX+sub_X + subMv_int.x - ( N/2 - 1 );
            // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
            int rightSpam = ctuX+cuX+sub_X + subMv_int.x + ( N/2 );
            int rightSlack = frameWidth - 1 - rightSpam;
            // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
            int topSlack = ctuY+cuY+sub_Y + subMv_int.y - ( N/2 - 1 );
            // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
            int bottomSpam = ctuY+cuY+sub_Y + subMv_int.y + ( N/2 );
            int bottomSlack = frameHeight - 1 - bottomSpam;

            // Fetch reference window (i.e., samples that may be used for filtering)
            bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
            int currSample; // Used to avoid if/else structures during left/right correction
            int properIdx; // This variable is used to update the index of the reference sample without accessing global memory until the proper index is found. It avoid segmentation fault when the motion vectors point outside the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
            
            // TODO: Unroll the following loop (or not?)
            for(int row=0; row<windowHeight; row++){
                for(int col=0; col<windowWidth; col++){           
                    // First computes the "individual" left, right, top and bottom corrections, disconsidering any other correction
                    leftCorrect = select(true,false,leftSlack+col>=0); // moving right (i.e., moving column-wise) increases left slack and decreases right slack
                    rightCorrect = select(true,false,rightSlack-col+7>=0); // this +7 corrects the difference between the reference block and the reference window
                    topCorrect = select(true,false,topSlack+row>=0); // moving downwards (i.e., moving row-wise) increses the top slack and decreases the bottom slack
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
                    currSample = (short) referenceFrameSamples[properIdx];
                    referenceWindow[row*windowWidth+col] = currSample;
                }
            }

            // This is used to export the reference window
            /*
            if(wg==targetWg && iter==targetIter && sub_X==0 && sub_Y==0 && lid==targetLid){
                printf("REFERENCE WINDOW\n");
                printf("  Iter: %d WG: %d subX: %d subY: %d cuIdx: %d LT: %dx%d RT: %dx%d\n", iter, wg, sub_X, sub_Y, cuIdx, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);

                for(int i=0; 1<windowHeight; i++){
                    for(int j=0; j<windowWidth; j++){
                        printf("%d,", referenceWindow[i*windowWidth+j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/

            // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
            predBlock = horizontal_vertical_filter_new(referenceWindow, subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     

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
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s0;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s1;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s2;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s3;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s4;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s5;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s6;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s7;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s8;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s9;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+2] = predBlock.sa;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sb;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+0] = predBlock.sc;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+1] = predBlock.sd;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+2] = predBlock.se;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sf;
           
            // Compute the SATD 4x4 for the current sub-block and accumulate on the private accumulator
            if(VECTORIZED_MEMORY){
                satd = satd_4x4(currentCU_subBlock[pass], predBlock);
            }
            else{
                satd = satd_4x4(original_block, predBlock);
            }
            cumulativeSATD += (long) satd;
        }
        
        // Each position of local_cumulativeSATD will hold the cumulativeSATD of the sub-blocks of current workitem
        // Wait until all workitems compute their cumulativeSATD and reduce (i.e., sum all SATDs) to the first position of local_cumulativeSATD
        local_cumulativeSATD[lid] = cumulativeSATD;
        barrier(CLK_LOCAL_MEM_FENCE);


        /* Only necessary when we are exporting the predicted CU
        if(wg==targetWg && lid==targetLid && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                    // printf("%d,", predCU_then_error[i*128+j]);
                }
                // printf("\n");
            }
            
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        // TODO: Only the first item of each sub-group is used to reduce the SATDs. It may be posible to accelerarte using multiple ids
        // Reduce the smaller SATDs, compute the cost and update CPMVs
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the borders of each CU
            int virtual_cuIdx = lid;
            for(int i=1; i<itemsPerCu; i++){
                local_cumulativeSATD[virtual_lid] += local_cumulativeSATD[virtual_lid+i];
            }

            Cpmvs zeroCpmvs;
            zeroCpmvs.LT.x = 0;
            zeroCpmvs.LT.y = 0;
            zeroCpmvs.RT.x = 0;
            zeroCpmvs.RT.y = 0;
            zeroCpmvs.LB.x = 0;
            zeroCpmvs.LB.y = 0;
            // Although affine prediction with 3 CPs has a starting point derived from affine with 2 CPs, the predictor used for bitrate is always zero
            bitrate = calc_affine_bits(AFFINE_MV_PRECISION_QUARTER, nCP, lCurrCpmvs[virtual_cuIdx], zeroCpmvs);

            // TODO: This "+4" represents the ruiBits of the VTM-12.0 encoder, and it is the base-bitrate for using affine. The "+4" when using low delay with a single reference frame. Improve this when using multiple reference frames
            currCost[virtual_cuIdx] = local_cumulativeSATD[virtual_lid] + (long) getCost(bitrate + 4, lambda);

            // If the current CPMVs are not better than the previous (rd-cost wise), the best CPMVs are not updated but the next iteration continues from the current CPMVs
            if(currCost[virtual_cuIdx] < bestCost[virtual_cuIdx]){
                bestCost[virtual_cuIdx] = currCost[virtual_cuIdx];
                bestDist[virtual_cuIdx] = local_cumulativeSATD[virtual_lid];

                lBestCpmvs[virtual_cuIdx] = lCurrCpmvs[virtual_cuIdx];
            }
        }

        // #########################################################################################################
        // ###### AT THIS POINT WE HAVE COMPUTED THE COST OF THE CURRENT CPMVs. IF IT IS THE LAST ITERATION   ######
        // ###### THERE IS NO NEED TO PERFORM THE GRADIENT REFINEMENT SINCE THE UPDATED MV WILL NOT BE TESTED ######
        // #########################################################################################################

        if(iter == numGradientIter){
            break;
        }

        // #########################################################################################
        // ###### HERE IT STARTS COMPUTING THE GRADIENTS AND BUILDING THE SYSTEM OF EQUATIONS ######
        // #########################################################################################

        int centerSample;
        // Compute general case of gradient. It will be necessary to refill the border and corner values
        // TODO: These 64 passes are valid for aligned CUs only. For unaligned CUs, it is possible to have fewer samples per CTU
        // Each workitem will compute the gradient on 64 positions (128x128 samples, 256 workitems)
        // This part makes no distinction between subgroups. All data (gradient and predicted samples) are stored on __local or __global memory. This is better to optimize memory transactions
        for(int pass=0; pass<64; pass++){
            centerSample = pass*256 + lid;

            // Values at the border of CUs and CTUs are not valid. However, we can compute the values at the border of CUs to maintain regularity in the kernel and overwrite these values in sequence. Valeus at the borders of a CTU CANNOT be computed since we do not have the neighboring samples
            int isValid = !((centerSample%CTU_WIDTH==0) || (centerSample%CTU_WIDTH==CTU_WIDTH-1) || (centerSample/CTU_WIDTH==0) || (centerSample/CTU_WIDTH==CTU_HEIGHT-1));

            // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
            if(isValid){
                // Stride memory to location of current CTU (1 wg processes all CUs with a given size for one CTU
                // TODO: Verify this behavior when processing unaligned CUs
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample-CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+1]-2*predCU_then_error[centerSample-1] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample+CTU_WIDTH-1];    
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample+CTU_WIDTH-1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+CTU_WIDTH]-2*predCU_then_error[centerSample-CTU_WIDTH] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH+1];
            }
            else{
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
            }
        }
        
        // Wait until all workitems have computed their gradients
        barrier(CLK_LOCAL_MEM_FENCE);

        // TODO: It may be possible to use workitems of the same warp to fill the border of different CUs, but it would require computing a different value for cuY and cuX
        // if(lid%itemsPerCu==0){
        // @ OPTIMIZATION 1
        // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
        // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
        // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
        // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the borders of each CU
            int virtual_cuIdx = lid;
            int virtual_cuX = (virtual_cuIdx%cuColumnsPerCtu) * cuWidth; // CU position inside the CTU [0,128]
            int virtual_cuY = (virtual_cuIdx/cuColumnsPerCtu) * cuHeight;

            // Fills the first and last rows of gradient with the correct values (upper/lower row values)
            for(int col=0; col<cuWidth; col++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];
            }
            // Fills the first and last columns of the gradient with the correct values (left/right column values)
            for(int row=0; row<cuHeight; row++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];
            }
            // Fills the four corners
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];        
        }
        
        // Wait until the border values are filled. They are required in the next stage
        barrier(CLK_LOCAL_MEM_FENCE);
                
        // Since the prediction will not be used again in the same iteration, it is possible to reuse the __local array to store the error (pred = pred-orig). The error will be used to build the system of equations
        // Each workitem computes the error of a subset of sub-blocks (the same sub-blocks it predicted earlier)
        if(VECTORIZED_MEMORY){
            // TODO: Maybe this stridePerPass can be computed exactly as "idx" when computing the system of equations, using itemsPerCu
            // TODO: Verify if this works when processing unaligned CUs
            stridePerPass = (cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
            for(int pass=0; pass<4; pass++){   
                int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
                int sub_X, sub_Y;
                sub_Y = (index/subBlockColumnsPerCu)<<2;
                sub_X = (index%subBlockColumnsPerCu)<<2;

                int cuOffset = (cuY+sub_Y)*CTU_WIDTH + cuX+sub_X;
                short4 linePred, lineDiff;
                
                // Read one line from predicted sub-block, subtract from original CU, overwrite predCU_then_error
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
                
                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
            }   
        }
        else{
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // This error computation is not correct when encoding CUs smaller than 128x128. 
            // It is necessary to use cuIdx, cuX and cuY to correct the indexing
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        // Wait until all woritems have conducted their work over the predCU_then_error buffer
        barrier(CLK_LOCAL_MEM_FENCE); 

        /* Only necessary when we are exporting the prediction error
        barrier(CLK_LOCAL_MEM_FENCE);
        if(wg==targetWg && lid==0 && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); //
        //*/ 
        

        // ######################################################################
        // ###### HERE IT STARTS BUILDING THE PARTIAL SYSTEMS OF EQUATIONS ######
        // ######################################################################

        // Holds the "complete" system of equations to be solved during gradient-ME. For 2 CPs, only the first [5][4] indices are used.
        // __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 
        // The long variable is used when building the system of equations (only integers). The double variable is used for solving the system
        __private long private_pEqualCoeff[7][7];
        __private double private_dEqualCoeff[7][7];
        // The double variable is purely based on the affine parameters, the int variable is used after rounding the double to the proper precision
        double8 dDeltaMv = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int8 intDeltaMv;

        // TODO: Use these "iParaNum" and "affineParaNum" in the place of all these 6's and 7's
        int iParaNum = select(5, 7, nCP==3);
        int affineParaNum = iParaNum - 1;

        // TODO: Improve this to be adaptive to number of CPs
        // Initialize the system of equations
        for(int i=0; i<7; i++){
            for(int j=0; j<7; j++){
                private_pEqualCoeff[i][j] = 0;
            }
        }

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        // TODO:
        // The partial system only depends on the prediction error and gradient at a single position inside the CU
        // The "idx" variable assumes values in the range [0,n], where each CU has n samples
        // cuX and cuY are combined to derive the position of the error/gradient inside the CTU

        // Currently, each workitem only processes the samples inside its own CU. This makes the logic easier, but may cause performance degradation
        // Whenever there is a vertical partition, two sequential workitems may point to non-sequential memory segments: since the next sample is inside another CU, the workitem must go ot the next row of samples to stay inside its CU

        // CONSIDER THE FOLLOWING IDEA:
        // Use memory access similar to gradient computation, where sequantial workitems always evaluate sequential samples and make coalesced accesses. With this approach, one workitem may compute the partial system of a sample inside another CU
        // This causes some synchronization issues, given that the workitems will need to access the global/local memory to store the partial systems (it is not possible to sum them in private memory since they possibly belong to different CUs)
        
        // Maybe the performance is not that affected by the current approach:
        // In the worst case which is a 16x16 CU, we still have 16 sequential and aligned samples (the same size as a half warp) that must be fetched from global memory, whici sums 64 bytes of data
        // For CUs with width=32 we have an entire warp fetching 32 sequential and aligned gradient values, totalizing 128 bytes of data (maximum data obtained in single transaction for NVIDIA)
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        // Build the system of equations
        long tmp_atomic;
        // Computes a fraction of the entire system on private memory for the current workitem
        // It is necessary to go through the 128x128 samples, therefore each workitem processes 64 positions (128x128 divided by 256 workitems)
        // TODO: The number of passes may change when processing aligned and unaligned CUs. This must be verified
        // TODO: I believe the upper limit of "pass"A is always 64
        for(int pass=0; pass<(cuWidth*cuHeight)/itemsPerCu; pass++){           
            int iC[6], j, k, cx, cy;
            int idx = pass*itemsPerCu + lid%itemsPerCu; // lid%itemsPerCu represents the index of this item inside the sub-group
            
            // j and k -> Position of the sample inside the CTU [0,128]
            j = cuY + idx/cuWidth; 
            k = cuX + idx%cuWidth;       

            // cy and cx -> Center of the sub-block inside the CU
            cy = (((idx/cuWidth) >> 2) << 2) + 2; // cy represents the center of the sub-block (4x4) of current sample. If sample is in y=11, cy=10 (the center of the third block: 2, 6, 10, 14, ...)
            cx = (((idx%cuWidth) >> 2) << 2) + 2; // cx represents the center of the sub-block (4x4) of current sample. If sample is in x=4, cx=6 (the center of the second block: 2, 6, 10, 14, ...)

            // TODO: Improve this to avoid using if/else (maybe the if/else will no interfere in performance since all workitems will be true or false)
            if(nCP==2){
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] + cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] - cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }
            else{
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[4] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[5] = cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }

            // TODO: Test if using atomic operations (adding the sub-systems) directly over global memory improves performance
            for(int col=0; col<2*nCP; col++){
                for(int row=0; row<2*nCP; row++){
                    tmp_atomic = (long)iC[col] * (long)iC[row];
                    private_pEqualCoeff[col + 1][row] += tmp_atomic;
                }

                tmp_atomic = ((long)iC[col] * (long)predCU_then_error[j*CTU_WIDTH + k]) << 3;
                private_pEqualCoeff[col + 1][2*nCP] += tmp_atomic;
                // pEqualCoeff[col + 1][affineParamNum] += ((int64_t)iC[col] * pResidue[idx]) << 3;
            }
        }

        // TODO: Improve this to be adaptive to number of CPs
        // Copy the private fraction of the system to the global memory
        for(int col=0; col<7; col++){
            for(int row=0; row<7; row++){
                // Strides to the memory location of this workgroup by wg*256*7*7 and to the location of this workitem by id*7*7
                global_pEqualCoeff[wg*256*7*7 + lid*7*7 + col*7 + row] = private_pEqualCoeff[col][row];
            }
        }

        // Wait until all workitems have copied their systems to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // #########################################################################
        // ###### HERE IT SOLVES THE SYSTEM OF EQUATIONS AND UPDATE THE CPMVs ######
        // #########################################################################
        
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the global memory
            int virtual_cuIdx = lid;
            int virtual_cuX = (virtual_cuIdx%cuColumnsPerCtu) * cuWidth; // CU position inside the CTU [0,128]
            int virtual_cuY = (virtual_cuIdx/cuColumnsPerCtu) * cuHeight;

            // Copy values from first workitem
            // TODO: Make these for loops adaptive to number of CPs
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<7; row++){
                    // Stride memory to the local of this WG
                    private_pEqualCoeff[col][row] = global_pEqualCoeff[wg*256*7*7 + virtual_lid*7*7 + col*7 + row];
                }
            }

            // Reduce remaining workitems by adding over the first values
            for(int item=1; item<itemsPerCu; item++){
                // TODO: Make these for loops adaptive to number of CPs
                // for(int col=0; col<7; col++){
                for(int col=1; col<2*nCP+1; col++){
                    for(int row=0; row<7; row++){
                        private_pEqualCoeff[col][row] += global_pEqualCoeff[wg*256*7*7 + (virtual_lid+item)*7*7 + col*7 + row];
                    }
                }

            }

            // At this point, private_pEqualCoeff holds the final system of equations for the current CU (multiple partial systems were reduced in the previous for loop)

            // Copy the system to private memory using double type to solve the system in sequence
            // TODO: Make these for loops adaptive to number of CPs
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<7; row++){
                    private_dEqualCoeff[col][row] = (double) private_pEqualCoeff[col][row];
                }
            }

            // Export the final system of equations for one CU
            /*
            if(wg==targetWg && virtual_cuIdx==targetCuIdx){
                //printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, cuIdx);
                printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, lid); //lid equals virtual_cuIdx
                for(int col=0; col<iParaNum; col++){
                    for(int row=0; row<iParaNum; row++){
                        printf("%ld,", private_pEqualCoeff[col][row]);
                        // printf("%f,", private_dEqualCoeff[col][row]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/


            // BEGIN solving the system of equations
            int iOrder = affineParaNum;
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
                    break;
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

            // Export progress for a given WG
            /*
            // if(wg==targetWg && lid%itemsPerCu==0 && cuIdx==targetCuIdx){
            if(wg==targetWg && virtual_lid==targetCuIdx){
                printf("Iter: %d WG: %d Width: %d Height: %d X: %d Y: %d cuIdx: %d \n", iter, wg, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY, virtual_cuIdx);
                printf("\tCurrent CPMVs.       LT: %dx%d        RT: %dx%d    LB: %dx%d    isSpread: %d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y, isSpread);
            }
            //*/

            // Update the current MVs and clip to allowed values
            lCurrCpmvs[virtual_cuIdx].LT.x += intDeltaMv.s0;
            lCurrCpmvs[virtual_cuIdx].LT.y += intDeltaMv.s1;
            lCurrCpmvs[virtual_cuIdx].RT.x += intDeltaMv.s2;
            lCurrCpmvs[virtual_cuIdx].RT.y += intDeltaMv.s3;
            lCurrCpmvs[virtual_cuIdx].LB.x += intDeltaMv.s4;
            lCurrCpmvs[virtual_cuIdx].LB.y += intDeltaMv.s5;

            lCurrCpmvs[virtual_cuIdx] = clampCpmvs(lCurrCpmvs[virtual_cuIdx], MV_MIN, MV_MAX);
            
            lCurrCpmvs[virtual_cuIdx] = clipCpmvs(lCurrCpmvs[virtual_cuIdx], ctuX+virtual_cuX, ctuY+virtual_cuY, cuWidth, cuHeight, frameWidth, frameHeight);


            // Export progress for a given WG
            /*
            if(wg==targetWg && lid<cusPerCtu && virtual_cuIdx==targetCuIdx){
                printf("\tIteration %d  WxH %dx%d @ XY %dx%d\n", iter, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY);
                printf("\tCurrent deltas. deltaLT: %dx%d   deltaRT: %dx%d   deltaLB: %dx%d\n", intDeltaMv.s0, intDeltaMv.s1, intDeltaMv.s2, intDeltaMv.s3, intDeltaMv.s4, intDeltaMv.s5);
                printf("\tUpdated CPMVs.       LT: %dx%d        RT: %dx%d        LB: %dx%d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y);
                printf("\tCurrent COST: %ld\n", currCost[virtual_cuIdx]);
                printf("\n-----\n");
            }
            //*/

            // The next iteration will perform prediction with the new CPMVs and verify if this improves performance

        } // if(lid==0) for solving the system and updating the MVs
        
        // All workitems must wait until the CPMVs are updated before starting the next prediction
        barrier(CLK_LOCAL_MEM_FENCE);

    } // end for the five iterations of gradient


    // Write the best CPMVs and corresponding costs to global memory and return to host
    // @ OPTIMIZATION 3
    // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
    // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
    // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
    // if(lid%itemsPerCu==0){
    //     int use_opt3 = 0;
    if(lid < cusPerCtu){
        int use_opt3 = 1;

        int cuDimensionIdx = wg%NUM_CU_SIZES;
        // Stride to the position of this CTU, then for the position of this CU size, and finally for the current CU index
        int virtual_cuIdx = lid;
        int returnArrayIdx;
        if(use_opt3)
            returnArrayIdx = ctuIdx*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuDimensionIdx] + virtual_cuIdx;
        else
            returnArrayIdx = ctuIdx*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuDimensionIdx] + cuIdx;

        // Export the updated cost/CPMV information for a given CU at the end of each iteration
        /*
        if(ctuIdx==49 && cuWidth==64 && cuHeight==64){           
            // printf("\nCTU %d - CU %d\n  LT_X %d\n  LT_Y %d\n  RT_X %d\n  RT_Y %d\n  returnArrayIdx %d\n\n", ctuIdx, cuIdx, lBestCpmvs[cuIdx].LT.x, lBestCpmvs[cuIdx].LT.y, lBestCpmvs[cuIdx].RT.x, lBestCpmvs[cuIdx].RT.y, returnArrayIdx);
            printf("\nCTU %d - CU %d\n  Cost %ld\n  Index %d\n", ctuIdx, virtual_cuIdx, bestCost[virtual_cuIdx], returnArrayIdx);

        }
        //*/
        if(use_opt3){
            gBestCost[returnArrayIdx] = bestCost[virtual_cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[virtual_cuIdx];
        }
        else{
            gBestCost[returnArrayIdx] = bestCost[cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[cuIdx];
        }
        
    }
}


__kernel void affine_gradient_mult_sizes_HA_2CPs(__global short *referenceFrameSamples, __global short *currentFrameSamples,const int frameWidth, const int frameHeight, const float lambda, __global short *horizontalGrad, __global short *verticalGrad, __global long *global_pEqualCoeff, __global long *gBestCost, __global Cpmvs *gBestCpmvs, __global long *debug, __global short *retCU){
    // TODO: Improve this to use 2 and 3 CPs
    int nCP = 2;
    // Used to debug the information of specific workitems and encoding stages
    int targetIter = 0;
    int targetWg = 16;
    int targetCuIdx = 0;
    int targetLid = 0;

    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    // Based on the WG index, the correct block size is selected
    // This assumes that each WG will process a single CU size inside a single CTU, considering ALIGNED CUs only
    int cuWidth = HA_WIDTH_LIST[wg%HA_NUM_CU_SIZES];
    int cuHeight = HA_HEIGHT_LIST[wg%HA_NUM_CU_SIZES];
    int cusPerCtu = HA_CUS_PER_CTU[wg%HA_NUM_CU_SIZES];
    int itemsPerCu = wgSize/cusPerCtu;
    int subBlocksPerCu = (cuWidth/4)*(cuHeight/4);
    int subBlockColumnsPerCu = cuWidth/4;
    int subBlockRowsPerCu = cuHeight/4;
    int cuIdx = lid/itemsPerCu;

    
    // Index of current CTU. Every HA_NUM_CU_SIZES workgroups share the same ctuIdx: one for each CU size
    int ctuIdx = wg/HA_NUM_CU_SIZES;
    // Derive position of current CTU based on WG, and position of current CU based on lid
    int ctusPerRow = ceil((float) frameWidth/CTU_WIDTH);
    int ctuX = (ctuIdx%ctusPerRow) * CTU_WIDTH;  // CTU position inside the frame
    int ctuY = (ctuIdx/ctusPerRow) * CTU_HEIGHT; 
    int cuColumnsPerCtu = CTU_WIDTH/cuWidth;

    int cuX = HA_ALL_X_POS[wg%HA_NUM_CU_SIZES][cuIdx]; // CU position inside the CTU [0,128]
    int cuY = HA_ALL_Y_POS[wg%HA_NUM_CU_SIZES][cuIdx];

    // Variables to keep track of the current and best MVs/costs
    // TODO: Even if the current CTU holds a single 128x128 CU, we are allocating MAX_CUS_PER_CTU positions in the array
    __local Cpmvs lBestCpmvs[MAX_HA_CUS_PER_CTU], lCurrCpmvs[MAX_HA_CUS_PER_CTU];
    __local long currCost[MAX_HA_CUS_PER_CTU], bestCost[MAX_HA_CUS_PER_CTU], bestDist[MAX_HA_CUS_PER_CTU];
    
    // TODO: Design a motion vector predicton algorithm. Currently MPV is always zero
    Cpmvs predCpmvs;
    predCpmvs.LT.x = 0;
    predCpmvs.LT.y = 0;
    predCpmvs.RT.x = 0;
    predCpmvs.RT.y = 0;
    predCpmvs.LB.x = 0;
    predCpmvs.LB.y = 0;

    // Hold a fraction of the total distortion of current CPMVs (each workitem uses one position)
    // TODO: Modify this 256 to use a MACRO value. Different architectures may support fewer workitems per WG
    __local long local_cumulativeSATD[256];

    // TODO: All these VECTORIZED_MEMORY if/elses are used to control memory access: either using vload/vstore operations, or indexing the values one by one
    // Fetch current CU for private memory. Only the samples used by this workitem are fetched to decrease memory bandwidth
    int16 currentCU_subBlock[2]; // For HALF-ALIGNED CUs, wach workitem is responsible for AT MOST TWO sub-blocks
    int currentCU[CTU_WIDTH*CTU_HEIGHT];
    int totalArea = (cuWidth*cuHeight)*HA_CUS_PER_CTU[wg%HA_NUM_CU_SIZES]; // Total area encoded by this WG. Depending on the CU size and its group (type of HALF-ALIGNMENT), each WG can encode 50%, 25% or 12.5% of a CTU
    int nSubblocksForWG = totalArea/(4*4); // Number of subblocks encoded by this WG
    // TODO: Correct this 256 to be adaptive to workgroup size
    int nPasses = nSubblocksForWG/256; // We have 256 WGs. The number of subblocks encoded by each workitem is the number of passes. It may assume values: 2, 1, or 0.5 !!!!! Half subblocks per WI occurs for CUs 16x16 of G3 and G4, where we have only 8 CUs per CTU
    
    // if(lid%itemsPerCu==0) // First LID of each sub-group
    //     printf("wg,lid %d,%d -> @(%dx%d) %d passes\n", wg, lid, cuX, cuY, nPasses);

    if(VECTORIZED_MEMORY){
        int stridePerPass = itemsPerCu; //(cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
        for(int pass=0; pass<nPasses; pass++){   
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu) << 2;
            sub_X = (index%subBlockColumnsPerCu) << 2;
            int offset = (ctuY + cuY + sub_Y)*frameWidth + ctuX + cuX + sub_X;
            currentCU_subBlock[pass].lo.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].lo.hi = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.hi = convert_int4(vload4(offset/4, currentFrameSamples));
        }
    }
    else{
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // THIS WILL NOT WORK BECAUSE THE INDEXING WAS NOT CORRECTED TO SUPPORT MORE BLOCK SIZES
        // cuX and cuY were modified and must be combined with ctuX and ctuY
        // correct this code if VECTORIZED_MEMORY=0

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
    
    // At first, this holds the predicted samples for the entire CTU
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
    int numGradientIter = select(4, 5, nCP==2); // Number of iteration in gradient ME search (i.e., number of CPMV updates after predicted MV)
    // TODO: Maybe it is faster to make all workitems write to the same local variable than using if()+barrier
    // Initial MVs from AMVP and initialize rd-cost
    if(lid%itemsPerCu==0){ // The first id of each sub-group initializes the CPMVs of its CU
        bestCost[cuIdx] = MAX_LONG;
        lCurrCpmvs[cuIdx] = predCpmvs;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // sync all workitems after initializing the MV
    
    // Starts the motion estimation based on gradient and optical flow
    __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 

    // Here each workitem verify if its CU lies completely within the frame
    // This is used to avoid performing the prediction of CUs "outside the frame" when the last CTU row is composed of partil CUs (i.e., vertical resolution not multiple of 128)
    int isWithinFrame =   (ctuX + cuX + cuWidth <= frameWidth)
                        && (ctuY + cuY + cuHeight <= frameHeight);

     for(int iter=0; iter<numGradientIter+1; iter++){ // +1 because we need to conduct the initial prediction (with AMVP) in addition to the gradient-ME
        
        // ###############################################################################
        // ###### HERE IT STARTS THE PREDICTION OF THE BLOCK AND COMPUTES THE COSTS ######
        // ###############################################################################
  
        // Reset SATD for current iteration
        cumulativeSATD = 0;
        local_cumulativeSATD[lid] = 0;

        // nPasses is the number of sub-blocks predicted by each workitem. This is wither 1 or 2 for HALF-ALIGNED blocks
        int stridePerPass = itemsPerCu;
        int nPasses = (cuWidth*cuHeight*HA_CUS_PER_CTU[wg%HA_NUM_CU_SIZES]*4)/(CTU_WIDTH*CTU_HEIGHT);

        for(int pass=0; pass<isWithinFrame*nPasses; pass++){   // When isWithinFrame==0 we do not conduct any passes and avoid performing the prediction and distortion
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu) << 2;
            sub_X = (index%subBlockColumnsPerCu) << 2;
            
            // Derive sub-MVs and determine if they are too spread (when spread, all sub-blocks have the same MV)
            // TODO: Create a wrapper function that receives the nCPs and calls the proper sub-MV derivation
            if(nCP==2){
                subMv_and_spread = deriveMv2Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                subMv_and_spread = deriveMv3Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }

            subMv = subMv_and_spread.xy;
            isSpread = subMv_and_spread.z;

            // These deltas are used during PROF
            // TODO: Create a wrapper function that receives the nCPs and calls the proper function for each CP
            // While enablePROF=0 this part of the code makes no difference
            ///*
            if(nCP==2){
                deltaHorVec = getHorizontalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                deltaHorVec = getHorizontalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            //*/

            subMv = roundAndClipMv(subMv, ctuX+cuX, ctuY+cuY,  cuWidth, cuHeight, frameWidth, frameHeight);

            // Export the sub-MV for current sub-block
            /*
            if(wg==targetWg && lid==targetLid){
                printf("iter %d, WG %d, lid %d, WH=%d XY=%d LT %dx%d, RT %dx%d\n", iter, wg, lid, cuWidth, cuHeight, ctuX+cuX, ctuY+cuY, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);
                printf("  subXY %dx%d -->> %dx%d\n", sub_X, sub_Y, subMv.x, subMv.y);               
            }
            //*/

            // Split the int and fractional part of the MVs to perform prediction/filtering
            // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
            int2 subMv_int, subMv_frac;
            subMv_int.x  = subMv.x >> 4;
            subMv_frac.x = subMv.x & 15;
            subMv_int.y  = subMv.y >> 4;
            subMv_frac.y = subMv.y & 15;

            // Top-left corner of the reference block in integer pixels
            int refPosition = (ctuY+cuY+sub_Y)*frameWidth + ctuX+cuX+sub_X + subMv_int.y*frameWidth + subMv_int.x;

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
            int leftSlack = ctuX+cuX+sub_X + subMv_int.x - ( N/2 - 1 );
            // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
            int rightSpam = ctuX+cuX+sub_X + subMv_int.x + ( N/2 );
            int rightSlack = frameWidth - 1 - rightSpam;
            // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
            int topSlack = ctuY+cuY+sub_Y + subMv_int.y - ( N/2 - 1 );
            // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
            int bottomSpam = ctuY+cuY+sub_Y + subMv_int.y + ( N/2 );
            int bottomSlack = frameHeight - 1 - bottomSpam;

            // Fetch reference window (i.e., samples that may be used for filtering)
            bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
            int currSample; // Used to avoid if/else structures during left/right correction
            int properIdx; // This variable is used to update the index of the reference sample without accessing global memory until the proper index is found. It avoid segmentation fault when the motion vectors point outside the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
            
            // TODO: Unroll the following loop (or not?)
            for(int row=0; row<windowHeight; row++){
                for(int col=0; col<windowWidth; col++){           
                    // First computes the "individual" left, right, top and bottom corrections, disconsidering any other correction
                    leftCorrect = select(true,false,leftSlack+col>=0); // moving right (i.e., moving column-wise) increases left slack and decreases right slack
                    rightCorrect = select(true,false,rightSlack-col+7>=0); // this +7 corrects the difference between the reference block and the reference window
                    topCorrect = select(true,false,topSlack+row>=0); // moving downwards (i.e., moving row-wise) increses the top slack and decreases the bottom slack
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
                    currSample = (short) referenceFrameSamples[properIdx];
                    referenceWindow[row*windowWidth+col] = currSample;
                }
            }

            // This is used to export the reference window
            /*
            // if(wg==targetWg && iter==targetIter && sub_X==0 && sub_Y==0 && lid==targetLid){
            if(wg==targetWg && iter==targetIter){
                printf("REFERENCE WINDOW\n");
                printf("  Iter: %d WG: %d subX: %d subY: %d cuIdx: %d LT: %dx%d RT: %dx%d\n", iter, wg, sub_X, sub_Y, cuIdx, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);

                for(int i=0; 1<windowHeight; i++){
                    for(int j=0; j<windowWidth; j++){
                        printf("%d,", referenceWindow[i*windowWidth+j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/

            // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
            predBlock = horizontal_vertical_filter_new(referenceWindow, subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     


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
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s0;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s1;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s2;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s3;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s4;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s5;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s6;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s7;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s8;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s9;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+2] = predBlock.sa;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sb;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+0] = predBlock.sc;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+1] = predBlock.sd;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+2] = predBlock.se;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sf;

            // Compute the SATD 4x4 for the current sub-block and accumulate on the private accumulator
            if(VECTORIZED_MEMORY){
                satd = satd_4x4(currentCU_subBlock[pass], predBlock);
            }
            else{
                satd = satd_4x4(original_block, predBlock);
            }
            cumulativeSATD += (long) satd;
        }
       
        // Each position of local_cumulativeSATD will hold the cumulativeSATD of the sub-blocks of current workitem
        // Wait until all workitems compute their cumulativeSATD and reduce (i.e., sum all SATDs) to the first position of local_cumulativeSATD
        local_cumulativeSATD[lid] = cumulativeSATD;
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Only necessary when we are exporting the predicted CTU
        if(wg==targetWg && lid==targetLid && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                    // printf("%d,", predCU_then_error[i*128+j]);
                }
                // printf("\n");
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        /* Only necessary when we are exporting the predicted CU
        if(wg==targetWg && lid==targetLid && iter==targetIter){
            printf("Exporting... cuX=%d cuY=%d\n", cuX, cuY);
            for(int i=0; i<cuHeight; i++){
                for(int j=0; j<cuWidth; j++){
                    retCU[(cuY+i)*128+cuX+j] = predCU_then_error[(cuY+i)*128+cuX+j];
                    // printf("%d,", predCU_then_error[i*128+j]);
                }
                // printf("\n");
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        // TODO: Only the first item of each sub-group is used to reduce the SATDs. It may be posible to accelerarte using multiple ids
        // Reduce the smaller SATDs, compute the cost and update CPMVs
        if(lid%itemsPerCu==0){
            for(int i=1; i<itemsPerCu; i++){
                local_cumulativeSATD[lid] += local_cumulativeSATD[lid+i];
            }

            // TODO: Review the implementation of calc_affine_bits(). Verify how it behaves when current MV is equal to predicted MV, and it is necessary to add an offset (ruiCost) to the number of bits when computing the cost with different reference frames
            bitrate = calc_affine_bits(AFFINE_MV_PRECISION_QUARTER, nCP, lCurrCpmvs[cuIdx], predCpmvs);

            // TODO: This "+4" represents the ruiBits of the VTM-12.0 encoder, and it is the base-bitrate for using affine. The "+4" when using low delay with a single reference frame. Improve this when using multiple reference frames
            currCost[cuIdx] = local_cumulativeSATD[lid] + (long) getCost(bitrate + 4, lambda);

            // If the current CPMVs are not better than the previous (rd-cost wise), the best CPMVs are not updated but the next iteration continues from the current CPMVs
            if(currCost[cuIdx] < bestCost[cuIdx]){
                bestCost[cuIdx] = currCost[cuIdx];
                bestDist[cuIdx] = local_cumulativeSATD[lid];

                lBestCpmvs[cuIdx] = lCurrCpmvs[cuIdx];
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
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!
        //      IMPORTANT
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!
        // TODO: The general case of gradient is not optimized for HALF-ALIGNED CUs. For these CUs, each CU group covers either 25% or 50% of the CTU area, but the following code is computing the gradient for 100% of the samples
        // It may be possible to optimize the schedule so the gradient is computed only for the valid CUs
        // Compute general case of gradient. It will be necessary to refill the border and corner values
        // Each workitem will compute the gradient on 64 positions (128x128 samples, 256 workitems)
        // This part makes no distinction between subgroups. All data (gradient and predicted samples) are stored on __local or __global memory. This is better to optimize memory transactions
        for(int pass=0; pass<64; pass++){
            centerSample = pass*256 + lid;

            // Values at the border of CUs and CTUs are not valid. However, we can compute the values at the border of CUs to maintain regularity in the kernel and overwrite these values in sequence. Valeus at the borders of a CTU CANNOT be computed since we do not have the neighboring samples
            int isValid = !((centerSample%CTU_WIDTH==0) || (centerSample%CTU_WIDTH==CTU_WIDTH-1) || (centerSample/CTU_WIDTH==0) || (centerSample/CTU_WIDTH==CTU_HEIGHT-1));

            // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
            if(isValid){
                // Stride memory to location of current CTU (1 wg processes all CUs with a given size for one CTU
                // TODO: Verify this behavior when processing unaligned CUs
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample-CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+1]-2*predCU_then_error[centerSample-1] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample+CTU_WIDTH-1];    
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample+CTU_WIDTH-1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+CTU_WIDTH]-2*predCU_then_error[centerSample-CTU_WIDTH] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH+1];
            }
            else{
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
            }
        }
        
        // Wait until all workitems have computed their gradients
        barrier(CLK_LOCAL_MEM_FENCE);

        // TODO: It may be possible to use workitems of the same warp to fill the border of different CUs, but it would require computing a different value for cuY and cuX
        // if(lid%itemsPerCu==0){
        // @ OPTIMIZATION 1
        // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
        // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
        // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
        // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the borders of each CU
            int virtual_cuIdx = lid;
            int virtual_cuX = HA_ALL_X_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx]; // CU position inside the CTU [0,128]
            int virtual_cuY = HA_ALL_Y_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx];

            // Fills the first and last rows of gradient with the correct values (upper/lower row values)
            for(int col=0; col<cuWidth; col++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];
            }
            // Fills the first and last columns of the gradient with the correct values (left/right column values)
            for(int row=0; row<cuHeight; row++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];
            }
            // Fills the four corners
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];        
        }
        
        // Wait until the border values are filled. They are required in the next stage
        barrier(CLK_LOCAL_MEM_FENCE);
                
        // Since the prediction will not be used again in the same iteration, it is possible to reuse the __local array to store the error (pred = pred-orig). The error will be used to build the system of equations
        // Each workitem computes the error of a subset of sub-blocks (the same sub-blocks it predicted earlier)
        
        // TODO: Correct this 256 to be adaptive to workgroup size
        nPasses = nSubblocksForWG/256; // We have 256 WGs. The number of subblocks encoded by each workitem is the number of passes. It may assume values: 2 or 1
        if(VECTORIZED_MEMORY){
            // TODO: Maybe this stridePerPass can be computed exactly as "idx" when computing the system of equations, using itemsPerCu
            stridePerPass = itemsPerCu; // Number of sub-blocks between two passes of the same workitem
            for(int pass=0; pass<nPasses; pass++){   
                int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
                int sub_X, sub_Y;
                sub_Y = (index/subBlockColumnsPerCu) << 2;
                sub_X = (index%subBlockColumnsPerCu) << 2;

                int cuOffset = (cuY+sub_Y)*CTU_WIDTH + cuX+sub_X;
                short4 linePred, lineDiff;
                
                // Read one line from predicted sub-block, subtract from original CU, overwrite predCU_then_error
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
                
                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
            }   
        }
        else{
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // This error computation is not correct when encoding CUs smaller than 128x128. 
            // It is necessary to use cuIdx, cuX and cuY to correct the indexing
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        // Wait until all woritems have conducted their work over the predCU_then_error buffer
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Only necessary when we are exporting the prediction error
        barrier(CLK_LOCAL_MEM_FENCE);
        if(wg==targetWg && lid==0 && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); //
        //*/ 
        

        // ######################################################################
        // ###### HERE IT STARTS BUILDING THE PARTIAL SYSTEMS OF EQUATIONS ######
        // ######################################################################

        // Holds the "complete" system of equations to be solved during gradient-ME. For 2 CPs, only the first [5][4] indices are used.
        // __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 
        // The long variable is used when building the system of equations (only integers). The double variable is used for solving the system
        __private long private_pEqualCoeff[7][7];
        __private double private_dEqualCoeff[7][7];
        // The double variable is purely based on the affine parameters, the int variable is used after rounding the double to the proper precision
        double8 dDeltaMv = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int8 intDeltaMv;

        // TODO: Use these "iParaNum" and "affineParaNum" in the place of all these 6's and 7's
        int iParaNum = select(5, 7, nCP==3);
        int affineParaNum = iParaNum - 1;

        // TODO: Improve this to be adaptive to number of CPs
        // Initialize the system of equations
        for(int i=0; i<7; i++){
            for(int j=0; j<7; j++){
                private_pEqualCoeff[i][j] = 0;
            }
        }

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        // TODO:
        // The partial system only depends on the prediction error and gradient at a single position inside the CU
        // The "idx" variable assumes values in the range [0,n], where each CU has n samples
        // cuX and cuY are combined to derive the position of the error/gradient inside the CTU

        // Currently, each workitem only processes the samples inside its own CU. This makes the logic easier, but may cause performance degradation
        // Whenever there is a vertical partition, two sequential workitems may point to non-sequential memory segments: since the next sample is inside another CU, the workitem must go ot the next row of samples to stay inside its CU

        // CONSIDER THE FOLLOWING IDEA:
        // Use memory access similar to gradient computation, where sequantial workitems always evaluate sequential samples and make coalesced accesses. With this approach, one workitem may compute the partial system of a sample inside another CU
        // This causes some synchronization issues, given that the workitems will need to access the global/local memory to store the partial systems (it is not possible to sum them in private memory since they possibly belong to different CUs)
        
        // Maybe the performance is not that affected by the current approach:
        // In the worst case which is a 16x16 CU, we still have 16 sequential and aligned samples (the same size as a half warp) that must be fetched from global memory, whici sums 64 bytes of data
        // For CUs with width=32 we have an entire warp fetching 32 sequential and aligned gradient values, totalizing 128 bytes of data (maximum data obtained in single transaction for NVIDIA)
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        // Build the system of equations
        long tmp_atomic;
        // Computes a fraction of the entire system on private memory for the current workitem
        // It is necessary to go through the 128x128 samples, therefore each workitem processes 64 positions (128x128 divided by 256 workitems)
        // TODO: The number of passes may change when processing aligned and unaligned CUs. This must be verified
        // TODO: I believe the upper limit of "pass"A is always 64
        for(int pass=0; pass<(cuWidth*cuHeight)/itemsPerCu; pass++){           
            int iC[6], j, k, cx, cy;
            int idx = pass*itemsPerCu + lid%itemsPerCu; // lid%itemsPerCu represents the index of this item inside the sub-group
            
            // j and k -> Position of the sample inside the CTU [0,128]
            j = cuY + idx/cuWidth; 
            k = cuX + idx%cuWidth;       

            // cy and cx -> Center of the sub-block inside the CU
            cy = (((idx/cuWidth) >> 2) << 2) + 2; // cy represents the center of the sub-block (4x4) of current sample. If sample is in y=11, cy=10 (the center of the third block: 2, 6, 10, 14, ...)
            cx = (((idx%cuWidth) >> 2) << 2) + 2; // cx represents the center of the sub-block (4x4) of current sample. If sample is in x=4, cx=6 (the center of the second block: 2, 6, 10, 14, ...)

            // TODO: Improve this to avoid using if/else (maybe the if/else will no interfere in performance since all workitems will be true or false)
            if(nCP==2){
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] + cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] - cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }
            else{
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[4] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[5] = cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }
            
            // TODO: Test if using atomic operations (adding the sub-systems) directly over global memory improves performance
            for(int col=0; col<2*nCP; col++){
                for(int row=0; row<2*nCP; row++){
                    tmp_atomic = (long)iC[col] * (long)iC[row];
                    private_pEqualCoeff[col + 1][row] += tmp_atomic;
                }

                tmp_atomic = ((long)iC[col] * (long)predCU_then_error[j*CTU_WIDTH + k]) << 3;
                private_pEqualCoeff[col + 1][2*nCP] += tmp_atomic;
                // pEqualCoeff[col + 1][affineParamNum] += ((int64_t)iC[col] * pResidue[idx]) << 3;
            }
        }

        // TODO: Improve this to be adaptive to number of CPs
        // Copy the private fraction of the system to the global memory
        for(int col=0; col<2*nCP+1; col++){
            for(int row=0; row<2*nCP+1; row++){
                // Strides to the memory location of this workgroup by wg*256*7*7 and to the location of this workitem by id*7*7
                global_pEqualCoeff[wg*256*7*7 + lid*7*7 + col*7 + row] = private_pEqualCoeff[col][row];
            }
        }

        // Wait until all workitems have copied their systems to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // #########################################################################
        // ###### HERE IT SOLVES THE SYSTEM OF EQUATIONS AND UPDATE THE CPMVs ######
        // #########################################################################
        
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the global memory
            int virtual_cuIdx = lid;
            int virtual_cuX = HA_ALL_X_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx]; // CU position inside the CTU [0,128]
            int virtual_cuY = HA_ALL_Y_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx];

            // Copy values from first workitem
            // TODO: Make these for loops adaptive to number of CPs
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<2*nCP+1; row++){
                    // Stride memory to the local of this WG
                    private_pEqualCoeff[col][row] = global_pEqualCoeff[wg*256*7*7 + virtual_lid*7*7 + col*7 + row];
                }
            }

            // Reduce remaining workitems by adding over the first values
            for(int item=1; item<itemsPerCu; item++){
                // TODO: Make these for loops adaptive to number of CPs
                // for(int col=0; col<7; col++){
                for(int col=1; col<2*nCP+1; col++){
                    for(int row=0; row<2*nCP+1; row++){
                        private_pEqualCoeff[col][row] += global_pEqualCoeff[wg*256*7*7 + (virtual_lid+item)*7*7 + col*7 + row];
                    }
                }

            }

            // At this point, private_pEqualCoeff holds the final system of equations for the current CU (multiple partial systems were reduced in the previous for loop)

            // TODO: Make these for loops adaptive to number of CPs
            // Copy the system to private memory using double type to solve the system in sequence
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<2*nCP+1; row++){
                    private_dEqualCoeff[col][row] = (double) private_pEqualCoeff[col][row];
                }
            }

            // Export the final system of equations for one CU
            /*
            if(wg==targetWg && virtual_cuIdx==targetCuIdx){
                //printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, cuIdx);
                printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, lid); //lid equals virtual_cuIdx
                for(int col=0; col<iParaNum; col++){
                    for(int row=0; row<iParaNum; row++){
                        printf("%ld,", private_pEqualCoeff[col][row]);
                        // printf("%f,", private_dEqualCoeff[col][row]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/


            // BEGIN solving the system of equations
            int iOrder = affineParaNum;
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
                    break;
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

            // Export progress for a given WG
            /*
            // if(wg==targetWg && lid%itemsPerCu==0 && cuIdx==targetCuIdx){
            if(wg==targetWg && virtual_lid==targetCuIdx){
                printf("Iter: %d WG: %d Width: %d Height: %d X: %d Y: %d cuIdx: %d \n", iter, wg, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY, virtual_cuIdx);
                printf("\tCurrent CPMVs.       LT: %dx%d        RT: %dx%d    LB: %dx%d    isSpread: %d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y, isSpread);
            }
            //*/

            // Update the current MVs and clip to allowed values
            lCurrCpmvs[virtual_cuIdx].LT.x += intDeltaMv.s0;
            lCurrCpmvs[virtual_cuIdx].LT.y += intDeltaMv.s1;
            lCurrCpmvs[virtual_cuIdx].RT.x += intDeltaMv.s2;
            lCurrCpmvs[virtual_cuIdx].RT.y += intDeltaMv.s3;
            lCurrCpmvs[virtual_cuIdx].LB.x += intDeltaMv.s4;
            lCurrCpmvs[virtual_cuIdx].LB.y += intDeltaMv.s5;

            lCurrCpmvs[virtual_cuIdx] = clampCpmvs(lCurrCpmvs[virtual_cuIdx], MV_MIN, MV_MAX);
            
            lCurrCpmvs[virtual_cuIdx] = clipCpmvs(lCurrCpmvs[virtual_cuIdx], ctuX+virtual_cuX, ctuY+virtual_cuY, cuWidth, cuHeight, frameWidth, frameHeight);


            // Export progress for a given WG
            /*
            if(wg==targetWg && lid<cusPerCtu && virtual_cuIdx==targetCuIdx){
                printf("\tIteration %d  WxH %dx%d @ XY %dx%d\n", iter, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY);
                printf("\tCurrent deltas. deltaLT: %dx%d   deltaRT: %dx%d   deltaLB: %dx%d\n", intDeltaMv.s0, intDeltaMv.s1, intDeltaMv.s2, intDeltaMv.s3, intDeltaMv.s4, intDeltaMv.s5);
                printf("\tUpdated CPMVs.       LT: %dx%d        RT: %dx%d        LB: %dx%d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y);
                printf("\tCurrent COST: %ld\n", currCost[virtual_cuIdx]);
                printf("\n-----\n");
            }
            //*/

            // The next iteration will perform prediction with the new CPMVs and verify if this improves performance

        } // if(lid==0) for solving the system and updating the MVs
        
        // All workitems must wait until the CPMVs are updated before starting the next prediction
        barrier(CLK_LOCAL_MEM_FENCE);

    } // end for the five iterations of gradient


    // Write the best CPMVs and corresponding costs to global memory and return to host
    // @ OPTIMIZATION 3
    // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
    // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
    // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
    // if(lid%itemsPerCu==0){
    //     int use_opt3 = 0;
    if(lid < cusPerCtu){
        int use_opt3 = 1;

        int cuDimensionIdx = wg%HA_NUM_CU_SIZES;
        // Stride to the position of this CTU, then for the position of this CU size, and finally for the current CU index
        int virtual_cuIdx = lid;
        int returnArrayIdx;
        if(use_opt3)
            returnArrayIdx = ctuIdx*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuDimensionIdx] + virtual_cuIdx;
        else
            returnArrayIdx = ctuIdx*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuDimensionIdx] + cuIdx;

        // Export the updated cost/CPMV information for a given CU at the end of each iteration
        /*
        if(ctuIdx==0 && cuWidth==64 && cuHeight==16){           
            printf("\nCTU %d - CU %d\n  LT_X %d\n  LT_Y %d\n  RT_X %d\n  RT_Y %d\n  returnArrayIdx %d\n\n", ctuIdx, virtual_cuIdx, lBestCpmvs[virtual_cuIdx].LT.x, lBestCpmvs[virtual_cuIdx].LT.y, lBestCpmvs[virtual_cuIdx].RT.x, lBestCpmvs[virtual_cuIdx].RT.y, returnArrayIdx);
            // printf("\nCTU %d - CU %d\n  Cost %ld\n  Index %d\n", ctuIdx, virtual_cuIdx, bestCost[virtual_cuIdx], returnArrayIdx);

        }
        //*/
        if(use_opt3){
            gBestCost[returnArrayIdx] = bestCost[virtual_cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[virtual_cuIdx];
        }
        else{
            gBestCost[returnArrayIdx] = bestCost[cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[cuIdx];
        }
        
    }
}

__kernel void affine_gradient_mult_sizes_HA_3CPs(__global short *referenceFrameSamples, __global short *currentFrameSamples,const int frameWidth, const int frameHeight, const float lambda, __global short *horizontalGrad, __global short *verticalGrad, __global long *global_pEqualCoeff, __global long *gBestCost, __global Cpmvs *gBestCpmvs, __global long *debug, __global short *retCU){
    // TODO: Improve this to use 2 and 3 CPs
    int nCP = 3;
    // Used to debug the information of specific workitems and encoding stages
    int targetIter = 0;
    int targetWg = 16;
    int targetCuIdx = 0;
    int targetLid = 0;

    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    // Based on the WG index, the correct block size is selected
    // This assumes that each WG will process a single CU size inside a single CTU, considering ALIGNED CUs only
    int cuWidth = HA_WIDTH_LIST[wg%HA_NUM_CU_SIZES];
    int cuHeight = HA_HEIGHT_LIST[wg%HA_NUM_CU_SIZES];
    int cusPerCtu = HA_CUS_PER_CTU[wg%HA_NUM_CU_SIZES];
    int itemsPerCu = wgSize/cusPerCtu;
    int subBlocksPerCu = (cuWidth/4)*(cuHeight/4);
    int subBlockColumnsPerCu = cuWidth/4;
    int subBlockRowsPerCu = cuHeight/4;
    int cuIdx = lid/itemsPerCu;

    
    // Index of current CTU. Every HA_NUM_CU_SIZES workgroups share the same ctuIdx: one for each CU size
    int ctuIdx = wg/HA_NUM_CU_SIZES;
    
    // Derive position of current CTU based on WG, and position of current CU based on lid
    int ctusPerRow = ceil((float) frameWidth/CTU_WIDTH);
    int ctuX = (ctuIdx%ctusPerRow) * CTU_WIDTH;  // CTU position inside the frame
    int ctuY = (ctuIdx/ctusPerRow) * CTU_HEIGHT; 
    int cuColumnsPerCtu = CTU_WIDTH/cuWidth;

    int cuX = HA_ALL_X_POS[wg%HA_NUM_CU_SIZES][cuIdx]; // CU position inside the CTU [0,128]
    int cuY = HA_ALL_Y_POS[wg%HA_NUM_CU_SIZES][cuIdx];
    
    // Variables to keep track of the current and best MVs/costs
    // TODO: Even if the current CTU holds a single 128x128 CU, we are allocating MAX_CUS_PER_CTU positions in the array
    __local Cpmvs lBestCpmvs[MAX_HA_CUS_PER_CTU], lCurrCpmvs[MAX_HA_CUS_PER_CTU];
    __local long currCost[MAX_HA_CUS_PER_CTU], bestCost[MAX_HA_CUS_PER_CTU], bestDist[MAX_HA_CUS_PER_CTU];
    
    int inheritCpmvs = 1;
    Cpmvs predCpmvs;
    predCpmvs.LT.x = 0;
    predCpmvs.LT.y = 0;
    predCpmvs.RT.x = 0;
    predCpmvs.RT.y = 0;
    predCpmvs.LB.x = 0;
    predCpmvs.LB.y = 0;

    if(inheritCpmvs){
        int use_opt3 = 1;

        int cuDimensionIdx = wg%HA_NUM_CU_SIZES;
        // Stride to the position of this CTU, then for the position of this CU size, and finally for the current CU index
        int returnArrayIdx;

        returnArrayIdx = ctuIdx*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuDimensionIdx] + cuIdx;

        // Export the updated cost/CPMV information for a given CU at the end of each iteration
        /*
        if(ctuIdx==49 && cuWidth==64 && cuHeight==64){           
            // printf("\nCTU %d - CU %d\n  LT_X %d\n  LT_Y %d\n  RT_X %d\n  RT_Y %d\n  returnArrayIdx %d\n\n", ctuIdx, cuIdx, lBestCpmvs[cuIdx].LT.x, lBestCpmvs[cuIdx].LT.y, lBestCpmvs[cuIdx].RT.x, lBestCpmvs[cuIdx].RT.y, returnArrayIdx);
            printf("\nCTU %d - CU %d\n  Cost %ld\n  Index %d\n", ctuIdx, virtual_cuIdx, bestCost[virtual_cuIdx], returnArrayIdx);

        }
        //*/

        // Read the result for affine with 2 CPs in the same CU
        predCpmvs = gBestCpmvs[returnArrayIdx];   

        int shift = MAX_CU_DEPTH;

        int vx2 = (predCpmvs.LT.x << shift) - ((predCpmvs.RT.y - predCpmvs.LT.y) << (shift + (int)log2((float)cuHeight) - (int)log2((float)cuWidth)));        
        int vy2 = (predCpmvs.LT.y << shift) + ((predCpmvs.RT.x - predCpmvs.LT.x) << (shift + (int)log2((float)cuHeight) - (int)log2((float)cuWidth)));

        int offset = (1 << (shift - 1));
        vx2 = (vx2 + offset - (vx2 >= 0)) >> shift;
        vy2 = (vy2 + offset - (vy2 >= 0)) >> shift;

        predCpmvs.LB.x = vx2;
        predCpmvs.LB.y = vy2;

        // Adjust the precision of derived CPMVs
        predCpmvs.LB.x = clamp(vx2, -(1 << 17), (1 << 17) - 1);
        predCpmvs.LB.y = clamp(vy2, -(1 << 17), (1 << 17) - 1);

        int2 mv2;
        mv2.x = predCpmvs.LB.x;
        mv2.y = predCpmvs.LB.y;
        mv2 = roundAffinePrecInternal2Amvr(mv2, 4);
        mv2 = clipMv(mv2, ctuX+cuX, ctuY+cuY, cuWidth, cuHeight, frameWidth, frameHeight);
        predCpmvs.LB.x = mv2.x;
        predCpmvs.LB.y = mv2.y;
    }

    // Hold a fraction of the total distortion of current CPMVs (each workitem uses one position)
    // TODO: Modify this 256 to use a MACRO value. Different architectures may support fewer workitems per WG
    __local long local_cumulativeSATD[256];

    // TODO: All these VECTORIZED_MEMORY if/elses are used to control memory access: either using vload/vstore operations, or indexing the values one by one
    // Fetch current CU for private memory. Only the samples used by this workitem are fetched to decrease memory bandwidth
    int16 currentCU_subBlock[2]; // For HALF-ALIGNED CUs, wach workitem is responsible for AT MOST TWO sub-blocks
    int currentCU[CTU_WIDTH*CTU_HEIGHT];
    int totalArea = (cuWidth*cuHeight)*HA_CUS_PER_CTU[wg%HA_NUM_CU_SIZES]; // Total area encoded by this WG. Depending on the CU size and its group (type of HALF-ALIGNMENT), each WG can encode 50%, 25% or 12.5% of a CTU
    int nSubblocksForWG = totalArea/(4*4); // Number of subblocks encoded by this WG
    // TODO: Correct this 256 to be adaptive to workgroup size
    int nPasses = nSubblocksForWG/256; // We have 256 WGs. The number of subblocks encoded by each workitem is the number of passes. It may assume values: 2, 1, or 0.5 !!!!! Half subblocks per WI occurs for CUs 16x16 of G3 and G4, where we have only 8 CUs per CTU
    
    // if(lid%itemsPerCu==0) // First LID of each sub-group
    //     printf("wg,lid %d,%d -> @(%dx%d) %d passes\n", wg, lid, cuX, cuY, nPasses);

    if(VECTORIZED_MEMORY){
        int stridePerPass = itemsPerCu; //(cuHeight*2)/(CTU_WIDTH/cuWidth); // Number of sub-blocks between two passes of the same workitem
        for(int pass=0; pass<nPasses; pass++){   
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu) << 2;
            sub_X = (index%subBlockColumnsPerCu) << 2;
            int offset = (ctuY + cuY + sub_Y)*frameWidth + ctuX + cuX + sub_X;
            currentCU_subBlock[pass].lo.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].lo.hi = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.lo = convert_int4(vload4(offset/4, currentFrameSamples));
            offset += frameWidth;
            currentCU_subBlock[pass].hi.hi = convert_int4(vload4(offset/4, currentFrameSamples));
        }
    }
    else{
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // THIS WILL NOT WORK BECAUSE THE INDEXING WAS NOT CORRECTED TO SUPPORT MORE BLOCK SIZES
        // cuX and cuY were modified and must be combined with ctuX and ctuY
        // correct this code if VECTORIZED_MEMORY=0

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
    
    // At first, this holds the predicted samples for the entire CTU
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
    int numGradientIter = select(4, 5, nCP==2); // Number of iteration in gradient ME search (i.e., number of CPMV updates after predicted MV)
    // TODO: Maybe it is faster to make all workitems write to the same local variable than using if()+barrier
    // Initial MVs from AMVP and initialize rd-cost
    if(lid%itemsPerCu==0){ // The first id of each sub-group initializes the CPMVs of its CU
        bestCost[cuIdx] = MAX_LONG;
        lCurrCpmvs[cuIdx] = predCpmvs;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // sync all workitems after initializing the MV
    
    // Starts the motion estimation based on gradient and optical flow
    __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 

    // Here each workitem verify if its CU lies completely within the frame
    // This is used to avoid performing the prediction of CUs "outside the frame" when the last CTU row is composed of partil CUs (i.e., vertical resolution not multiple of 128)
    int isWithinFrame =   (ctuX + cuX + cuWidth <= frameWidth)
                        && (ctuY + cuY + cuHeight <= frameHeight);

    for(int iter=0; iter<numGradientIter+1; iter++){ // +1 because we need to conduct the initial prediction (with AMVP) in addition to the gradient-ME
        
        // ###############################################################################
        // ###### HERE IT STARTS THE PREDICTION OF THE BLOCK AND COMPUTES THE COSTS ######
        // ###############################################################################
  
        // Reset SATD for current iteration
        cumulativeSATD = 0;
        local_cumulativeSATD[lid] = 0;

        // nPasses is the number of sub-blocks predicted by each workitem. This is wither 1 or 2 for HALF-ALIGNED blocks
        int stridePerPass = itemsPerCu;
        int nPasses = (cuWidth*cuHeight*HA_CUS_PER_CTU[wg%HA_NUM_CU_SIZES]*4)/(CTU_WIDTH*CTU_HEIGHT);

        for(int pass=0; pass<isWithinFrame*nPasses; pass++){   // When isWithinFrame==0 we do not conduct any passes and avoid performing the prediction and distortion
            int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
            int sub_X, sub_Y;
            sub_Y = (index/subBlockColumnsPerCu) << 2;
            sub_X = (index%subBlockColumnsPerCu) << 2;
            
            // Derive sub-MVs and determine if they are too spread (when spread, all sub-blocks have the same MV)
            // TODO: Create a wrapper function that receives the nCPs and calls the proper sub-MV derivation
            if(nCP==2){
                subMv_and_spread = deriveMv2Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                subMv_and_spread = deriveMv3Cps_and_spread(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }

            subMv = subMv_and_spread.xy;
            isSpread = subMv_and_spread.z;

            // These deltas are used during PROF
            // TODO: Create a wrapper function that receives the nCPs and calls the proper function for each CP
            // While enablePROF=0 this part of the code makes no difference
            ///*
            if(nCP==2){
                deltaHorVec = getHorizontalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF2Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            else{
                deltaHorVec = getHorizontalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
                deltaVerVec = getVerticalDeltasPROF3Cps(lCurrCpmvs[cuIdx], cuWidth, cuHeight, sub_X, sub_Y, bipred);
            }
            //*/

            subMv = roundAndClipMv(subMv, ctuX+cuX, ctuY+cuY,  cuWidth, cuHeight, frameWidth, frameHeight);

            // Export the sub-MV for current sub-block
            /*
            if(wg==targetWg && lid==targetLid){
                printf("iter %d, WG %d, lid %d, WH=%d XY=%d LT %dx%d, RT %dx%d\n", iter, wg, lid, cuWidth, cuHeight, ctuX+cuX, ctuY+cuY, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);
                printf("  subXY %dx%d -->> %dx%d\n", sub_X, sub_Y, subMv.x, subMv.y);               
            }
            //*/

            // Split the int and fractional part of the MVs to perform prediction/filtering
            // CPMVs are represented in 1/16 pixel accuracy (4 bits for frac), that is why we use >>4 and &15 to get int and frac
            int2 subMv_int, subMv_frac;
            subMv_int.x  = subMv.x >> 4;
            subMv_frac.x = subMv.x & 15;
            subMv_int.y  = subMv.y >> 4;
            subMv_frac.y = subMv.y & 15;

            // Top-left corner of the reference block in integer pixels
            int refPosition = (ctuY+cuY+sub_Y)*frameWidth + ctuX+cuX+sub_X + subMv_int.y*frameWidth + subMv_int.x;

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
            int leftSlack = ctuX+cuX+sub_X + subMv_int.x - ( N/2 - 1 );
            // Positive right slack means we have enough columns to the right. Negative represents the number of columns outside (to the right) the frame                                
            int rightSpam = ctuX+cuX+sub_X + subMv_int.x + ( N/2 );
            int rightSlack = frameWidth - 1 - rightSpam;
            // Positive top slack means we have enough rows to the top. Negative represents the number of rows outside (to the top) the frame
            int topSlack = ctuY+cuY+sub_Y + subMv_int.y - ( N/2 - 1 );
            // Positive bottom slack means we have enough rows to the bottom. Negative represents the number of rows outside (to the bottom) the frame
            int bottomSpam = ctuY+cuY+sub_Y + subMv_int.y + ( N/2 );
            int bottomSlack = frameHeight - 1 - bottomSpam;

            // Fetch reference window (i.e., samples that may be used for filtering)
            bool leftCorrect, rightCorrect, topCorrect, bottomCorrect, topLeftCorrect, topRightCorrect, bottomLeftCorrect, bottomRightCorrect; // Used to verify, for each sample, if it lies "outside" the frame
            int currSample; // Used to avoid if/else structures during left/right correction
            int properIdx; // This variable is used to update the index of the reference sample without accessing global memory until the proper index is found. It avoid segmentation fault when the motion vectors point outside the reference frame and it is necessary to "correct" the index to a sample inside the frame during a "virtual padding"
            
            // TODO: Unroll the following loop (or not?)
            for(int row=0; row<windowHeight; row++){
                for(int col=0; col<windowWidth; col++){           
                    // First computes the "individual" left, right, top and bottom corrections, disconsidering any other correction
                    leftCorrect = select(true,false,leftSlack+col>=0); // moving right (i.e., moving column-wise) increases left slack and decreases right slack
                    rightCorrect = select(true,false,rightSlack-col+7>=0); // this +7 corrects the difference between the reference block and the reference window
                    topCorrect = select(true,false,topSlack+row>=0); // moving downwards (i.e., moving row-wise) increses the top slack and decreases the bottom slack
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
                    currSample = (short) referenceFrameSamples[properIdx];
                    referenceWindow[row*windowWidth+col] = currSample;
                }
            }

            // This is used to export the reference window
            /*
            // if(wg==targetWg && iter==targetIter && sub_X==0 && sub_Y==0 && lid==targetLid){
            if(wg==targetWg && iter==targetIter){
                printf("REFERENCE WINDOW\n");
                printf("  Iter: %d WG: %d subX: %d subY: %d cuIdx: %d LT: %dx%d RT: %dx%d\n", iter, wg, sub_X, sub_Y, cuIdx, lCurrCpmvs[cuIdx].LT.x, lCurrCpmvs[cuIdx].LT.y, lCurrCpmvs[cuIdx].RT.x, lCurrCpmvs[cuIdx].RT.y);

                for(int i=0; 1<windowHeight; i++){
                    for(int j=0; j<windowWidth; j++){
                        printf("%d,", referenceWindow[i*windowWidth+j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/

            // This line computes the complete prediction: horizontal filtering, vertical filtering, and PROF
            predBlock = horizontal_vertical_filter_new(referenceWindow, subMv_int, frameWidth, frameHeight, 4, 4, subMv_frac.x, subMv_frac.y, isSpread, deltaHorVec, deltaVerVec, enablePROF);     


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
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s0;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s1;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s2;
            predCU_then_error[(cuY+sub_Y+0)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s3;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s4;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s5;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+2] = predBlock.s6;
            predCU_then_error[(cuY+sub_Y+1)*CTU_WIDTH + cuX+sub_X+3] = predBlock.s7;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+0] = predBlock.s8;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+1] = predBlock.s9;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+2] = predBlock.sa;
            predCU_then_error[(cuY+sub_Y+2)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sb;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+0] = predBlock.sc;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+1] = predBlock.sd;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+2] = predBlock.se;
            predCU_then_error[(cuY+sub_Y+3)*CTU_WIDTH + cuX+sub_X+3] = predBlock.sf;

            // Compute the SATD 4x4 for the current sub-block and accumulate on the private accumulator
            if(VECTORIZED_MEMORY){
                satd = satd_4x4(currentCU_subBlock[pass], predBlock);
            }
            else{
                satd = satd_4x4(original_block, predBlock);
            }
            cumulativeSATD += (long) satd;
        }
       
        // Each position of local_cumulativeSATD will hold the cumulativeSATD of the sub-blocks of current workitem
        // Wait until all workitems compute their cumulativeSATD and reduce (i.e., sum all SATDs) to the first position of local_cumulativeSATD
        local_cumulativeSATD[lid] = cumulativeSATD;
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Only necessary when we are exporting the predicted CTU
        if(wg==targetWg && lid==targetLid && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                    // printf("%d,", predCU_then_error[i*128+j]);
                }
                // printf("\n");
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        /* Only necessary when we are exporting the predicted CU
        if(wg==targetWg && lid==targetLid && iter==targetIter){
            printf("Exporting... cuX=%d cuY=%d\n", cuX, cuY);
            for(int i=0; i<cuHeight; i++){
                for(int j=0; j<cuWidth; j++){
                    retCU[(cuY+i)*128+cuX+j] = predCU_then_error[(cuY+i)*128+cuX+j];
                    // printf("%d,", predCU_then_error[i*128+j]);
                }
                // printf("\n");
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // necessary to avoid writing down the error that is overwritten on predCU_then_error
        //*/

        // TODO: Only the first item of each sub-group is used to reduce the SATDs. It may be posible to accelerarte using multiple ids
        // Reduce the smaller SATDs, compute the cost and update CPMVs
       if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the borders of each CU
            int virtual_cuIdx = lid;
            for(int i=1; i<itemsPerCu; i++){
                local_cumulativeSATD[virtual_lid] += local_cumulativeSATD[virtual_lid+i];
            }

            Cpmvs zeroCpmvs;
            zeroCpmvs.LT.x = 0;
            zeroCpmvs.LT.y = 0;
            zeroCpmvs.RT.x = 0;
            zeroCpmvs.RT.y = 0;
            zeroCpmvs.LB.x = 0;
            zeroCpmvs.LB.y = 0;
            // Although affine prediction with 3 CPs has a starting point derived from affine with 2 CPs, the predictor used for bitrate is always zero
            bitrate = calc_affine_bits(AFFINE_MV_PRECISION_QUARTER, nCP, lCurrCpmvs[virtual_cuIdx], zeroCpmvs);

            // TODO: This "+4" represents the ruiBits of the VTM-12.0 encoder, and it is the base-bitrate for using affine. The "+4" when using low delay with a single reference frame. Improve this when using multiple reference frames
            currCost[virtual_cuIdx] = local_cumulativeSATD[virtual_lid] + (long) getCost(bitrate + 4, lambda);

            // If the current CPMVs are not better than the previous (rd-cost wise), the best CPMVs are not updated but the next iteration continues from the current CPMVs
            if(currCost[virtual_cuIdx] < bestCost[virtual_cuIdx]){
                bestCost[virtual_cuIdx] = currCost[virtual_cuIdx];
                bestDist[virtual_cuIdx] = local_cumulativeSATD[virtual_lid];

                lBestCpmvs[virtual_cuIdx] = lCurrCpmvs[virtual_cuIdx];
            }
        }

        // #########################################################################################################
        // ###### AT THIS POINT WE HAVE COMPUTED THE COST OF THE CURRENT CPMVs. IF IT IS THE LAST ITERATION   ######
        // ###### THERE IS NO NEED TO PERFORM THE GRADIENT REFINEMENT SINCE THE UPDATED MV WILL NOT BE TESTED ######
        // #########################################################################################################

        if(iter == numGradientIter){
            break;
        }

        // #########################################################################################
        // ###### HERE IT STARTS COMPUTING THE GRADIENTS AND BUILDING THE SYSTEM OF EQUATIONS ######
        // #########################################################################################

        int centerSample;
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!
        //      IMPORTANT
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!
        // TODO: The general case of gradient is not optimized for HALF-ALIGNED CUs. For these CUs, each CU group covers either 25% or 50% of the CTU area, but the following code is computing the gradient for 100% of the samples
        // It may be possible to optimize the schedule so the gradient is computed only for the valid CUs
        // Compute general case of gradient. It will be necessary to refill the border and corner values
        // Each workitem will compute the gradient on 64 positions (128x128 samples, 256 workitems)
        // This part makes no distinction between subgroups. All data (gradient and predicted samples) are stored on __local or __global memory. This is better to optimize memory transactions
        for(int pass=0; pass<64; pass++){
            centerSample = pass*256 + lid;

            // Values at the border of CUs and CTUs are not valid. However, we can compute the values at the border of CUs to maintain regularity in the kernel and overwrite these values in sequence. Valeus at the borders of a CTU CANNOT be computed since we do not have the neighboring samples
            int isValid = !((centerSample%CTU_WIDTH==0) || (centerSample%CTU_WIDTH==CTU_WIDTH-1) || (centerSample/CTU_WIDTH==0) || (centerSample/CTU_WIDTH==CTU_HEIGHT-1));

            // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
            if(isValid){
                // Stride memory to location of current CTU (1 wg processes all CUs with a given size for one CTU
                // TODO: Verify this behavior when processing unaligned CUs
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample-CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+1]-2*predCU_then_error[centerSample-1] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample+CTU_WIDTH-1];    
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = predCU_then_error[centerSample+CTU_WIDTH-1]-predCU_then_error[centerSample-CTU_WIDTH-1] + 2*predCU_then_error[centerSample+CTU_WIDTH]-2*predCU_then_error[centerSample-CTU_WIDTH] + predCU_then_error[centerSample+CTU_WIDTH+1]-predCU_then_error[centerSample-CTU_WIDTH+1];
            }
            else{
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + centerSample] = 0;
            }
        }
        
        // Wait until all workitems have computed their gradients
        barrier(CLK_LOCAL_MEM_FENCE);

        // TODO: It may be possible to use workitems of the same warp to fill the border of different CUs, but it would require computing a different value for cuY and cuX
        // if(lid%itemsPerCu==0){
        // @ OPTIMIZATION 1
        // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
        // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
        // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
        // TODO: Maybe we can improve this to avoid computing the gradient of samples "outside the frame"?
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the borders of each CU
            int virtual_cuIdx = lid;
            int virtual_cuX = HA_ALL_X_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx]; // CU position inside the CTU [0,128]
            int virtual_cuY = HA_ALL_Y_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx];

            // Fills the first and last rows of gradient with the correct values (upper/lower row values)
            for(int col=0; col<cuWidth; col++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+CTU_WIDTH];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + col+(cuHeight-2)*CTU_WIDTH];
            }
            // Fills the first and last columns of the gradient with the correct values (left/right column values)
            for(int row=0; row<cuHeight; row++){
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];

                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH + 1];
                verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + row*CTU_WIDTH+cuWidth-2];
            }
            // Fills the four corners
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + 0] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH + 1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + CTU_WIDTH+cuWidth-2];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+1];
            
            horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];
            verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-1)*CTU_WIDTH+cuWidth-1] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + virtual_cuY*CTU_WIDTH+virtual_cuX + (cuHeight-2)*CTU_WIDTH+cuWidth-2];        
        }
        
        // Wait until the border values are filled. They are required in the next stage
        barrier(CLK_LOCAL_MEM_FENCE);
                
        // Since the prediction will not be used again in the same iteration, it is possible to reuse the __local array to store the error (pred = pred-orig). The error will be used to build the system of equations
        // Each workitem computes the error of a subset of sub-blocks (the same sub-blocks it predicted earlier)
        
        // TODO: Correct this 256 to be adaptive to workgroup size
        nPasses = nSubblocksForWG/256; // We have 256 WGs. The number of subblocks encoded by each workitem is the number of passes. It may assume values: 2 or 1
        if(VECTORIZED_MEMORY){
            // TODO: Maybe this stridePerPass can be computed exactly as "idx" when computing the system of equations, using itemsPerCu
            stridePerPass = itemsPerCu; // Number of sub-blocks between two passes of the same workitem
            for(int pass=0; pass<nPasses; pass++){   
                int index = pass*stridePerPass + lid%itemsPerCu; // lid%itemsPerCu represents the index of the current id inside its sub-group (each sub-group processes one CU)
                int sub_X, sub_Y;
                sub_Y = (index/subBlockColumnsPerCu) << 2;
                sub_X = (index%subBlockColumnsPerCu) << 2;

                int cuOffset = (cuY+sub_Y)*CTU_WIDTH + cuX+sub_X;
                short4 linePred, lineDiff;
                
                // Read one line from predicted sub-block, subtract from original CU, overwrite predCU_then_error
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
                
                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].lo.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.lo) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);

                cuOffset += CTU_WIDTH;
                linePred = vload4(cuOffset/4, predCU_then_error);
                lineDiff = convert_short4(currentCU_subBlock[pass].hi.hi) - linePred;
                vstore4(lineDiff, cuOffset/4, predCU_then_error);
            }   
        }
        else{
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // This error computation is not correct when encoding CUs smaller than 128x128. 
            // It is necessary to use cuIdx, cuX and cuY to correct the indexing
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        // Wait until all woritems have conducted their work over the predCU_then_error buffer
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Only necessary when we are exporting the prediction error
        barrier(CLK_LOCAL_MEM_FENCE);
        if(wg==targetWg && lid==0 && iter==targetIter){
            for(int i=0; i<128; i++){
                for(int j=0; j<128; j++){
                    retCU[i*128+j] = predCU_then_error[i*128+j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); //
        //*/ 
        

        // ######################################################################
        // ###### HERE IT STARTS BUILDING THE PARTIAL SYSTEMS OF EQUATIONS ######
        // ######################################################################

        // Holds the "complete" system of equations to be solved during gradient-ME. For 2 CPs, only the first [5][4] indices are used.
        // __local long local_pEqualCoeff[MAX_CUS_PER_CTU][7][7]; 
        // The long variable is used when building the system of equations (only integers). The double variable is used for solving the system
        __private long private_pEqualCoeff[7][7];
        __private double private_dEqualCoeff[7][7];
        // The double variable is purely based on the affine parameters, the int variable is used after rounding the double to the proper precision
        double8 dDeltaMv = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int8 intDeltaMv;

        // TODO: Use these "iParaNum" and "affineParaNum" in the place of all these 6's and 7's
        int iParaNum = select(5, 7, nCP==3);
        int affineParaNum = iParaNum - 1;

        // TODO: Improve this to be adaptive to number of CPs
        // Initialize the system of equations
        for(int i=0; i<7; i++){
            for(int j=0; j<7; j++){
                private_pEqualCoeff[i][j] = 0;
            }
        }

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        // TODO:
        // The partial system only depends on the prediction error and gradient at a single position inside the CU
        // The "idx" variable assumes values in the range [0,n], where each CU has n samples
        // cuX and cuY are combined to derive the position of the error/gradient inside the CTU

        // Currently, each workitem only processes the samples inside its own CU. This makes the logic easier, but may cause performance degradation
        // Whenever there is a vertical partition, two sequential workitems may point to non-sequential memory segments: since the next sample is inside another CU, the workitem must go ot the next row of samples to stay inside its CU

        // CONSIDER THE FOLLOWING IDEA:
        // Use memory access similar to gradient computation, where sequantial workitems always evaluate sequential samples and make coalesced accesses. With this approach, one workitem may compute the partial system of a sample inside another CU
        // This causes some synchronization issues, given that the workitems will need to access the global/local memory to store the partial systems (it is not possible to sum them in private memory since they possibly belong to different CUs)
        
        // Maybe the performance is not that affected by the current approach:
        // In the worst case which is a 16x16 CU, we still have 16 sequential and aligned samples (the same size as a half warp) that must be fetched from global memory, whici sums 64 bytes of data
        // For CUs with width=32 we have an entire warp fetching 32 sequential and aligned gradient values, totalizing 128 bytes of data (maximum data obtained in single transaction for NVIDIA)
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        // Build the system of equations
        long tmp_atomic;
        // Computes a fraction of the entire system on private memory for the current workitem
        // It is necessary to go through the 128x128 samples, therefore each workitem processes 64 positions (128x128 divided by 256 workitems)
        // TODO: The number of passes may change when processing aligned and unaligned CUs. This must be verified
        // TODO: I believe the upper limit of "pass"A is always 64
        for(int pass=0; pass<(cuWidth*cuHeight)/itemsPerCu; pass++){           
            int iC[6], j, k, cx, cy;
            int idx = pass*itemsPerCu + lid%itemsPerCu; // lid%itemsPerCu represents the index of this item inside the sub-group
            
            // j and k -> Position of the sample inside the CTU [0,128]
            j = cuY + idx/cuWidth; 
            k = cuX + idx%cuWidth;       

            // cy and cx -> Center of the sub-block inside the CU
            cy = (((idx/cuWidth) >> 2) << 2) + 2; // cy represents the center of the sub-block (4x4) of current sample. If sample is in y=11, cy=10 (the center of the third block: 2, 6, 10, 14, ...)
            cx = (((idx%cuWidth) >> 2) << 2) + 2; // cx represents the center of the sub-block (4x4) of current sample. If sample is in x=4, cx=6 (the center of the second block: 2, 6, 10, 14, ...)

            // TODO: Improve this to avoid using if/else (maybe the if/else will no interfere in performance since all workitems will be true or false)
            if(nCP==2){
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] + cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k] - cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }
            else{
                iC[0] = horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[1] = cx * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[2] = verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[3] = cx * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[4] = cy * horizontalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
                iC[5] = cy * verticalGrad[wg*CTU_WIDTH*CTU_HEIGHT + j*CTU_WIDTH + k];
            }
            
            // TODO: Test if using atomic operations (adding the sub-systems) directly over global memory improves performance
            for(int col=0; col<2*nCP; col++){
                for(int row=0; row<2*nCP; row++){
                    tmp_atomic = (long)iC[col] * (long)iC[row];
                    private_pEqualCoeff[col + 1][row] += tmp_atomic;
                }

                tmp_atomic = ((long)iC[col] * (long)predCU_then_error[j*CTU_WIDTH + k]) << 3;
                private_pEqualCoeff[col + 1][2*nCP] += tmp_atomic;
                // pEqualCoeff[col + 1][affineParamNum] += ((int64_t)iC[col] * pResidue[idx]) << 3;
            }
        }

        // TODO: Improve this to be adaptive to number of CPs
        // Copy the private fraction of the system to the global memory
        for(int col=0; col<2*nCP+1; col++){
            for(int row=0; row<2*nCP+1; row++){
                // Strides to the memory location of this workgroup by wg*256*7*7 and to the location of this workitem by id*7*7
                global_pEqualCoeff[wg*256*7*7 + lid*7*7 + col*7 + row] = private_pEqualCoeff[col][row];
            }
        }

        // Wait until all workitems have copied their systems to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // #########################################################################
        // ###### HERE IT SOLVES THE SYSTEM OF EQUATIONS AND UPDATE THE CPMVs ######
        // #########################################################################
        
        if(lid < cusPerCtu){
            int virtual_lid = lid*itemsPerCu; // This is used to obtain the first lid inside each CU. Then, these virtual lids will be used to index the global memory
            int virtual_cuIdx = lid;
            int virtual_cuX = HA_ALL_X_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx]; // CU position inside the CTU [0,128]
            int virtual_cuY = HA_ALL_Y_POS[wg%HA_NUM_CU_SIZES][virtual_cuIdx];

            // Copy values from first workitem
            // TODO: Make these for loops adaptive to number of CPs
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<2*nCP+1; row++){
                    // Stride memory to the local of this WG
                    private_pEqualCoeff[col][row] = global_pEqualCoeff[wg*256*7*7 + virtual_lid*7*7 + col*7 + row];
                }
            }

            // Reduce remaining workitems by adding over the first values
            for(int item=1; item<itemsPerCu; item++){
                // TODO: Make these for loops adaptive to number of CPs
                // for(int col=0; col<7; col++){
                for(int col=1; col<2*nCP+1; col++){
                    for(int row=0; row<2*nCP+1; row++){
                        private_pEqualCoeff[col][row] += global_pEqualCoeff[wg*256*7*7 + (virtual_lid+item)*7*7 + col*7 + row];
                    }
                }

            }

            // At this point, private_pEqualCoeff holds the final system of equations for the current CU (multiple partial systems were reduced in the previous for loop)

            // TODO: Make these for loops adaptive to number of CPs
            // Copy the system to private memory using double type to solve the system in sequence
            // for(int col=0; col<7; col++){
            for(int col=1; col<2*nCP+1; col++){
                for(int row=0; row<2*nCP+1; row++){
                    private_dEqualCoeff[col][row] = (double) private_pEqualCoeff[col][row];
                }
            }

            // Export the final system of equations for one CU
            /*
            if(wg==targetWg && virtual_cuIdx==targetCuIdx){
                //printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, cuIdx);
                printf("SYSTEM OF EQUATIONS, WG=%d, cuIdx=%d\n",wg, lid); //lid equals virtual_cuIdx
                for(int col=0; col<iParaNum; col++){
                    for(int row=0; row<iParaNum; row++){
                        printf("%ld,", private_pEqualCoeff[col][row]);
                        // printf("%f,", private_dEqualCoeff[col][row]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            //*/


            // BEGIN solving the system of equations
            int iOrder = affineParaNum;
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
                    break;
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

            // Export progress for a given WG
            /*
            // if(wg==targetWg && lid%itemsPerCu==0 && cuIdx==targetCuIdx){
            if(wg==targetWg && virtual_lid==targetCuIdx){
                printf("Iter: %d WG: %d Width: %d Height: %d X: %d Y: %d cuIdx: %d \n", iter, wg, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY, virtual_cuIdx);
                printf("\tCurrent CPMVs.       LT: %dx%d        RT: %dx%d    LB: %dx%d    isSpread: %d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y, isSpread);
            }
            //*/

            // Update the current MVs and clip to allowed values
            lCurrCpmvs[virtual_cuIdx].LT.x += intDeltaMv.s0;
            lCurrCpmvs[virtual_cuIdx].LT.y += intDeltaMv.s1;
            lCurrCpmvs[virtual_cuIdx].RT.x += intDeltaMv.s2;
            lCurrCpmvs[virtual_cuIdx].RT.y += intDeltaMv.s3;
            lCurrCpmvs[virtual_cuIdx].LB.x += intDeltaMv.s4;
            lCurrCpmvs[virtual_cuIdx].LB.y += intDeltaMv.s5;

            lCurrCpmvs[virtual_cuIdx] = clampCpmvs(lCurrCpmvs[virtual_cuIdx], MV_MIN, MV_MAX);
            
            lCurrCpmvs[virtual_cuIdx] = clipCpmvs(lCurrCpmvs[virtual_cuIdx], ctuX+virtual_cuX, ctuY+virtual_cuY, cuWidth, cuHeight, frameWidth, frameHeight);


            // Export progress for a given WG
            /*
            if(wg==targetWg && lid<cusPerCtu && virtual_cuIdx==targetCuIdx){
                printf("\tIteration %d  WxH %dx%d @ XY %dx%d\n", iter, cuWidth, cuHeight, ctuX+virtual_cuX, ctuY+virtual_cuY);
                printf("\tCurrent deltas. deltaLT: %dx%d   deltaRT: %dx%d   deltaLB: %dx%d\n", intDeltaMv.s0, intDeltaMv.s1, intDeltaMv.s2, intDeltaMv.s3, intDeltaMv.s4, intDeltaMv.s5);
                printf("\tUpdated CPMVs.       LT: %dx%d        RT: %dx%d        LB: %dx%d\n", lCurrCpmvs[virtual_cuIdx].LT.x, lCurrCpmvs[virtual_cuIdx].LT.y, lCurrCpmvs[virtual_cuIdx].RT.x, lCurrCpmvs[virtual_cuIdx].RT.y, lCurrCpmvs[virtual_cuIdx].LB.x, lCurrCpmvs[virtual_cuIdx].LB.y);
                printf("\tCurrent COST: %ld\n", currCost[virtual_cuIdx]);
                printf("\n-----\n");
            }
            //*/

            // The next iteration will perform prediction with the new CPMVs and verify if this improves performance

        } // if(lid==0) for solving the system and updating the MVs
        
        // All workitems must wait until the CPMVs are updated before starting the next prediction
        barrier(CLK_LOCAL_MEM_FENCE);

    } // end for the five iterations of gradient


    // Write the best CPMVs and corresponding costs to global memory and return to host
    // @ OPTIMIZATION 3
    // This conditional is an improvement over testing lid%itemsPerCu==0. Using lid%itemsPerCu==0 will require multiple warps being executed simultaneously, with a single workitem doing useful work and the remaining idle
    // This improvement will use multiple itens inside the same warp, reducing the number of processors being occupied
    // virtual_variables are used to simulate the behavior of testing lid%itemsPerCu==0
    // if(lid%itemsPerCu==0){
    //     int use_opt3 = 0;

    

    if(lid < cusPerCtu){
        int use_opt3 = 1;

        int cuDimensionIdx = wg%HA_NUM_CU_SIZES;
        // Stride to the position of this CTU, then for the position of this CU size, and finally for the current CU index
        int virtual_cuIdx = lid;
        int returnArrayIdx;
        if(use_opt3)
            returnArrayIdx = ctuIdx*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuDimensionIdx] + virtual_cuIdx;
        else
            returnArrayIdx = ctuIdx*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuDimensionIdx] + cuIdx;

        // Export the updated cost/CPMV information for a given CU at the end of each iteration
        /*
        if(wg==targetWg && virtual_cuIdx==targetCuIdx){           
            printf("\nCTU %d - CU %d\n  LT_X %d\n  LT_Y %d\n  RT_X %d\n  RT_Y %d\n  Cost %ld\n  returnArrayIdx %d\n\n", ctuIdx, virtual_cuIdx, lBestCpmvs[virtual_cuIdx].LT.x, lBestCpmvs[virtual_cuIdx].LT.y, lBestCpmvs[virtual_cuIdx].RT.x, lBestCpmvs[virtual_cuIdx].RT.y, bestCost[virtual_cuIdx], returnArrayIdx);
            // printf("\nCTU %d - CU %d\n  Cost %ld\n  Index %d\n", ctuIdx, virtual_cuIdx, bestCost[virtual_cuIdx], returnArrayIdx);
        }
        //*/
        if(use_opt3){
            gBestCost[returnArrayIdx] = bestCost[virtual_cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[virtual_cuIdx];
        }
        else{
            gBestCost[returnArrayIdx] = bestCost[cuIdx];
            gBestCpmvs[returnArrayIdx] = lBestCpmvs[cuIdx];
        }
        
    }
}
