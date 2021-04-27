#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#endif

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