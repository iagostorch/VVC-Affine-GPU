/*
ctu 		samples of current CTU
yPos		yPos of current CTU
xPos		xPos of current CTU
currPOC		POC of current CTU
L0_refPics	array with samples from the references pictures in list0
L1_refPics	array with samples from the references pictures in list1
*/
void encode_ctu(ctu, yPos, xPos, currFrame, L0_refPics[numRefs][width][height], L1_refPics[numRefs][width][height]):
	
	ref_frame_idx = select_affine_mvp_reference()
	
	// This call must be corrected: the return is either the function return itself, or the function is void and the return is via parameter
	mvp_affine = compute_affine_CPmvp(ctu, yPos, xPos, ref_frame_idx, cpmvp)

	// Perform affine with 2 or  CPs for each CU inside current CTU
	// If we design a mechanism to skip affine with 3 CPs, it may be interesting to perform only 2 CPs first, and perform 3 CPs on demand in another loop
	// This for will be improved to perform affine for several blocks simultaneously
	for each cu in ctu: 

		// Compute Y/X position and width/height of current CU
		// The actual computation depends on how we will iterate over blocks in the CTU
		yPosCU = yPos + offset
		xPosCU = xPos + offset
		widthCU = 
		heightCU = 

		bestCP = 2
		bestMV = 
		bestCost = MAX_DOUBLE
		bestList = -1
		bestRef = -1

		/*****     ME with 2 CPs     *****/
		nCPs = 2
		// For the current block, perform ME for all the references in LIST0. This will be made parallel in the future
		for each refPic in L0:
			tmpCost = MAX_DOUBLE
			tmpMV = 
			affine_ME(cu, yPosCU, xPosCU, widthCU, heightCU, mvp_affine, nCPs, refPic, tmpCost, tmpMV)
			if(tmpCost<bestCost):
				bestCP = 2
				bestList = L0
				bestRef = refPic
				bestCost = tmpCost
				bestMV = tmpMV
		
		// For the current block, perform ME for all the references in LIST1. This will be made parallel in the future
		for each refPic in L1:
			tmpCost = MAX_DOUBLE
			tmpMV = 
			affine_ME(cu, yPosCU, xPosCU, widthCU, heightCU, mvp_affine, nCPs, refPic, tmpCost, tmpMV)
			if(tmpCost<bestCost):
				bestCP = 2
				bestList = L1
				bestRef = refPic
				bestCost = tmpCost
				bestMV = tmpMV				

		// Find out how to perform bi-prediction in affine
		bipred_affine()


		/*****     ME with 3 CPs     *****/
		nCPs = 3
		// For the current block, perform ME for all the references in LIST0. This will be made parallel in the future
		for each refPic in L0:
			tmpCost = MAX_DOUBLE
			tmpMV = 
			affine_ME(cu, yPosCU, xPosCU, widthCU, heightCU, mvp_affine, nCPs, refPic, tmpCost, tmpMV)
			if(tmpCost<bestCost):
				bestCP = 3
				bestList = L0
				bestRef = refPic
				bestCost = tmpCost
				bestMV = tmpMV
		
		// For the current block, perform ME for all the references in LIST1. This will be made parallel in the future
		for each refPic in L1:
			tmpCost = MAX_DOUBLE
			tmpMV = 
			affine_ME(cu, yPosCU, xPosCU, widthCU, heightCU, mvp_affine, nCPs, refPic, tmpCost, tmpMV)
			if(tmpCost<bestCost):
				bestCP = 3
				bestList = L1
				bestRef = refPic
				bestCost = tmpCost
				bestMV = tmpMV				

		// Find out how to perform bi-prediction in affine
		bipred_affine()

		/*****     FINISH AFFINE     *****/
		/*
		Here we know the best number of CPs, the best CPMV for these CPs, and the best reference and ref list
		Must define some method to select between different combinations of block sizes. Maybe store the data of MV and cost in a matrix, and then
		perform another search trying to minimize the sum of costs... It is also necessary to define how the final motion compensatio will be done: since
		it is not returning the predicted blocks (only the best MV and cost), it will be necessary to perform MC once again to get the prediction and residual.
		If predicted samples are returned by the function, it is necessary to only combine them to form one CTU again. Mustaccess the tradeoff between
		(1) writing/reading to OpenCL shared memory multiple times and perform MC only once, and (2) avoid using shared memory and perform an extra MC
		*/


/*
These parameters must be defined
*/
int select_affine_mvp_reference():
	/*
	This functin will select what will be the reference frame for affine AMVP.
	Since it is not possible to access neighboring blocks in the same frame during AMVP (due to parallelization purposes), it is necesary
	that this information come from previous frames. Some possibilities are looking for the last encoded frame or frame with the smallest
	PCO difference in relation to current
	*/


/*
ctu 			samples of current CTU
yPos			yPos of current CTU
xPos			xPos of current CTU
ref_frame_idx 	index of frame used as reference to generate MVP 
cpmvp 			[ret] MVP for the CPs of current ctu
Como essa função precisa retornar vários valores (mvX e mvY para cada CP), acho que será preciso usar memobj (acho que openCL não aceita structs)
	## It is necessary to define how the predictors will be returned
*/
void compute_affine_CPmvp(ctu, yPos, xPos, ref_frame_idx, cpmvp):
	/*
	This function will use information from current CTU together with information from the reference frame to generate the preditors for CPs in current CTU.
	It may be usefull to compute the CPMVPs for 2 and 3 CPs together, even if 3 CPs are not always used.
	Depending on the AMVP algorithm, different information may be used in this function, and the signature must be updated:
		-- If MVs from the reference frame are used, we need current position and the MVs from reference frame
		-- If we perform some similarity test (such as a fast ME todefine predictors), then we need the current position, current samples, and samples from reference
	*/

/*
currBlock 	samples of current block
yPos 		Y position of currentBlock inside the frame
xPos 		X position of currentBlock inside the frame
width 		width of currBlock
height 		height of currBlock
mvp 		MVPs for the 2/3 CPs
nCPs		number of CPs to be used (2 or 3)
refPic 		Samples with reference picture
bestCost	[ret] Best cost for this refPic
bestMV 		[ret] MV leading to best cost in this refPic
*/
void affine_ME(currBlock, yPos, xPos, width, height, mvp, nCPs, refPic, bestCost, bestMV):
	/*
	This function will perform the actual motion estimation for the current block inside the reference frame, using the mvp as a starting point for the CPMVs.
	The best set of MVs will be stored in bestMV, and theirs costs in bestCost
	*/

	bestRD = MAX_DOUBLE
	bestDist = MAX_DOUBLE
	bestCost = MAX_DOUBLE

	tmpRD = MAX_DOUBLE
	tmpDist = MAX_DOUBLE
	tmpCost = MAX_DOUBLE

	if(nCPS==2):
		// This for is an abstraction of our ME algorithm. First it is necessary to define WHAT and HOW position will be tested,
		// and then we will implement it in a clever manner. ME is intended to be performed in a parallel manner, all MVs at once
		for each mv in mvCandidates:
			predBlock = affineBlockPrediction(currBlock, yPos, xPos, width, height, mvp, nCPs, refPic, mv)
			tmpDist = computeDistortion(predBlock, currBlock, width, height)
			tmpCost = computeCost(mvp, mv, nCPs)
			tmpRD = computeRD(tmpDist, tmpCost)

			if(tmpRD < bestRD):
				bestRD = tmpRD
				bestDist = tmpDist
				bestCost = tmpCost
				bestMV = mv

	else if(nCPs==3):
		// This for is an abstraction of our ME algorithm. First it is necessary to define WHAT and HOW position will be tested,
		// and then we will implement it in a clever manner. ME is intended to be performed in a parallel manner, all MVs at once
		for each mv in mvCandidates:
			predBlock = affineBlockPrediction(currBlock, yPos, xPos, width, height, mvp, nCPs, refPic, mv)
			tmpDist = computeDistortion(predBlock, currBlock, width, height)
			tmpCost = computeCost(mvp, mv, nCPs)
			tmpRD = computeRD(tmpDist, tmpCost)

			if(tmpRD < bestRD):
				bestRD = tmpRD
				bestDist = tmpDist
				bestCost = tmpCost
				bestMV = mv

	else:
		// Must not get here. Exit execution with error report



/*
currBlock 	samples of current block
yPos 		Y position of currentBlock inside the frame
xPos 		X position of currentBlock inside the frame
width 		width of currBlock
height 		height of currBlock
mvp 		MVPs for the 2/3 CPs
nCPs		number of CPs to be used (2 or 3)
refPic 		Samples with reference picture
mv 			MV to be used
*/
void affineBlockPrediction(currBlock, yPos, xPos, width, height, mvp, nCPs, refPic, mv):
	
	subMVs[] = zeros() // An array to hold the MV of each 4x4 sub-block inside currBlock. The dimension is width/4 x height/4 (for LUMA).
	predictedBlock[] = blankBlock() // Matrix to store the predicted samples

	if(nCPs==2):
		mvLT = mv[0]	// MVs of 2 CPs
		mvRT = mv[1]

		// This nested for derive the MV for each sub-block, based on the first equation 3-0 from document T-2002
		// In sequence, it performs the motion compensations (filtering and such)		
		for(h=0; h+=4; h<height):
			for(w=0; w+=4; w<width):
				subMVs[h/4][w/4] = deriveMv2Cps(h, w, mvLT, mvRT, width)
				predictedBlock[h][w] = x // Actual motion compensation comes here: filtering, clipping, precision correction and such.				

	else if(nCPs==3):
		mvLT = mv[0]	// MVs of 3 CPs
		mvRT = mv[1]
		mvLB = mv[2]

		// This nested for derive the MV for each sub-block, based on the second equation 3-0 from document T-2002
		// In sequence, it performs the motion compensations (filtering and such)
		for(h=0; h+=4; h<height):
			for(w=0; w+=4; w<width):
				subMVs[h/4][w/4] = deriveMv3Cps(h, w, mvLT, mvRT, mvLB, width, height)
				predictedBlock[h][w] = x // Actual motion compensation comes here: filtering, clipping, precision correction and such.				
				
	else:
		// Must not get here. Exit execution with error report



/*
predBlock 	samples of predicted block
origBlock 	samples of original block
width 		width of block
height 		height of block
*/
double computeDistortion(predBlock, origBlock, width, height):
	dist = 0
	for h in height:
		for w in width:
			// This function is computing SSE (sum of squared errors)
			// It must be corrected with the proper distortion function (SAD, SATD, SSE, ...)
			 dist += (predBlock-refBlock)^2
	return dist

/*
mvp 		predicted MVs for the CPs
mv 			actual MV for the CPs
nCPs 		number of CPs
*/
double computeCost(mvp, mv, nCPs):
	// This function computes the number of bits (or rate) required to use the input mv based on the predicted MV
	// I must study to understand how this computation is performed to update it here
	return cost


/*
dist 	distortion of a prediction
cost 	cost of a prediction
*/
double computeRD(dist, cost):
	// This functin will compute the rate-distortion of a prediction based on the cost, distortion and lambda. The lambda may be one
 	// input parameter or not, not sure at the moment

	return rd


/*
y		Y position of 44x sub-block inside the PU
x 		X position of 44x sub-block inside the PU
mvLT 	CPMV of left-top position
mvRT 	CPMV of right-top position
width 	width of PU where this sub-block is inside
*/
void deriveMv2Cps(y, x, mvLT, mvRT, width):
	// This function will derive the MV of one 4x4 sub-block based on its position and the 2 CPMVs. 
	//>>>>>>>>>  REMINDER: IT IS NECESSARY TO USE THE POSITION AT THE CENTER OF THE BLOCK: IF X/Y IS THE TOP-LEFT SAMPLE, APPLY AN OFFSET OF +2 TO BOTH X AND Y ON EQUATION
	// I must define how the return of this function will work: either a return statement or overwritting a parameter
	// Apply the first equation 3-0 from document T-2002 to perform the derivation
	mv = equation_3_0()
	return mv


/*
y		Y position of 44x sub-block inside the PU
x 		X position of 44x sub-block inside the PU
mvLT 	CPMV of left-top position
mvRT 	CPMV of right-top position
mvLB	CPMV of bottom-left position
width 	width of PU where this sub-block is inside
height 	height of PU where this sub-block is inside
*/
void deriveMv3Cps(y, x, mvLT, mvRT, mvLB, width, height):
	// This function will derive the MV of one 4x4 sub-block based on its position and the 3 CPMVs. 
	//>>>>>>>>>  REMINDER: IT IS NECESSARY TO USE THE POSITION AT THE CENTER OF THE BLOCK: IF X/Y IS THE TOP-LEFT SAMPLE, APPLY AN OFFSET OF +2 TO BOTH X AND Y ON EQUATION
	// I must define how the return of this function will work: either a return statement or overwritting a parameter
	// Apply the second equation 3-0 from document T-2002 to perform the derivation
	mv = equation_3_0()
	return mv
