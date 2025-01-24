#include <CL/cl.h>
#include "typedef.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
using namespace std;

#include <time.h>
#include <sys/time.h>

#include<vector>
#include<boost/program_options.hpp>

namespace po = boost::program_options;

float kernelExecutionTime[4] = {0, 0, 0, 0};
float resultsEssentialReadingTime[4] = {0, 0, 0, 0};;
float resultsEntireReadingTime[4] = {0, 0, 0, 0};
float samplesWritingTime;

// memory (in byes) used for parameters in each kernel
long memBytes_refSamples[4], memBytes_currSamples[4], memBytes_horizontalGrad[4], memBytes_verticalGrad[4], memBytes_equations[4], memBytes_returnCosts[4], memBytes_returnCpmvs[4], memBytes_debug[4], memBytes_returnCu[4], memBytes_frameWidth[4], memBytes_frameHeight[4], memBytes_lambda[4], memBytes_totalBytes[4];

// used for the timestamps
typedef struct DateAndTime {
    int year;
    int month;
    int day;
    int hour;
    int minutes;
    int seconds;
    int msec;
} DateAndTime;

DateAndTime date_and_time;
DateAndTime startWriteSamples, endReadingDistortion; // Used to track the processing time
struct timeval tv;
struct tm *timeStruct; 

void print_timestamp(char* messagePreffix){
    gettimeofday(&tv, NULL);
    timeStruct = localtime(&tv.tv_sec);
    date_and_time.hour = timeStruct->tm_hour;
    date_and_time.minutes = timeStruct->tm_min;
    date_and_time.seconds = timeStruct->tm_sec;
    date_and_time.msec = (int) (tv.tv_usec / 1000);
                // hh:mm:ss:ms
    printf("%s @ %02d:%02d:%02d.%03d\n", messagePreffix,date_and_time.hour, date_and_time.minutes, date_and_time.seconds, date_and_time.msec );
}

void probe_error(cl_int error, char* message){
    if (error != CL_SUCCESS ) {
        printf("Code %d, %s", error, message);
        return;
    }
}

int checkReportParameters(po::variables_map vm){
    
    int errors = 0;
    
    cout << "-=-= INPUT PARAMETERS =-=-" << endl;
    
    if( vm["DeviceIndex"].defaulted()){
        cout << "  Device index not set. Using standard value of " << vm["DeviceIndex"].as<int>() << "." << endl;
    }
    else{
        cout << "  Device Index=" << vm["DeviceIndex"].as<int>() << endl;
    }

    if(vm["CpmvLogFile"].defaulted() ){
        cout << "  CPMVs log file not set. The output will not be written to any file." << endl;
    }
    else{
        cout << "  CpmvLogFile=" << vm["CpmvLogFile"].as<string>() <<  endl;
    }

    if( ! vm["QP"].empty() ){
        cout << "  QP=" << vm["QP"].as<int>() << endl;
    }
    else{
        cout << "  [!] ERROR: QP not set." << endl;
        errors++;
    }

    if( ! vm["FramesToBeEncoded"].empty() ){
        cout << "  FramesToBeEncoded=" << vm["FramesToBeEncoded"].as<int>() << endl;
    }
    else{
        cout << "  [!] ERROR: FramesToBeEncoded not set." << endl;
        errors++;
    }

    if ( ! vm["Resolution"].empty() ){
        cout << "  Resolution=" << vm["Resolution"].as<string>() << endl;
    }
    else{
        cout << "  [!] ERROR: Resolution not set." << endl;
        errors++;
    }

    if ( ! vm["OriginalFrames"].empty() ){
        cout << "  InputOriginalFrame=" << vm["OriginalFrames"].as<string>() << endl;
    }
    else{
        cout << "  [!] ERROR: Input original frames not set." << endl;
        errors++;
    }

    if ( ! vm["ReferenceFrames"].empty() ){
        cout << "  InputReferenceFrame=" << vm["ReferenceFrames"].as<string>() << endl;
    }
    else{
        cout << "  [!] ERROR: Input reference frames not set." << endl;
        errors++;
    }

    return errors;
}

// Report the total memory used by memory objects and scalar kernel arguments (non memory objects)
void accessMemoryUsage(int pred, cl_mem ref_samples_mem_obj, cl_mem curr_samples_mem_obj, int frameWidth, int frameHeight, float lambda, cl_mem horizontal_grad_mem_obj, cl_mem vertical_grad_mem_obj, cl_mem equations_mem_obj, cl_mem return_costs_mem_obj, cl_mem return_cpmvs_mem_obj, cl_mem debug_mem_obj, cl_mem cu_mem_obj){
    int error;
    size_t retSize;
    long retData, totalBytes;

    const char* alignment = pred < 2 ? "FULL" : "HALF";
    int cps = pred%2 == 0 ? 2 : 3;


    printf("\n\n\n  MEMORY (in BYTES) USED BY THE KERNEL PARAMETERS -- %s %d CPs\n", alignment, cps);
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    totalBytes = 0;
    error = clGetMemObjectInfo(ref_samples_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of ref_samples_mem_obj\n");
    printf("ref_samples_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_refSamples[pred] = retData;

    error = clGetMemObjectInfo(curr_samples_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of curr_samples_mem_obj\n");
    printf("curr_samples_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_currSamples[pred] = retData;

    retData = sizeof(frameWidth);
    printf("frameWidth,%ld\n", retData);
    totalBytes += retData;
    memBytes_frameWidth[pred] = retData;

    retData = sizeof(frameHeight);
    printf("frameHeight,%ld\n", retData);
    totalBytes += retData;
    memBytes_frameHeight[pred] = retData;

    retData = sizeof(lambda);
    printf("lambda,%ld\n", retData);
    totalBytes += retData;
    memBytes_lambda[pred] = retData;

    error = clGetMemObjectInfo(horizontal_grad_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of horizontal_grad_mem_obj\n");
    printf("horizontal_grad_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_horizontalGrad[pred] = retData;

    error = clGetMemObjectInfo(vertical_grad_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of vertical_grad_mem_obj\n");
    printf("vertical_grad_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_verticalGrad[pred] = retData;

    error = clGetMemObjectInfo(equations_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of equations_mem_obj\n");
    printf("equations_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_equations[pred] = retData;

    error = clGetMemObjectInfo(return_costs_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of return_costs_mem_obj\n");
    printf("return_costs_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_returnCosts[pred] = retData;

    error = clGetMemObjectInfo(return_cpmvs_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of return_cpmvs_mem_obj\n");
    printf("return_cpmvs_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_returnCpmvs[pred] = retData;
    
    error = clGetMemObjectInfo(debug_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of debug_mem_obj\n");
    printf("debug_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_debug[pred] = retData;

    error = clGetMemObjectInfo(cu_mem_obj, CL_MEM_SIZE, sizeof(long), &retData, &retSize);
    probe_error(error, (char*)"Error returning size of cu_mem_obj\n");
    printf("cu_mem_obj,%ld\n", retData);
    totalBytes += retData;
    memBytes_returnCu[pred] = retData;

    memBytes_totalBytes[pred] = totalBytes;
    printf("TOTAL_BYTES,%ld\n", totalBytes);
    
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n\n");
    //*/
}

// Read data from memory objects into arrays
void readMemobjsIntoArray(int PRED, cl_command_queue command_queue, int nWG, int itemsPerWG, int nCtus, int testingAlignedCus, cl_mem return_costs_mem_obj, cl_mem return_cpmvs_mem_obj, cl_mem debug_mem_obj, cl_mem cu_mem_obj, cl_mem equations_mem_obj, cl_mem horizontal_grad_mem_obj, cl_mem vertical_grad_mem_obj, long *return_costs, Cpmvs *return_cpmvs, long *debug_data, short *return_cu, long *return_equations, short *horizontal_grad, short *vertical_grad){    
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    
    error  = clEnqueueReadBuffer(command_queue, return_costs_mem_obj, CL_TRUE, 0, 
            nCtus * (testingAlignedCus ? TOTAL_ALIGNED_CUS_PER_CTU : TOTAL_HALF_ALIGNED_CUS_PER_CTU) * sizeof(cl_long), return_costs, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return costs\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error = clEnqueueReadBuffer(command_queue, return_cpmvs_mem_obj, CL_TRUE, 0, 
            nCtus * (testingAlignedCus ? TOTAL_ALIGNED_CUS_PER_CTU : TOTAL_HALF_ALIGNED_CUS_PER_CTU) * sizeof(Cpmvs), return_cpmvs, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return CPMVs\n");    
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;   

    resultsEssentialReadingTime[PRED] += nanoSeconds;
    
    // The following memory reads are not essential, they only get some debugging information. This is not considered during performance estimation.
    
    int DEBUG = 0;
    
    if(DEBUG){
        error = clEnqueueReadBuffer(command_queue, debug_mem_obj, CL_TRUE, 0, 
                nWG*itemsPerWG*4 * sizeof(cl_long), debug_data, 0, NULL, &read_event);   
        probe_error(error, (char*)"Error reading return debug\n");    
        error = clWaitForEvents(1, &read_event);
        probe_error(error, (char*)"Error waiting for read events\n");
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing read\n");
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
        nanoSeconds += read_time_end-read_time_start;
        
        error = clEnqueueReadBuffer(command_queue, cu_mem_obj, CL_TRUE, 0, 
                128*128 * sizeof(cl_short), return_cu, 0, NULL, &read_event);  
        probe_error(error, (char*)"Error reading return CU\n");    
        error = clWaitForEvents(1, &read_event);
        probe_error(error, (char*)"Error waiting for read events\n");
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing read\n");
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
        nanoSeconds += read_time_end-read_time_start;
        
        error = clEnqueueReadBuffer(command_queue, equations_mem_obj, CL_TRUE, 0, 
                nWG*itemsPerWG*7*7 * sizeof(cl_long), return_equations, 0, NULL, &read_event);  
        probe_error(error, (char*)"Error reading return equations\n");    
        error = clWaitForEvents(1, &read_event);
        probe_error(error, (char*)"Error waiting for read events\n");
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing read\n");
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
        nanoSeconds += read_time_end-read_time_start;

        error = clEnqueueReadBuffer(command_queue, horizontal_grad_mem_obj, CL_TRUE, 0, 
                nWG * 128 * 128 * sizeof(cl_short), horizontal_grad, 0, NULL, &read_event);  
        error = clWaitForEvents(1, &read_event);
        probe_error(error, (char*)"Error waiting for read events\n");
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing read\n");
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
        nanoSeconds += read_time_end-read_time_start;
        
        error |= clEnqueueReadBuffer(command_queue, vertical_grad_mem_obj, CL_TRUE, 0, 
                nWG * 128 * 128 * sizeof(cl_short), vertical_grad, 0, NULL, &read_event);  
        probe_error(error, (char*)"Error reading gradients\n");
        error = clWaitForEvents(1, &read_event);
        probe_error(error, (char*)"Error waiting for read events\n");
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing read\n");
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
        
        nanoSeconds += read_time_end-read_time_start;
        resultsEntireReadingTime[PRED] += nanoSeconds;
    }
    
    probe_error(error, (char*)"Error reading returned memory objects into malloc'd arrays\n");
}

void reportAffineResultsMaster_new(int printCpmvToTerminal, int exportCpmvToFile, string cpmvFilePreffix, int pred, int nWG, int frameWidth, int frameHeight, long *return_costs, Cpmvs *return_cpmvs, int poc, int ref){
    const int list = 0;
    string type;

    switch(pred){
        case FULL_2CP:
            type = "_FULL_2CPs_";
            break;
        case FULL_3CP:
            type = "_FULL_3CPs_";
            break;
        case HALF_2CP:
            type = "_HALF_2CPs_";
            break;
        case HALF_3CP:
            type = "_HALF_3CPs_";
            break;
        default:
            printf("Wrong pred in reportAffineResultsMaster\n");
            exit(0);
    };
    

    printf("Reporting results POC=%d refIdx=%d\n", poc, ref);

    string exportFileName;
    FILE *cpmvFile;
    int dataIdx, currX, currY, nCus;

    vector<string> sizesFullStr, sizesHalfStr, sizesStr;
    vector<int> sizesFullEnum, sizesHalfEnum, sizesEnum;

    for(int i=0; i<NUM_CU_SIZES; i++){
        sizesFullStr.push_back(to_string(WIDTH_LIST[i]) + "x" + to_string(HEIGHT_LIST[i]));
        sizesFullEnum.push_back(_128x128+i);
    }
        
    
    for(int i=0; i<HA_NUM_CU_SIZES; i++){
        sizesHalfStr.push_back(to_string(HA_WIDTH_LIST[i]) + "x" + to_string(HA_HEIGHT_LIST[i]));
        sizesHalfEnum.push_back(HA_64x32+i);
    }

    // First report of the sequence. Create files with headers
    if(exportCpmvToFile==1 && poc==1 && ref==0){ 
        printf("Writing headers\n");
        if(pred<=FULL_3CP){ // FULL_2CP or FULL_3CP
            sizesEnum = sizesFullEnum;
            sizesStr = sizesFullStr;
            for(vector<string>::iterator itSizesFull = sizesFullStr.begin(); itSizesFull<sizesFullStr.end(); itSizesFull++){
                exportFileName = cpmvFilePreffix + type + *itSizesFull + ".csv";
                cpmvFile = fopen( exportFileName.c_str(), "w" );
                fprintf(cpmvFile, "POC,List,Ref,CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
                fclose(cpmvFile);
            }
        }
        else{ // HALF_2CP or HALF_3CP
            sizesEnum = sizesHalfEnum;
            sizesStr = sizesHalfStr;
            for(vector<string>::iterator itSizesHalf = sizesHalfStr.begin(); itSizesHalf<sizesHalfStr.end(); itSizesHalf++){
                exportFileName = cpmvFilePreffix + type + *itSizesHalf + ".csv";
                // cout << "FILE HALF: " << exportFileName << endl;
                cpmvFile = fopen( exportFileName.c_str(), "w" );
                fprintf(cpmvFile, "POC,List,Ref,CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
                fclose(cpmvFile);
            }    
        }
    }

    printf("Exporting results for POC=%d and refIdx=%d\n", poc, ref);
    int frameStride = 0; // currFrame*(nWG/NUM_CU_SIZES)*TOTAL_ALIGNED_CUS_PER_CTU;


    if(pred<=FULL_3CP){
        sizesEnum = sizesFullEnum;
        sizesStr = sizesFullStr;
    }
    else{
        sizesEnum = sizesHalfEnum;
        sizesStr = sizesHalfStr;  
    }

    for(vector<int>::iterator itEnum = sizesEnum.begin(); itEnum<sizesEnum.end(); itEnum++){
        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs %s\n", sizesFullStr[*itEnum].c_str() );
            printf("POC,List,Ref,CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        } 
        
        if(pred<=FULL_3CP) // FULL
            nCus = *itEnum==_16x16 ? 64 : RETURN_STRIDE_LIST[*itEnum+1] - RETURN_STRIDE_LIST[*itEnum] ;
        else // HALF
            nCus = *itEnum==HA_16x16_U123 ? 32 : HA_RETURN_STRIDE_LIST[*itEnum+1] - HA_RETURN_STRIDE_LIST[*itEnum] ;

        exportFileName = cpmvFilePreffix + type + sizesStr[*itEnum] + ".csv";
        cpmvFile = fopen( exportFileName.c_str(), "a" );

        int NUM = pred<=FULL_3CP ? NUM_CU_SIZES : HA_NUM_CU_SIZES ;

        for(int ctu=0; ctu<nWG/NUM; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                // CU position inside CTU
                if(pred<=FULL_3CP){ // FULL
                    currY = cuIdx*WIDTH_LIST[*itEnum];
                    currY = currY/128;
                    currY = currY*HEIGHT_LIST[*itEnum];
                                               
                    currX = cuIdx*WIDTH_LIST[*itEnum];
                    currX = currX%128;
                }
                else{ // HALF
                    currY = HA_ALL_Y_POS[*itEnum][cuIdx];
                    currX = HA_ALL_X_POS[*itEnum][cuIdx];
                }
                
                // Add CTU Position
                int ctuColumnsInFrame = ceil((float)frameWidth/128);
                currY += (ctu/ctuColumnsInFrame)*128;
                currX += (ctu%ctuColumnsInFrame)*128;               

                if(pred<=FULL_3CP) // FULL
                    dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[*itEnum] + cuIdx;
                else // HALF
                    dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[*itEnum] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", poc, list, ref, ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", poc, list, ref, ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        fclose(cpmvFile);        
    }
}

// Export affine results to the terminal or writing files
// TODO: It must be improved to be shorter and reuse code
void reportAffineResultsMaster(int printCpmvToTerminal, int exportCpmvToFile, string cpmvFilePreffix, int pred, int nWG, int frameWidth, int frameHeight, long *return_costs, Cpmvs *return_cpmvs, int currFrame){    
    const int testingAlignedCus = pred < 2 ? 1 : 0;
    
    const int cps = pred%2 == 0 ? 2 : 3;
    string exportFileName;
    FILE *cpmvFile;
    int cuSizeIdx, dataIdx, currX, currY, nCus;

    string currFrameStr = to_string(currFrame);
    
    if(testingAlignedCus){

        int frameStride = 0; // currFrame*(nWG/NUM_CU_SIZES)*TOTAL_ALIGNED_CUS_PER_CTU;

        string predPreffix = (cps==2 ? "_FULL_2CPs" : "_FULL_3CPs");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 128x128\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = _128x128;
        nCus = 1;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_128x128_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128;
                currX = (ctu*128)%frameWidth;

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");


        if(printCpmvToTerminal){
            printf("CUs 128x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = _128x64;
        nCus = 2;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_128x64_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_128x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_128x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 64x128\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x128;
        nCus = 2;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_64x128_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x128[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x128[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 64x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x64;
        nCus = 4;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_64x64_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");
            
        if(printCpmvToTerminal){
            printf("CUs 64x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x32;
        nCus = 8;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_64x32_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x32[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x32[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");
            
        if(printCpmvToTerminal){
            printf("CUs 32x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _32x64;
        nCus = 8;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_32x64_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_32x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_32x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 32x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _32x32;
        nCus = 16;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_32x32_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_32x32[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_32x32[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 64x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x16;
        nCus = 16;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_64x16_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x16[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x16[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");


        if(printCpmvToTerminal){
            printf("CUs 16x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _16x64;
        nCus = 16;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_16x64_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_16x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_16x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 32x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _32x16;
        nCus = 32;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_32x16_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_32x16[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_32x16[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 16x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _16x32;
        nCus = 32;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_16x32_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_16x32[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_16x32[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 16x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _16x16;
        nCus = 64;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_16x16_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_16x16[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_16x16[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");
    }


    // Report the results for HALF-ALIGNED CUS
    if(!testingAlignedCus){


        int frameStride = 0; // currFrame*(nWG/HA_NUM_CU_SIZES)*TOTAL_HALF_ALIGNED_CUS_PER_CTU;

        string predPreffix = (cps==2? "_HALF_2CPs" : "_HALF_3CPs");
        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 64x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = HA_64x32;
        nCus = HA_CUS_PER_CTU[cuSizeIdx];
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_64x32_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 32x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = HA_32x64;
        nCus = HA_CUS_PER_CTU[cuSizeIdx];
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_32x64_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 64x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_64x16_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
                
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            
            cuSizeIdx = HA_64x16_G1;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_64x16_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 16x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_16x64_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
                
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            
            cuSizeIdx = HA_16x64_G1;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x64_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 32x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_32x32_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
                
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            
            cuSizeIdx = HA_32x32_G1;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_32x32_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 32x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_32x16_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
                
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            
            cuSizeIdx = HA_32x16_G1;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_32x16_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_32x16_G3;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 16x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_16x32_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
                
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            
            cuSizeIdx = HA_16x32_G1;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x32_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x32_G3;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 16x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + predPreffix + "_16x16_" + currFrameStr + ".csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
                
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            
            cuSizeIdx = HA_16x16_G1;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x16_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x16_G3;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x16_G4;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[frameStride + dataIdx], return_cpmvs[frameStride + dataIdx].LT.x, return_cpmvs[frameStride + dataIdx].LT.y, return_cpmvs[frameStride + dataIdx].RT.x, return_cpmvs[frameStride + dataIdx].RT.y, return_cpmvs[frameStride + dataIdx].LB.x, return_cpmvs[frameStride + dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        if(printCpmvToTerminal)
            printf("\n");
    
    }
}

void reportTimingResults(int N_FRAMES){
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("TIMING RESULTS (nanoseconds)\n");
    printf("Writing,%f\n", samplesWritingTime);
    
    printf("FULL_2CP_EXEC,%f\n", kernelExecutionTime[FULL_2CP]);
    printf("FULL_2CP_READ,%f\n", resultsEssentialReadingTime[FULL_2CP]);
    printf("FULL_2CP_DEBUG,%f\n", resultsEntireReadingTime[FULL_2CP]);

    printf("FULL_3CP_EXEC,%f\n", kernelExecutionTime[FULL_3CP]);
    printf("FULL_3CP_READ,%f\n", resultsEssentialReadingTime[FULL_3CP]);
    printf("FULL_3CP_DEBUG,%f\n", resultsEntireReadingTime[FULL_3CP]);

    printf("HALF_2CP_EXEC,%f\n", kernelExecutionTime[HALF_2CP]);
    printf("HALF_2CP_READ,%f\n", resultsEssentialReadingTime[HALF_2CP]);
    printf("HALF_2CP_DEBUG,%f\n", resultsEntireReadingTime[HALF_2CP]);

    printf("HALF_3CP_EXEC,%f\n", kernelExecutionTime[HALF_3CP]);
    printf("HALF_3CP_READ,%f\n", resultsEssentialReadingTime[HALF_3CP]);
    printf("HALF_3CP_DEBUG,%f\n", resultsEntireReadingTime[HALF_3CP]);

    printf("TOTAL_TIME(%dx),%f\n", N_FRAMES, samplesWritingTime+kernelExecutionTime[FULL_2CP] + kernelExecutionTime[FULL_3CP] + kernelExecutionTime[HALF_2CP] + kernelExecutionTime[HALF_3CP] + resultsEssentialReadingTime[FULL_2CP] + resultsEssentialReadingTime[FULL_3CP] + resultsEssentialReadingTime[HALF_2CP] + resultsEssentialReadingTime[HALF_3CP]);





    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");
}

void reportMemoryUsage(){
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("MEMORY USAGE RESULTS (bytes)\n");
    for(int pred=0; pred<N_PREDS; pred++){
        const char* mode = pred==FULL_2CP ? "HALF_2CP" : (pred==FULL_3CP ? "HALF_3CP" : (pred==HALF_2CP ? "HALF_2CP" : "HALF_3CP"));

        printf("%s,refSamples,%ld\n", mode, memBytes_refSamples[pred]);
        printf("%s,currSamples,%ld\n", mode, memBytes_currSamples[pred]);
        printf("%s,horizontalGrad,%ld\n", mode, memBytes_horizontalGrad[pred]);
        printf("%s,verticalGrad,%ld\n", mode, memBytes_verticalGrad[pred]);
        printf("%s,equations,%ld\n", mode, memBytes_equations[pred]);

        printf("%s,returnCosts,%ld\n", mode, memBytes_returnCosts[pred]);
        printf("%s,returnCpmvs,%ld\n", mode, memBytes_returnCpmvs[pred]);
        printf("%s,debug,%ld\n", mode, memBytes_debug[pred]);
        printf("%s,returnCu,%ld\n", mode, memBytes_returnCu[pred]);

        printf("%s,frameWidth,%ld\n", mode, memBytes_frameWidth[pred]);
        printf("%s,frameHeight,%ld\n", mode, memBytes_frameHeight[pred]);
        printf("%s,lambda,%ld\n", mode, memBytes_lambda[pred]);
        printf("%s,TOTAL_BYTES,%ld\n", mode, memBytes_totalBytes[pred]);
    }
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");
}
void testReferences(int N_FRAMES, string refFilaNamePreffix, int inputQp){
    printf("-=-=-= Artificial references used for debugging =-=-=-=-\n");
    printf("Input QP = %d\n", inputQp);
    int qp;
    int numRefs;
    int circularBufferRefs[MAX_REFS] = {-1, -1, -1, -1};
    int circularBufferIsLT[MAX_REFS] = {0, 0, 0, 0}; // is long term ref?
    int tempA, tempB;
    for(int f=1; f<N_FRAMES; f++){
        qp = inputQp + qpOffset[f%8];
        numRefs = min(4, f);
        if(f<5){ // Ref list not full yet
            tempA = circularBufferRefs[0];
            circularBufferRefs[0] = f-1;
            tempB = circularBufferRefs[1];
            circularBufferRefs[1] = tempA; 
            tempA = circularBufferRefs[2];
            circularBufferRefs[2] = tempB;
            tempB = circularBufferRefs[3];
            circularBufferRefs[3] = tempA;

            circularBufferIsLT[3] =   circularBufferRefs[3]%8==0 ? 1 : 0;           
        }
        else{
            tempA = circularBufferRefs[0];
            circularBufferRefs[0] = f-1;
            tempB = circularBufferRefs[1];
            circularBufferRefs[1] = circularBufferIsLT[1]==0 ? tempA : ( tempA%8==0 && tempA!=circularBufferRefs[0] ? tempA : circularBufferRefs[1] );            //(circularBufferRefs[1]%8==0 && tempA%8!=0) ? circularBufferRefs[1] : tempA;
            tempA = circularBufferRefs[2];
            circularBufferRefs[2] = circularBufferIsLT[2]==0 ? tempB : ( tempB%8==0 && tempB!=circularBufferRefs[1] ? tempB : circularBufferRefs[2] );            // (circularBufferRefs[2]%8==0 && tempB%8!=0) ? circularBufferRefs[2] : tempB;
            tempB = circularBufferRefs[3];
            circularBufferRefs[3] = circularBufferIsLT[3]==0 ? tempA : ( tempA%8==0 && tempA!=circularBufferRefs[2] ? tempA : circularBufferRefs[3] );            // (circularBufferRefs[3]%8==0 && tempA%8!=0) ? circularBufferRefs[3] : tempA;

            circularBufferIsLT[3] =   circularBufferRefs[3]%8==0 ? 1 : 0;
            circularBufferIsLT[2] = ( circularBufferRefs[2]%8==0 && circularBufferIsLT[3] ) ? 1 : 0;
            circularBufferIsLT[1] = ( circularBufferRefs[1]%8==0 && circularBufferIsLT[2] ) ? 1 : 0;
        }
        

        // printf("POC %3d   QP %d : [L0 %2d %2d %2d %2d]  LT[ %d %d %d %d ]\n", f, qp, circularBufferRefs[0], circularBufferRefs[1], circularBufferRefs[2], circularBufferRefs[3], circularBufferIsLT[0], circularBufferIsLT[1], circularBufferIsLT[2], circularBufferIsLT[3] );
        printf("POC %3d   QP %d motionLambda %f : [L0 %d", f, qp, fullLambdas[qp], circularBufferRefs[0] );
        for(int r=1; r<numRefs; r++){
            printf(" %d", circularBufferRefs[r]);
        }
        printf("]\n");
    }
}

void removeOldTraces(string cpmvFilePreffix){
    printf("Removing older outputs with identical names...\n");

    string temp;

    vector<string> types;
    types.push_back("FULL_2CPs");
    types.push_back("FULL_3CPs");
    types.push_back("HALF_2CPs");
    types.push_back("HALF_3CPs");
    
    vector<string> sizes;
    for(int i=0; i<NUM_CU_SIZES; i++)
        sizes.push_back(to_string(WIDTH_LIST[i]) + "x" + to_string(HEIGHT_LIST[i]));
    
    for(vector<string>::iterator itType = types.begin(); itType != types.end(); itType++){
        for(vector<string>::iterator itSize = sizes.begin(); itSize != sizes.end(); itSize++){
            temp = cpmvFilePreffix + "_" + *itType + "_" + *itSize + ".csv";
            // cout << temp << endl;
            remove(temp.c_str());
        }
    }
    
    // remove((cpmvFilePreffix + "_FULL_2CPs_128x128.csv").c_str());
    // remove((cpmvFilePreffix + "_FULL_3CPs_128x128.csv").c_str());
    // remove((cpmvFilePreffix + "_HALF_2CPs_128x128.csv").c_str());
    // remove((cpmvFilePreffix + "_HALF_3CPs_128x128.csv").c_str());

    // remove((cpmvFilePreffix + "_FULL_2CPs_128x128.csv").c_str());
    // remove((cpmvFilePreffix + "_FULL_3CPs_128x128.csv").c_str());
    // remove((cpmvFilePreffix + "_HALF_2CPs_128x128.csv").c_str());
    // remove((cpmvFilePreffix + "_HALF_3CPs_128x128.csv").c_str());


    //                                 "_FULL_2CPs_" HALF/3
    //  = cpmvFilePreffix + predPreffix + "_16x16.csv";
    //         cpmvFile = fopen(exportFileName.c_str(),"w");

}

int getNumCtus(int frameWidth, int frameHeight){
    pair<int,int> inputRes(frameWidth, frameHeight);

    for( unsigned int i=0; i<availableRes.size(); i++){
        if(inputRes == pair<int,int>( get<0>(availableRes[i]), get<1>(availableRes[i]) )){
            return get<2>(availableRes[i]); 
        }
    }
    // In case it didn't match with any of the available resolutions, return an error code
   return 0;
}

// memBytes_frameWidth[4], memBytes_frameHeight[4], memBytes_lambda[4], memBytes_totalBytes[4];