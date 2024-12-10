#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include "constants.h"
// #include "typedef.h"
#include "main_aux_functions.h"

using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

// Pad the borders of a frame with black samples to be a multiple of a given block size
int* pad_borders(int* original, int origWidth, int origHeight, int blockWidth, int blockHeight){
    
    int newWidth = origWidth/blockWidth + blockWidth*(origWidth%blockWidth != 0); // Round width for the next multiple of block width and height
    int newHeight = origHeight/blockHeight + blockHeight*(origWidth%blockHeight != 0);

    int* padded = (int*) malloc(sizeof(int) * newWidth*newHeight);

    // Fill the padded frame filling the borders with zeros
    for(int h=0; h<newHeight; h++){
        for(int w=0; w<newWidth; w++){
            if(h<origHeight && w<origWidth){ // current sample lies withing original range
                padded[h*newWidth + w] = original[h*newWidth + w];
            }
            else{   // Current sample lies outside the original range
                padded[h*newWidth + w] = 0;
            }
        }
    }

    return padded;

}

int main(int argc, char *argv[]) {

    print_timestamp((char*)"START HOST");

    float lambda;

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("affine.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Error objects for detecting problems in OpenCL
    cl_int error, error_1, error_2, error_3, error_4;

    // Get platform and device information
    cl_platform_id *platform_id = NULL;
    cl_uint ret_num_platforms;

    error = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    probe_error(error, (char*)"Error querying available platforms\n");
    
    platform_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * ret_num_platforms); // Malloc space for all ret_num_platforms platforms
    
    error = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    probe_error(error, (char*)"Error querying platform IDs\n");

    // List available platforms
    char platform_name[128] = {0};
    // List name of platforms available and assign to proper CPU/GPU IDs
    cout << "Idx    Platform Name" << endl;
    for (cl_uint ui=0; ui< ret_num_platforms; ++ui){
        error = clGetPlatformInfo(platform_id[ui], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        if (platform_name != NULL){
            cout << ui << "      " << platform_name << endl;
        }
    }

    // Scan all platforms looking for CPU and GPU devices.
    // This results in errors when searching for GPU devices on CPU platforms, for instance. Not a problem
    cl_device_id cpu_device_ids[5] = {NULL, NULL, NULL, NULL, NULL};
    cl_uint ret_cpu_num_devices;
    int assigned_cpus = 0; // Keeps number of available devices
    
    cl_device_id gpu_device_ids[5] = {NULL, NULL, NULL, NULL, NULL};
    cl_uint ret_gpu_num_devices;
    int assigned_gpus = 0; // Keeps number of available devices

    cl_device_id tmp_device_ids[5] = {NULL, NULL, NULL, NULL, NULL};

    // Scan all platforms...
    printf("\n");
    for(cl_uint p=0; p<ret_num_platforms; p++){
        error = clGetPlatformInfo(platform_id[p], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        printf("Scanning platform %d...\n", p);
     
        // Query all CPU devices on current platform, and copy them to global CPU devices list (cpu_device_ids)
        error = clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_CPU, 0, 
            NULL, &ret_cpu_num_devices);
        error |= clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_CPU, ret_cpu_num_devices, tmp_device_ids, NULL);
        probe_error(error, (char*)"\tError querying CPU device IDs\n"); // GPU platforms do not have CPU devices
        
        for(cl_uint d=0; d<ret_cpu_num_devices; d++){
                cpu_device_ids[assigned_cpus] = tmp_device_ids[d];
                assigned_cpus++;
        }
        
        // Query all GPU devices on current platform, and copy them to global GPU devices list (gpu_device_ids)
        error = clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_GPU, 0, 
            NULL, &ret_gpu_num_devices);
        error |= clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_GPU, ret_gpu_num_devices, tmp_device_ids, NULL);
        probe_error(error, (char*)"\tError querying GPU device IDs\n");  // CPU platforms do not have GPU devices
        
        for(cl_uint d=0; d<ret_gpu_num_devices; d++){
                gpu_device_ids[assigned_gpus] = tmp_device_ids[d];
                assigned_gpus++;
        }
    }
    printf("\n");

    char device_name[1024];
    char device_extensions[1024];

    // List the ID and name for each CPU and GPU device
    for(int cpu=0; cpu<assigned_cpus; cpu++){
        error = clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        error&= clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_EXTENSIONS, 1024 * sizeof(char), device_extensions, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");
        
        cout << "CPU " << cpu << endl;
        cout << "\tid " << cpu_device_ids[cpu] << endl << "\t" <<  device_name << endl;
        cout << "\tExtensions: " << device_extensions << endl;
    }
    for(int gpu=0; gpu<assigned_gpus; gpu++){
        error = clGetDeviceInfo(gpu_device_ids[gpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");
        error&= clGetDeviceInfo(gpu_device_ids[gpu], CL_DEVICE_EXTENSIONS, 1024 * sizeof(char), device_extensions, NULL);
        
        cout << "GPU " << gpu << endl;
        cout << "\tid " << gpu_device_ids[gpu] << endl << "\t" <<  device_name << endl;
        cout << "\tExtensions: " << device_extensions << endl;
    }

    
    // Create "target" device and assign proper IDs
    cl_device_id device_id = NULL; 
    
    // Select what CPU or GPU will be used based on parameters
    if(argc==8){
        if(!strcmp(argv[1],"CPU")){
            if(stoi(argv[2]) < assigned_cpus){
                cout << "COMPUTING ON CPU " << argv[2] << endl;        
                device_id = cpu_device_ids[stoi(argv[2])];    
            }
            else{
                cout << "Incorrect CPU number. Only " << assigned_cpus << " CPUs are detected" << endl;
                exit(0);    
            }
        }
        else if(!strcmp(argv[1],"GPU")){
            if(stoi(argv[2]) < assigned_gpus){
                cout << "COMPUTING ON GPU " << argv[2] << endl;        
                device_id = gpu_device_ids[stoi(argv[2])];    
            }
            else{
                cout << "Incorrect GPU number. Only " << assigned_gpus << " GPUs are detected" << endl;
                exit(0);    
            }
        }
        else{
            cout << "Incorrect usage. First parameter must be either CPU or GPU" << endl;
            exit(0);
        }

        // if(!strcmp(argv[3],"22")){
        //     cout << "Using QP=22" << endl;
        //     lambda = lambdas[QP22];

        // }
        // else if(!strcmp(argv[3],"27")){
        //     cout << "Using QP=27" << endl;
        //     lambda = lambdas[QP27];
        // }
        // else if(!strcmp(argv[3],"32")){
        //     cout << "Using QP=32" << endl;
        //     lambda = lambdas[QP32];
        // }
        // else if(!strcmp(argv[3],"37")){
        //     cout << "Using QP=37" << endl;
        //     lambda = lambdas[QP37];
        // }
        // else{
        //     cout << "Incorrect usage. Third parameter must be the QP value, one of the following: 22, 27, 32, 37" << endl;
        //     exit(0);
        // }
    }
    else{
        cout << "\n\n\nFailed to specify the input parameters. Proper execution has the form of" << endl;
        cout << "./main <CPU or GPU> <# of CPU or GPU device> <QP in set [22,27,32,37]> <original_frame_file> <reference_frame_file> <preffix for exported CPMV files> <total number of frames in file>\n\n\n" << endl;
        exit(0);
    }
    
    size_t ret_val;
    cl_uint max_compute_units;
    error = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &ret_val);
    error|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, ret_val, &max_compute_units, NULL);
    probe_error(error, (char*)"Error querying maximum number of compute units of device\n");
    cout << "-- Max compute units " << max_compute_units << endl;

    ///////////////////////////////////////////////////////////////
    /////         STARTS BY CREATING A CONTEXT, QUEUE,        /////
    ///////////////////////////////////////////////////////////////

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &error);
    probe_error(error, (char*)"Error creating context\n");

    // Create a command queue
    // Profiling enabled to measure execution time. 
    // TODO: Remove this profiling when perform actual computation, it may slowdown the processing (for experiments and etc)
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    // TODO: This should be an input parameter
    const int frameWidth  = 1920; // 1920 or 3840
    const int frameHeight = 1080; // 1080 or 2160

    const int TEST_FULL_2CP = 1;
    const int TEST_FULL_3CP = 1;
    const int TEST_FULL = TEST_FULL_2CP || TEST_FULL_3CP;
    const int TEST_HALF_2CP = 1;
    const int TEST_HALF_3CP = 1;
    const int TEST_HALF = TEST_HALF_2CP || TEST_HALF_3CP;

    // TODO: This should be computed based on the frame resolution
    const int nCtus = frameHeight==1080 ? 135 : 510; //135 or 510 for 1080p and 2160p  ||  1080p videos have 120 entire CTUs plus 15 partial CTUs || 4k videos have 480 entire CTUs plus 30 partial CTUs
    const int itemsPerWG = 256;  // Each workgroup has 256 workitems
    int testingAlignedCus; // Toggle between predicting ALIGNED or HALF_ALIGNED CUs
    int nWG; // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs

    // Read the frame data into the matrix
    string currFileName = argv[4];  // File with samples from current frame
    string refFilaNamePreffix = currFileName.substr(0, currFileName.find("original"));
    string refFileName = argv[5];   // File with samples from reference frame
    string cpmvFilePreffix = argv[6];   // Preffix of exported files containing CPMV information

    int N_FRAMES = stoi(argv[7]);
    int inputQp = stoi(argv[3]);

    testReferences(N_FRAMES, refFilaNamePreffix, inputQp);

    // return 1;

    ifstream currFile, refFile;
    currFile.open(currFileName);
    refFile.open(refFileName);

    if (!currFile.is_open() || !refFile.is_open()) {     // validate file open for reading 
    // if (!refFile.is_open()) {     // validate file open for reading 
        perror (("error while opening samples files" ));
        return 1;
    }

    string refLine, refVal, currLine, currVal;

    const int FRAME_SIZE = frameWidth*frameHeight;

    unsigned short *reference_frame = (unsigned short*) malloc(sizeof(short) * FRAME_SIZE * N_FRAMES);
    unsigned short *reference_buffer = (unsigned short*) malloc(sizeof(short) * FRAME_SIZE * MAX_REFS);
    unsigned short *current_frame   = (unsigned short*) malloc(sizeof(short) * FRAME_SIZE * N_FRAMES);

    print_timestamp((char*)"START READ .csv");

    // Read the samples from reference frame into the reference array
    for(int f=0; f<N_FRAMES; f++){
        for(int h=0; h<frameHeight; h++){
            // getline(currFile, currLine, '\n');
            getline(refFile, refLine, '\n');
            stringstream currStream(currLine), refStream(refLine); 
            
            for(int w=0; w<frameWidth; w++){
                // getline(currStream, currVal, ',');
                getline(refStream, refVal, ',');
                // current_frame[f*frameWidth*frameHeight +   h*frameWidth + w] = stoi(currVal);
                reference_frame[f*frameWidth*frameHeight + h*frameWidth + w] = stoi(refVal);
            }
        }
    }

    print_timestamp((char*)"FINISHED READ .csv");
    
    int label_circularBufferRefs[MAX_REFS] = {-1, -1, -1, -1}; // poc of frame in ref list
    int label_circularBufferIsLT[MAX_REFS] = {0, 0, 0, 0}; // is long term ref?
    int label_tempA, label_tempB; // poc of frame in temp buffers

    // Initialize memory objects to be used as circular buffer for references along with two temporary buffers to help swapping references inside the buffer
    cl_mem memObj_tempA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                FRAME_SIZE * sizeof(short), NULL, &error_1);   
    cl_mem memObj_tempB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                FRAME_SIZE * sizeof(short), NULL, &error_2);   
    probe_error(error_1 || error_2, (char*)"Error creatin temp memObjs\n");

    cl_mem memObj_circularBufferRefs[MAX_REFS];
    for(int f=0; f<N_FRAMES; f++){
        memObj_circularBufferRefs[f] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                FRAME_SIZE * sizeof(short), NULL, &error_1);   
        
        probe_error(error_1, (char*)"Error initializing array of memory objects\n");
    }
    cl_event write_event, read_event;
    
    // These buffers are for storing the reference samples and current samples
    cl_mem ref_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,  // This will be deprecated
            FRAME_SIZE * sizeof(short), NULL, &error_1);    
    cl_mem curr_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            FRAME_SIZE * sizeof(short), NULL, &error_2);                
    error = error_1 || error_2;

    probe_error(error, (char*)"Error creating memory buffers\n");
    
    double nanoSeconds = 0;

    
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////       CREATE A PROGRAM (OBJECT) BASED ON .cl FILE AND BUILD IT TO TARGET DEVICE       /////
    /////         CREATE A KERNEL BY ASSIGNING A NAME FOR THE RECENTLY COMPILED PROGRAM         /////
    /////                           LOADS THE ARGUMENTS FOR THE KERNEL                          /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create a program from the kernel source (specifically for the device in the context variable)
    // Each one will be compiled with a different MACRO/CONSTANT to optimize 2 and 3 CPs
    cl_program program_2CP, program_3CP;
    
    program_2CP = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &error);
    program_3CP = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &error);
    probe_error(error, (char*)"Error creating program from source\n");

    // Build the program
    // -cl-nv-verbose is used to show memory and registers used by the kernel
    // -cl-nv-maxrregcount=241 is used to set the maximum number of registers per kernel. Using a large value and modifying in each compilation makes no difference in the code, but makes the -cl-nv-verbose flag work properly
    
    char buildOptions_2CP[100], buildOptions_3CP[100];
    
    sprintf(buildOptions_2CP, "-DnCP=%d", 2);
    sprintf(buildOptions_3CP, "-DnCP=%d", 3);

    error = clBuildProgram(program_2CP, 1, &device_id, buildOptions_2CP, NULL, NULL);
    probe_error(error, (char*)"Error building the program with 2 CPs\n");
    // Show debugging information when the build is not successful
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_2CP, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program_2CP, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }

    error = clBuildProgram(program_3CP, 1, &device_id, buildOptions_3CP, NULL, NULL);
    probe_error(error, (char*)"Error building the program with 3 CPs\n");
    // Show debugging information when the build is not successful
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_3CP, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program_3CP, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    /////               COMPILES THE FOUR KERNELS INTO DIFFERENT OBJECTS                /////
    /////////////////////////////////////////////////////////////////////////////////////////

    cl_kernel kernel; // The object that holds the compiled kernel to be enqueued
    cl_kernel kernel_FULL_2CP, kernel_FULL_3CP, kernel_HALF_2CP, kernel_HALF_3CP;

    print_timestamp((char*)"START BUILD KERNELS");


    kernel_FULL_2CP = clCreateKernel(program_2CP, "affine_gradient_mult_sizes", &error);
    probe_error(error, (char*)"Error creating kernel for FULL 2 CPs\n"); 
    kernel_FULL_3CP = clCreateKernel(program_3CP, "affine_gradient_mult_sizes", &error);
    probe_error(error, (char*)"Error creating kernel for FULL 3 CPs\n"); 
    kernel_HALF_2CP = clCreateKernel(program_2CP, "affine_gradient_mult_sizes_HA", &error);
    probe_error(error, (char*)"Error creating kernel for HALF 2 CPs\n"); 
    kernel_HALF_3CP = clCreateKernel(program_3CP, "affine_gradient_mult_sizes_HA", &error);
    probe_error(error, (char*)"Error creating kernel for HALF 3 CPs\n"); 

    print_timestamp((char*)"FINISH BUILD KERNELS");

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////                  VARIABLES SHARED BETWEEN THE EXECUTION OF ALL KERNELS                /////
    /////                        2 CPs AND 3 CPs, ALIGNED AND HALF ALIGNED                      /////
    /////           THIS INCLUDES CONSTANTS, VARIABLES USED FOR CONTROL AND PROFILING           /////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Used to access optimal workgroup size
    size_t size_ret;
    cl_uint preferred_size, maximum_size;

    // Used to profile execution time of kernel
    cl_event event;
    cl_ulong time_start, time_end;

    // Used to set the workgroup sizes
    size_t global_item_size, local_item_size;

    // Used to export the kernel results into proper files or the terminal
    string exportFileName;
    
    int reportToTerminal = 0;
    int reportToFile = 1;
  

    int MAX_TOTAL_CUS_PER_CTU = max(TOTAL_ALIGNED_CUS_PER_CTU, TOTAL_HALF_ALIGNED_CUS_PER_CTU); // used to allocate memory enough for the worst case
    int MAX_nWGs = nCtus * max(NUM_CU_SIZES, HA_NUM_CU_SIZES); // used to allocate memory enough for the worst case

    print_timestamp((char*)"START ALLOCATE MEMORY");

    // ----------------------------
    //
    //     Creates buffers for GPU

    // These memory objects hold the best cost and respective CPMVs for each 128x128 CTU
    // nCtus * TOTAL_ALIGNED/HA_CUS_PER_CTU accounts for all aligned/HalfAligned CUs (and all sizes) inside each CTU
    cl_mem return_costs_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                nCtus * MAX_TOTAL_CUS_PER_CTU * sizeof(cl_long), NULL, &error_1);   
    cl_mem return_cpmvs_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                nCtus * MAX_TOTAL_CUS_PER_CTU * sizeof(Cpmvs), NULL, &error_2);
    // This memory object is used to share data among workitems of the same workgroup. __local memory is not enough for such amount of data
    cl_mem horizontal_grad_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                MAX_nWGs * 128*128 * sizeof(cl_short), NULL, &error_3);   
    cl_mem vertical_grad_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                MAX_nWGs * 128*128 * sizeof(cl_short), NULL, &error_4); 
    error = error_1 | error_2 | error_3 | error_4;
    // 7*7 is the dimension of the system of equations. Each workitem inside each WG hold its own system    
    cl_mem equations_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                MAX_nWGs*itemsPerWG*7*7 * sizeof(cl_long), NULL, &error_1); // maybe it is possible to use cl_int here

    // These memory objects are used to store intermediate data and debugging information from the kernel
    cl_mem debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                MAX_nWGs*itemsPerWG*4 * sizeof(cl_long), NULL, &error_2);     
    cl_mem cu_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                128*128 * sizeof(cl_short), NULL, &error_3);

    error |= error_1 | error_2 | error_3;
    probe_error(error,(char*)"Error creating memory object for shared data and debugging information\n");

    // ----------------------------
    //
    //     Creates buffers for HOST

    // These dynamic arrays retrieve the information from the kernel to the host
    // Useful information returned by kernel: bestSATD and bestCMP of each CU
    long *return_costs =        (long*) malloc(sizeof(long) * nCtus * MAX_TOTAL_CUS_PER_CTU);
    Cpmvs *return_cpmvs =       (Cpmvs*) malloc(sizeof(Cpmvs) * nCtus * MAX_TOTAL_CUS_PER_CTU);
    // Debug information returned by kernel
    long *debug_data =          (long*)  malloc(sizeof(long)  * MAX_nWGs*itemsPerWG*4);
    short *return_cu =          (short*) malloc(sizeof(short) * 128*128);
    long *return_equations =    (long*)  malloc(sizeof(long)  * MAX_nWGs*itemsPerWG*7*7);
    short *horizontal_grad =    (short*) malloc(sizeof(short) * 128*128*MAX_nWGs);
    short *vertical_grad =      (short*) malloc(sizeof(short) * 128*128*MAX_nWGs);

    print_timestamp((char*)"FINISH ALLOCATE MEMORY");

    cl_ulong write_time_start;
    cl_ulong write_time_end;
    cl_event copy_event;

    print_timestamp((char*)"START GPU KERNEL");
    
    // Used to debug the circular buffer used to keep reference frames
    unsigned short* testArray[4];
    testArray[0] = (unsigned short*) malloc(FRAME_SIZE * sizeof(unsigned short));
    testArray[1] = (unsigned short*) malloc(FRAME_SIZE * sizeof(unsigned short));
    testArray[2] = (unsigned short*) malloc(FRAME_SIZE * sizeof(unsigned short));
    testArray[3] = (unsigned short*) malloc(FRAME_SIZE * sizeof(unsigned short));
    
    int numRefs = -1;
    for(int curr=0; curr<N_FRAMES; curr++){
        
        cl_int currFrame = curr;
        // We start at POC=1 since there is no AME in intra frames
        int poc = curr + 1;

        // printf("Processing frame %d (POC %d)\n", curr, poc);

        numRefs = min(4, poc);
        lambda =  fullLambdas[ inputQp + qpOffset[poc%8] ];

        if(poc<5){ // Ref list is not full yet. Always update all positions
            
            // TODO: Aff if(poc>x) to avoid doing backup of data that do not need to be written into other memory positions. Until the whole reference array is filled with long term references this is common
            
            // Copy ref[0] into tempA
            label_tempA = label_circularBufferRefs[0];
            error  = clEnqueueCopyBuffer(command_queue, memObj_circularBufferRefs[0], memObj_tempA, 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
            error |= clFinish(command_queue);
            probe_error(error, (char*) "Error copying ref[0] into temp memObj\n");

            // Fill ref[0] with new frame
            label_circularBufferRefs[0] = poc-1;
            error = clEnqueueWriteBuffer(command_queue, memObj_circularBufferRefs[0], CL_TRUE, 0, FRAME_SIZE * sizeof(short), reference_frame+currFrame*FRAME_SIZE, 0, NULL, &write_event); 
            error |= clFinish(command_queue);
            probe_error(error, (char*) "Error writing new ref into ref[0]\n");

            if(numRefs>1){
                // Copy ref[1] into tempB
                label_tempB = label_circularBufferRefs[1];
                error  = clEnqueueCopyBuffer(command_queue, memObj_circularBufferRefs[1], memObj_tempB, 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error copying ref[1] into temp memObj\n");

                // Copy tempA into ref[1]
                label_circularBufferRefs[1] = label_tempA;
                error  = clEnqueueCopyBuffer(command_queue, memObj_tempA, memObj_circularBufferRefs[1], 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error copying temp into ref[1] memObj\n");
            }

            if(numRefs>2){
                // Copy ref[2] into tempA
                label_tempA = label_circularBufferRefs[2];
                error  = clEnqueueCopyBuffer(command_queue, memObj_circularBufferRefs[2], memObj_tempA, 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error copying ref[2] into temp memObj\n");

                // Copy tempB into ref[2]
                label_circularBufferRefs[2] = label_tempB;
                error  = clEnqueueCopyBuffer(command_queue, memObj_tempB, memObj_circularBufferRefs[2], 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error copying temp into ref[2] memObj\n");
            }

            if(numRefs>3){
                // Copy tempA into ref[3]
                label_circularBufferRefs[3] = label_tempA;
                error  = clEnqueueCopyBuffer(command_queue, memObj_tempA, memObj_circularBufferRefs[3], 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error copying temp into ref[0] memObj\n");
            }
            
            label_circularBufferIsLT[3] = label_circularBufferRefs[3]%8==0 ? 1 : 0;

        }
        else{ // Ref list is full

            int update = 0;
            
            // Copy ref[0] into tempA
            label_tempA = label_circularBufferRefs[0];
            error  = clEnqueueCopyBuffer(command_queue, memObj_circularBufferRefs[0], memObj_tempA, 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
            error |= clFinish(command_queue);
            probe_error(error, (char*) "Error copying ref[0] into temp memObj\n");

            // Fill ref[0] with new frame
            label_circularBufferRefs[0] = poc-1;
            error = clEnqueueWriteBuffer(command_queue, memObj_circularBufferRefs[0], CL_TRUE, 0, FRAME_SIZE * sizeof(short), reference_frame+currFrame*FRAME_SIZE, 0, NULL, &write_event); 
            error |= clFinish(command_queue);
            probe_error(error, (char*) "Error writing new ref into ref[0]\n");

            // Update ref[1] ?
            update = label_circularBufferIsLT[1]==0 ? 1 : ( label_tempA%8==0 && label_tempA!=label_circularBufferRefs[0] ? 1 : 0 ); 
            if(update){
                // Copy ref[1] into tempB
                label_tempB = label_circularBufferRefs[1];
                error  = clEnqueueCopyBuffer(command_queue, memObj_circularBufferRefs[1], memObj_tempB, 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error copying ref[1] into temp memObj\n");

                // Copy tempA into ref[1]
                label_circularBufferRefs[1] = label_tempA;
                error  = clEnqueueCopyBuffer(command_queue, memObj_tempA, memObj_circularBufferRefs[1], 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                error |= clFinish(command_queue);
                probe_error(error, (char*) "Error writing new ref into ref[1]\n");

                // Update ref[2]?
                update = label_circularBufferIsLT[2]==0 ? 1 : ( label_tempB%8==0 && label_tempB!=label_circularBufferRefs[1] ? 1 : 0 ); 
                if(update){
                    // Copy ref[2] into tempA
                    label_tempA = label_circularBufferRefs[2];
                    error  = clEnqueueCopyBuffer(command_queue, memObj_circularBufferRefs[2], memObj_tempA, 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                    error |= clFinish(command_queue);
                    probe_error(error, (char*) "Error copying ref[2] into temp memObj\n");

                    // Copy tempB into ref[2]
                    label_circularBufferRefs[2] = label_tempB;
                    error  = clEnqueueCopyBuffer(command_queue, memObj_tempB, memObj_circularBufferRefs[2], 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                    error |= clFinish(command_queue);
                    probe_error(error, (char*) "Error writing new ref into ref[2]\n");

                    // Update ref[3]?
                    update = label_circularBufferIsLT[3]==0 ? 1 : ( label_tempA%8==0 && label_tempA!=label_circularBufferRefs[3] ? 1 : 0 ); 
                    if(update){
                        // Copy tempA into ref[3]
                        label_circularBufferRefs[3] = label_tempA;
                        error  = clEnqueueCopyBuffer(command_queue, memObj_tempA, memObj_circularBufferRefs[3], 0, 0, FRAME_SIZE * sizeof(cl_short), 0, NULL, &copy_event );
                        error |= clFinish(command_queue);
                        probe_error(error, (char*) "Error writing new ref into ref[3]\n");
                    }
                }
            }
            // Check which ones are long term (LT) refereneces
            label_circularBufferIsLT[3] =   label_circularBufferRefs[3]%8==0 ? 1 : 0;
            label_circularBufferIsLT[2] = ( label_circularBufferRefs[2]%8==0 && label_circularBufferIsLT[3] ) ? 1 : 0;
            label_circularBufferIsLT[1] = ( label_circularBufferRefs[1]%8==0 && label_circularBufferIsLT[2] ) ? 1 : 0;
        }

        error  = clEnqueueReadBuffer(command_queue, memObj_circularBufferRefs[0], CL_TRUE, 0, FRAME_SIZE * sizeof(cl_short), testArray[0], 0, NULL, &read_event);
        probe_error(error, (char*)"Error reading teste\n");
        error |= clEnqueueReadBuffer(command_queue, memObj_circularBufferRefs[1], CL_TRUE, 0, FRAME_SIZE * sizeof(cl_short), testArray[1], 0, NULL, &read_event);
        probe_error(error, (char*)"Error reading teste\n");
        error |= clEnqueueReadBuffer(command_queue, memObj_circularBufferRefs[2], CL_TRUE, 0, FRAME_SIZE * sizeof(cl_short), testArray[2], 0, NULL, &read_event);
        probe_error(error, (char*)"Error reading teste\n");
        error |= clEnqueueReadBuffer(command_queue, memObj_circularBufferRefs[3], CL_TRUE, 0, FRAME_SIZE * sizeof(cl_short), testArray[3], 0, NULL, &read_event);
        probe_error(error, (char*)"Error reading teste\n");

        // Summary of references and lambdas per frame. Consider the case where POC=0 has all samples equal zero
        printf("POC %3d   QP %d motionLambda %f : [L0 %d", poc, inputQp+qpOffset[poc%8], lambda, testArray[0][5000] );
        for(int r=1; r<numRefs; r++){
            printf(" %d", testArray[r][5000]);
        }
        printf("]\n");

    }

    return 1;


    for(int curr=0; curr<N_FRAMES; curr++){

        cl_int currFrame = curr;

        printf("PROCESSING FRAME %d\n", currFrame);

        print_timestamp((char*)"START WRITE SAMPLES MEMOBJ");
        // Write reference and original samples into memory object
        nanoSeconds = 0.0;
        error  = clEnqueueWriteBuffer(command_queue, ref_samples_mem_obj, CL_TRUE, 0, 
                FRAME_SIZE * sizeof(short), reference_frame+currFrame*FRAME_SIZE, 0, NULL, &write_event); 
        error = clWaitForEvents(1, &write_event);
        probe_error(error, (char*)"Error waiting for write events\n");  
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing write\n");
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
        nanoSeconds += write_time_end-write_time_start;

        error |= clEnqueueWriteBuffer(command_queue, curr_samples_mem_obj, CL_TRUE, 0, 
                FRAME_SIZE * sizeof(short), current_frame+currFrame*FRAME_SIZE, 0, NULL, &write_event);      
        error = clWaitForEvents(1, &write_event);
        probe_error(error, (char*)"Error waiting for write events\n");  
        error = clFinish(command_queue);
        probe_error(error, (char*)"Error finishing write\n");
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
        nanoSeconds += write_time_end-write_time_start;
        // printf("Partial read %0.3f miliseconds \n", (write_time_end-write_time_start) / 1000000.0);

        samplesWritingTime += nanoSeconds;
        nanoSeconds = 0.0;

        probe_error(error, (char*)"Error copying data from memory to buffers LEGACY\n");

        print_timestamp((char*)"FINISH WRITE SAMPLES MEMOBJ");

        // ------------------------------------------------------
        //
        //  IF WE ARE CONDUCTING AFFINE OVER FULLY-ALIGNED BLOCKS...
        //
        // ------------------------------------------------------       
        if(TEST_FULL){
            // Update variables based on alignment
            testingAlignedCus = 1; // Toggle between predict ALIGNED or HALF_ALIGNED CUs
            nWG = nCtus * (testingAlignedCus ? NUM_CU_SIZES : HA_NUM_CU_SIZES);     // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs
            
            for(int PRED=FULL_2CP; PRED<=FULL_3CP; PRED++){
                // Select specific kernel based on iteration
                // printf("Current Affine Code = %d...\n", PRED);
                if(PRED==FULL_2CP){
                    // printf("Predicting FULLY-ALIGNED blocks with 2 CPs...\n");
                    print_timestamp((char*)"START EXEC FULL 2 CPs");
                    kernel = kernel_FULL_2CP;
                }                
                else if(PRED==FULL_3CP){
                    // printf("Predicting FULLY-ALIGNED blocks with 3 CPs...\n");
                    print_timestamp((char*)"START EXEC FULL 3 CPs");
                    kernel = kernel_FULL_3CP;
                }

                // Query for work groups sizes information
                error = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
                error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
                error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
                error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);
                
                probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
                // cout << "-- Preferred WG size multiple " << preferred_size << endl;
                // cout << "-- Maximum WG size " << maximum_size << endl;

                // Set the arguments of the kernel
                error_1  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&ref_samples_mem_obj);
                error_1 |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&curr_samples_mem_obj);
                error_1 |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&frameWidth);
                error_1 |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&frameHeight);
                error_1 |= clSetKernelArg(kernel, 4, sizeof(cl_float), (void *)&lambda);
                error_1 |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&horizontal_grad_mem_obj);
                error_1 |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&vertical_grad_mem_obj);
                error_1 |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&equations_mem_obj);
                error_1 |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&return_costs_mem_obj);
                error_1 |= clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&return_cpmvs_mem_obj);
                error_1 |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&debug_mem_obj);
                error_1 |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&cu_mem_obj);  
                probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

                // Report the number of bytes used by all kernel parameters (memory objects and scalars)
                // accessMemoryUsage(PRED, ref_samples_mem_obj, curr_samples_mem_obj, frameWidth, frameHeight, lambda, horizontal_grad_mem_obj, vertical_grad_mem_obj, equations_mem_obj, return_costs_mem_obj, return_cpmvs_mem_obj, debug_mem_obj, cu_mem_obj);

                // Execute the OpenCL kernel on the list
                // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
                global_item_size = nWG*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
                local_item_size = itemsPerWG; 
                
                error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                        &global_item_size, &local_item_size, 0, NULL, &event);
                probe_error(error, (char*)"Error enqueuing kernel\n");

                error = clWaitForEvents(1, &event);
                probe_error(error, (char*)"Error waiting for events\n");
                
                error = clFinish(command_queue);
                probe_error(error, (char*)"Error finishing\n");

                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
                nanoSeconds = time_end-time_start;
                
                kernelExecutionTime[PRED] += nanoSeconds;
                nanoSeconds = 0;

                if(PRED==FULL_2CP){
                    print_timestamp((char*)"FINISH EXEC FULL 2 CPs");
                    print_timestamp((char*)"START READ FULL 2 CPs");
                }                
                else if(PRED==FULL_3CP){
                    print_timestamp((char*)"FINISH EXEC FULL 3 CPs");
                    print_timestamp((char*)"START READ FULL 3 CPs");
                }

                // Read affine results from memory objects into host arrays
                readMemobjsIntoArray(PRED, command_queue, nWG, itemsPerWG, nCtus, testingAlignedCus, return_costs_mem_obj, return_cpmvs_mem_obj,debug_mem_obj, cu_mem_obj, equations_mem_obj, horizontal_grad_mem_obj, vertical_grad_mem_obj, return_costs, return_cpmvs, debug_data, return_cu, return_equations, horizontal_grad, vertical_grad);

                if(PRED==FULL_2CP){
                    print_timestamp((char*)"FINISH READ FULL 2 CPs");
                    if(reportToFile)
                        print_timestamp((char*)"START EXPORT FULL 2 CPs");
                }                
                else if(PRED==FULL_3CP){
                    print_timestamp((char*)"FINISH READ FULL 3 CPs");
                    if(reportToFile)
                        print_timestamp((char*)"START EXPORT FULL 3 CPs");
                        
                }

                // Report affine results (CPMVs and costs) to terminal or writing to files
                reportAffineResultsMaster(reportToTerminal, reportToFile, cpmvFilePreffix, PRED, nWG, frameWidth, frameHeight, return_costs, return_cpmvs, currFrame);

                if(reportToFile){
                    if(PRED==FULL_2CP){
                        print_timestamp((char*)"FINISH EXPORT FULL 2 CPs");
                    }                
                    else if(PRED==FULL_3CP){
                        print_timestamp((char*)"FINISH EXPORT FULL 3 CPs");
                    }
                }
            }
        }

        // ------------------------------------------------------
        //
        //  IF WE ARE CONDUCTING AFFINE OVER HALF-ALIGNED BLOCKS
        //
        // ------------------------------------------------------
        if(TEST_HALF){
            // Update variables based on alignment
            testingAlignedCus = 0; // Toggle between predict ALIGNED or HALF_ALIGNED CUs
            nWG = nCtus * (testingAlignedCus ? NUM_CU_SIZES : HA_NUM_CU_SIZES);     // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs
            
            for(int PRED=HALF_2CP; PRED<=HALF_3CP; PRED++){
                // Select specific kernel based on iteration
                // printf("Current Affine Code = %d...\n", PRED);
                if(PRED==HALF_2CP){
                    print_timestamp((char*)"START EXEC HALF 2 CPs");
                    kernel = kernel_HALF_2CP;
                }                
                else if(PRED==HALF_3CP){
                    print_timestamp((char*)"START EXEC HALF 3 CPs");
                    kernel = kernel_HALF_3CP;

                }
                    
                // Query for work groups sizes information
                error = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
                error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
                error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
                error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);
                
                probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
                // cout << "-- Preferred WG size multiple " << preferred_size << endl;
                // cout << "-- Maximum WG size " << maximum_size << endl;
                
                // Set the arguments of the kernel
                error_1  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&ref_samples_mem_obj);
                error_1 |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&curr_samples_mem_obj);
                error_1 |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&frameWidth);
                error_1 |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&frameHeight);
                error_1 |= clSetKernelArg(kernel, 4, sizeof(cl_float), (void *)&lambda);
                error_1 |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&horizontal_grad_mem_obj);
                error_1 |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&vertical_grad_mem_obj);
                error_1 |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&equations_mem_obj);
                error_1 |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&return_costs_mem_obj);
                error_1 |= clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&return_cpmvs_mem_obj);
                error_1 |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&debug_mem_obj);
                error_1 |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&cu_mem_obj);    
                probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

                // Report the number of bytes used by all kernel parameters (memory objects and scalars)
                // accessMemoryUsage(PRED, ref_samples_mem_obj, curr_samples_mem_obj, frameWidth, frameHeight, lambda, horizontal_grad_mem_obj, vertical_grad_mem_obj, equations_mem_obj, return_costs_mem_obj, return_cpmvs_mem_obj, debug_mem_obj, cu_mem_obj);

                // Execute the OpenCL kernel on the list
                // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
                global_item_size = nWG*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
                local_item_size = itemsPerWG; 
                
                error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                        &global_item_size, &local_item_size, 0, NULL, &event);
                probe_error(error, (char*)"Error enqueuing kernel\n");

                error = clWaitForEvents(1, &event);
                probe_error(error, (char*)"Error waiting for events\n");
                
                error = clFinish(command_queue);
                probe_error(error, (char*)"Error finishing\n");
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
                nanoSeconds = time_end-time_start;
                
                kernelExecutionTime[PRED] += nanoSeconds;
                nanoSeconds = 0;
                
                if(PRED==HALF_2CP){
                    print_timestamp((char*)"FINISH EXEC HALF 2 CPs");
                    print_timestamp((char*)"START READ HALF 2 CPs");
                }                
                else if(PRED==HALF_3CP){
                    print_timestamp((char*)"FINISH EXEC HALF 3 CPs");
                    print_timestamp((char*)"START READ HALF 3 CPs");

                }

                // Read affine results from memory objects into host arrays
                readMemobjsIntoArray(PRED, command_queue, nWG, itemsPerWG, nCtus, testingAlignedCus, return_costs_mem_obj, return_cpmvs_mem_obj,debug_mem_obj, cu_mem_obj, equations_mem_obj, horizontal_grad_mem_obj, vertical_grad_mem_obj, return_costs, return_cpmvs, debug_data, return_cu, return_equations, horizontal_grad, vertical_grad);

                if(PRED==HALF_2CP){
                    print_timestamp((char*)"FINISH READ HALF 2 CPs");
                    if(reportToFile)
                        print_timestamp((char*)"START EXPORT HALF 2 CPs");
                }                
                else if(PRED==HALF_3CP){
                    print_timestamp((char*)"FINISH READ HALF 3 CPs");
                    if(reportToFile)
                        print_timestamp((char*)"START EXPORT HALF 3 CPs");

                }

                // Report affine results to terminal or writing files
                reportAffineResultsMaster(reportToTerminal, reportToFile, cpmvFilePreffix, PRED, nWG, frameWidth, frameHeight, return_costs, return_cpmvs, currFrame);

                if(reportToFile){
                    if(PRED==HALF_2CP){
                        print_timestamp((char*)"FINISH EXPORT HALF 2 CPs");
                    }                
                    else if(PRED==HALF_3CP){
                        print_timestamp((char*)"FINISH EXPORT HALF 3 CPs");

                    }                
                }
            }
        }
    }
    print_timestamp((char*)"FINISH GPU KERNEL");


    reportTimingResults(N_FRAMES);
    // reportMemoryUsage();

    // FREE MEMORY USED BY HALF-ALIGNED BLOCKS
    error  = clReleaseMemObject(return_costs_mem_obj);
    error |= clReleaseMemObject(return_cpmvs_mem_obj);
    error |= clReleaseMemObject(debug_mem_obj);
    error |= clReleaseMemObject(horizontal_grad_mem_obj);
    error |= clReleaseMemObject(vertical_grad_mem_obj);
    error |= clReleaseMemObject(equations_mem_obj);
    probe_error(error,(char*)"Error releasing memory objects between ALIGNED and HALF-ALIGNED blocks\n");

    free(return_costs);
    free(return_cpmvs);
    free(debug_data);
    free(return_equations);
    free(horizontal_grad);
    free(vertical_grad);

    // -----------------------------------------------------------------
    //
    //  ALL THIS DEBUGGING INFORMATION MUST BE CORRECTED NOW THAT WE
    //  CONDUCT MULTIPLE AFFINE PREDICTIONS ON THE SAME MAIN PROGRAM
    //  AND FREE THE DYNAMIC ARRAYS AFTER TRACING THE RESULTS
    //
    // -----------------------------------------------------------------

    /* Print the contents of debug_data. BEWARE of the data types (long, short, int, ...)
    printf("Debug array...\n");
    // for(int i=0; i<nWG*itemsPerWG; i++){
    for(int j=0; j<128; j++){
        for(int k=0; k<128; k++){
            int iC[4];
            iC[0] = debug_data[j*128*4 + k*4 + 0];
            iC[1] = debug_data[j*128*4 + k*4 + 1];
            iC[2] = debug_data[j*128*4 + k*4 + 2];
            iC[3] = debug_data[j*128*4 + k*4 + 3];
            printf("j=%d,k=%d,iC[0]=%d,iC[1]=%d,iC[2]=%d,iC[3]=%d\n", j, k, iC[0], iC[1], iC[2], iC[3]);
        }
    //    printf("[%d] = %ld -> %d\n", i, debug_data[2*i], debug_data[2*i+1]);
    }
    //*/
  
    /* Print returned CU
    printf("Return CU\n");
    for(int i=0; i<128; i++){
        for(int j=0; j<128; j++){
            printf("%d,",return_cu[i*128+j]);
        }
        printf("\n");
    }
    //*/

    /* Print Gradients on a predefined position (SINGLE CU SIZE AND CTU, indexing must be adjusted)
    printf("Horizontal Gradient\n");
    for(int i=0; i<128; i++){
        for(int j=0; j<128; j++){
            printf("%d,",horizontal_grad[17*128*128 + i*128+j]);
        }
        printf("\n");
    }

    printf("Vertical Gradient\n");
    for(int i=0; i<128; i++){
        for(int j=0; j<128; j++){
            printf("%d,",vertical_grad[17*128*128 + i*128+j]);
        }
        printf("\n");
    }
    //*/

    ////////////////////////////////////////////////////
    /////         FREE SOME MEMORY SPACE           /////
    ////////////////////////////////////////////////////

    // Clean up
    error = clFlush(command_queue);
    error |= clFinish(command_queue);
    error |= clReleaseKernel(kernel);
    error |= clReleaseProgram(program_2CP);
    error |= clReleaseProgram(program_3CP);
    error |= clReleaseMemObject(ref_samples_mem_obj);
    error |= clReleaseMemObject(curr_samples_mem_obj);   
    error |= clReleaseMemObject(cu_mem_obj);
    error |= clReleaseCommandQueue(command_queue);
    error |= clReleaseContext(context);
    probe_error(error, (char*)"Error releasing  OpenCL objects\n");
    
    free(source_str);
    free(platform_id);
    free(reference_frame);
    free(current_frame);
    free(return_cu);
 
    print_timestamp((char*)"FINISH HOST");

    return 0;
}