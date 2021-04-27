#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <string.h>
#include <limits.h>
#include <assert.h>

using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

void probe_error(cl_int error, char* message){
    if (error != CL_SUCCESS ) {
        printf("Code %d, %s", error, message);
        return;
    }
}

int findBestSad(int* sad, int frameWidth, int frameHeight, int blockTop, int blockLeft, int srWidth, int srHeight){
    int minSad = INT_MAX;
    int minIdx = -1;

    int forTop, forBottom, forLeft, forRight;

    forTop = max(0, blockTop-srHeight);
    forLeft = max(0, blockLeft-srWidth);
    forBottom = min(frameHeight, blockTop+srHeight);
    forRight = min(frameWidth, blockLeft+srWidth);

    for(int row=forTop; row<forBottom; row++){
        for(int col=forLeft; col<forRight; col++){
            if(sad[row*frameWidth+col]<minSad){
                minSad = sad[row*frameWidth+col];
                minIdx = row*frameWidth+col;
            }
        }
    }

    // for(int i=0; i<n; i++){
    //     if (sad[i] < minSad){
    //         minSad = sad[i];
    //         minIdx = i;
    //     }
    // }

    assert(minIdx!=0 && minSad!= INT_MAX);

    return minIdx;
}

int main(int argc, char *argv[]) {
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("pseudo_ME.cl", "r");
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
    cl_uint cpu_id;
    cl_uint gpu_id;
    cl_uint ret_num_platforms;

    error = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    probe_error(error, (char*)"Error querying available platforms\n");
    
    platform_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * ret_num_platforms); // Malloc space for all ret_num_platforms platforms
    
    error = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    probe_error(error, (char*)"Error querying platform IDs\n");


    // TODO: Improve this if/else to work with any hardware setting
    char platform_name[128] = {0};
    // List name of platforms available and assign to proper IDs
    cout << "Idx    Platform Name" << endl;
    for (cl_uint ui=0; ui< ret_num_platforms; ++ui){
        error = clGetPlatformInfo(platform_id[ui], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        if (platform_name != NULL){
            cout << ui << "      " << platform_name << endl;
            if (!strcmp(platform_name, "Intel(R) CPU Runtime for OpenCL(TM) Applications")){
                cpu_id = ui;
            }
            if (!strcmp(platform_name, "Intel(R) OpenCL HD Graphics")){
                gpu_id = ui;
            }
        }
    }

    // TODO: Proper way would be to declare arrays for CPU and GPU devices, since clGetDeviceIDs requires number of devices and array to store IDs
    cl_device_id cpu_device_id = NULL; 
    cl_uint ret_cpu_num_devices;
    
    cl_device_id gpu_device_id = NULL; 
    cl_uint ret_gpu_num_devices;

    // List information of devices in each platform   
    error = clGetDeviceIDs( platform_id[cpu_id], CL_DEVICE_TYPE_CPU, 0, 
            NULL, &ret_cpu_num_devices);
    error |= clGetDeviceIDs( platform_id[cpu_id], CL_DEVICE_TYPE_CPU, ret_cpu_num_devices, 
            &cpu_device_id, NULL);
    probe_error(error, (char*)"Error querying CPU device IDs\n");

    error = clGetDeviceIDs( platform_id[gpu_id], CL_DEVICE_TYPE_GPU, 0, 
            NULL, &ret_gpu_num_devices);
    error |= clGetDeviceIDs( platform_id[gpu_id], CL_DEVICE_TYPE_GPU, ret_gpu_num_devices, 
            &gpu_device_id, NULL);
    probe_error(error, (char*)"Error querying GPU device IDs\n");


    cout << "CPU Platform information... " << endl;
    cout << "\tplatform_id" << " " << platform_id[cpu_id] << endl;
    cout << "\tdevice_id" << " " << cpu_device_id << endl;
    cout << "\tret_num_devices" << " " << ret_cpu_num_devices << endl;
    

    cout << "GPU Platform information... " << endl;
    cout << "\tplatform_id" << " " << platform_id[gpu_id] << endl;
    cout << "\tdevice_id" << " " << gpu_device_id << endl;
    cout << "\tret_num_devices" << " " << ret_gpu_num_devices << endl;
    

    // Create "target" device and assign proper IDs
    cl_device_id device_id = NULL; 
    cl_uint ret_num_devices;
    
    // Select if we are using CPU or GPU
    if(argc>1){
        if(!strcmp(argv[1],"CPU")){
            cout << "COMPUTING ON CPU" << endl;
            device_id = cpu_device_id;
            ret_num_devices = ret_cpu_num_devices;
        } else if(!strcmp(argv[1],"GPU")){
            cout << "COMPUTING ON GPU" << endl;
            device_id = gpu_device_id;
            ret_num_devices = ret_gpu_num_devices;
        } else{
            cout << "Ignoring argument, COMPUTING ON CPU" << endl;
            device_id = cpu_device_id;
            ret_num_devices = ret_cpu_num_devices;
        }
    }
    else{
        cout << "Failed to specify CPU or GPU, COMPUTING ON CPU" << endl;
        device_id = cpu_device_id;
        ret_num_devices = ret_cpu_num_devices; 
    }
    

    size_t ret_val;
    cl_uint max_compute_units;
    error = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &ret_val);
    error|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, ret_val, &max_compute_units, NULL);
    probe_error(error, (char*)"Error querying maximum number of compute units of device\n");
    cout << "-- Max compute units " << max_compute_units << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////         STARTS BY CREATING A CONTEXT, QUEUE, AND MOVING DATA INTO THE BUFFERS         /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &error);
    probe_error(error, (char*)"Error creating context\n");

    // Create a command queue
    // Profiling enabled to measure execution time. 
    // TODO: Remove this profiling when perform actual computation, it may slowdown the processing (for experiments and etc)
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    // Read the frame data into the matrix
    string currFileName = "data/current.csv";         // File with samples from current frame
    string refFileName = "data/reference.csv";     // File with samples from reference frame
    int const frameWidth = 832;
    int const frameHeight = 480;
    int const blockWidth = 64;
    int const blockHeight = 64;
    int const srWidth = 64; // Search range width and height (origin +/- width/height)
    int const srHeight = 64;

    unsigned int currFrame[frameHeight][frameWidth], refFrame[frameHeight][frameWidth];

    ifstream currFile, refFile;
    currFile.open(currFileName);
    refFile.open(refFileName);

    if (!currFile.is_open() || !refFile.is_open()) {     /* validate file open for reading */
        perror (("error while opening files" ));
        return 1;
    }

    string currLine, currVal, refLine, refVal;
    
    for(int h=0; h<frameHeight; h++){
        
        getline(currFile, currLine, '\n');
        getline(refFile, refLine, '\n');
        stringstream currStream(currLine), refStream(refLine); 
        
        for(int w=0; w<frameWidth; w++){
            getline(currStream, currVal, ',');
            getline(refStream, refVal, ',');
            // cout << origVal << ",";// << endl;;
            currFrame[h][w] = stoi(currVal);
            refFrame[h][w] = stoi(refVal);
        }
        // cout << endl;
    }

    // // Print current or reference samples
    // for(int h=0; h<frameHeight; h++){
    //     for(int w=0; w<frameWidth; w++){
    //         // cout << currFrame[h][w] << ",";
    // //         // cout << refFrame[h][w] << ",";
    //     }
    //     // cout << endl;
    // }


    int const n_frame = frameWidth*frameHeight;
    int const n_block = blockWidth*blockHeight;
    int const n_searchRange = (srWidth*2)*(srHeight*2);
    int unsigned block_samples[n_block];

    int blockTop = 380;     // Position of a "target block" to conduck ME
    int blockLeft = 300;

    // Copy samples from a specific block (blockTop, blockLeft) to a separate array
    for(int r=0; r<blockHeight; r++){
        for(int c=0; c<blockWidth; c++){
            block_samples[r*blockWidth + c] = currFrame[r+blockTop][c+blockLeft];
         }
    }
    

    const int FRAME_SIZE = n_frame;
    const int BLOCK_SIZE = n_block;
    const int SEARCH_RANGE_SIZE = n_searchRange;

    unsigned int *reference = (unsigned int*) malloc(sizeof(int) * FRAME_SIZE);
    unsigned int *samples = (unsigned  int*) malloc(sizeof(int) * BLOCK_SIZE);

    for(int i=0; i<FRAME_SIZE; i++){
        reference[i] = refFrame[i/frameWidth][i%frameWidth];
    }
    for (int i=0; i<BLOCK_SIZE; i++){
        samples[i] = block_samples[i];
    }

    // for(int i=0; i<REF_SIZE; i++)
    //     cout << reference[i] << endl;
    // printf("\n");

    // for(int i=0; i<BLOCK_SIZE; i++)
    //     cout << samples[i] << endl;
    
    cl_mem reference_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            FRAME_SIZE * sizeof(int), NULL, &error_1);
    cl_mem samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            BLOCK_SIZE * sizeof(int), NULL, &error_2);
    cl_mem sad_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            FRAME_SIZE * sizeof(int), NULL, &error_3);
    probe_error(error_1&error_2&error_3, (char*)"Error creating memory buffers\n");

    
    error_1 = clEnqueueWriteBuffer(command_queue, reference_mem_obj, CL_TRUE, 0,
            FRAME_SIZE * sizeof(int), reference, 0, NULL, NULL);
    error_2 = clEnqueueWriteBuffer(command_queue, samples_mem_obj, CL_TRUE, 0, 
            BLOCK_SIZE * sizeof(int), samples, 0, NULL, NULL);
    // error_3 = clEnqueueWriteBuffer(command_queue, sad_mem_obj, CL_TRUE, 0, 
    //         REF_SIZE * sizeof(int), sad, 0, NULL, NULL);            
    probe_error(error_1&error_2, (char*)"Error copying data from memory to buffers LEGACY\n");

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////       CREATE A PROGRAM (OBJECT) BASED ON .cl FILE AND BUILD IT TO TARGET DEVICE       /////
    /////         CREATE A KERNEL BY ASSIGNING A NAME FOR THE RECENTLY COMPILED PROGRAM         /////
    /////                           LOADS THE ARGUMENTS FOR THE KERNEL                          /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create a program from the kernel source (specifically for the device in the context variable)
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &error);
    probe_error(error, (char*)"Error creating program from source\n");

    // Build the program
    error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    probe_error(error, (char*)"Error building the program\n");
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "FS_ME", &error);
    probe_error(error, (char*)"Error creating kernel\n");

    // Set the arguments of the kernel
    error_1  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&reference_mem_obj);
    error_1 |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&samples_mem_obj);
    error_1 |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&sad_mem_obj);
    error_1 |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&n_frame);
    error_1 |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&srWidth);
    error_1 |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&srHeight);
    error_1 |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&frameWidth);
    error_1 |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&frameHeight);
    error_1 |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&blockWidth);
    error_1 |= clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&blockHeight);
    error_1 |= clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&blockTop);
    error_1 |= clSetKernelArg(kernel, 11, sizeof(cl_int), (void *)&blockLeft);
    probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

    // Query for work groups sizes information
    size_t size_ret;
    cl_uint preferred_size, maximum_size;
    error = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
    error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);
    
    probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
    cout << "-- Preferred WG size multiple " << preferred_size << endl;
    cout << "-- Maximum WG size " << maximum_size << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////         ENQUEUE KERNEL (PUT IT TO BE EXECUTED WHEN THE DEVICE IS AVAILABLE)           /////
    /////         TRACK EXECUTION TIME. TODO: REMOVE TIME INFORMATION IF NOT NECESSARY          /////
    /////           RETRIEVE THE RESULT OF COMPUTATION BY READING THE MEMORY BUFFERS            /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Execute the OpenCL kernel on the list
    // Created an event to find out when queue is finished
    cl_event event;
    size_t global_item_size = SEARCH_RANGE_SIZE; // Process the entire lists
    size_t local_item_size = min(64, SEARCH_RANGE_SIZE); // Divide work items into groups of 64
    error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, &event);
    probe_error(error, (char*)"Error enqueuing kernel\n");

    clWaitForEvents(1, &event);
    clFinish(command_queue);
    
    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end-time_start;
    printf("OpenCl Execution time is: %0.3f miliseconds \n",nanoSeconds / 1000000.0);

    // Read the SAD memory buffer on the device to the local variable SAD
    int *sad = (int*)malloc(sizeof(int)*FRAME_SIZE);
    error = clEnqueueReadBuffer(command_queue, sad_mem_obj, CL_TRUE, 0, 
            FRAME_SIZE * sizeof(int), sad, 0, NULL, NULL);
    probe_error(error, (char*)"Error reading from buffer to memory\n");

    int minIdx = findBestSad(sad, frameWidth, frameHeight, blockTop, blockLeft, srWidth, srHeight);
    cout << "Id: " << minIdx << endl;
    cout << "\tMV yx  " << minIdx/frameWidth-blockTop << "x" << minIdx%frameWidth-blockLeft << endl;
    cout << "\tSAD    " << sad[minIdx] << endl;


    // for(int r=0; r<frameHeight; r++){
    //     for(int c=0; c<frameWidth; c++){
    //         cout << sad[r*frameWidth+c] << ",";
    //     }
    //     cout << endl;
    // }
 

    ////////////////////////////////////////////////////
    /////         FREE SOME MEMORY SPACE           /////
    ////////////////////////////////////////////////////
        
    // Clean up
    error = clFlush(command_queue);
    error = clFinish(command_queue);
    error = clReleaseKernel(kernel);
    error = clReleaseProgram(program);
    error = clReleaseMemObject(reference_mem_obj);
    error = clReleaseMemObject(samples_mem_obj);
    error = clReleaseMemObject(sad_mem_obj);
    error = clReleaseCommandQueue(command_queue);
    error = clReleaseContext(context);
    free(reference);
    free(samples);
    free(sad);
    
    return 0;
}