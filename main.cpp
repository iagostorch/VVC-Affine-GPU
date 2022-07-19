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
#include "typedef.h"

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
    cl_uint ret_num_devices;
    
    // Select what CPU or GPU will be used based on parameters
    if(argc==7){
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

        if(!strcmp(argv[3],"22")){
            cout << "Using QP=22" << endl;
            lambda = lambdas[QP22];

        }
        else if(!strcmp(argv[3],"27")){
            cout << "Using QP=27" << endl;
            lambda = lambdas[QP27];
        }
        else if(!strcmp(argv[3],"32")){
            cout << "Using QP=32" << endl;
            lambda = lambdas[QP32];
        }
        else if(!strcmp(argv[3],"37")){
            cout << "Using QP=37" << endl;
            lambda = lambdas[QP37];
        }
        else{
            cout << "Incorrect usage. Third parameter must be the QP value, one of the following: 22, 27, 32, 37" << endl;
            exit(0);
        }
    }
    else{
        cout << "\n\n\nFailed to specify the input parameters. Proper execution has the form of" << endl;
        cout << "./main <CPU or GPU> <# of CPU or GPU device> <QP in set [22,27,32,37]> <original_frame_file> <reference_frame_file> <preffix for exported CPMV files>\n\n\n" << endl;
        exit(0);
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

    // TODO: This should be an input parameter
    const int frameWidth  = 1920;
    const int frameHeight = 1080;

    // Read the frame data into the matrix
    string currFileName = argv[4];  // File with samples from current frame
    string refFileName = argv[5];   // File with samples from reference frame
    string cpmvFilePreffix = argv[6];   // Preffix of exported files containing CPMV information

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

    unsigned short *reference_frame = (unsigned short*) malloc(sizeof(short) * FRAME_SIZE);
    unsigned short *current_frame   = (unsigned short*) malloc(sizeof(short) * FRAME_SIZE);

    // Read the samples from reference frame into the reference array
    for(int h=0; h<frameHeight; h++){
        getline(currFile, currLine, '\n');
        getline(refFile, refLine, '\n');
        stringstream currStream(currLine), refStream(refLine); 
        
        for(int w=0; w<frameWidth; w++){
            getline(currStream, currVal, ',');
            getline(refStream, refVal, ',');
            current_frame[h*frameWidth + w] = stoi(currVal);
            reference_frame[h*frameWidth + w] = stoi(refVal);
        }
    }
    
    // These buffers are for storing the reference samples and current samples
    cl_mem ref_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            FRAME_SIZE * sizeof(short), NULL, &error_1);    
    cl_mem curr_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            FRAME_SIZE * sizeof(short), NULL, &error_2);                
    error = error_1 || error_2;

    probe_error(error, (char*)"Error creating memory buffers\n");
    
    double nanoSeconds = 0;
    // These variabels are used to profile the time spend writing to memory objects "clEnqueueWriteBuffer"
    cl_ulong write_time_start;
    cl_ulong write_time_end;
    cl_event write_event;

    error  = clEnqueueWriteBuffer(command_queue, ref_samples_mem_obj, CL_TRUE, 0, 
            FRAME_SIZE * sizeof(short), reference_frame, 0, NULL, &write_event); 
    error = clWaitForEvents(1, &write_event);
    probe_error(error, (char*)"Error waiting for write events\n");  
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing write\n");
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
    nanoSeconds += write_time_end-write_time_start;

    error |= clEnqueueWriteBuffer(command_queue, curr_samples_mem_obj, CL_TRUE, 0, 
            FRAME_SIZE * sizeof(short), current_frame, 0, NULL, &write_event);      
    error = clWaitForEvents(1, &write_event);
    probe_error(error, (char*)"Error waiting for write events\n");  
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing write\n");
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
    nanoSeconds += write_time_end-write_time_start;
    // printf("Partial read %0.3f miliseconds \n", (write_time_end-write_time_start) / 1000000.0);

    probe_error(error, (char*)"Error copying data from memory to buffers LEGACY\n");

    printf("OpenCl WriteBuffer time is: %0.3f miliseconds \n",nanoSeconds / 1000000.0);


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
    // -cl-nv-verbose is used to show memory and registers used by the kernel
    // -cl-nv-maxrregcount=241 is used to set the maximum number of registers per kernel. Using a large value and modifying in each compilation makes no difference in the code, but makes the -cl-nv-verbose flag work properly
    srand (time(NULL));
    // TODO: Check if the number of registers is enough for the application
    int maxReg = 255;//+rand()%5;
    char argBuild[39];
    char *pt1 = "-cl-nv-maxrregcount=";
    char *pt2 = "-cl-nv-verbose";
    snprintf(argBuild, sizeof(argBuild), "%s%d %s", pt1, maxReg, pt2);
    cout << "\n\n\n@@@@@\n"<< argBuild<< "\n@@@@@\n\n\n";
    error = clBuildProgram(program, 1, &device_id, argBuild, NULL, NULL);
    
    // Build for non-NVIDIA devices
    // error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    probe_error(error, (char*)"Error building the program\n");
    // Show debugging information when the build is not successful
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
        free(log);
    }

    int testingAlignedCus = 0; // Toggle between predict ALIGNED or HALF_ALIGNED CUs

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, testingAlignedCus? "affine_gradient_mult_sizes" : "affine_gradient_mult_sizes_HA", &error);
    probe_error(error, (char*)"Error creating kernel\n");

    // TODO: Declare these on the correct place. nCtus=120 is specific for 1080p videos
    int itemsPerWG = 256;               // Each workgroup has 256 workitems
    int nCtus = 135;                    // 1080p videos have 120 entire CTUs plus 15 partial CTUs
    int nWG = nCtus * (testingAlignedCus ? NUM_CU_SIZES : HA_NUM_CU_SIZES);     // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs

    // These memory objects hold the best cost and respective CPMVs for each 128x128 CTU
    // nCtus * TOTAL_ALIGNED/HA_CUS_PER_CTU accounts for all aligned/HalfAligned CUs (and all sizes) inside each CTU
    cl_mem return_costs_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nCtus * (testingAlignedCus ? TOTAL_ALIGNED_CUS_PER_CTU : TOTAL_HALF_ALIGNED_CUS_PER_CTU) * sizeof(cl_long), NULL, &error_1);   
    cl_mem return_cpmvs_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nCtus * (testingAlignedCus ? TOTAL_ALIGNED_CUS_PER_CTU : TOTAL_HALF_ALIGNED_CUS_PER_CTU) * sizeof(Cpmvs), NULL, &error_2);
    error = error_1 | error_2;
    probe_error(error,(char*)"Error creating memory object for cost and CPMVs of each WG\n");

    // These memory objects are used to retrieve debugging information from the kernel
    cl_mem debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG*itemsPerWG*4 * sizeof(cl_long), NULL, &error);  
    cl_mem cu_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            128*128 * sizeof(cl_short), NULL, &error_1);  
    // This memory object is used to share data among workitems of the same workgroup. __local memory is not enough for such amount of data
    cl_mem horizontal_grad_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * 128*128 * sizeof(cl_short), NULL, &error_2);   
    cl_mem vertical_grad_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * 128*128 * sizeof(cl_short), NULL, &error_3);  

    // 7*7 is the dimension of the system of equations. Each workitem inside each WG hold its own system    
    cl_mem equations_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
        nWG*itemsPerWG*7*7 * sizeof(cl_long), NULL, &error_4); // maybe it is possible to use cl_int here
    error = error | error_1 | error_2 | error_3 | error_4;
    probe_error(error,(char*)"Error creating memory object for shared data and debugging information\n");

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

    // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
    cl_event event;
    cl_ulong time_start;
    cl_ulong time_end;

    size_t global_item_size = nWG*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
    size_t local_item_size = itemsPerWG; 
    
    // TODO: Correct the following line so it is possible to compare against the maximum size (i.e., try a specific local_size, but reduce to maximum_size if necessary)
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
    printf("OpenCl Execution time is: %0.3f miliseconds \n",nanoSeconds / 1000000.0);

    // Useful information returnbed by kernel: bestSATD and bestCMP of each CU
    long *return_costs = (long*) malloc(sizeof(long) * nCtus * (testingAlignedCus ? TOTAL_ALIGNED_CUS_PER_CTU : TOTAL_HALF_ALIGNED_CUS_PER_CTU));
    Cpmvs *return_cpmvs = (Cpmvs*) malloc(sizeof(Cpmvs) * nCtus * (testingAlignedCus ? TOTAL_ALIGNED_CUS_PER_CTU : TOTAL_HALF_ALIGNED_CUS_PER_CTU));
    // Debug information returned by kernel
    long *debug_data =       (long*)  malloc(sizeof(long)  * nWG*itemsPerWG*4);
    short *return_cu =       (short*) malloc(sizeof(short) * 128*128);
    long *return_equations = (long*)  malloc(sizeof(long)  * nWG*itemsPerWG*7*7);
    short *horizontal_grad = (short*) malloc(sizeof(short) * 128*128*nWG);
    short *vertical_grad =   (short*) malloc(sizeof(short) * 128*128*nWG);

    // These variabels are used to profile the time spend reading from memory objects "clEnqueueReadBuffer"
    // The "essential time" corresponds to reading the main results from the device. The other "non-essential" includes debugging information
    cl_ulong read_time_start;
    cl_ulong read_time_end;
    cl_event read_event;

    nanoSeconds = 0;

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

    printf("OpenCl Essential ReadBuffer time is: %0.3f miliseconds \n",nanoSeconds / 1000000.0);

    // The following memory reads are not essential, they only get some debugging information. This is not considered during performance estimation.
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

    printf("OpenCl All (+DEBUG) ReadBuffer time is: %0.3f miliseconds \n",nanoSeconds / 1000000.0);
    
    error = clEnqueueReadBuffer(command_queue, horizontal_grad_mem_obj, CL_TRUE, 0, 
            nWG * 128 * 128 * sizeof(cl_short), horizontal_grad, 0, NULL, &read_event);  
    
    error |= clEnqueueReadBuffer(command_queue, vertical_grad_mem_obj, CL_TRUE, 0, 
            nWG * 128 * 128 * sizeof(cl_short), vertical_grad, 0, NULL, &read_event);  
    probe_error(error, (char*)"Error reading gradients\n");


    probe_error(error, (char*)"Error reading returned memory objects into malloc'd arrays\n");

    // ###################################################
    // FROM NOW ON WE ARE EXPORTING THE RESULTS INTO THE
    // TERMINAL AND INTO DISTINCT FILES AS WELL
    // ###################################################

    int cuSizeIdx;  // Used to index the current CU size in RETURN_STRIDE_LIST
    int nCus;       // Number of aligned CUs inside CTU
    int dataIdx;    // Pointer to the position with the ME results for current CU

    // Enable/disable exporting results to distinct files and printing in terminal
    int exportCpmvToFile = 1;
    int printCpmvToTerminal = 0;
    string exportFileName;
    FILE *cpmvFile;

    // Top-left corner of the current CU inside the frame when exporting the results
    int currX, currY;

    // Report the results for ALIGNED CUS
    if(testingAlignedCus){
        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 128x128\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = _128x128;
        nCus = 1;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_128x128.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128;
                currX = (ctu*128)%frameWidth;

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");


        if(printCpmvToTerminal){
            printf("CUs 128x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = _128x64;
        nCus = 2;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_128x64.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_128x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_128x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 64x128\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x128;
        nCus = 2;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_64x128.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x128[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x128[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");
        
        if(printCpmvToTerminal){
            printf("CUs 64x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x64;
        nCus = 4;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_64x64.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 64x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x32;
        nCus = 8;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_64x32.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x32[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x32[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("CUs 32x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _32x64;
        nCus = 8;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_32x64.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_32x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_32x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");   


        if(printCpmvToTerminal){
            printf("CUs 32x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _32x32;
        nCus = 16;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_32x32.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_32x32[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_32x32[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");    

        if(printCpmvToTerminal){
            printf("CUs 64x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _64x16;
        nCus = 16;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_64x16.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_64x16[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_64x16[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");  
    

        if(printCpmvToTerminal){
            printf("CUs 16x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _16x64;
        nCus = 16;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_16x64.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_16x64[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_16x64[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");  

        if(printCpmvToTerminal){
            printf("CUs 32x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _32x16;
        nCus = 32;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_32x16.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_32x16[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_32x16[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n"); 

        if(printCpmvToTerminal){
            printf("CUs 16x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _16x32;
        nCus = 32;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_16x32.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_16x32[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_16x32[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n"); 

        if(printCpmvToTerminal){
            printf("CUs 16x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        cuSizeIdx = _16x16;
        nCus = 64;
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_16x16.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + Y_POS_16x16[cuIdx];
                currX = (ctu*128)%frameWidth + X_POS_16x16[cuIdx];

                dataIdx = ctu*TOTAL_ALIGNED_CUS_PER_CTU + RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;

                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n"); 
    }

    // Report the results for HALF-ALIGNED CUS
    if(!testingAlignedCus){
        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 64x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = HA_64x32;
        nCus = HA_CUS_PER_CTU[cuSizeIdx];
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_64x32.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 32x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        cuSizeIdx = HA_32x64;
        nCus = HA_CUS_PER_CTU[cuSizeIdx];
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_32x64.csv";
            cpmvFile = fopen(exportFileName.c_str(),"w");
            fprintf(cpmvFile,"CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }
        for(int ctu=0; ctu<nWG/HA_NUM_CU_SIZES; ctu++){
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");      

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 64x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_64x16.csv";
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
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_64x16_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");  

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 16x64\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_16x64.csv";
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
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x64_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 32x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_32x32.csv";
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
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_32x32_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 32x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_32x16.csv";
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
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_32x16_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_32x16_G3;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 16x32\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_16x32.csv";
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
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x32_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x32_G3;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");

        if(printCpmvToTerminal){
            printf("Motion Estimation results...\n");
            printf("CUs 16x16\n");
            printf("CTU,idx,X,Y,Cost,LT_X,LT_Y,RT_X,RT_Y,LB_X,LB_Y\n");
        }    
        
        if (exportCpmvToFile){
            exportFileName = cpmvFilePreffix + "_16x16.csv";
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
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x16_G2;
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }

            cuSizeIdx = HA_16x16_G34; // G34 corresponds to two sets of CUs (G3 and G4) combined into a single set to optimize resources usage (G3 and G4 alone would have less CUs than workitems)
            nCus = HA_CUS_PER_CTU[cuSizeIdx];
            for(int cuIdx=0; cuIdx<nCus; cuIdx++){
                currY = ((ctu*128)/frameWidth)*128 + HA_ALL_Y_POS[cuSizeIdx][cuIdx];
                currX = (ctu*128)%frameWidth + HA_ALL_X_POS[cuSizeIdx][cuIdx];

                dataIdx = ctu*TOTAL_HALF_ALIGNED_CUS_PER_CTU + HA_RETURN_STRIDE_LIST[cuSizeIdx] + cuIdx;
                
                if(printCpmvToTerminal){
                    printf("%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
                
                if (exportCpmvToFile){
                    fprintf(cpmvFile, "%d,%d,%d,%d,%ld,%d,%d,%d,%d,%d,%d\n", ctu, cuIdx, currX, currY, return_costs[dataIdx], return_cpmvs[dataIdx].LT.x, return_cpmvs[dataIdx].LT.y, return_cpmvs[dataIdx].RT.x, return_cpmvs[dataIdx].RT.y, return_cpmvs[dataIdx].LB.x, return_cpmvs[dataIdx].LB.y);          
                }
            }
        }
        if (exportCpmvToFile){
            fclose(cpmvFile);
        }
        printf("\n");
       
    }

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
  
    //* Print returned CU
    printf("Return CU\n");
    for(int i=0; i<128; i++){
        for(int j=0; j<128; j++){
            printf("%d,",return_cu[i*128+j]);
        }
        printf("\n");
    }
    //*/

    //* Print Gradients on a predefined position (SINGLE CU SIZE AND CTU, indexing must be adjusted)
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
    error |= clReleaseProgram(program);
    error |= clReleaseMemObject(ref_samples_mem_obj);
    error |= clReleaseMemObject(curr_samples_mem_obj);   
    error |= clReleaseMemObject(return_costs_mem_obj);
    error |= clReleaseMemObject(return_cpmvs_mem_obj);
    error |= clReleaseMemObject(debug_mem_obj);
    error |= clReleaseMemObject(cu_mem_obj);
    error |= clReleaseMemObject(horizontal_grad_mem_obj);
    error |= clReleaseMemObject(vertical_grad_mem_obj);
    error |= clReleaseMemObject(equations_mem_obj);
    error |= clReleaseCommandQueue(command_queue);
    error |= clReleaseContext(context);
    probe_error(error, (char*)"Error releasing  OpenCL objects\n");
    
    free(source_str);
    free(platform_id);
    free(reference_frame);
    free(current_frame);
    free(return_costs);
    free(return_cpmvs);
    free(debug_data);
    free(return_cu);
    free(return_equations);
 
    return 0;
}