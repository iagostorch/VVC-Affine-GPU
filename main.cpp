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

    // TODO: These IDs should be arrays for the case when there are more than 1 CPU or 1 GPU
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
    // List name of platforms available and assign to proper CPU/GPU IDs
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
    if(argc==3){
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
        cout << "Failed to specify the input parameters. Proper execution has the form of" << endl;
        cout << "./main <CPU or GPU> <input_file>" << endl;
        exit(0);
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

    // TODO: This information must be gathered from the input file. 
    // This should be improved soon
    int const frameWidth = 1920;
    int const frameHeight = 1080;

/*
    REMINDER:   THIS INFORMATION IS NOT BEING USED. IT IS HERE AS A TEMPLATE TO BE USED IN THE FUTURE
                THIS ALLOWS READING A FRAME REPRESENTED AS .csv INTO A MATRIX AND PROCESS IT INSIDE THE PROGRAM
*/
    /*
    // Read the frame data into the matrix
    string currFileName = "data/current.csv";       // File with samples from current frame
    string refFileName = "data/reference.csv";      // File with samples from reference frame

    unsigned int currFrame[frameHeight][frameWidth], refFrame[frameHeight][frameWidth];

    ifstream currFile, refFile;
    currFile.open(currFileName);
    refFile.open(refFileName);

    if (!currFile.is_open() || !refFile.is_open()) {     // validate file open for reading 
        perror (("error while opening samples files" ));
        return 1;
    }

    string currLine, currVal, refLine, refVal;

    // Read a file of frame samples into the matrix
    
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

    const int FRAME_SIZE = frameWidth*frameHeight;
    const int BLOCK_SIZE = blockWidth*blockHeight;

    unsigned int *reference = (unsigned int*) malloc(sizeof(int) * FRAME_SIZE);   
 
    // Copy reference frame to an 1D array
   for(int i=0; i<FRAME_SIZE; i++){
        reference[i] = refFrame[i/frameWidth][i%frameWidth];
    }
    //*/

    // Open a file with information from affine ME and performs some processing over it
    string affineInfoFilename = argv[2];
    ifstream affineInfoFile;
    affineInfoFile.open(affineInfoFilename); 
    if (!affineInfoFile.is_open()) {     /* validate file open for reading */
        perror (("error while opening affine info file" ));
        return 1;
    }

    // Compute number of lines in the input file
    int rows=0;
    ifstream file;
    file.open(affineInfoFilename);
    string tmp;
    while (getline(file, tmp))
        rows++;
    file.close();

    string line, val;
    // TODO: The number of lines n_lines must be a multiple of the local_item_size, declared further. Improve the code to verify this condition and abort execution otherwise
    int n_lines = rows-1; // MV files have an empty line at the end
    int n_columns = 17; // TODO: IMPROVE THIS TO INFER NUMBER OF COLUMNS FROM FILE

    int mv_data[n_lines][n_columns];
    
    // Read a file with affine info into the matrix. The first line is not processed since it is the header
    getline(affineInfoFile, line, '\n');
    for(int l=0; l<n_lines; l++){
        getline(affineInfoFile, line, '\n');
        
        stringstream stream(line);
        
        for(int col=0; col<n_columns; col++){
            getline(stream, val, ',');
            mv_data[l][col] = stoi(val);
        }       
    }

    // These MVs are read from the file. They will be used to cross-check the computation at the end
    int *file_sub_mvs_x = (int*) malloc(sizeof(int) * n_lines);
    int *file_sub_mvs_y = (int*) malloc(sizeof(int) * n_lines);
    // These MVs are computed in the kernel
    int *sub_mvs_x = (int*) malloc(sizeof(int) * n_lines);
    int *sub_mvs_y = (int*) malloc(sizeof(int) * n_lines);
    
    // Remaining parameters read from the input affine information file
    int *LT_x          = (int*) malloc(sizeof(int) * n_lines);
    int *LT_y          = (int*) malloc(sizeof(int) * n_lines);
    int *RT_x          = (int*) malloc(sizeof(int) * n_lines);
    int *RT_y          = (int*) malloc(sizeof(int) * n_lines);
    int *LB_x          = (int*) malloc(sizeof(int) * n_lines);
    int *LB_y          = (int*) malloc(sizeof(int) * n_lines);
    int *subBlock_x    = (int*) malloc(sizeof(int) * n_lines);
    int *subBlock_y    = (int*) malloc(sizeof(int) * n_lines);
    int *pu_x          = (int*) malloc(sizeof(int) * n_lines);
    int *pu_y          = (int*) malloc(sizeof(int) * n_lines);
    int *pu_width      = (int*) malloc(sizeof(int) * n_lines);
    int *pu_height     = (int*) malloc(sizeof(int) * n_lines);
    bool *bipred       = (bool*) malloc(sizeof(bool) * n_lines);
    int *nCP           = (int*) malloc(sizeof(int) * n_lines);
    
    // Index of the parameters on the file
    // 0           1       2      3         4     5        6       7       8      9       10      11      12       13    14       15       16
    // Frame	 cu_X	 cu_Y	 bipred    nCP	 LT_X	 LT_Y	 RT_X	 RT_Y	 LB_X	 LB_Y	 width	 height	    x	 y	 MV_X	 MV_y

    // TODO: Improve the code to avoid using a matrix and then the 1D arrays -> read the data directly into the respective 1D arrays
    // Move data from the main matrix into the specific arrays
    for(int i=0; i<n_lines; i++){
        pu_x[i]   = mv_data[i][1];
        pu_y[i]   = mv_data[i][2];
        bipred[i] = mv_data[i][3];
        nCP[i]    = mv_data[i][4];
        LT_x[i]   = mv_data[i][5];
        LT_y[i]   = mv_data[i][6];
        RT_x[i]   = mv_data[i][7];
        RT_y[i]   = mv_data[i][8];
        LB_x[i]   = mv_data[i][9];
        LB_y[i]   = mv_data[i][10];
        pu_width[i]  = mv_data[i][11];
        pu_height[i] = mv_data[i][12];
        subBlock_x[i]   = mv_data[i][13];
        subBlock_y[i]   = mv_data[i][14];
        file_sub_mvs_x[i] = mv_data[i][15];
        file_sub_mvs_y[i] = mv_data[i][16];
    }

    // TODO: Correct the size of the mem_obj when porting to a real scenario
    // Create memory objects to send input data into the kernel
    cl_mem pu_x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_1);    
    cl_mem pu_y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_2);    
    cl_mem bipred_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(bool), NULL, &error_3);    
    cl_mem nCP_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_4);
    
        error = error_1 || error_2 || error_3 || error_4;

    cl_mem subMVs_x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_1);    
    cl_mem subMVs_y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_2);
    cl_mem LT_x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_3);    
    cl_mem LT_y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_4);
    
        error = error || error_1 || error_2 || error_3 || error_4;

    cl_mem RT_x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_1);       
    cl_mem RT_y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_2);
    cl_mem LB_x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_3);    
    cl_mem LB_y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_4);            
    
    error = error || error_1 || error_2 || error_3 || error_4;
    
    cl_mem subBlock_x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_1);    
    cl_mem subBlock_y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_2);                                                         
    cl_mem pu_width_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_3);    
    cl_mem pu_height_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            n_lines * sizeof(int), NULL, &error_4);   

    error = error || error_1 || error_2 || error_3 || error_4;

    probe_error(error, (char*)"Error creating memory buffers\n");


    // Copy data from the 1D arrays into the memory objects
    error  = clEnqueueWriteBuffer(command_queue, pu_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), pu_x, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, pu_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), pu_y, 0, NULL, NULL);
    error |= clEnqueueWriteBuffer(command_queue, bipred_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(bool), bipred, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, nCP_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), nCP, 0, NULL, NULL);  
    error |= clEnqueueWriteBuffer(command_queue, subMVs_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), sub_mvs_x, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, subMVs_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), sub_mvs_y, 0, NULL, NULL);   
    error |= clEnqueueWriteBuffer(command_queue, LT_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), LT_x, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, LT_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), LT_y, 0, NULL, NULL); 
    error |= clEnqueueWriteBuffer(command_queue, RT_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), RT_x, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, RT_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), RT_y, 0, NULL, NULL); 
    error |= clEnqueueWriteBuffer(command_queue, LB_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), LB_x, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, LB_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), LB_y, 0, NULL, NULL); 
    error |= clEnqueueWriteBuffer(command_queue, subBlock_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), subBlock_x, 0, NULL, NULL);       
    error |=  clEnqueueWriteBuffer(command_queue, subBlock_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), subBlock_y, 0, NULL, NULL); 
    error |= clEnqueueWriteBuffer(command_queue, pu_width_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), pu_width, 0, NULL, NULL);       
    error |= clEnqueueWriteBuffer(command_queue, pu_height_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), pu_height, 0, NULL, NULL);                   
    probe_error(error, (char*)"Error copying data from memory to buffers LEGACY\n");

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
    cl_kernel kernel = clCreateKernel(program, "affine", &error);
    probe_error(error, (char*)"Error creating kernel\n");
    
    // Set the arguments of the kernel
    error_1  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&subMVs_x_mem_obj);
    error_1 |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&subMVs_y_mem_obj);

    error_1 |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&LT_x_mem_obj);
    error_1 |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&LT_y_mem_obj);
    error_1 |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&RT_x_mem_obj);
    error_1 |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&RT_y_mem_obj);
    error_1 |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&LB_x_mem_obj);
    error_1 |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&LB_y_mem_obj);

    error_1 |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&pu_x_mem_obj);
    error_1 |= clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&pu_y_mem_obj);
    error_1 |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&pu_width_mem_obj);
    error_1 |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&pu_height_mem_obj);
    error_1 |= clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&subBlock_x_mem_obj);
    error_1 |= clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *)&subBlock_y_mem_obj);

    error_1 |= clSetKernelArg(kernel, 14, sizeof(cl_mem), (void *)&bipred_mem_obj);
    error_1 |= clSetKernelArg(kernel, 15, sizeof(cl_mem), (void *)&nCP_mem_obj);

    error_1 |= clSetKernelArg(kernel, 16, sizeof(cl_int), (void *)&frameWidth);
    error_1 |= clSetKernelArg(kernel, 17, sizeof(cl_int), (void *)&frameHeight);


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
    size_t global_item_size = n_lines; // TODO: Correct these sizes (global and local) when considering a real scenario
    size_t local_item_size = 100;
    // TODO: Correct the following line so it is possible to compare against the maximum size (i.e., try a specific local_size, but reduce to maximum_size if necessary)
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


    // Read the sub-MVs memory buffer on the device to the local variables
    error_1 = clEnqueueReadBuffer(command_queue, subMVs_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), sub_mvs_x, 0, NULL, NULL);
    error_2 = clEnqueueReadBuffer(command_queue, subMVs_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), sub_mvs_y, 0, NULL, NULL);            
    probe_error(error_1&error_2, (char*)"Error reading from buffer to memory\n");
        
    // Cross-check if the computed MVs are equal to the MVs computed by VTM-12
    for(int i=0; i<n_lines; i++){
        if((sub_mvs_x[i]!=file_sub_mvs_x[i])||(sub_mvs_y[i]!=file_sub_mvs_y[i])){
                cout << "ERROR IN FILE " << argv[2] << " LINE " << i << endl;
                cout << "    File MVs: " << file_sub_mvs_x[i] << "x" << file_sub_mvs_y[i] << endl;
                cout << "    Comp MVs: " << sub_mvs_x[i] << "x" << sub_mvs_y[i] << endl;
                cin.get();
        }       
    }

    ////////////////////////////////////////////////////
    /////         FREE SOME MEMORY SPACE           /////
    ////////////////////////////////////////////////////

    // TODO: Add all malloc'd variables and memory objects in this list        
    // Clean up
    error = clFlush(command_queue);
    error = clFinish(command_queue);
    error = clReleaseKernel(kernel);
    error = clReleaseProgram(program);
    error = clReleaseMemObject(subMVs_x_mem_obj);
    error = clReleaseMemObject(subMVs_y_mem_obj);
    error = clReleaseCommandQueue(command_queue);
    error = clReleaseContext(context);
    free(sub_mvs_x);
    free(sub_mvs_y);
    
    return 0;
}