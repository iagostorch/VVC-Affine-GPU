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

    // List the ID and name for each CPU and GPU device
    for(int cpu=0; cpu<assigned_cpus; cpu++){
        error = clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");
        
        cout << "CPU " << cpu << endl;
        cout << "\tid " << cpu_device_ids[cpu] << endl << "\t" <<  device_name << endl;
    }
    for(int gpu=0; gpu<assigned_gpus; gpu++){
        error = clGetDeviceInfo(gpu_device_ids[gpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");
        
        cout << "GPU " << gpu << endl;
        cout << "\tid " << gpu_device_ids[gpu] << endl << "\t" <<  device_name << endl;
    }

    
    // Create "target" device and assign proper IDs
    cl_device_id device_id = NULL; 
    cl_uint ret_num_devices;
    
    // Select what CPU or GPU will be used based on parameters
    if(argc==4){
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
    }
    else{
        cout << "Failed to specify the input parameters. Proper execution has the form of" << endl;
        cout << "./main <CPU or GPU> <# of CPU or GPU device> <input_file>" << endl;
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

    // TODO: This file is not usen when searching for the MVs. It is a debugging feature from now on
    // Open a file with information from affine ME and performs some processing over it
    // The information form this file will be used to perform prediction after the reference frame
    string affineInfoFilename = argv[3];
    ifstream affineInfoFile;
    affineInfoFile.open(affineInfoFilename); 
    if (!affineInfoFile.is_open()) {     // validate file open for reading 
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
    int n_lines = rows-1; // MV files have an empty line at the end
    int n_columns = 19; // TODO: IMPROVE THIS TO INFER NUMBER OF COLUMNS FROM FILE
    
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

    /*
    //These variables are not used anymore. They are used on the "affine" kernel, that uses an input file to conduct MC
    // TODO: MVs read from file should NOT be used when performing motion estimation. They are here for debugging the sub-mv derivation functions
    // These MVs are read from the file. They will be used to cross-check the computation at the end
    int *file_sub_mvs_x = (int*) malloc(sizeof(int) * n_lines);
    int *file_sub_mvs_y = (int*) malloc(sizeof(int) * n_lines);
    // These MVs are computed in the kernel and returned to main program (they should be the same as file_sub_mvs_x and y during debugging)
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
    */

    // Index of the parameters on the file
    // 0         1            2      3   4     5       6           7     8    9      10      11      12      13      14      15      16      17      18
    // POC	frameWidth	frameHeight	puX	puY	puWidth	puHeight	biPred	nCP	LT_X	LT_Y	RT_X	RT_Y	LB_X	LB_Y	subX	subY	mv_X	mv_Y

    const int frameWidth  = mv_data[0][1];
    const int frameHeight = mv_data[0][2];
    const int blockWidth  = mv_data[0][5];
    const int blockHeight = mv_data[0][6];

    /* // These assignments based on the input files are not used anymore
    // TODO: Improve the code to avoid using a matrix and then the 1D arrays -> read the data directly into the respective 1D arrays
    // Move data from the main matrix into the specific arrays
    for(int i=0; i<n_lines; i++){
        pu_x[i]   = mv_data[i][3];
        pu_y[i]   = mv_data[i][4];
        pu_width[i]  = mv_data[i][5];
        pu_height[i] = mv_data[i][6];
        bipred[i] = mv_data[i][7];
        nCP[i]    = mv_data[i][8];
        LT_x[i]   = mv_data[i][9];
        LT_y[i]   = mv_data[i][10];
        RT_x[i]   = mv_data[i][11];
        RT_y[i]   = mv_data[i][12];
        LB_x[i]   = mv_data[i][13];
        LB_y[i]   = mv_data[i][14];
        
        subBlock_x[i]   = mv_data[i][15];
        subBlock_y[i]   = mv_data[i][16];
        file_sub_mvs_x[i] = mv_data[i][17];
        file_sub_mvs_y[i] = mv_data[i][18];
    }
    */

    // Read the frame data into the matrix
    string currFileName = "data/original_1.csv";                // File with samples from current frame
    string refFileName = "data/reconstructed_0.csv";      // File with samples from reference frame

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
    const int BLOCK_SIZE = blockWidth*blockHeight;

    unsigned int *reference_frame = (unsigned int*) malloc(sizeof(int) * FRAME_SIZE);
    unsigned int *current_frame   = (unsigned int*) malloc(sizeof(int) * FRAME_SIZE);

    // Read the samples from reference frame into the reference array
    for(int h=0; h<frameHeight; h++){
        getline(currFile, currLine, '\n');
        getline(refFile, refLine, '\n');
        stringstream currStream(currLine), refStream(refLine); 
        
        for(int w=0; w<frameWidth; w++){
            getline(currStream, currVal, ',');
            getline(refStream, refVal, ',');
            // cout << origVal << ",";// << endl;;
            current_frame[h*frameWidth + w] = stoi(currVal);
            reference_frame[h*frameWidth + w] = stoi(refVal);
        }
    }
    
    
    /* These memory objects are based on the input file and they are not used anymore
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
    */

    // These buffers are for storing the reference samples and current samples and predicted/filtered samples
    cl_mem ref_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            FRAME_SIZE * sizeof(int), NULL, &error_1);    
    cl_mem curr_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            FRAME_SIZE * sizeof(int), NULL, &error_2);                
//     cl_mem filtered_samples_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
//             BLOCK_SIZE * sizeof(int), NULL, &error_3);   
    error = error_1 || error_2;

    probe_error(error, (char*)"Error creating memory buffers\n");

    /* These memory objects are based on the input file and they are not used anymore
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
    */

    error  = clEnqueueWriteBuffer(command_queue, ref_samples_mem_obj, CL_TRUE, 0, 
            FRAME_SIZE * sizeof(int), reference_frame, 0, NULL, NULL); 
    error |= clEnqueueWriteBuffer(command_queue, curr_samples_mem_obj, CL_TRUE, 0, 
            FRAME_SIZE * sizeof(int), current_frame, 0, NULL, NULL);                  
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
    // -cl-nv-verbose is used to show memory and registers used by the kernel
    // -cl-nv-maxrregcount=241 is used to set the maximum number of registers per kernel. Using a large value and modifying in each compilation makes no difference in the code, but makes the -cl-nv-verbose flag work properly
    srand (time(NULL));
    int maxReg = 160+rand()%50;
    char argBuild[39];
    char *pt1 = "-cl-nv-maxrregcount=";
    char *pt2 = "-cl-nv-verbose";
    snprintf(argBuild, sizeof(argBuild), "%s%d%s", pt1, maxReg, pt2);
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
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "naive_affine_2CPs_CU_128x128", &error);
    probe_error(error, (char*)"Error creating kernel\n");
    
    // TODO: Declare these on the correct place
    int itemsPerWG = 256;
    int nWG = 16;
    // These memory objects hold the best SATD and respective CPMVs for each WG
    cl_mem wgs_SATDs_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * sizeof(int), NULL, &error);   
    cl_mem wgs_LT_X_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * sizeof(cl_short), NULL, &error_1);   
    cl_mem wgs_LT_Y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * sizeof(cl_short), NULL, &error_2);   
    cl_mem wgs_RT_X_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * sizeof(cl_short), NULL, &error_3);   
    cl_mem wgs_RT_Y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG * sizeof(cl_short), NULL, &error_4);   
    
    error = error | error_1 | error_2 | error_3 | error_4;
    
    // These memory objects are used to retrieve debugging information from the kernel
    cl_mem debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG*itemsPerWG * sizeof(int), NULL, &error_1);  
    cl_mem cu_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            128*128 * sizeof(int), NULL, &error_2);  

    error = error | error_1 | error_2;
    
    probe_error(error_1,(char*)"Error creating memory object for SATDs and CPMVs of each WG\n");

    // Set the arguments of the kernel
    error_1  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&ref_samples_mem_obj);
    error_1 |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&curr_samples_mem_obj);
    error_1 |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&frameWidth);
    error_1 |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&frameHeight);
    error_1 |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&wgs_SATDs_mem_obj);
    error_1 |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&debug_mem_obj);
    error_1 |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&cu_mem_obj);
    error_1 |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&wgs_LT_X_mem_obj);
    error_1 |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&wgs_LT_Y_mem_obj);
    error_1 |= clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&wgs_RT_X_mem_obj);
    error_1 |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&wgs_RT_Y_mem_obj);

    /* Arguments for the "affine" kernel, which uses the input file to perform motion compensation of a single CU
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

    error_1 |= clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *)&ref_samples_mem_obj);
    error_1 |= clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *)&curr_samples_mem_obj);
    error_1 |= clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *)&filtered_samples_mem_obj);
    */

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

    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end-time_start;
    printf("OpenCl Execution time is: %0.3f miliseconds \n",nanoSeconds / 1000000.0);

    /* // Read the sub-MVs memory buffer on the device to the local variables. Used in the "affine" kernel, which uses the input file to conduct MC
    error_1 = clEnqueueReadBuffer(command_queue, subMVs_x_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), sub_mvs_x, 0, NULL, NULL);
    error_2 = clEnqueueReadBuffer(command_queue, subMVs_y_mem_obj, CL_TRUE, 0, 
            n_lines * sizeof(int), sub_mvs_y, 0, NULL, NULL);   
    */

    // Useful information returnbed by kernel: bestSATD and bestCMP of each WG
    int *return_satds = (int*) malloc(sizeof(int) * nWG);
    short *return_LT_X = (short*) malloc(sizeof(int) * nWG);
    short *return_LT_Y = (short*) malloc(sizeof(int) * nWG);
    short *return_RT_X = (short*) malloc(sizeof(int) * nWG);
    short *return_RT_Y = (short*) malloc(sizeof(int) * nWG);
    // Debug information returned by kernel
    int *debug_data = (int*) malloc(sizeof(int) * nWG*itemsPerWG);
    int *return_cu = (int*) malloc(sizeof(int) * 128*128);


    error  = clEnqueueReadBuffer(command_queue, wgs_SATDs_mem_obj, CL_TRUE, 0, 
            nWG * sizeof(int), return_satds, 0, NULL, NULL);
    error |= clEnqueueReadBuffer(command_queue, wgs_LT_X_mem_obj, CL_TRUE, 0, 
            nWG * sizeof(cl_short), return_LT_X, 0, NULL, NULL);
    error |= clEnqueueReadBuffer(command_queue, wgs_LT_Y_mem_obj, CL_TRUE, 0, 
            nWG * sizeof(cl_short), return_LT_Y, 0, NULL, NULL);
    error |= clEnqueueReadBuffer(command_queue, wgs_RT_X_mem_obj, CL_TRUE, 0, 
            nWG * sizeof(cl_short), return_RT_X, 0, NULL, NULL);
    error |= clEnqueueReadBuffer(command_queue, wgs_RT_Y_mem_obj, CL_TRUE, 0, 
            nWG * sizeof(cl_short), return_RT_Y, 0, NULL, NULL);

    error |= clEnqueueReadBuffer(command_queue, debug_mem_obj, CL_TRUE, 0, 
            nWG*itemsPerWG * sizeof(int), debug_data, 0, NULL, NULL);   
    error |= clEnqueueReadBuffer(command_queue, cu_mem_obj, CL_TRUE, 0, 
            128*128 * sizeof(int), return_cu, 0, NULL, NULL);   

    probe_error(error, (char*)"Error reading returned memory objects into malloc'd arrays\n");

    // Search the returned values for the best SATD and corresponding CPMVs
    int bestSATD = (1<<30);
    short best_LT_X, best_LT_Y, best_RT_X, best_RT_Y;
    for(int i=0; i<nWG; i++){
        if(return_satds[i]<bestSATD){
            bestSATD = return_satds[i];
            best_LT_X = return_LT_X[i];
            best_LT_Y = return_LT_Y[i];
            best_RT_X = return_RT_X[i];
            best_RT_Y = return_RT_Y[i];
        }
        printf("WG %d -> SATD=%d\n", i, return_satds[i]);
    }

    printf("@@@ BEST CPMVs:\n");
    printf("___ SATD  = %d\n",bestSATD);
    printf("___ LT XY = %dx%d\n",best_LT_X,best_LT_Y);
    printf("___ RT XY = %dx%d\n",best_RT_X,best_RT_Y);

    // printf("\n Debug array\n");
    // for(int i=0; i<nWG*itemsPerWG; i++){
    //    // printf("gid %4.d -> %d \n",i,debug_data[i]);
    //    // printf("%d\n", debug_data[i]);
    //    // printf("\n");
    // }
    // printf("\n Current CU\n");
    // for(int i=0; i<128; i++){
    //     for(int j=0; j<128; j++){
    //         printf("%d,",return_cu[i*128+j]);
    //     }
    //     printf("\n");
    // }

    /*
    // Cross-check if the computed MVs are equal to the MVs computed by VTM-12
    // Used on "affine" kernel, that uses an input file to conduct MC
    for(int i=0; i<n_lines; i++){
        if((sub_mvs_x[i]!=file_sub_mvs_x[i])||(sub_mvs_y[i]!=file_sub_mvs_y[i])){
                cout << "ERROR IN FILE " << argv[3] << " LINE " << i << endl;
                cout << "    File MVs: " << file_sub_mvs_x[i] << "x" << file_sub_mvs_y[i] << endl;
                cout << "    Comp MVs: " << sub_mvs_x[i] << "x" << sub_mvs_y[i] << endl;
                cin.get();
        }       
    }
    */

    ////////////////////////////////////////////////////
    /////         FREE SOME MEMORY SPACE           /////
    ////////////////////////////////////////////////////

    // Clean up
    error = clFlush(command_queue);
    error |= clFinish(command_queue);
    error |= clReleaseKernel(kernel);
    error |= clReleaseProgram(program);
    /* // Memory objects of "affine" kernel
    error |= clReleaseMemObject(pu_x_mem_obj);
    error |= clReleaseMemObject(pu_y_mem_obj);
    error |= clReleaseMemObject(bipred_mem_obj);
    error |= clReleaseMemObject(nCP_mem_obj);
    error |= clReleaseMemObject(subMVs_x_mem_obj);
    error |= clReleaseMemObject(subMVs_y_mem_obj);
    error |= clReleaseMemObject(LT_x_mem_obj);
    error |= clReleaseMemObject(LT_y_mem_obj);
    error |= clReleaseMemObject(RT_x_mem_obj);
    error |= clReleaseMemObject(RT_y_mem_obj);
    error |= clReleaseMemObject(LB_x_mem_obj);
    error |= clReleaseMemObject(LB_y_mem_obj);
    error |= clReleaseMemObject(subBlock_x_mem_obj);
    error |= clReleaseMemObject(subBlock_y_mem_obj);
    error |= clReleaseMemObject(pu_width_mem_obj);
    error |= clReleaseMemObject(pu_height_mem_obj);
    */
    error |= clReleaseMemObject(ref_samples_mem_obj);
    error |= clReleaseMemObject(curr_samples_mem_obj);
    error |= clReleaseCommandQueue(command_queue);
    error |= clReleaseContext(context);
    probe_error(error, (char*)"Error releasing  OpenCL objects\n");
    
    free(source_str);
    free(platform_id);
    free(reference_frame);
    free(current_frame);
    /* //Arrays used on the "affine" kernel
    free(file_sub_mvs_x);
    free(file_sub_mvs_y);
    free(sub_mvs_x);
    free(sub_mvs_y);
    free(LT_x);
    free(LT_y);
    free(RT_x);
    free(RT_y);
    free(LB_x);
    free(LB_y);
    free(subBlock_x);
    free(subBlock_y);
    free(pu_x);
    free(pu_y);
    free(pu_width);
    free(pu_height);
    free(bipred);
    free(nCP);
    */
 
    return 0;
}