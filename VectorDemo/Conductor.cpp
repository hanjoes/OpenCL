//
//  Conductor.cpp
//  VectorDemo
//
//  Created by Hanzhou Shi on 5/26/15.
//
//

#include <iostream>

#include <OpenCL/opencl.h>
#include <fstream>
#include <mach/mach_time.h>
#include <cmath>
#include <sstream>

#include "Conductor.h"

using namespace std;
using namespace GPUTest::VectorDemo;

Conductor::~Conductor() {

}

void Conductor::getPlatformInfo() {
    cl_platform_id *platformId = new cl_platform_id[MAX_PLATFORM_NUM];
    cl_uint numPlatforms;
    cl_uint ret = (cl_uint) clGetPlatformIDs(MAX_PLATFORM_NUM, platformId, &numPlatforms);
    if (ret != CL_SUCCESS)
        cout << "Get platform info failed: " << ret << endl;

    for (auto i = 0; i != numPlatforms; ++i) {
        getDetailedPlatformInfo(platformId[i]);
    }
    // free resource
    delete[] platformId;
}

void Conductor::getDetailedPlatformInfo(cl_platform_id id) {
    size_t retSize;
    MIS::const_iterator it = platformConstMap.begin();
    for (; it != platformConstMap.end(); ++it) {
        memset(buf, 0, MAX_BUF_LENGTH);
        clGetPlatformInfo(id, (cl_platform_info) it->first, MAX_BUF_LENGTH, buf, &retSize);
        cout << it->second << ": " << buf << endl;
    }
    getDevicesInfo(id);
}

void Conductor::dumbCalculation(const VI &op1, const VI &op2) {
    cout << "----------------------" << endl;
    cout << "Executing on single CPU core..." << endl;
    int n = nloops;
    uint64_t start = mach_absolute_time();
    while (n--) {
        for (auto i = 0; i != op1.size(); ++i) {
            result.push_back(op1[i] + op2[i]);
        }
    }
    uint64_t end = mach_absolute_time();
    cout << "Serial calculation on single core: " << ((end - start) / pow(10, 6)) << "ms" << endl;
}

void Conductor::getDevicesInfo(cl_platform_id pId) {
    cl_device_id *deviceId = new cl_device_id[MAX_DEVICES_NUM];
    cl_uint numDevices;
    clGetDeviceIDs(pId, CL_DEVICE_TYPE_ALL, MAX_DEVICES_NUM, deviceId, &numDevices);
    cout << "Number of Devices: " << numDevices << endl;
    for (auto i = 0; i != numDevices; ++i) {
        getDetailedDeviceInfo(deviceId[i]);
    }
    // free resources
    delete[] deviceId;
}

void Conductor::getDetailedDeviceInfo(cl_device_id dId) {
    Device device;
    device.id = dId;

    size_t retSize;
    // only get those important information
    memset(buf, 0, sizeof(buf));
    clGetDeviceInfo(dId, CL_DEVICE_NAME, MAX_BUF_LENGTH, buf, &retSize);
    device.deviceName = buf;

    cl_device_type type;
    clGetDeviceInfo(dId, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, &retSize);
    device.type = type;

    cl_uint num;
    clGetDeviceInfo(dId, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &num, &retSize);
    device.vendorId = num;

    clGetDeviceInfo(dId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, &retSize);
    device.numComputeUnits = num;

    clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, &retSize);
    device.maxWorkItemDimensions = num;

    size_t size;
    clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, &retSize);
    device.maxWorkGroupSize = size;

    clGetDeviceInfo(dId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &num, &retSize);
    device.maxClockFrequency = num;

    insertDevice(device);
}

void Conductor::translateDeviceType(cl_device_type type) {
    cout << "CL_DEVICE_TYPE: ";
    switch (type) {
        case CL_DEVICE_TYPE_CPU:
            cout << "CL_DEVICE_TYPE_CPU" << endl;
            break;
        case CL_DEVICE_TYPE_GPU:
            cout << "CL_DEVICE_TYPE_GPU" << endl;
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            cout << "CL_DEVICE_TYPE_ACCELORATOR" << endl;
            break;
        case CL_DEVICE_TYPE_CUSTOM:
            cout << "CL_DEVICE_TYPE_CUSTOM" << endl;
            break;
        default:
            cout << "CL_DEVICE_TYPE_DEFAULT" << endl;
            break;
    }
}

void Conductor::insertDevice(Device device) {

    MTVD::iterator it = this->deviceMap.find(device.type);
    if (it != this->deviceMap.end()) {
        it->second.push_back(device);
    }
    else {
        VD vd;
        vd.push_back(device);
        this->deviceMap[device.type] = vd;
    }
}

void Conductor::parallelCompute(VI &op1, VI &op2, cl_device_type type) {
    cl_int err;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem op1Buf;
    cl_mem op2Buf;
    cl_mem retBuf;
    MTVD::const_iterator it = deviceMap.find(type);
    for (auto device : it->second) {
        cout << "----------------------" << endl;
        printDevice(device);
        // currently we only have one CPU device.
        context = clCreateContext(NULL, 1, &(device.id), NULL, NULL, NULL);

        // prepare the kernel
        // get the program instance
        string programSource;
        getProgramSource(programSource);
        const char *source = programSource.c_str();
        program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
        // build the program
        clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        // create kernel
        kernel = clCreateKernel(program, "vadd", &err);

        // initialize the required memory objects.
        // TODO: could be expensive
        size_t numItems = op1.size();
        op1Buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numItems * sizeof(int), &op1[0],
                                &err);
        op2Buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numItems * sizeof(int), &op2[0],
                                &err);
        // memory mapping.
        retBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, numItems * sizeof(int), NULL, &err);

        // set arguments
        cl_uint dev = 1;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &op1Buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &op2Buf);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &retBuf);
        clSetKernelArg(kernel, 3, sizeof(dev), &dev);// pass device type to kernel
        clSetKernelArg(kernel, 4, sizeof(size_t), &numItems);// pass in total number of work

        // launch kernel
        // let the device figure out the local_work_size,
        // and this command is not waiting for any other commands' finishing.
        queue = clCreateCommandQueue(context, device.id, (cl_command_queue_properties)NULL, &err);

        size_t globalWorkSize;
        size_t localWorkSize;
        if (type == CL_DEVICE_TYPE_CPU) {
            // in case of CPU, the global work size is just the number of cores.
            // and the local work size is 1 since each core can only execute one
            // thread at a time.
            globalWorkSize = device.numComputeUnits;
            localWorkSize = 1;
        }
        else if (type == CL_DEVICE_TYPE_GPU) {
            // send kernel-enqueue command
            // the global work size need to be factor of the total amount of work
            // the divisor here is chosen randomly, just ensure global work size is
            // multiple of local work size
            globalWorkSize = (numItems / 4) / 64;
            // the local work size here is picked randomly, just smaller than 512
            localWorkSize = 256;
        }
        else {
            // currently only support two kinds of devices...
        }
        uint64_t start = mach_absolute_time();
        int n = nloops;
        while (n--) {
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                   &globalWorkSize,
                                   &localWorkSize,
                                   0, NULL, NULL);
        }

        // wait for the kernel to finish
        clFinish(queue);
        uint64_t end = mach_absolute_time();
        // map device data back to host memory.
        int *dataOut = (int *) clEnqueueMapBuffer(queue, retBuf,
                                                  CL_TRUE, CL_MAP_READ, 0,
                                                  numItems * sizeof(int), 0, NULL, NULL, &err);
        checkResult(dataOut, numItems);

        // release resource
        clReleaseMemObject(op1Buf);
        clReleaseMemObject(op2Buf);
        clReleaseMemObject(retBuf);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        cout << "Parallel calculation on this device took "
             << ((end - start) / pow(10, 6)) << "ms" << endl;

    }
}

void Conductor::getProgramSource(string &programSource) {
    ifstream in("/Users/hanjoes/Dropbox/Master Project/GPU/OpenCL/VectorDemo/VectorArithmetic.cl", ios::in | ios::binary);
    in.seekg(0, ios::end);
    programSource.resize((unsigned long) in.tellg());
    in.seekg(0, ios::beg);
    in.read(&programSource[0], programSource.size());
}

void Conductor::checkResult(int *dataOut, size_t numItems) {
    for (auto i = 0; i < numItems; ++i) {
        if (dataOut[i] != result[i]) {
            cout << "Result incorrect." << endl;
            return;
        }
    }
    cout << "Result correct." << endl;
}

void Conductor::printDevice(const Device &device) {
    cout << "CL_DEVICE_NAME: " << device.deviceName << endl;
    translateDeviceType(device.type);
    cout << "CL_DEVICE_VENDOR_ID: " << device.vendorId << endl;
    cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << device.numComputeUnits << endl;
    cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << device.maxWorkItemDimensions << endl;
    cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.maxWorkGroupSize << endl;
    cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << device.maxClockFrequency << " MHz" << endl;
}

Conductor::Conductor(int nloops) : nloops(nloops) {
    platformConstMap[CL_PLATFORM_PROFILE] = "CL_PLATFORM_PROFILE";
    platformConstMap[CL_PLATFORM_VERSION] = "CL_PLATFORM_VERSION";
    platformConstMap[CL_PLATFORM_NAME] = "CL_PLATFORM_NAME";
    platformConstMap[CL_PLATFORM_VENDOR] = "CL_PLATFORM_VENDOR";
    platformConstMap[CL_PLATFORM_EXTENSIONS] = "CL_PLATFORM_EXTENSIONS";
}
