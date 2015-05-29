//
//  Conductor.h
//  VectorDemo
//
//  Created by Hanzhou Shi on 5/26/15.
//
//

#ifndef __VectorArithmetic__Conductor__
#define __VectorArithmetic__Conductor__

#include <Opencl/opencl.h>

#include <vector>
#include <string>
#include <map>


namespace GPUTest {
    namespace VectorDemo {

        typedef struct {
            cl_device_id id;
            cl_device_type type;
            size_t numComputeUnits;
            size_t maxWorkGroupSize;
            size_t maxClockFrequency;
            cl_uint vendorId;
            cl_uint maxWorkItemDimensions;
            std::string deviceName;

        } Device;
        typedef std::vector<Device> VD;
        typedef std::map<cl_device_type, VD> MTVD;

        typedef std::vector<int> VI;
        typedef std::map<int, std::string> MIS;

        // currently we could only have one platform
        const static cl_uint MAX_PLATFORM_NUM = 1;
        // maximum number of devices in a platform
        const static cl_uint MAX_DEVICES_NUM = 8;
        // max buffer length constant, 1KB
        const static int MAX_BUF_LENGTH = 1024;

        class Conductor {

        private:
            // this field stores parameter constants
            MIS platformConstMap;
            // general purpose buffer
            char buf[MAX_BUF_LENGTH];
            // device map
            MTVD deviceMap;
            // temporary result
            VI result;
            // number of loops
            int nloops;

        public:

            Conductor(int nloops);

            ~Conductor();

            void getPlatformInfo();

            void dumbCalculation(const VI &op1, const VI &op2);

            void parallelCompute(VI &op1, VI &op2, cl_device_type type);


        private:

            void getDetailedPlatformInfo(cl_platform_id id);

            void getDevicesInfo(cl_platform_id pId);

            void getDetailedDeviceInfo(cl_device_id dId);

            void translateDeviceType(cl_device_type type);

            void insertDevice(Device dId);

            void getProgramSource(std::string &programSource);

            void checkResult(int *dataOut, size_t numItems);

            void printDevice(const Device &device);
        };
    }
}

#endif /* defined(__VectorArithmetic__Conductor__) */
