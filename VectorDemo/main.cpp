//
//  main.cpp
//  VectorDemo
//
//  Created by Hanzhou Shi on 5/26/15.
//
//

#include <iostream>

#include "Conductor.h"

using namespace GPUTest::VectorDemo;

int main(int argc, const char * argv[]) {

    // hard-coded initialization of two input array.
    VI op1;
    VI op2;
    for (auto i = 0; i != (1 << 23); ++i) {
        op1.push_back(i);
        op2.push_back(i);
    }
    
    // create the Conductor
    int nloops = 100;
    Conductor c(nloops);
    // query platform information
    c.getPlatformInfo();

    // single CPU core vector add
    c.dumbCalculation(op1, op2);
    // CPU parallel calculation
    c.parallelCompute(op1, op2, CL_DEVICE_TYPE_CPU);
    // GPU parallel calculation
    c.parallelCompute(op1, op2, CL_DEVICE_TYPE_GPU);

    return 0;
}
