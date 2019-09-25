#pragma once

#include "tensorflow/core/framework/op_kernel.h"

using namespace std;

const string preStr = "==C++== ";

void print(string output);
void print(float output);
void print(int output);
void print(const Eigen::VectorXf& v);


