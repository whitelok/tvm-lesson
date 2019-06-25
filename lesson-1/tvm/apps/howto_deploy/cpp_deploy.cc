/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sys/time.h>

static uint64_t getCurrentTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(void) {
  tvm::runtime::Module mod_dylib =
      tvm::runtime::Module::LoadFromFile("/home/karll/TVMDebug/super_resolution.so");

  std::ifstream json_in("/home/karll/TVMDebug/super_resolution.graph", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();
  
  std::ifstream params_in("/home/karll/TVMDebug/super_resolution.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLGPU;
  int device_cpu = kDLCPU;
  int device_gpu = kDLGPU;
  int device_cpu_id = 0;
  int device_gpu_id = 0;
  int device_id = 0;

  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_dylib, device_type, device_id);

  DLTensor* x;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, 1, 224, 224};
  int nbytes_float32 = 4;
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_cpu, device_cpu_id, &x);
  // load image data saved in binary
  std::ifstream data_fin("/home/karll/TVMDebug/cat.bin", std::ios::binary);
  data_fin.read(static_cast<char*>(x->data), 1 * 224 * 224 * nbytes_float32);
  
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("1", x);
  
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  uint64_t start_time = getCurrentTime();
  for(int i = 0; i < 1000; i++) {
      run();
  } 
  uint64_t finish_time = getCurrentTime();
  std::cout << "done(" << (finish_time - start_time) / 1000 / 1000<< " ms)." << std::endl;

  DLTensor* y;
  int out_ndim = 4;
  int64_t out_shape[4] = {1, 1, 672, 672};
  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_cpu, device_cpu_id, &y);

  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  get_output(0, y);

  auto y_iter = static_cast<float*>(y->data);
  for (int i = 0; i < 100; i++){
    std::cout << y_iter[i] << ", ";
  }

  TVMArrayFree(x);
  TVMArrayFree(y);
  
  return 0;
}
