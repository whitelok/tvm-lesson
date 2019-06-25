TVM_ROOT=../tvm
NNVM_PATH=nnvm
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -lpthread

.PHONY: clean all

all: lib/cpp_deploy_normal

# Deploy using pre-built libtvm_runtime.so
lib/cpp_deploy_normal: cpp_deploy.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -ltvm_runtime
