# -*- coding: utf-8 -*-

# File name: tvm_python_infer_debug.py
# Author: Karl Luo
# Date created: 03/08/2019
# Date last modified: 03/08/2019

import os
import nnvm
import tvm
import onnx
import json
import time
import numpy as np
import codecs
from PIL import Image
from tvm.contrib import graph_runtime
import nnvm.compiler


if '__main__' == __name__:
    # loading model
    lib = tvm.module.load("model/super_resolution.so")
    with open('model/super_resolution.graph', 'r') as _f:
        graph = nnvm.graph.load_json(_f.read())
    with open('model/super_resolution.params', 'rb') as _f:
        params = nnvm.compiler.load_param_dict(_f.read())

    # loading image
    img = Image.open('data/cat.png').resize((224, 224))
    img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
    img_y, img_cb, img_cr = img_ycbcr.split()
    x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

    # setting input information
    input_name = '1'
    shape_dict = {input_name: x.shape}

    # infer with gpu
    ctx = tvm.gpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    # execute
    running_times = 10000
    print("Started infer %s times..." % str(running_times))
    start_time = time.time()
    for _ in range(running_times):
        m.run()
    end_time = time.time()
    print("Average Infer time: %s" % str((end_time - start_time) / running_times))
    # get outputs
    output_shape = (1, 1, 672, 672)
    tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()

    print("Output:")
    print(tvm_output)
