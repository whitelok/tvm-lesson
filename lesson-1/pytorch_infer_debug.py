# -*- coding: utf-8 -*-

# File name: pytorch_infer_debug.py
# Author: Karl Luo
# Date created: 03/08/2019
# Date last modified: 03/08/2019

import onnx
import time
from PIL import Image
import caffe2.python.onnx.backend as backend
import numpy as np

if '__main__' == __name__:
    # Load the ONNX model
    model = onnx.load("model/super_resolution.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

    img = Image.open('data/cat.png').resize((224, 224))
    img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
    img_y, img_cb, img_cr = img_ycbcr.split()
    x = np.array(img_y)[np.newaxis, np.newaxis, :, :].astype(np.float32)

    rep = backend.prepare(model, device="CUDA:0") # or "CPU"

    running_times = 10000
    print("Started infer %s times..." % str(running_times))
    start_time = time.time()
    for _ in range(running_times):
        outputs = rep.run(x)
    end_time = time.time()
    print("Average infer time: %s " % str((end_time - start_time) / running_times))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0].astype(np.float32))
