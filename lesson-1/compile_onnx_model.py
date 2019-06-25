# -*- coding: utf-8 -*-

# File name: compile_onnx_model.py
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


def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except BaseException:
        import urllib
        urllib.urlretrieve(url, path)

if '__main__' == __name__:
    # download onnx model
    model_url = ''.join(['https://gist.github.com/zhreshold/',
                        'bcda4716699ac97ea44f791c24310193/raw/',
                        '93672b029103648953c4e5ad3ac3aadf346a4cdc/',
                        'super_resolution_0.2.onnx'])
    download(model_url, 'model/super_resolution.onnx')
    onnx_model = onnx.load_model('model/super_resolution.onnx')

    # download input image
    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    download(img_url, 'data/cat.png')
    img = Image.open('data/cat.png').resize((224, 224))
    img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
    img_y, img_cb, img_cr = img_ycbcr.split()
    x = np.array(img_y)[np.newaxis, np.newaxis, :, :]
    # saving demo image
    x.astype("float32").tofile("data/cat.bin")

    # load onnx model
    sym, params = nnvm.frontend.from_onnx(onnx_model)
    # optimize with GPU
    target = 'cuda'

    # assume first input name is data
    input_name = sym.list_input_names()[0]
    print("All input : %s " % sym.list_input_names())
    print("Input name: %s " % input_name)
    print('Input shape: %s' % str(x.shape))
    shape_dict = {input_name: x.shape}
    graph_json_str = None
    nnvm.compiler.build_config(opt_level=3)
    graph, lib, params = nnvm.compiler.build(
        sym, target, shape_dict, params=params)

    # saving model
    lib.export_library("model/super_resolution.so")
    with open('model/super_resolution.graph', 'w') as _f:
        _f.write(graph.json())
    with open('model/super_resolution.params', 'wb') as _f:
        _f.write(nnvm.compiler.save_param_dict(params))
