# Christofs models in Triton

Lets quickly get a few onnx models working in NVIDIA Triton Inference Server.
Here we start with the two models built by Christof.

The model conversion from ONNX is the most straight forward. If you start with a PyTorch model, ONNX should do a good job in exporting it. 
ONNX can be straight served in Triton using the onnx-runtime backend. Here we choose the NVIDIA TensorRT backend. Lets get started, three steps, all in the NVIDIA NGC containers:

* from PyT to ONNX: torch.onnx.export(...)
* from ONNX to plan file: /usr/src/tensorrt/bin/trtexec --onnx=model_head.onnx --saveEngine=model.plan --fp16
* create a config.pbtxt and place everything in the model structure below
** each model below the model repository has its own folder
** each model folder has a config.pbtxt and a version folder (1, .... onwards) keeping the model itself
** a model folder contains for a certain backend, e.g. tensorflow 
** a different model folder can serve a model for another backend

```
model_repo/
├── both_together
│   ├── 1
│   └── config.pbtxt
├── dali
│   ├── 1
│   │   ├── dali.py
│   │   └── __pycache__
│   │       ├── dali.cpython-310.pyc
│   │       └── dali.cpython-38.pyc
│   └── config.pbtxt
├── model_head
│   ├── 1
│   │   └── model.plan
│   └── config.pbtext
└── model_headless
    ├── 1
    │   └── model.plan
    └── config.pbtxt
```


# Model Definitions for Triton
This is in deeplab_v3_models

