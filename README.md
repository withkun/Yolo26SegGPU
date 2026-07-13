# Yolo26SegGPU
YOLO26 instance segmentation postprocess with GPU


# OpenCV可下载最新版本:
https://github.com/opencv/opencv

https://github.com/opencv/opencv/releases/download/5.0.0/opencv-5.0.0-windows.exe


# CUDA可下载最新版本:
https://developer.nvidia.com/cuda-downloads

https://developer.download.nvidia.com/compute/cuda/13.3.1/local_installers/cuda_13.3.1_windows.exe


# CUDNN可下载最新版本:
https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/

https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.24.0.43_cuda13-archive.zip


# TensorRT根据本机CUDA下载对应版本:
https://developer.nvidia.com/tensorrt/download/10x

https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.1.0/zip/TensorRT-Enterprise-11.1.0.106-Windows-amd64-cuda-13.3-Release-external.zip


# 设置环境与执行:
SET PATH=d:\3rd_party\opencv_5.0.0\x64\vc16\bin;d:\3rd_party\cuda13.3_lib\bin;d:\3rd_party\TensorRT-11.1.0.106\bin;

Yolo26SegGPU.exe  -model_yolo=d:/WORK/YOLO26/runs/segment/train/weights/best_dyn.onnx  -input_dims=image:1,1,960,1280;1,1,1216,1920;1,1,1216,1920  -input_file=d:/WORK/测试图片/*.bmp;  -output_dir=runs/results > run.log 2>&1

Yolo26SegGPU.exe "C:/WORK/Yolo26SegGPU/yolo26s-seg.onnx" "C:/WORK/Yolo26SegGPU/images/rg28_uv_001_001.png"
Yolo26SegGPU.exe "C:/WORK/Yolo26SegGPU/yolo26s-seg.engine" "C:/WORK/Yolo26SegGPU/images/rg28_uv_001_001.png"

# 参数说明:
"C:/WORK/Yolo26SegGPU/yolo26s-seg.engine": 分割模型, 可使用onnx模型或序列化后的engine模型(初始化速度快)
"C:/WORK/Yolo26SegGPU/images/rg28_uv_001_001.png": 测试图像, 可使用图片目录或视频文件


# YOLO26系列结构差异:
### YOLO26n (nano): 主干网络仅3个CSP块, 检测头单尺度输出, 参数量<3MB, 适合Jetson Orin Nano等低算力设备.
### YOLO26s (small): 增加Neck模块与轻量注意力机制, 支持双尺度特征融合, 参数量约7MB.
### YOLO26m (medium): 完整PANet结构+动态标签分配, 参数量约18MB, 是精度与速度的黄金平衡点.
### YOLO26l (large): 扩展感受野至128×128, 新增跨层特征重校准模块, 参数量约35MB.
### YOLO26x (extra-large): 4尺度输出+自适应锚点生成, 参数量约58MB, 专为小目标密集场景优化.


# 版本选择策略
## 1.硬件约束优先
低算力设备(如树莓派, Jetson Nano): 选择n版本, 例如YOLOv8n在Tesla T4上推理延迟仅1.8ms, YOLO26n在Jetson Orin Nano上可达42FPS.
中等算力设备(如移动端GPU, 消费级摄像头): s版本是首选, 如YOLOv8s在移动端AI应用中广泛使用.
高算力设备(如A100, V100): 可考虑l或x版本, 但需注意显存占用(如YOLO26x在1280输入下显存占用达11.7GB).
## 2.任务需求导向
实时性要求高(如自动驾驶, 视频监控): 优先选择m版本, 例如YOLOv8m在产线相机(1920×1080@30fps)上实现99.2%漏检率控制, 而l版本因延迟导致帧率跌破25fps.
高精度需求(如医学图像分析, 精密检测): x版本是唯一选择, 但需接受计算开销(如YOLOv8x的FLOPs高达257G, 训练成本成倍上升).
小目标检测: x版本优势显著, 例如YOLO26x在NWPU VHR-10数据集上比l版本在0.5m以下目标检测AP提升11.4%, 但推理时间增加2.3倍.
## 3.性价比权衡
m版本是性价比拐点: 以YOLO26为例, 从m到l版本, mAP仅提升2.7个点, 但延迟增加63%, 显存翻倍; 从l到x版本, mAP仅提升0.9个点, 延迟再涨50%.
知识蒸馏替代方案: 若无法直接使用x版本, 可通过知识蒸馏将能力迁移到s或m版本, 既保留部分高性能特性, 又兼顾部署效率.