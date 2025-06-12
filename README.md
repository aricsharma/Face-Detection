# Face Detection Model Comparison

This project presents a comparative analysis of four widely used face detection models: Haar Cascade, DNN-Caffe, HoG, and DNN-YOLO. The goal is to evaluate their performance under various conditions such as lighting changes, occlusion, face orientation, and image scale.

## Models Compared

- **Haar Cascade**  
  Fast and lightweight, but limited by poor performance with occlusions and non-frontal faces.

- **DNN-Caffe**  
  High accuracy and robustness; performs well across diverse real-world scenarios.

- **HoG (Dlib)**  
  CPU-efficient and suitable for medium to large faces; requires upscaling for smaller faces.

- **DNN-YOLO**  
  Excellent speed and accuracy trade-off, ideal for high-resolution and real-time applications.

## Evaluation Criteria

- **Accuracy**  
  Tested on datasets with variations in crowd density, facial expressions, and occlusions.

- **Speed**  
  Measured on both CPU and GPU environments.

- **Robustness**  
  Evaluated under varying lighting conditions, partial occlusions, and different face sizes.

## Key Findings

- **Best Overall**: DNN-Caffe for strong accuracy and acceptable performance across scenarios.
- **Best for Real-Time (GPU)**: DNN-YOLO, offering high throughput with reliable precision.
- **Best for Limited Resources**: Haar Cascade for simple and fast deployments.
- **Best CPU-Only Option**: HoG for efficient detection of medium to large-sized faces.

## Future Scope

- Optimize detection pipelines using parallel processing techniques.
- Extend evaluation using larger and more diverse datasets.
- Integrate adaptive learning for dynamic environments and streaming input.

## Tech Stack

- **Languages**: Python
- **Libraries**: OpenCV, Dlib, YOLO (Darknet / OpenCV DNN)
- **Tools**: Jupyter Notebook, Matplotlib for visualization
