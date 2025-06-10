This project presents a comparative analysis of four popular face detection models: Haar Cascade, DNN-Caffe, HoG, and DNN-YOLO. The aim is to evaluate their performance across various conditions including lighting, occlusion, face orientation, and image scale.

Models Compared

Haar Cascade: Fast and lightweight but struggles with occlusions and non-frontal faces.

DNN-Caffe: High accuracy and robustness; suitable for varied real-world scenarios.

HoG (Dlib): Efficient on CPUs; less effective for small faces without upscaling.

DNN-YOLO: Best speed and accuracy trade-off; great for high-resolution and real-time applications.

Evaluation Criteria

Accuracy: Assessed using diverse datasets (Crowd, Expressions, Occlusions).

Speed: Measured on CPU and GPU where applicable.

Robustness: Tested under occlusions, lighting variation, and different face sizes.

Key Findings

Best Overall: DNN-Caffe for general use with strong accuracy and acceptable speed.

Best for Real-Time on GPU: DNN-YOLO with high throughput and precision.

Best for Low Resources: Haar Cascade (limited scenarios).

Best CPU-only Option: HoG for medium to large face sizes.

Future Scope

Optimize models for parallel processing

Extend evaluation to larger datasets

Explore adaptive learning for dynamic environments
