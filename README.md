# A-Net
[A-Net: An A-shape Lightweight Neural Network for Real-time Surface Defect Segmentation](https://ieeexplore.ieee.org/abstract/document/10352342)

**Abstract:** Surface defect segmentation is a critical task in industrial quality control. Existing neural network architectures often face challenges in providing both real-time performance and high accuracy, limiting their practical applicability in time-sensitive, resource-constrained industrial setting. To bridge this gap, we introduce A-Net, an A-shape lightweight neural network specifically designed for real-time surface defect segmentation. Initially, A-Net introduces a pioneering A-shaped architecture tailored to efficiently handle both low-level details and high-level semantic information. Secondly, a series of lightweight feature extraction blocks are designed, explicitly engineered to meet the stringent demands of industrial defect segmentation. Finally, rigorous evaluations across multiple industry-standard benchmarks demonstrate A-Net's exceptional efficiency and high performance. Compared to the well-estabilished U-Net, A-Net achieves comparable or superior intersection over union (IoU) scores with gains of −0.21%, −0.3%, +4.7%, and +5.94% on NEU-seg, DAGM-seg, MCSD-seg, and MT dataset, respectively. Remarkably, A-Net does so with only 0.39M parameters, a 98.8% reduction, and 0.44G floating point operations (FLOPs), a 99% decrease in computational load. Besides, A-Net shows extremely fast inference speed on edge device without GPU because of its low FLOPs. A-Net contributes to the development of effective and efficient defect segmentation networks, suitable for real-world industrial applications with limited resources.


![The architecture of A-Net](https://github.com/Max-Chenb/A-Net/blob/main/images/architecture.png)  
<div align="center">
The architecture of A-Net
</div>

![111](https://github.com/Max-Chenb/A-Net/blob/main/images/results_neu.png)  
<div align="center">
Results on NEU dataset
</div>

![111](https://github.com/Max-Chenb/A-Net/blob/main/images/inference_speed_cpu.png)
<div align="center">
Inference speed test on CPU
</div>
