#pragma once
#include <cuda_runtime.h>

void launch_nms(
    const float* d_boxes,
    const float* d_scores,
    int*         d_keep,
    int          num_boxes,
    float        iou_threshold,
    cudaStream_t stream = 0
);
