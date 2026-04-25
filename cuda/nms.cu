#include <cuda_runtime.h>
#include "nms.h"

__device__ float iou(const float* a, const float* b) {
    float ax1 = a[0] - a[2] / 2, ay1 = a[1] - a[3] / 2;
    float ax2 = a[0] + a[2] / 2, ay2 = a[1] + a[3] / 2;
    float bx1 = b[0] - b[2] / 2, by1 = b[1] - b[3] / 2;
    float bx2 = b[0] + b[2] / 2, by2 = b[1] + b[3] / 2;

    float inter_w = fmaxf(0.f, fminf(ax2, bx2) - fmaxf(ax1, bx1));
    float inter_h = fmaxf(0.f, fminf(ay2, by2) - fmaxf(ay1, by1));
    float inter   = inter_w * inter_h;
    float uni     = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter;
    return (uni > 0) ? inter / uni : 0.f;
}

__global__ void nms_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ scores,
    int*         keep,
    int          num_boxes,
    float        iou_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes || keep[i] == 0) return;
    for (int j = 0; j < i; j++) {
        if (keep[j] == 0) continue;
        if (scores[j] >= scores[i] && iou(boxes + j*4, boxes + i*4) > iou_threshold) {
            keep[i] = 0;
            return;
        }
    }
}

void launch_nms(
    const float* d_boxes,
    const float* d_scores,
    int*         d_keep,
    int          num_boxes,
    float        iou_threshold,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks  = (num_boxes + threads - 1) / threads;
    nms_kernel<<<blocks, threads, 0, stream>>>(d_boxes, d_scores, d_keep, num_boxes, iou_threshold);
}
