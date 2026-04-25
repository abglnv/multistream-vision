#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "nms.h"

namespace py = pybind11;

py::array_t<int> run_nms(
    py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    float iou_threshold  = 0.45f,
    float conf_threshold = 0.25f
) {
    auto b = boxes.unchecked<2>();   // [N, 4]
    auto s = scores.unchecked<1>(); // [N]
    int N = b.shape(0);

    std::vector<int> h_keep(N, 0);
    for (int i = 0; i < N; i++)
        h_keep[i] = (s(i) >= conf_threshold) ? 1 : 0;

    float *d_boxes, *d_scores;
    int   *d_keep;
    cudaMalloc(&d_boxes,  N * 4 * sizeof(float));
    cudaMalloc(&d_scores, N     * sizeof(float));
    cudaMalloc(&d_keep,   N     * sizeof(int));

    cudaMemcpy(d_boxes,  boxes.data(),  N * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores.data(), N     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keep,   h_keep.data(), N     * sizeof(int),   cudaMemcpyHostToDevice);
     
    launch_nms(d_boxes, d_scores, d_keep, N, iou_threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(h_keep.data(), d_keep, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_boxes); cudaFree(d_scores); cudaFree(d_keep);

    auto result = py::array_t<int>(N);
    std::copy(h_keep.begin(), h_keep.end(), result.mutable_data());
    return result;
}

PYBIND11_MODULE(nms_cuda, m) {
    m.doc() = "CUDA NMS for YOLOv8 output";
    m.def("run_nms", &run_nms,
          py::arg("boxes"),
          py::arg("scores"),
          py::arg("iou_threshold")  = 0.45f,
          py::arg("conf_threshold") = 0.25f);
}
