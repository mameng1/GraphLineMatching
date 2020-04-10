// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> LinePoolAlign_forward_cuda(const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width);

at::Tensor LinePoolAlign_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& rois,
                                 const at::Tensor& con_idx_x,
                                 const at::Tensor& con_idx_y,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Interface for Python
std::tuple<at::Tensor, at::Tensor, at::Tensor> LinePoolAlign_forward(const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width) {
  if (input.type().is_cuda()) {
     CHECK_INPUT(input);
     CHECK_INPUT(rois);
    return LinePoolAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width);
  }
  AT_ERROR("Not implemented on the CPU");

}

at::Tensor LinePoolAlign_backward(const at::Tensor& grad,
                                 const at::Tensor& rois,
                                 const at::Tensor& con_idx_x,
                                 const at::Tensor& con_idx_y,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  if (grad.type().is_cuda()) {
    CHECK_INPUT(grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(con_idx_x);
    CHECK_INPUT(con_idx_y);
    return LinePoolAlign_backward_cuda(grad, rois, con_idx_x, con_idx_y, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &LinePoolAlign_forward, "LinePoolAlign forward (CUDA)");
  m.def("backward", &LinePoolAlign_backward, "LinePoolAlign backward (CUDA)");
}
