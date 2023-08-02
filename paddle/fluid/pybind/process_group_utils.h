// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

namespace paddle {
namespace pybind {

template <typename DeviceContext, typename T>
struct ConcatDenseTensor {
  void operator()(const DeviceContext &context,
                  const std::vector<phi::DenseTensor> &in,
                  phi::DenseTensor *out,
                  int axis = 0) {
    phi::funcs::ConcatFunctor<DeviceContext, T> concat_functor;
    concat_functor(context, in, axis, out);
  }
};

template <typename DeviceContext, typename T>
struct SplitDenseTensor {
  void operator()(const DeviceContext &context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor *> *out,
                  int axis = 0) {
    std::vector<const phi::DenseTensor *> shape_refer;

    // std::cout << "bug2" << std::endl;
    // std::cout << "out->size " << out->size() << std::endl;
    // std::cout << "in:" << in.dims() << std::endl;

    shape_refer.reserve(out->size());
    // std::cout << "bug3" << std::endl;
    for (auto *p_tensor : *out) {
      // p_tensor->dims() = 1;

      // phi::DenseTensor* tmp = new phi::DenseTensor();
      // std::cout << "in dims:" << in.dims() << std::endl;
      if (in.dims().size() == 1) {
        // std::cout << "here's johnny!" << std::endl;
        p_tensor->Resize(phi::make_ddim({1, 1}));
        // std::cout << "p_tensor" << p_tensor->dims() << std::endl;
      }
      // else {
      //   std::cout << "p_tensor->dims()" << p_tensor->dims() << std::endl;
      //   p_tensor->Resize(p_tensor->dims());
      // }
      // if (in.dtype() == phi::DataType::INT64) {
      //   context.template Alloc<int64_t>(tmp);
      // } else {
      //   context.template Alloc<float>(tmp);
      // }tmp->Resize(phi::make_ddim({1, 1}));
      // p_tensor->Resize(phi::make_ddim({1, 1}));
      shape_refer.emplace_back(p_tensor);
    }

    // std::cout << "bug4" << std::endl;
    phi::funcs::SplitFunctor<DeviceContext, T> split_functor;
    // std::cout << "bug5" << std::endl;
    // std::cout << "out: " << out << std::endl;
    // std::cout << shape_refer << std::endl;
    // std::cout << "in: " << in << std::endl;
    // cpu_ctx = static_cast<const phi::XPUContext &>(context);
    // data_t* p = (data_t*) malloc(num * sizeof(data_t));
    // phi::backends::xpu::MemcpySyncD2H(p, enctrypted_vector, num *
    // sizeof(data_t), place, *xpu_ctx); phi::Copy(context, cpu_seq, place,
    // true, &seq); split_functor(cpu_ctx, in, shape_refer, axis, out);
    split_functor(context, in, shape_refer, axis, out);
    // std::cout << "bug5。1" << std::endl;
  }
};

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T>
struct ConcatDenseTensor<platform::CustomDeviceContext, T> {
  void operator()(const platform::CustomDeviceContext &context,
                  const std::vector<phi::DenseTensor> &in,
                  phi::DenseTensor *out,
                  int axis UNUSED = 0) {
    auto *out_data = out->data<T>();
    auto *device = phi::DeviceManager::GetDeviceWithPlace(context.GetPlace());
    size_t offset = 0;
    phi::stream::Stream stream_wrapper(context.GetPlace(), context.stream());

    for (const auto &tensor : in) {
      const auto *in_data = tensor.data<T>();
      if (out_data + offset != in_data) {
        device->MemoryCopyD2D(out_data + offset,
                              in_data,
                              tensor.numel() * sizeof(T),
                              &stream_wrapper);
      }
      offset += tensor.numel();
    }
  }
};

template <typename T>
struct SplitDenseTensor<platform::CustomDeviceContext, T> {
  void operator()(const platform::CustomDeviceContext &context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor *> *out,
                  int axis UNUSED = 0) {
    auto *in_data = in.data<T>();
    auto *device = phi::DeviceManager::GetDeviceWithPlace(context.GetPlace());
    size_t offset = 0;
    phi::stream::Stream stream_wrapper(context.GetPlace(), context.stream());

    for (auto *p_tensor : *out) {
      auto *out_data = p_tensor->data<T>();
      if (out_data != in_data + offset) {
        device->MemoryCopyD2D(out_data,
                              in_data + offset,
                              p_tensor->numel() * sizeof(T),
                              &stream_wrapper);
      }
      offset += p_tensor->numel();
    }
  }
};
#endif

template <typename DeviceContext>
void ConcatDenseTensorWithType(const DeviceContext &dev_ctx,
                               const std::vector<phi::DenseTensor> &t_list,
                               phi::DenseTensor *p_out,
                               phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      ConcatDenseTensor<DeviceContext, bool>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensor<DeviceContext, uint8_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT8:
      ConcatDenseTensor<DeviceContext, int8_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensor<DeviceContext, int32_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensor<DeviceContext, int64_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT16:
      ConcatDenseTensor<DeviceContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::BFLOAT16:
      ConcatDenseTensor<DeviceContext, phi::dtype::bfloat16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensor<DeviceContext, float>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT64:
      ConcatDenseTensor<DeviceContext, double>()(dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}

#ifdef PADDLE_WITH_XPU
template <>
void ConcatDenseTensorWithType(const phi::XPUContext &dev_ctx,
                               const std::vector<phi::DenseTensor> &t_list,
                               phi::DenseTensor *p_out,
                               phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      ConcatDenseTensor<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensor<phi::XPUContext, float>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensor<phi::XPUContext, int32_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensor<phi::XPUContext, int64_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensor<phi::XPUContext, uint8_t>()(dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}
#endif

template <typename DeviceContext>
void SplitDenseTensorWithType(const DeviceContext &dev_ctx,
                              const phi::DenseTensor &t_in,
                              std::vector<phi::DenseTensor *> *p_list,
                              phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      // std::cout << "BOOL" << std::endl;
      SplitDenseTensor<DeviceContext, bool>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::UINT8:
      // std::cout << "UINT8" << std::endl;
      SplitDenseTensor<DeviceContext, uint8_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT8:
      // std::cout << "INT8" << std::endl;
      SplitDenseTensor<DeviceContext, int8_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT32:
      // std::cout << "INT32" << std::endl;
      SplitDenseTensor<DeviceContext, int32_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT64:
      // std::cout << "INT64" << std::endl;
      SplitDenseTensor<DeviceContext, int64_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT16:
      // std::cout << "FLOAT16" << std::endl;
      SplitDenseTensor<DeviceContext, phi::dtype::float16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::BFLOAT16:
      // std::cout << "BFLOAT16" << std::endl;
      SplitDenseTensor<DeviceContext, phi::dtype::bfloat16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT32:
      // std::cout << "FLOAT32" << std::endl;
      SplitDenseTensor<DeviceContext, float>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT64:
      // std::cout << "FLOAT64" << std::endl;
      SplitDenseTensor<DeviceContext, double>()(dev_ctx, t_in, p_list);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}

#ifdef PADDLE_WITH_XPU
template <>
void SplitDenseTensorWithType(const phi::XPUContext &dev_ctx,
                              const phi::DenseTensor &t_in,
                              std::vector<phi::DenseTensor *> *p_list,
                              phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      // std::cout << "FLOAT16" << std::endl;
      SplitDenseTensor<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT32:
      // std::cout << "FLOAT32" << std::endl;
      SplitDenseTensor<phi::XPUContext, float>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT32:
      // std::cout << "INT32" << std::endl;
      SplitDenseTensor<phi::XPUContext, int32_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT64:
      // std::cout << "INT64" << std::endl;
      SplitDenseTensor<phi::XPUContext, int64_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::UINT8:
      // std::cout << "UINT8" << std::endl;
      SplitDenseTensor<phi::XPUContext, uint8_t>()(dev_ctx, t_in, p_list);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}
#endif

void ConcatTensor(const phi::DeviceContext &dev_ctx,
                  const std::vector<phi::DenseTensor> &tensor_list,
                  const Tensor *tensor) {
  auto *dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensor->impl()).get();

  const auto &place = dev_ctx.GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ConcatDenseTensorWithType(static_cast<const phi::GPUContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not support GPU, please "
        "recompile or reinstall Paddle with GPU support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    ConcatDenseTensorWithType(static_cast<const phi::XPUContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not support XPU, please "
        "recompile or reinstall Paddle with XPU support."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    ConcatDenseTensorWithType(
        static_cast<const platform::CustomDeviceContext &>(dev_ctx),
        tensor_list,
        dense_tensor,
        tensor->dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not compiled with "
        "CUSTOM_DEVICE, please recompile or reinstall Paddle with "
        "CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    ConcatDenseTensorWithType(static_cast<const phi::CPUContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Concat tensor not supported on place (%s)", place));
  }
}

void SplitTensor(const phi::DeviceContext &dev_ctx,
                 const phi::DenseTensor &tensor,
                 const std::vector<Tensor> *tensor_list) {
  // std::cout << "CXK1" << std::endl;
  std::vector<phi::DenseTensor *> dense_list;
  for (auto &tensor : *tensor_list) {
    auto *p_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()).get();
    // std::cout << "p_tensor1" << *p_tensor << std::endl;
    dense_list.emplace_back(p_tensor);
  }
  // std::cout << "CXK2" << std::endl;

  const auto &place = dev_ctx.GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    SplitDenseTensorWithType(static_cast<const phi::GPUContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split tensor since it's not support GPU, please "
        "recompile or reinstall Paddle with GPU support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    // std::cout << "CXK3" << std::endl;
    SplitDenseTensorWithType(static_cast<const phi::XPUContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
    // std::cout << "CXK4" << std::endl;
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with XPU, "
        "please recompile or reinstall Paddle with XPU support."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    // std::cout << "CXK5" << std::endl;
    SplitDenseTensorWithType(
        static_cast<const platform::CustomDeviceContext &>(dev_ctx),
        tensor,
        &dense_list,
        tensor.dtype());
    // std::cout << "CXK6" << std::endl;
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with CUSTOM_DEVICE, "
        "please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    // std::cout << "CXK7" << std::endl;
    SplitDenseTensorWithType(static_cast<const phi::CPUContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
    // std::cout << "CXK8" << std::endl;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split tensor not supported on place (%s)", place));
  }
  // std::cout << "CXK9" << std::endl;
}

inline std::vector<int64_t> GetDefaultSplitSizes(const phi::DenseTensor &tensor,
                                                 int world_size) {
  return std::vector<int64_t>(world_size, tensor.dims()[0] / world_size);
}

}  //  namespace pybind
}  //  namespace paddle
