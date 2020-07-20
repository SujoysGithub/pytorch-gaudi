#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using binary_fn_foreach_alpha = std::vector<Tensor>(*)(TensorList, Scalar alpha);

DECLARE_DISPATCH(binary_fn_foreach_alpha, foreach_tensor_add_scalar_stub);
}} // namespace at::native
