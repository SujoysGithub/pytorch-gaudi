
#include <ATen/native/ForeachOps.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

std::vector<Tensor> _foreach_add(TensorList tensors, Scalar scalar) {
  
  return foreach_tensor_add_scalar_stub(tensors[0].device().type(), tensors, scalar);
}
DEFINE_DISPATCH(foreach_tensor_add_scalar_stub);

}}
