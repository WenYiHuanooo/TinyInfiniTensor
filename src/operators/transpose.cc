#include "operators/transpose.h"
#include "core/tensor.h"
#include <optional>
#include <unordered_map>
#include <vector>

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
  auto rank = input->getRank();
  if (permute.empty()) {
    for (size_t i = 0; i < rank; ++i) {
      transposePermute[i] = i;
    }
  } else {
    IT_ASSERT(rank == permute.size());
    transposePermute = std::move(permute);
  }
  IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs) {
  const auto A = inputs[0];
  auto input_dim = A->getDims();
  auto output_dim = input_dim;
  int rank = A->getRank();
  
  // =================================== 作业
  // ===================================
  // TODO：修改 output_dim，返回正确的 transpose 后的 shape
  // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
  // =================================== 作业
  // ===================================
  std::unordered_map<int, int> dim_pos;
  for (int i = 0; i < rank; ++ i) {
    dim_pos[i] = input_dim[i];
  }
  for (int i = 0; i < rank; ++ i) {
    output_dim[i] = dim_pos[transposePermute[i]];
  }
  return std::make_optional<std::vector<Shape>>({output_dim});
}

std::string TransposeObj::toString() const {
  std::ostringstream os;
  os << type.toString() << "[" << getGuid() << "]";
  os << "(";
  os << vecToString(inputs[0]->getDims()) << ",";
  os << "input=" << inputs[0]->getGuid() << ",";
  os << "output=" << outputs[0]->getGuid() << ")";
  return os.str();
}
}; // namespace infini
