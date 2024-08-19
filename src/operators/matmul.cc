#include "operators/matmul.h"
#include "core/tensor.h"
#include "utils/operator_utils.h"
#include <optional>
#include <vector>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
     << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
     << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // =================================== 作业
  // ===================================
  // TODO：返回经过 matmul 操作后的 shape
  // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
  // =================================== 作业
  // ===================================
  const auto A = inputs[0];
  const auto B = inputs[1];
  auto shape_A = A->getDims();
  auto shape_B = B->getDims();
  Shape output_shape = infer_broadcast(shape_A, shape_B);
//   2 3 5 4
//   1 3 2 5
  auto rank = A->getRank();
  if (transA && !transB) {
    output_shape[rank - 2] = shape_A[rank - 1];
    output_shape[rank - 1] = shape_B[rank - 1];
  } else if (!transA && transB) {
    output_shape[rank - 2] = shape_A[rank - 2];
    output_shape[rank - 1] = shape_B[rank - 2];
  } else if (transA && transB) {
    output_shape[rank - 2] = shape_A[rank - 1];
    output_shape[rank - 1] = shape_B[rank - 2];
  } else {
    output_shape[rank - 2] = shape_A[rank - 2];
    output_shape[rank - 1] = shape_B[rank - 1];
  }
  return std::make_optional<vector<Shape>>({output_shape});
}

} // namespace infini
