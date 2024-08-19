#include "core/graph.h"
#include "core/blob.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <queue>
#include <utility>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  ops.push_back(op);
  for (auto &input : op->getInputs()) {
    if (input) {
      input->addTarget(op);
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  for (auto &output : op->getOutputs()) {
    if (output) {
      output->setSource(op);
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  for (const auto &tensor : tensors)
    oss << tensor << "\n";

  oss << "Graph operators:\n";
  for (const auto &op : ops) {
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors())
      preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors())
      succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;
  std::unordered_set<OperatorObj *> flags;
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    // Any node is move to sorted in this loop.
    auto modified = false;
    for (auto const &op : ops) {
      if (auto const &inputs = op->getInputs();
          flags.find(op.get()) == flags.end() &&
          std::all_of(inputs.begin(), inputs.end(),
                      [&flags](auto const &input) {
                        auto ptr = input->getSource().get();
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  this->ops = std::move(sorted);
  return this->sorted = true;
}

void GraphObj::optimize() {
  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来实现指定的图优化规则
  // 图优化规则如下：
  // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose
  // 算子，且做的是相反的操作，可以将其全部删除）
  // 2.
  // 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
  // =================================== 作业
  // ===================================
  auto transpose_reverse = [](Shape permute1, Shape permute2) -> bool {
    if (permute1.size() != permute2.size()) {
      return false;
    }
    int permute1_size = permute1.size();
    for (int i = 0; i < permute1_size; ++i) {
      if (permute2[permute1[i]] != i) {
        return false;
      }
    }
    return true;
  };
  auto IsSwapLastDim = [](Shape permute) -> bool {
    auto size = permute.size();
    return permute[size - 1] == static_cast<int>(size - 2) && permute[size - 2] == static_cast<int>(size - 1);
  };
  topo_sort();
  for (auto ite = ops.begin(); ite != ops.end();) {
    auto op = *ite;
    bool should_erase = false;
    if (op->getOpType() == OpType::Transpose) {
      auto successors = op->getSuccessors();
      if (successors.size() > 1)
        continue;
      auto succ_op = successors[0];
      if (succ_op->getOpType() == OpType::Transpose) {
        auto transpose_op = as<TransposeObj>(op);
        auto transpose_succ_op =
            as<TransposeObj>(succ_op);
        if (transpose_reverse(transpose_op->getPermute(),
                              transpose_succ_op->getPermute())) {
          auto input = op->getInputs(0);
          input->removeTarget(op);
          auto output = succ_op->getOutput();
          for (auto tgt : output->getTargets()) {
            input->addTarget(tgt);
            tgt->removePredecessors(succ_op);
            if (auto source = input->getSource(); source != nullptr) {
              tgt->addPredecessors(source);
              source->removeSuccessors(op);
              source->addSuccessors(tgt);
            }
            tgt->replaceInput(output, input);
          }
          removeTensor(op->getOutput());
          removeTensor(output);
          removeOperator(succ_op);
          removeOperator(op);
          should_erase = true;
        }
      }
    } else if (op->getOpType() == OpType::MatMul) {
      auto pre_op = op->getPredecessors()[0];
      if (pre_op->getOpType() == OpType::Transpose) {
        auto pre_output = pre_op->getOutput();
        auto transpose_op = as<TransposeObj>(pre_op);
        auto permute = transpose_op->getPermute();
        if (IsSwapLastDim(permute)) {
          auto matmul_op = as<MatmulObj>(op);
          if (matmul_op->getInputs(0) == pre_output &&
              !matmul_op->getTransA()) {
            matmul_op->setTransA(true);
            std::swap(permute[permute.size() - 1], permute[permute.size() - 2]);
          }
          if (matmul_op->getInputs(1) == pre_output &&
              !matmul_op->getTransB()) {
            matmul_op->setTransB(true);
            std::swap(permute[permute.size() - 1], permute[permute.size() - 2]);
          }
          auto pre_input = pre_op->getInputs(0);
          auto source = pre_input->getSource();
          if (source != nullptr) {
            source->removeSuccessors(pre_op);
            source->addSuccessors(matmul_op);
            op->addPredecessors(source);
          }
          pre_input->removeTarget(pre_op);
          pre_input->addTarget(op);
          op->removePredecessors(pre_op);
          int idx = matmul_op->getInputs(0) == pre_output ? 0 : 1;
          op->inputs[idx] = pre_input;
          removeOperator(pre_op);
          removeTensor(pre_output);
          should_erase = true;
        }
      }
    }
    if (!should_erase) ite++;
  }
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    auto oldOutputs = op->getOutputs();
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      auto newShape = ans.value()[i];
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        auto tensor = this->getTensor(fuid);
        tensor->setShape(newShape);
      }
    }
  }
}

void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);

  // =================================== 作业
  // ===================================
  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor
  // 绑定内存
  // =================================== 作业
  // ===================================
  for (auto tensor : tensors) {
    size_t ptr = allocator.alloc(1024);
    auto blob = make_ref<BlobObj>(runtime, (void *)ptr);
    tensor->setDataBlob(blob);
  }
  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime,
            std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                tensor->getRuntime()->toString() + " to " +
                runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors)
    addTensor(t);
  return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    IT_ASSERT(
        !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

} // namespace infini
