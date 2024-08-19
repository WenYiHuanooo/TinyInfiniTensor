#include "core/allocator.h"
#include "core/common.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = runtime->alloc(ALLOCATOR_SIZE);

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
  Block *first_block = (Block *)ptr;
  first_block->blockSize = ALLOCATOR_SIZE;
  first_block->addr = (size_t)ptr;
  free_list.push_front(first_block);
  first_block->pos = free_list.begin();
  allocated_blocks[first_block->addr] = first_block;
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
  this->ptr = nullptr;
}

// TODO：分配的内存地址需要仔细计算
size_t Allocator::alloc(size_t size) {
  if (this->ptr == nullptr) std::cout << "ptr is nullptr\n";
  // IT_ASSERT(this->ptr == nullptr);
  // pad the size to the multiple of alignment
  size = this->getAlignedSize(size);
  const std::size_t required_size = size + sizeof(Block);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来分配内存，返回起始地址偏移量
  // =================================== 作业
  // ===================================
  Block *alloc_block = nullptr;
  this->FindFirst(required_size, alloc_block);
  IT_ASSERT(alloc_block != nullptr);
  // 计算还有多少内存是没用的
  const std::size_t rest = alloc_block->blockSize - required_size;
  if (rest > 0) {
    Block *new_block = (Block *)((std::size_t)alloc_block + required_size);
    new_block->blockSize = rest;
    new_block->addr = alloc_block->addr + required_size;
    free_list.push_front(new_block);
    new_block->pos = free_list.begin();
  }
  free_list.erase(alloc_block->pos);
  alloc_block->blockSize = required_size;
  allocated_blocks[alloc_block->addr] = alloc_block;
  used += size;
  peak = std::max(peak, used);
  return alloc_block->addr;
}

void Allocator::FindFirst(const std::size_t size, Block *&block) {
  for (auto free_block : free_list) {
    if (free_block->blockSize >= size) {
      block = free_block;
      break;
    }
  }
}

void Allocator::free(size_t addr, size_t size) {
  // IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来回收内存
  // =================================== 作业
  // ===================================
  auto free_block = allocated_blocks[addr];
  free_list.push_front(free_block);
  free_block->pos = free_list.begin();
  used -= free_block->blockSize;
  allocated_blocks.erase(addr);
  MergeFreeBlock(free_block);
}

void Allocator::MergeFreeBlock(Block *&free_block) {
  if (free_block->pos != free_list.end()) {
    auto next_block = *(++free_block->pos);
    if (next_block->addr == (std::size_t)free_block + free_block->blockSize) {
      free_block->blockSize += next_block->blockSize;
      free_list.erase(next_block->pos);
    }
  }
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
  std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
            << std::endl;
}
} // namespace infini
