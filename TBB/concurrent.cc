#include <concepts>
#include <iostream>
#include <iterator>
#include <mutex>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <cmath>
#include <vector>
#include "tick.h"

int main (int argc, char *argv[]) {
  size_t n = 1 << 27;
  //concurrent_vector是并行的vector,可以不用锁锁起来使用，而且他的地址不是连续的而是随机的
  //它扩充的时候不会移动已有的元素
  tbb::concurrent_vector<float> a;

  //筛选>0的sin数
  //先推到线程局部的vector,再一次性推到concurrent_vector可以避免
  //频繁在concurrent_vector上产生锁竞争
  TICK(filter);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
  [&](tbb::blocked_range<size_t> r){
    std::vector<float> local_a;
    for(size_t i = r.begin(); i < r.end(); i ++){
      float val = std::sin(i);
      if(val > 0){
        local_a.push_back(val);
      }
    }
    auto it = a.grow_by(local_a.size());
    for(size_t i = 0; i < local_a.size(); i ++){
      *it ++ = local_a[i];
    }
  });
  TOCK(filter);

  //如果需要内存连续的方案
  std::vector<int> b;
  std::mutex mtx;

  TICK(filter2);
  b.reserve(n * 2 / 3);//先给b预留内存防止频繁扩容
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
  [&](tbb::blocked_range<size_t> r){
    std::vector<float> local_a;
    local_a.reserve(r.size());
    for(size_t i = r.begin(); i < r.end(); i ++){
      float val = std::sin(i);
      if(val > 0){
        local_a.push_back(val);
      }
    }
    std::lock_guard lck(mtx);
    std::copy(local_a.begin(), local_a.end(), std::back_inserter(a));
  });
  TOCK(filter);
  return 0;
}
