#include <iostream>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <vector>
#include <cmath>

int main (int argc, char *argv[]) {
  size_t n = 1 << 26;
  std::vector<float> a(n);

  //任务域，可以制定几个线程来对里面的任务执行
  tbb::task_arena ta(4);
  ta.execute([&]{
    tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
      a[i] = std::sin(i);
    });
  });

  //嵌套for
  tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
    tbb::parallel_for((size_t)0, (size_t)n, [&](size_t j){
      a[i * n + j] = std::sin(i) * std::sin(j);
    });
  });
  return 0;
}
