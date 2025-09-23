#include <cstddef>
#include <iostream>
#include <tbb/tbb.h>
#include <vector>
#include <cmath>
#include "tick.h"

int main (int argc, char *argv[]) {
  size_t n = 1 << 27;
  std::vector<float> a(n);

  //时间复杂度为O(n/c),c是线程数量
  size_t maxt = 4;
  tbb::task_group tg;
  //4个线程，每个线程处理n/c个数据
  for(size_t t = 0; t < maxt; t ++){
    auto beg = t * n / maxt;
    auto end = std::min(n, (t + 1) * n / maxt);
    tg.run([&, beg, end]{
      for(size_t i = beg; i < end; i ++){
        a[i] = std::sin(i);
      }
    });
  }
  tg.wait();

  //1.2s 串行则是8s
  TICK(for);
  //映射：parallel_for封装好了上面的操作
  //parallel_for:在一个范围内并行的执行for循环
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](tbb::blocked_range<size_t> r){
                      for(size_t i = r.begin(); i < r.end(); i ++){
                        a[i] = std::sin(i);
                      }
                    });
  TOCK(for);

  tbb::parallel_for_each(a.begin(), a.end(), [&](float &f){
    f = 32.f;
  });


  //二维区间for循环
  size_t n2 = 1 << 13;
  std::vector<float> b(n * n);
  tbb::parallel_for(tbb::blocked_range2d<size_t>(0, n, 0, n),
                    [&](tbb::blocked_range2d<size_t> r){
                      for(size_t i = r.cols().begin(); i < r.cols().end(); i ++){
                        for(size_t j = r.rows().begin(); j < r.rows().end(); j ++){
                          a[i * n + j] = std::sin(i) * std::sin(j);
                        }
                      }
                    });

  return 0;
}
