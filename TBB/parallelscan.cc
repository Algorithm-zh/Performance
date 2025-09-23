#include <iostream>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <cmath>
#include "tick.h"

int main (int argc, char *argv[]) {
  size_t n = 1 << 26;
  std::vector<float> a(n);

  TICK(scan);
  float res = tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), (float)0,
  [&](tbb::blocked_range<size_t> r, float local_res, auto is_final){//final表示当前是否在最终遍历
    for(size_t i = r.begin(); i < r.end(); i ++){
      local_res += std::sin(i);
      if(is_final){
        a[i] = local_res;
      }
    }
    return local_res;
  },[](float x, float y){
    return x + y;
  });
  TOCK(scan);
  std::cout << "parallel_scan" << std::endl;
  std::cout << a[n / 2] << std::endl;
  std::cout << res << std::endl;
  return 0;
}
