#include <iostream>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include <cmath>

//归约
int main (int argc, char *argv[]) {
  size_t n = 1 << 26;
  //初始值为0 
  float res = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), (float)0,
  [&](tbb::blocked_range<size_t> r, float local_res){
    for(size_t i = r.begin(); i < r.end(); i ++){
      local_res += std::sin(i);        
    }
    return local_res;
  }, [](float x, float y){
      return x + y;
  });
  std::cout << res << std::endl;
  return 0;
}
