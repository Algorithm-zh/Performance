#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tbb/parallel_sort.h>
#include "tick.h"

int main (int argc, char *argv[]) {
  size_t n = 1 << 24;

  std::vector<int> arr(n);
  std::generate(arr.begin(), arr.end(), std::rand);

  // sort: 9.34107s
  // tbb_parallel_sort: 1.68378s
  TICK(sort);
  std::sort(arr.begin(), arr.end(), std::less<int>{});
  TOCK(sort);

  std::vector<int> arr2(n);
  std::generate(arr2.begin(), arr2.end(), std::rand);

  TICK(tbb_parallel_sort);
  tbb::parallel_sort(arr2.begin(), arr2.end(), std::less<int>{});
  TOCK(tbb_parallel_sort);
  return 0;
}
