#include <cstddef>
#include <iostream>
#include <vector>
#include <cmath>
#include "tick.h"

int main (int argc, char *argv[]) {
  size_t n = 1 << 27;  
  std::vector<float> a(n);

  TICK(for);
  for(size_t i = 0; i < a.size(); i ++){
    a[i] = std::sin(i);
  }
  TOCK(for);

  return 0;
}
