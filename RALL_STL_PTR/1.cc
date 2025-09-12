#include <cmath>
#include <ranges>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

//cpp20允许函数参数自动推断
void myFunc(auto &&v)
{
  
  for(auto &&vi: v 
      | std::views::filter([] (auto &&x) {return x >= 10;})
      | std::views::transform([](auto &&x) {return std::sqrtf(x);})
      )
  {
    std::cout << vi << std::endl;
  }
}

int main (int argc, char *argv[]) {

  std::vector<int> v{1, 2, 3 ,4};
  int sum = 0;

  std::for_each(v.begin(), v.end(), [&](auto vi){
    sum += vi;
  });
  //从0开始连乘
  int sum2 = std::reduce(v.begin(), v.end(), 1, std::multiplies{});

  std::cout << sum << ' '  << sum2 << std::endl;
  return 0;
}
