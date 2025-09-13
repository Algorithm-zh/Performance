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
//如果一个类里全都是写基础类型(指针)组成，那这些类型被称为POD
//编译器将不会自动生成无参的构造函数去初始化这些成员
class Pig{
public:
  std::string m_name;
  int m_weight{0};
};

int main (int argc, char *argv[]) {

  std::vector<int> v{1, 2, 3 ,4};
  int sum = 0;

  std::for_each(v.begin(), v.end(), [&](auto vi){
    sum += vi;
  });
  //从0开始连乘
  int sum2 = std::reduce(v.begin(), v.end(), 1, std::multiplies{});

  std::cout << sum << ' '  << sum2 << std::endl;

  Pig pig;
  std::cout << pig.m_name << ' ' << pig.m_weight << std::endl;
  return 0;
}
