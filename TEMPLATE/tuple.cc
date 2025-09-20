#include <iostream>
#include <tuple>
#include <cmath>

// tuple可以将多个不同的类型打包成一个，尖括号里填各个元素的类型

std::tuple<bool, float> mysqrt(float x)
{
  if(x >= 0.f)
  {
    return {true, std::sqrt(x)};
  }else
  {
    return {false, 0.0f};
  }
}
int main (int argc, char *argv[]) {

  //当用于构造函数的时候。尖括号里的类型可以省去，c++17的新特性CTAD
  auto tup = std::tuple<int ,float, char>(3, 3.14f, 'a');
  auto first = std::get<0>(tup);
  auto second = std::get<1>(tup);
  auto third = std::get<2>(tup);
  //上面这种挨个get太慢，可以用17的新语法，结构化绑定
  auto &&[x, y, z] = tup;

  std::cout << first << ' ' << second << ' ' << third << std::endl;
  std::cout << x << ' ' << y << ' ' << z << std::endl;

  auto [success, value] = mysqrt(3.f);
  if(success)
  {
    std::cout << "成功！，value = " << value << std::endl;
  }else 
  {
    std::cout << "失败！" << std::endl;    
  }
  
  return 0;
}
