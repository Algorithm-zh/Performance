#include <iostream>
#include <vector>

//整数作为模板参数
template<int N>
void show_times(std::string msg)
{
  for(int i = 0; i < N; i ++)
  {
      std::cout << msg << std::endl;
  }
}
//多个模板参数
template<int N = 1, class T>
void show_times2(T msg)
{
  for(int i = 0; i < N; i ++)
  {
      std::cout << msg << std::endl;
  }
}
//部分特化
template<class T>
T sum(std::vector<T> const &arr)
{
  T res = 0;
  for(auto &v : arr)
  {
    res += v;
  }
  return res;
};
//编译器优化
template<bool debug>
int sumto(int n)
{
  int res = 0;
  for(int i = 1; i <= n; i ++)
  {
    res += i;
    //如果使用函数参数的方式的话每次运行他都要判断debug,会变慢
    //而如果换成模板参数的方式，编译器会直接优化掉这个if判断
    //编译器会将constexpr函数视为内联函数，所以在编译时若能求值,函数调用替换成结果值
    //constexpr可以判断debug是不是编译时即能得出的值，若不是则会报错
    if constexpr (debug)
    {
      std::cout << i << "-th: " << res << std::endl; 
    }
  }
  return res;
}
int main (int argc, char *argv[]) {
  show_times<1>("one");
  show_times<2>("two");
  show_times<3>("three");
  show_times2<4>(42);

  std::vector<int>a{1, 2, 3, 4};
  std::cout << sum(a) << std::endl;

  //模板尖括号内的参数也不能用运行时变量
  //可以通过加constexpr来解决(=右边也必须是编译期常量)
  constexpr bool debug = true;
  std::cout << sumto<debug>(4);

  return 0;
}
