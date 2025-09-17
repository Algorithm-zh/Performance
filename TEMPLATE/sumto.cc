#include <iostream>
#include "sumto.h"
//模板分离
//模板分离违反了开闭原则，不建议用
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

//必须在cpp文件里增加两个显示编译模板的声明(所有用到的情况都要写)
//编译器对模板的编译是惰性的，只有当.cpp文件里用到了这个模板，模板里的函数才会被定义
//而sumto.cc没有用到sumto<>函数的任何一份定义，所以main.cpp只能看到声明，所以他会认为没有定义
template int sumto<true>(int n);
template int sumto<false>(int n);
