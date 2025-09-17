#include <iostream>

//1.普通函数使用constexpr
//c++是不允许用变量声明数组大小的
//（有时候能编译成功那是因为编译器的扩展（为了兼容c99和旧代码)）
//而c++的标准从来都不允许变长数组
//而constexpr可以解决这个问题
//constexpr修饰的函数有一定的限制
//1、函数体尽量只包含一个return,多个可能会编译出错
//2、函数体可以包含其他语句，但是不能是运行期语句，只能是编译期语句
//编译器会把constexpr函数视为内联函数，所以在编译期若能求值，则会把函数调用直接替换为结果
constexpr int GetLen(int a, int b)
{
  return a + b;
}
//2.在类的构造函数里使用constexpr关键字
//constexpr修饰构造函数可以保证传递给构造函数的所有参数都是constexpr,也即对象的所有成员都是constexpr
//注意constexpr构造函数的函数体必须为空或者里面的内容是在编译期可以完成的代码，所有成员变量的初始化都放到初始化列表里
class Num
{
public:
  constexpr Num(int num1, int num2) : m_num1(num1), m_num2(num2)
  {

  }
  int m_num1;
  int m_num2;
};
//3.constexpr函数必须是内联的，不能分离声明和定义
int main (int argc, char *argv[]) {

  int array[GetLen(1, 2)];
  
  constexpr Num t(1, 2);
  const int *n = new int[3]{1,2,3};

  enum e
  {
    x = t.m_num1,
    y = t.m_num2,
  };
  return 0;
}
