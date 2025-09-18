#include <iostream>

//函数作为模板类型

void print_float(float n)
{
  std::cout << "Float " << n << std::endl;
}

void print_int(int n)
{
  std::cout << "Int " << n << std::endl;
}

template<class Func>
void call_twice(Func const &func)//使用引用避免拷贝，Func为什么是16字节，因为一个指针8个字节，捕获了counter和t两个指针
{
  func(0);
  func(1);
}

//函数作为返回值
auto make_twice(int fac)
{
  //这里不能写&，因为make_twice退出后fac就没了，&是引用的地址，fac地址处的值已经被回收了
  //所以使用&要保证lambda对象的生命周期不超过他捕获的所有引用的寿命
  return [=](int n)
  {
    return n * fac;
  };
}

int main (int argc, char *argv[]) {
  call_twice(print_float);
  call_twice(print_int);
  int t = 23;
  int counter = 0;
  //函数可以引用定义位置所有的变量，这个在函数式编程里叫做闭包
  auto myFunc = [&](int n)
  {
    counter ++;
    std::cout << "myFunc " << n * t << std::endl;
  };
  call_twice(myFunc);

  return 0;
}
