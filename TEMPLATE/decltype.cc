#include <iostream>
#include <functional>
#include <utility>
#include <vector>


//c++规定，int&& 可以自动转换为int const &, 但不能转换为int &
//当然有int &&在首先还是走右值引用的函数
void func(int const &i)
{
  std::cout << i << " const &" << std::endl;
}

void func(auto &&i)
{
  i += 1;
  std::cout << i << " &&" << std::endl;
}

void func(auto &i)
{
  std::cout << i << " &" << std::endl;
}

int& func()
{
  int a = 123;
  return a;
}

//用decltype算出t1和t2两个不同类型的相加后的结果
template<class T1, class T2>
auto add(std::vector<T1> const &a, std::vector<T2> const &b)
{
  using T0 = decltype(T1{} + T2{});
  std::vector<T0> ret;
  for(size_t i = 0; i < std::min(a.size(), b.size()); i ++)
  {
    ret.push_back(a[i] + b[i]);
  }
  return ret;
}



int main (int argc, char *argv[]) {
  
  int a = 1;
  int &b = a;
  const int &c = a;
  //decltype和auto类似，但是auto并不是什么都能推断
  //auto是根据变量的初始值来推导变量类型的，而decltype是根据exp表达式来推导类型的
  //如果exp是一个不被括号()包围的表达式，或者是一个类成员访问表达式，或者是一个单独的变量，decltype(exp)的类型和exp一致
  //如果exp是函数调用，则decltype(exp)的类型就和函数返回值的类型一致
  //如果exp是一个左值，或被括号()包围，decltype(exp)的类型就是exp的引用，假设exp的类型为T，则decltype(exp)的类型为T&
  decltype((a))d = a;//int &,因为套了层括号，变成了decltype(表达式)
  decltype(a)e = a;//int  而这个是decltype(变量名)
  func(3);

  //想要定义一个和表达式返回类型一样的变量
  auto p1 = func();//用auto的话类型可能不一样(func返回引用的时候，auto会自动decay,就没有引用了)
  decltype(auto) p = func();
  
  std::vector<int> a1{1, 2, 3};
  std::vector<float> b1{1.2, 2.3, 2.3};
  auto ret = add(a1, b1);

  return 0;
}
