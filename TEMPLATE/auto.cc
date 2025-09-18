#include <iostream>
#include <memory>
#include <map>

struct MyClass
{
  //auto不能用在类成员里,因为auto要编译阶段推导，但是成员要在构造函数调用后才会初始化
  // auto x = std::make_shared<int>();

};
//声明和实现分离后不能声明为auto
auto func()
{

};

//返回类型也可以是这样，懒汉单例模式
auto const &produce_table()
{
  static std::map<std::string, int> instance;
  return instance;
};

int main (int argc, char *argv[]) {

  int x = 233;
  auto const &ref = x;
  //ref = 43;//使用const可以让ref不能被写入，更安全
  
  return 0;
}
