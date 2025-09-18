#include <iostream>
#include <functional>
#include <vector>

//虽然<clas Func>可以让编译器对每个不同的lambda生成一次，有助于优化
//但是如果想分离声明和实现或者加快编译就要用别的方式
//1.std::functional容器
//2.无捕获的lambda可以传为函数指针

void call_twice(const std::function<int(int)> &func)
{
  std::cout << func(0) << std::endl;
  std::cout << func(1) << std::endl;
  std::cout << "Func大小:" << sizeof(func) << std::endl;
}
void call_twice2(int func(int))
{
  std::cout << func(0) << std::endl;
  std::cout << func(1) << std::endl;
  std::cout << "Func大小:" << sizeof(func) << std::endl;
}

std::function<int(int)> make_twice(int fac) 
{
  return [=](int n)
  {
    return n * fac; 
  };
}

//lambda用途举例1：yield模式
template<class Func>
void fetch_data(Func const &func)
{
  for(int i = 0; i < 3; i ++)
  {
    func(i);
    func(i + 0.5f);
  }
}


int main (int argc, char *argv[]) {
  auto twice = make_twice(2);
  call_twice(twice);
  //c语言的api没法用function，所以可以用无捕获的lambda.无捕获的lambda会退化成函数指针
  call_twice2([](auto n){
    return n * 2;
  });
  auto twice3 = []<class T>(T n)
  {
    return n * 2; 
  };

  //decay_t是一个类型转换工具模板，用于将给定类型的衰变后的类型返回
  //即去掉引用、const和volatile等限定符,int[]转为int*
  std::vector<int> res_i;
  std::vector<float> res_f;
  fetch_data([&](auto const &x){
    using T = std::decay_t<decltype(x)>;//得到准确的x的类型，然后去掉x的各种限定符,判断x的类型
    if constexpr (std::is_same_v<T, int>)
    {
      res_i.push_back(x);
    }
    else if constexpr(std::is_same_v<T, float>)
    {
      res_f.push_back(x);
    }
  });


  //lambda用途举例2:局部递归c++14开始,这里写auto &&dfs也可以
  auto dfs = [&](const auto &dfs, int index) ->void 
  {
    //... 
    dfs(dfs, index + 1);
  };
  //c++23新写法,不需要再传入dfs
  // auto dfs2 = [&](this auto &&dfs, int index) ->void
  // {
  //   dfs2(index + 1);
  // };
  dfs(dfs, 0);
  //dfs2(0);
  return 0;
}

