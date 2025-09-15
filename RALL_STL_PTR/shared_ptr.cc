#include <iostream>
#include <memory>
#include <format>

//为了解决unique_ptr不能拷贝和容易出错的问题
//出现了shared_ptr,他牺牲了效率换来了自由度
//而他之所以能够进行拷贝是因为他使用了引用计数
//拷贝会给他的计数器加1,智能指针被析构时计数器减1, 只有在计数器到0时才会销毁对象
struct C
{
  C()
  {
    std::cout << "分配内存" << std::endl;
  }
  ~C()
  {
    std::cout << "释放内存" << std::endl;
  }
  void do_something()
  {
    std::cout << "成员函数" << std::endl;
  }
};

void func(std::shared_ptr<C> p)
{
  //usecount 2 拷贝构造,计数+1 
  std::cout << std::format("use count:{0}\n", p.use_count());
  p->do_something();
}


int main (int argc, char *argv[]) {

  std::shared_ptr<C> p = std::make_shared<C>();
  //usecount 1 被创建出来，引用计数+1
  std::cout << std::format("use count:{0}\n", p.use_count());
  func(p);
  //usecount 1 出了func函数，p自动被析构了
  std::cout << std::format("use count:{0}\n", p.use_count());

 
  return 0;
}
