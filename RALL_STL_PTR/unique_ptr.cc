#include <iostream>
#include <memory>

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

void func(std::unique_ptr<C> p)
{
  p->do_something();
}

void func1(std::unique_ptr<C> &p)
{
  p->do_something();
}

void func2(C *p)
{
  p->do_something();
}

void func3(std::unique_ptr<C> p)
{
  p->do_something();
}
int main (int argc, char *argv[]) {

  std::unique_ptr<C> p = std::make_unique<C>();
  //这样就会报错，因为unique_ptr删除了拷贝函数
  //为什么要删除拷贝函数，因为三五法则
  //如果拷贝了指针，那么就会出现之前Vector那样重复释放的问题
  //func(p);
  //几个解决方式
  //1.使用引用
  func1(p);
  //2.获取原始指针,不需要移交给func
  func2(p.get());
  //3.移动构造,需要把控制权移交给func
  func3(std::move(p));
  
  return 0;
}
