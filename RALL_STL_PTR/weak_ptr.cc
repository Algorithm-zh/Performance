#include <iostream>
#include <memory>
#include <format>

//shared_ptr存在的问题：
//1.效率低
//2.会产生循环引用的问题
//所以weak_ptr产生了
//weak_ptr的构造和析构不会对计数器产生影响
//而且他可以查看计数器是否归0


struct B
{
  std::shared_ptr<B> m_child;
  // std::shared_ptr<B> m_parent;
  std::shared_ptr<B> m_parent;
};

void func(std::shared_ptr<B> p)
{
  std::cout << std::format("use count:{0}\n", p.use_count());
}


int main (int argc, char *argv[]) {

  auto parent = std::make_shared<B>();
  auto child = std::make_shared<B>();

  parent->m_child = child;
  child->m_parent = parent;

  //这样就会出现循环引用问题
  //计数器永远不会为0
  parent = nullptr;
  child = nullptr;
 
  return 0;
}
