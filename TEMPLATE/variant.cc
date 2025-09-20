#include <iostream>
#include <variant>

//这里用到了带auto的lambda，可以多次编译的特性，实现编译多个分支的效果
//std::visit std::variant这种模式称为静态多态
void print(std::variant<int, float> const &v)
{
  //visit可以自动用相应的类型调用lambda
  std::visit([&](auto const &t){
      std::cout << t << std::endl;
    }, v);
}

auto add(std::variant<int, float> const &v1, std::variant<int, float> const &v2)
{
  std::variant<int, float> ret;
  //visit里面的lambda可以有返回值，不过都得同样类型
  return std::visit([&](auto const &t1, auto const &t2) -> std::variant<int, float>{
    return t1 + t2;
  }, v1, v2);
}
//variant类似枚举类，
int main (int argc, char *argv[]) {
  std::variant<int, float> v = 3;

  //获取某个类型的值可以用get<type>，如果当前variant里不是这个类型就会抛出异常
  //<0>则是获取列表里第0个类型
  std::cout << std::get<0>(v) << std::endl;

  print(v);
  v = 3.14f;
  print(v);

  print(add(v, 1.2f));
  return 0;
}
