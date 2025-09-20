#include <optional>
#include <iostream>
#include <cmath>
//使用tuple还要返回成功与否，失败了还要制定一个值0.0f,比较麻烦，这时用optional更合适
std::optional<float> mysqrt(float x)
{
if(x >= 0.f)
  {
    return std::sqrt(x);
  }else
  {
    return std::nullopt;
  }
}
//optional类似在模仿指针，optional在这里和float*一样，而nullopt则是nullptr
//但是这里符合RALL思想，当设为nullopt时会自动释放内部的对象
//而且他还是存储在栈上的，比unique_ptr效率更高
int main (int argc, char *argv[]) {
  auto ret = mysqrt(3.f);
  if(ret)//或者ret.has_value，两者等价
  {
    std::cout << "成功，value = " << ret.value() << std::endl;
  }else
  {
    std::cout << "失败" << std::endl;
  }
  return 0;
}
