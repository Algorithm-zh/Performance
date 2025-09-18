#include <iostream>
#include <ostream>
#include <string>
#include <vector>
//模板函数例子，打印vector
//模板函数有惰性编译，多次编译的特点
//如果不用到这个函数，他不会去编译，如果T是不同的类型，他会编译多次

template<class T>
void print(const std::vector<T> & vec)
{
  std::cout << "{";
  for(size_t i = 0; i < vec.size(); i ++)
  {
    std::cout << vec[i];
    if(i != vec.size() - 1)
      std::cout << ", ";
  }
  std::cout << "}" << std::endl;
}

//os在<<左边，vec在<<右边，也就是cout << vec,然后他们会返回一个os，就可以实现链式编程了
template<class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
  os << "{";
  for(size_t i = 0; i < vec.size(); i ++)
  {
    os << vec[i];
    if(i != vec.size() - 1)
      os << ", ";
  }
  os << "}";
  return os;
}

int main (int argc, char *argv[]) {
  std::vector<int> a{1, 2, 3, 4};
  std::vector<std::string> b{"123", "asd", "qwe", "ads"};

  print(a);
  print(b);

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  
  return 0;
}
