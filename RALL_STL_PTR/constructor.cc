#include <cstdlib>
#include <cstring>
#include <iostream>
#include <format>

struct Demo{
  explicit Demo(std::string a, std::string b)
  {
    std::cout << std::format("Hello {0}, I am {1}\n", a, b);  
  }
};

//如果一个类里全都是写基础类型(指针)组成，那这些类型被称为POD
//编译器将不会在自动生成无参的构造函数里去初始化这些成员
//只会去掉用那些有无参构造函数类型的构造函数
//所以需要我们自己去指定初始化值
class Pig{
public:
  std::string m_name;
  int m_weight{};//括号里不写任何东西会默认初始化为0（指针为nullptr）(数组也会全部初始化为0)
  Demo m_demo{"Sam", "Lily"};//编译通过
  //Demo m_demo2 = {"Sam", "Lily"};//编译报错，使用了explicit,必须显示构造
};


//解决函数多返回值
Pig pigFunc()
{
  return {"sam", 120};
}

//编写自己的vector类
struct Vector{
  size_t m_size;
  int *m_data;

  explicit Vector(size_t n)
  {
    m_size = n;
    m_data = (int *)malloc(sizeof(int) * n);
  }
  //定义了析构函数，我们要么删除默认的拷贝构造要么自己定义一个新的
  //因为默认的是浅拷贝，只会对指针进行拷贝，不会对深层数组进行拷贝
  Vector(Vector const &other)
  {
    m_size = other.m_size;
    m_data = (int *)malloc(m_size * sizeof(int));
    memcpy(m_data, other.m_data, m_size * sizeof(int));
  }
  //拷贝赋值也是一样
  Vector &operator=(Vector const &other)
  {
    m_size = other.m_size;
    //内存的销毁重新分配通过realooc,可以利用现有的m_data,避免重新分配
    m_data = (int *)realloc(m_data, m_size * sizeof(int));
    memcpy(m_data, other.m_data, m_size * sizeof(int));
    return *this;
    //如果不考虑提高性能可以这样做 
    // this->~Vector();//先销毁现在的
    // new(this)Vector(other);//再重新构造
    // return *this;
  }

  //移动构造
  Vector(Vector &&other)
  {
    m_size = other.m_size;
    other.m_size = 0;
    m_data = other.m_data;
    other.m_data = nullptr;
  }
  //移动拷贝
  Vector &operator=(Vector &&other)
  {
    this->~Vector();
    //在已有的内存地址上构造一个新对象
    new(this)Vector(std::move(other));//placement new（定位new）
    return *this;
  }
  //placement new介绍
  /*
    placement new和普通的new相比，它不会分配新的内存，而是在已经分配好的内存上调用对象的构造函数
    char buffer[sizeof(MyClass)];
    MyClass* obj = new (buffer) MyClass(); // placement new
    注意：placement new不管理内存的释放，必须显示调用析构函数来释放对象占用的资源
    obj->~MyClass();
    注意使用是还要记得先把原来对象析构掉再new
  */

  ~Vector()
  {
    free(m_data);
  }
  size_t size()
  {
    return m_size;
  }

  void resize(size_t size)
  {
    m_size = size;
    m_data = (int *)realloc(m_data, m_size * sizeof(int));
  }

  int &operator[](size_t index)
  {
    return m_data[index];
  }
};

int main (int argc, char *argv[]) {
  
  //c++11里在没有定义任何构造函数时，编译器会自动生成一个参数个数和成员一样的构造函数
  //但是只能用{}来使用(为了兼容c++98)
  Pig pig{"zhansan", 120, Demo{"Tom", "Jieke"}};
  std::cout << pig.m_name << ' ' << pig.m_weight << std::endl;

  Pig pig2 = std::move(pigFunc());
  //内存交换swap只有O(1)的时间复杂度
  std::swap(pig, pig2);
  std::cout << pig.m_name << ' ' << pig.m_weight << std::endl;

  return 0;
}
