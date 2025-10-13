#include <iterator>
#include <ostream>
#include <vector>
#include <iostream>
using namespace std;
namespace std {
template <class T>
ostream& operator<<(ostream &os, vector<T> const &v){
  os << "{";
  auto it = v.begin();
  if(it != v.end()){
    os << *it;
    for(++it; it != v.end(); ++it){
      os << ',' << *it;
    }
  }
  os << "}";
  return os;
}
}
struct C{
  // vector<int> a(4);错，这样会被识别为函数，不能这样写
  vector<int> a = vector<int>(4);//强制调用显示构造函数
};
int main (int argc, char *argv[]) {
  vector<int> a(3);//默认初始化为0.不需要memset
  cout << "a[0] = " << a[0] << endl;
  cout << "a[1] = " << a[1] << endl;
  cout << "a[2] = " << a[2] << endl;//使用[]不能检测是否越界，性能比较好
  //cout << "a[5] = " << a.at(5) << endl;//at函数会检测索引是否越界，如果越界会抛出异常，但是他因为检测越界所以会有性能损失
  vector<int> b{2, 3, 1};//初始化列表，vector(initializer_list<int> list);

  C c;
  cout << "c.a[0] = " << c.a[0] << endl;
  cout << "c.a.size() = " << c.a.size() << endl;

  vector<int> d(4, 3);//全部初始化为3
  d.resize(6,4);//如果数组不足6个元素，他会用0或者我们指定的数填充新增的元素，原来的保持不变
  cout << d << endl;
  d.resize(4);//这样会删除多余的元素
  cout << d << endl;
  //获取首地址指针,在调用c语言的函数时很有用
  //生命周期是主对象也就是d管理的,不需要p来管理
  auto p = d.data();
  cout << p[0] << endl;
  cout << p[1] << endl;
  cout << p[2] << endl;
  d.resize(7, 7);
  //这里resize后，因为需要的容量大于原容量，所以需要分配一个更大的连续内存
  //然后把原数组的部分拷贝过去，导致元素的地址就变了，data返回的指针和迭代器对象都会失效
  //push_back也是这个道理
  //当然：如果目标长度小于原容量，他是不会重新分配的
  //vector扩容时他不是一点点扩容的，有可能你扩容5他会直接加到10,就是为了性能考虑
  cout << p[0] << endl;


  //capacity可以查询已经分配的内存的大小，即最大的容量
  //size返回的是已经存储的数据的数组长度，可以发现当resize指定一个新长度超过原来的容量的时候
  //会重新分配一段更大的容量的内存来存储数组
  vector<int> a2 {1,2,3,4,5};
  cout << a2.data() << ' ' << a2.size() << '/' << a2.capacity() << endl;
  a2.resize(2);
  cout << a2.data() << ' ' << a2.size() << '/' << a2.capacity() << endl;
  a2.resize(5);
  cout << a2.data() << ' ' << a2.size() << '/' << a2.capacity() << endl;
  a2.resize(7);
  //这里发现他多扩容了,就是预留的
  cout << a2.data() << ' ' << a2.size() << '/' << a2.capacity() << endl;
  a2.resize(14);
  //这里发现没有多扩容，说明resize(n)的逻辑是max(n, capacity * 2);
  cout << a2.data() << ' ' << a2.size() << '/' << a2.capacity() << endl;
  
  cout << endl;


  vector<int> a3{1,2,3,4,5};
  //使用reserve预留一定的容量，这样就能防止出现容量不足需要动态扩容影响性能
  cout << a3.data() << ' ' << a3.size() << '/' << a3.capacity() << endl;
  a3.reserve(12);
  cout << a3.data() << ' ' << a3.size() << '/' << a3.capacity() << endl;
  a3.resize(2);
  cout << a3.data() << ' ' << a3.size() << '/' << a3.capacity() << endl;
  a3.resize(5);
  cout << a3.data() << ' ' << a3.size() << '/' << a3.capacity() << endl;
  //shrink_to_fit可以释放多余的容量
  //clear只能标记为0,不会释放多余容量
  a3.shrink_to_fit();
  cout << a3.data() << ' ' << a3.size() << '/' << a3.capacity() << endl;


  //直接在某个位置插入一个列表，因为内部预先知道了要插入列表的长度，会一次完成扩容，比push_back重复调用高校
  a3.insert(a3.begin() + 1, {233, 322, 122, 233});
  //iterrator insert(const_iterator pos, initializer_list<int> lst);
  //参数类型是initializer_list，所以不能插入另一个vector
  cout << a3 << endl;
  //但是insert还有个重载时insert(const_iterator pos, it beg, it end);
  //也就是可以选择对方容器的一个自区间内的元素进行插入
  a3.insert(a3.end(), a2.begin(), a2.end());
  cout << a3 << endl;


  //对方容器甚至可以是个c语言风格的数组
  //std::begin在具有begin和end的成员函数的容器时会直接调用
  //对于c语言的数组会被特化为b和b + sizeof(b)/sizeof(b[0])
  //还有std::size也能用
  int b2[] = {23, 45, 6, 7};
  a.insert(a.end(), std::begin(b2), std::end(b2));
  cout << a << endl;

  //erase可以删除指定位置的一个元素，通过迭代器指定
  a.erase(a.begin());//复杂度为o(n),因为他需要移动pos之后的元素到前面
  cout << a << endl;
  a.erase(a.end());//复杂度为o(1)
  cout << a << endl;
  a.erase(a.begin(), a.end() - 2);//删除区间内的
  cout << a << endl;
  return 0;
}
