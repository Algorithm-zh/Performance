#include <ostream>
#include <iostream>
#include <set>
#include <vector>
using namespace std;
//set和vector的区别
/*
 * set内部是红黑树 迭代器是双向迭代器
 * 提供的运算符重载* != == ++ --
 * ++ --使用的是红黑树的遍历,而vector则是简单的指针移动
*
* 1.set会自动排序从小到大
* 2.set会去重
* 3.set元素在内存里是不连续的，不能按索引随机访问
* 4.set的元素可以高效的按值查找(红黑树o(logn))vector(o(n))
*
*/

// 检测类型是否为“容器”：有 begin(), end(), 且不是 string / string_view
template <typename T, typename = void>
struct is_container : std::false_type {};

// 排除 std::string 和 std::string_view（它们 technically 是容器，但通常不希望被遍历输出）
template <typename T>
struct is_container<T,
    std::void_t<
        decltype(std::begin(std::declval<T&>())),
        decltype(std::end(std::declval<T&>())),
        typename T::value_type
    >
> : std::bool_constant<
    !std::is_same_v<std::decay_t<T>, std::string> &&
    !std::is_same_v<std::decay_t<T>, std::string_view>
> {};
// 重载 operator<< 仅对“容器”生效
template <typename T>
std::enable_if_t<is_container<T>::value, std::ostream&>
operator<<(std::ostream& os, const T& container) {
    os << "{";
    auto it = std::begin(container);
    if (it != std::end(container)) {
        os << *it;
        ++it;
        for (; it != std::end(container); ++it) {
            os << "," << *it;
        }
    }
    os << "}";
    return os;
}


//set内部如何判定两个元素相等
//!(a<b)&&!(b<a)
//也就是set内部没有用==，而是用了两次比较运算符
//因此下面这种写法会直接把arch和any认为是同一个元素进行去重
//甚至可以不区分大小写进行去重
struct MyComp{
  bool operator()(string const &a, string const &b) const{
    return a[0] > b[0];
  }
};

int main (int argc, char *argv[]) {
  
  set<int> a{2,3 ,4,6,8, 5,1, 2};
  //cpp17结构化绑定
  auto [it, ret] = a.insert(2);//无需关心插入位置
  if(ret){
    cout << "插入成功, 元素：" << *it << endl;
  }else{
    cout << "插入失败" << endl;
  }
  cout << a << endl;
  //删除最大的元素
  a.erase(std::prev(a.end()));
  cout << a << endl;
  //返回大于等于x和大于x的元素的迭代器
  cout << *a.lower_bound(3) << ' ' << *a.upper_bound(3) << endl;

  //vector的构造函数能接受两个前向迭代器作为参数，set的迭代器符合要求
  vector<int> arr(a.lower_bound(2), a.upper_bound(7));
  cout << "vec:" << arr << endl;
  arr.push_back(2);
  cout << "vec:" << arr << endl;
  //set也接受前向迭代器，所以vector也能转为set,可以用来去重和排序
  set<int> arr2(arr.begin(), arr.end());
  //用assign可以利用已有的内存
  arr.assign(arr2.begin(), arr2.end());
  cout << "vec:" << arr << endl;


  //按字典序排序（ASCII🐎）
  set<string> b{"arch", "any", "zero", "Linux"};
  cout << b << endl;
  //元素类型，比较函数（如果不指定默认<）
  set<string, MyComp> c{"arch", "any", "zero", "Linux"};
  cout << c << endl;
  //二分法查找是否存在
  cout << c.count("zero") << endl;


  //multiset 只排序不去重
  multiset<int> bb = {1,2,1,2,3};
  cout << "bb:" << bb << endl;
  //删除所有的2
  bb.erase(bb.lower_bound(2),bb.upper_bound(2));
  cout << "bb:" << bb << endl;
  //equal_range,一次求出两个边界
  auto r = bb.equal_range(1);
  bb.erase(r.first, r.second);
  cout << "bb:" << bb << endl;

  
  return 0;
}
