#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>
#include <format>

//在C++ Ranges库中，视图（views）是对范围（ranges）的轻量级引用，它们提供了一种惰性求值的方式来转换和过滤数据
void viewFunc(std::vector<int>& vec)
{
  //只包含偶数的视图
  auto even_numbers = vec | std::views::filter([](int n){return n % 2 == 0;});
  //对剩余的偶数再进行平方
  auto squared_numbers = even_numbers | std::views::transform([](int n){return n * n;});

  for(auto v : squared_numbers)
  {
      std::cout << v << ' ';
  }
  std::cout << std::endl;

  //范围工厂函数
  //iota自定义点对象，一个通过重复递增初始值来生成元素序列的 range factory。可以是有限的或无限的。第一个是起始值，第二个是哨兵
  auto numbers = std::views::iota(1, 6);//[1, 2, 3, 4, 5]
  for(auto n : numbers)
  {
      std::cout << n << ' ';
  }
  std::cout << std::endl;

  //subrange
  auto subrange = std::ranges::subrange(vec.begin(), vec.end());
  auto first_three = subrange.advance(3);//取后三位
  for(auto n : first_three)
  {
      std::cout << n << ' ';
  }
  std::cout << std::endl;

  //join 将所有字符转大写并连起来
  std::vector<std::string> str{"hello", "world", "!"};
  auto link_str = str | std::views::join | std::views::transform([](unsigned char c) -> char{
    return static_cast<unsigned char>(std::toupper(c));
  });

  for(auto c: link_str)
  {
      std::cout << c;
  }
  std::cout << std::endl;
  // std::views::reverse：反转视图，用于反转原始范围的顺序。
  // std::views::take：取前N个元素的视图。
  // std::views::drop：丢弃前N个元素的视图。
  // std::views::join：连接视图，用于将多个范围连接成一个范围。
  // std::views::split：分割视图，用于根据分隔符将原始范围分割成多个子范围。
  // std::views::unique：移除连续重复的元素
}


//注意⚠️！：视图是对底层数据的引用，而不是数据的副本，所以必须确保数据在视图的生命周期内有效
auto get_view() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    return vec | std::views::filter([](int x) { return x % 2 == 0; });
}
void lazyView()
{
  //视图的一个关键特性为惰性求值，意味着视图的元素只有在被访问时才会被计算
  //numbers就是个无限序列，通过iota生成整数序列，但是实际的计算只发生在元素被访问时，即在循环中迭代时
  //可以看到iota不设置哨兵的时候就是无限序列，你要用到多少他就给你生成多少
  auto numbers = std::views::iota(1) | std::views::transform([](int n){
    std::cout << "Computing:" << n << std::endl;
    return n * n;
  });
  for(auto i : numbers | std::views::take(5))
  {
    std::cout << "Value: " << i << std::endl;
  }

  //这里就会出错，因为vec的生命周期出了get_view这个函数就没了，所以会导致未定义行为
  // auto view = get_view();
  // for (auto v : view) {
  //     std::cout << v << " ";
  // }

}


//范围算法，直接作用于范围，而不是迭代器对
void algorithm()
{
  std::vector<int> v{1, 2, 5, 6, 4, 3};
  std::ranges::sort(v);
  for(auto i : v)
  {
      std::cout << i << ' ';
  }
  //对范围内进行稳定排序
  std::ranges::stable_sort(v);
  //对范围内一部分元素进行排序,使前3个元素是整个范围中最小的3个元素
  std::ranges::partial_sort(v, v.begin() + 3);

  //查找第一个偶数
  auto it = std::ranges::find_if(v, [](int i){return i % 2 == 0;});

  std::cout << std::endl;
}



int main (int argc, char *argv[]) {
  
  std::vector vec{1, 2 ,4 ,5 , 6, 7};

  viewFunc(vec);

  lazyView();

  algorithm();

  return 0;
}
