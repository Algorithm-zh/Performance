#include <vector>
#include <list>
#include <iostream>
using namespace std;

template<class Ptr>
void print(Ptr begptr, Ptr endptr){
  for(Ptr ptr = begptr; ptr != endptr; ptr ++){
    auto value = *ptr;
    cout << value << endl;
  }
}

int main (int argc, char *argv[]) {
  list<char> a{'h', 'e', 'l', 'l', 'o'};
  //list<char>::iterator是一个特殊定义过的类型，其中的++ * !=等运算符被重载过
  //会把其对应到链表的curr = curr->next上
  //这样用起来像一个普通的指针，但是内部通过运算符的重载适配不同容器的类型
  //就是迭代器，迭代器是容器和算法之间的桥梁
  //vector这种连续的可随机访问容器的迭代器是单向迭代器，可以+ []
  //而list这种双向迭代器只有* ++ --运算符可以用
  //迭代器的本质就是弱引用,原容器解构了他就失效了
  list<char>::iterator begptr = a.begin();
  list<char>::iterator endptr = a.end();
  print(begptr, endptr); 
  return 0;
}
