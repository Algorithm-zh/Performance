#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <string_view>


std::vector<std::string> split(std::string s){
  std::vector<std::string> res;
  size_t pos = 0;
  while(true){
    auto newpos = s.find_first_of(" \t\v\f\n\r", pos);
    if(newpos == s.npos){
      res.push_back(s.substr(pos, newpos));
      break;
    }
    res.push_back(s.substr(pos, newpos - pos));
    pos = newpos + 1;
  }
  return res;
}

void func(char c[]){
  //数组退化为指针，因为sizeof返回的长度为char*的长度
  printf("func看到的字符串长度:%d\n", sizeof(c));
  //strlen找到\0后会返回结果,  也就是说你在中间写个\0他就直接返回之前的长度
  printf("使用strlen func看到的字符串长度:%d\n", strlen(c));
  c[2] = '\0';
  //printf内部也是调用的strlen,因此只输出了\0前部分
  printf("func新字符串：%s\n", c);
}

int main (int argc, char *argv[]) {
  char c = 'h';
  char s[] = "hello";
  printf("字符串：%s\n", s);
  //c语言：字符串的末尾必须有个0字符
  printf("字符串长度:%d\n", sizeof(s));//{'h','e','l','l','o','0'} 6
  func(s);


  std::string a = "asd";
  //c_str()保证返回0结尾的字符串首地址指针
  printf("a字符串长度:%d\n", strlen(a.c_str()));
  //data只保证返回长度为size()的连续内存的首地址指针，不保证0结尾
  printf("a字符串长度:%d\n", strlen(a.data()));
  //const char*可以隐式转换为string
  //反之必须调用c_str()
  
  //数字转字符串 全局函数std::to_string 内部是sprinf
  std::string b = std::to_string(23) + "haha" + std::to_string(42);
  std::cout << b << std::endl;
  //字符串转数字，std::sto* 而且会把开头数字后面的不是数字部分都去掉（开头必须是数字）
  //stoi stol stof stod... 
  size_t pos;
  auto res = std::stoi(b, &pos);
  std::cout << "转换后的数字为：" << res << "剩余部分开始位置为:" << pos << std::endl;


  //-1 npos, 直接到末尾 
  auto bb = b.substr(3,-1);
  std::cout << bb << std::endl;
  //从第三个字符开始找h字符
  std::cout << b.find('h', 3) << std::endl;


  //字符串转数字更好用的方式sstreamstring
  std::stringstream ss(b);
  int num;
  std::string str;
  ss >> num >> str;
  std::cout << num << ' ' << str << std::endl;


  //分割字符串find_first_of
  std::string cs = "hello world\tpyb teacher\ngood job";
  std::vector<std::string> vec = split(cs);
  for(const auto &vi : vec){
    std::cout << vi << std::endl;
  }

  //point 8B  len 8B  local_buf 16B(capacity 8B unused 8B) 
  std::cout << sizeof(cs) << std::endl;//32
  //vector则是24,因为他没有小字符串优化这个东西
  
  std::string s1 = "abcdefghijklnop";//经过测试最多长度为15，再长就不是在栈上了
  std::string_view s2 = s1;
  //切片 这样切片他会拷贝一个新的string对象，然后把字字符串拷贝到这个string对象里
  //浪费了内存,所以出现了string_view，只返回切片后的胖指针(ptr, len)
  //让新字符串和原字符串共享一片内存
  std::cout << s1.substr(2, 8) << std::endl;
  std::cout << s2.substr(1, 8) << std::endl;
  s1 = "abcdefghijklmnopqrst";
  std::cout << s1 << std::endl;
  //s1变大重新分配内存了，所以s2这个弱引用就失效了
  //有的时候不会失效是因为字符串太短，因此直接把他放到了栈上
  //会分配栈空间，如果长度小则存储在栈上
  std::cout << s2 << std::endl;



  std::string a4{"abc"};
  //可以看到string的capacity是15起步的，因为他会直接在栈上分配一个16大小的内存(还有一个是0，给c_str用的)
  //vector就不是，他甚至可以是0
  std::cout << a4.capacity() << std::endl;
  return 0;
}
