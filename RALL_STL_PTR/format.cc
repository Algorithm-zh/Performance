#include <bits/types/struct_timeval.h>
#include <ctime>
#include <format>
#include <iostream>
#include <chrono>
#include <sys/time.h>
void utc_time();
void local_time();
void local_time2();

void basicUsage()
{
  int age = 30;
  double pi = 3.14f;
  std::string name = "zhangsan";
  //{0} {1}分别表示第0个参数和第一个参数
  std::cout << std::format("My Name is {0} and I am {1} years old.", name, age) << std::endl;
  //:后面可以制定参数的格式选项，这里是保留小数点后三位
  std::cout << std::format("Pi is approximately {0:.3f}.", pi) << std::endl;

  //d十进制 x小写十六进制整数 X大写十六进制整数 o八进制 b二进制
  std::cout << std::format("{0:d} {0:x} {0:X} {0:o} {0:b}\n", 42);

  //<左对齐>右对齐^中间对齐, 数字是输出宽度，字符是对齐填充的字符
  std::cout << std::format("{:<10} | {:>10} | {:^10}\n", "left", "right", "center");
  std::cout << std::format("{:*<10} | {:#>10} | {:_^10}\n", "left", "right", "center");

  //强制输出正数的正号
  std::cout << std::format("{:+d} | {:+f}\n", 42, -3.14);

  //控制输出的数字的宽度填充
  std::cout << std::format("{:05}\n", 42);

  std::cout << std::format("{:=^60}\n", "=");

  //这个是linux系统的，可以比较精确的得到系统时间
  struct timeval tm;
  gettimeofday(&tm, nullptr);
  std::cout << tm.tv_sec * 1e3 + tm.tv_usec * 1e-3 << std::endl;

  utc_time();
  local_time();
  local_time2();
}

void utc_time()
{
  //格式化显示时间c++20开始支持直接格式化std::chrono::time_point
  //%Y四位年份 %m月份   %d月份中的第几天   %H小时   %M分钟    %S秒
  //UTC时间
  auto now = std::chrono::system_clock::now();
  std::cout << std::format("{:%Y-%m-%dT%H:%M:%S}\n", now);
}

void local_time()
{
  //本地时间
  auto now = std::chrono::system_clock::now();
  auto tz = std::chrono::current_zone();
  std::cout << std::format("{:%Y-%m-%dT%H:%M:%S}\n", tz->to_local(now));
}
//cpp20之前的方式
void local_time2()
{
  //本地时间(会根据系统时区将UTC转换为本地时间)
  struct tm local_tm;
  time_t now_t = time(NULL);
  localtime_r(&now_t, &local_tm);
  printf("%4d-%02d-%02dT%02d:%02d:%02d\n", local_tm.tm_year + 1900, local_tm.tm_mon + 1, local_tm.tm_mday, local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
}

int main (int argc, char *argv[]) {
  //使用printf iostreams以及format分别打印helloworld
  printf("Hello, %s!\n", "world");//运行时检查类型
  std::cout << "Hello, " << "world!" << std::endl;
  std::cout << std::format("Hello, {}!", "world") << std::endl;//编译期检查参数类型正确性

  basicUsage();

  return 0;
}
