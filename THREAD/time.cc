#include <iostream>
#include <ratio>
#include <thread>
#include <chrono>

int main (int argc, char *argv[]) {
  
  auto t0 = std::chrono::steady_clock::now();
  for(volatile int i = 0; i < 10000000; i ++);
  auto t1 = std::chrono::steady_clock::now();
  auto dt = t1 - t0;
  int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
  std::cout << "time elapsed: " << ms << " ms" << std::endl;

  //使用double作为类型
  //duration_cast可以在任意的duration类型之间转换，duration<T,R>，T表示类型，R是时间单位，不写就是秒
  //std::milli毫秒，std::micro微秒
  //milliseconds是duration<int64_t, std::milli>的类型别名
  using double_ms = std::chrono::duration<double, std::milli>;
  double mss = std::chrono::duration_cast<double_ms>(dt).count();
  std::cout << "time elapsed: " << mss << " ms" << std::endl;
  return 0;
}
