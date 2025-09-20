#include <algorithm>
#include <format>
#include <iostream>
#include <thread>
#include <vector>
//将线程放到全局数组里就可以延长其生命周期
std::vector<std::thread> pool;

//直接写个线程池的类可以在进程退出的时候自己调用线程池的析构函数，析构函数里join就可以省去手动join的麻烦
class ThreadPool
{
  std::vector<std::thread> m_pool;

public:
  void push_back(std::thread thr)
  {
    m_pool.push_back(std::move(thr));
  }
  ~ThreadPool()
  {
    for(auto &t : m_pool)t.join();
  }
};
ThreadPool tpool;

void download()
{
  for(int i = 0; i < 100; i ++)
  {
    if(i % 10 == 0)
    {
      std::cout << std::format("Downloading hello.zip (0%)\n", i);
    }
  }
}
void myfunc()
{
  std::thread t1([&](){
    download();
  });
  //pool.push_back(std::move(t1));
  tpool.push_back(std::move(t1));
}

int main (int argc, char *argv[]) {
  myfunc();
  for(auto &t : pool) t.join(); 
  return 0;
}
