#include <iostream>
#include <thread>
#include <future>
#include <format>

//不想让async创建线程而想手动创建线程可以使用promise
int download()
{
  for(int i = 0; i < 100; i ++)
  {
    if(i % 10 == 0)
    {
      std::cout << std::format("Downloading hello.zip (0%)\n", i);
    }
  }
  return 404;
}

int main (int argc, char *argv[]) {
  std::promise<int> pret;
  std::thread t1([&]{
    auto ret = download();
    pret.set_value(ret);//在线程返回的时候设置返回值
  });
  std::future<int> fret = pret.get_future();//在主线程里获future对象

  int ret = fret.get();//等待并获取线程返回值
  std::cout << ret << std::endl;
  return 0;
}
