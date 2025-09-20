#include <iostream>
#include <thread>
#include <future>
#include <format>

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

int main (int argc, char *argv[]) {
  //异步启动线程
  std::future<void> fret = std::async([&]{
    return download();
  });
  //这样他不会创建一个线程来执行，而是推迟运行直到调用get,可以实现惰性求值
  auto fret2 = std::async(std::launch::deferred, [&]{
    return download();
  });
  //如果想要实现浅拷贝实现共享同一个future对象，可以用shared
  std::shared_future<void> fret3 = std::async([&]{
    return download();
  });


  fret.get();//调用get看线程是否执行完毕，没有就等待他完成
  return 0;
}
