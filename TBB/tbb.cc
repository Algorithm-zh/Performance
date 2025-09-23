#include <chrono>
#include <iostream>
#include <tbb/tbb.h>
#include <string>
#include <thread>

//tbb是在一开始创建了核心数量个线程分别在不同的核心上运行
void download(std::string file){
  for(int i = 0; i < 10; i ++){
    std::cout << "Downloading " << file
              << " (" << i * 10 << "%)..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
  }
  std::cout << "Download complete: " << file << std::endl;
}

void interact(){
  std::string name;
  std::cin >> name;
  std::cout << "Hi, " << name << std::endl;
}

int main (int argc, char *argv[]) {

  //使用一个任务组tbb::task_group启动多个任务
  //一个任务不一定队员一个线程，如果任务数量超过cpu最大的线程数
  //会由TBB在用户层负责调度任务运行在多个预先分配好的线程
  //而不是由操作系统负责调度线程运行在多个物理核心
  tbb::task_group tg;
  tg.run([&]{
    download("hello.zip");
  });
  tg.run([&]{
    interact();
  });
  tg.wait();


  //使用parallel_invoke,他已经封装好了
  //wait也不需要调用
  tbb::parallel_invoke([&]{
    download("world.zip");
  }, [&]{
    interact();
  });

  return 0;
}


