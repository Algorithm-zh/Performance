#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

int main (int argc, char *argv[]) {
  std::condition_variable cv;
  std::mutex mtx;
  bool ready = false;

  std::thread t1([&]{
    std::unique_lock lck(mtx);
    //设置唤醒条件，只有返回true唤醒才有用
    cv.wait(lck, [&]{return ready;});

    std::cout << "t1 is awake" << std::endl;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  std::cout << "not ready" << std::endl;
  cv.notify_one();

  ready = true;
  std::cout << "ready" << std::endl;
  cv.notify_one();
  t1.join();
  return 0;
}
