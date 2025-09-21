#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <shared_mutex>

class MTVector{
  std::vector<int> m_arr;
  //shared_mutex上锁时需要指定是读还是写，如果是写则用lock，它只会让一个人写 
  //如果是读则用lock_shared,那么他就会允许好几个人一起读
  mutable std::shared_mutex m_mtx;

public:
  void push_back(int val){
    //m_mtx.lock();
    //m_arr.push_back(val);
    //m_mtx.unlock();
    
    //符合rall思想的写法
    std::unique_lock grd(m_mtx);
    m_arr.push_back(val);
  }
  size_t size() const{
    // m_mtx.lock_shared();
    // size_t ret = m_arr.size();
    // m_mtx.unlock_shared();

    //符合rall思想的写法
    std::shared_lock grd(m_mtx);
    return m_arr.size();
  }
};

int main (int argc, char *argv[]) {
  MTVector arr;  
  std::thread t1([&]{
    for(int i = 0; i < 1000; i ++){
      arr.push_back(i);
    }
  });
  std::thread t2([&]{
    for(int i = 0; i < 1000; i ++){
      arr.push_back(i);
    }
  });
  t1.join();
  t2.join();
  std::cout << arr.size() << std::endl;
  return 0;
}
