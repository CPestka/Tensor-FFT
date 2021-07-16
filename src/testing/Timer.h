#pragma once

//Provides some Objects that can be used time sections of code

#include <chrono>
#include <cstdint>

//Scope based timer that prints to console at destruction
class ScopeTimer{
 public:
  std::chrono::high_resolution_clock::time_point start;
  ScopeTimer(): start(std::chrono::high_resolution_clock::now() ){}
  virtual ~ScopeTimer(){};
  void PrintPeriod(){
    std::cout << std::chrono::high_resolution_clock::period::den << std::endl;
  }
};

class MinutesScopeTimer : public ScopeTimer {
public:
  ~MinutesScopeTimer(){
    std::cout << std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::high_resolution_clock::now() - this->start).count()
    << " Minutes" << std::endl;
  }
};

class SecondsScopeTimer : public ScopeTimer {
public:
  ~SecondsScopeTimer(){
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - this->start).count()
    << " Seconds" << std::endl;
  }
};

class MillisecondsScopeTimer : public ScopeTimer {
public:
  ~MillisecondsScopeTimer(){
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - this->start).count()
    << " Milliseconds" << std::endl;
  }
};

class MicrosecondsScopeTimer : public ScopeTimer {
public:
  ~MicrosecondsScopeTimer(){
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - this->start).count()
    << " Microseconds" << std::endl;
  }
};

class NanosecondsScopeTimer : public ScopeTimer {
public:
  ~NanosecondsScopeTimer(){
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - this->start).count()
    << " Nanoseconds" << std::endl;
  }
};

//Instantiate to set start time point, then call getTime..() to get the elapsed
//time.
class IntervallTimer{
 public:
  std::chrono::high_resolution_clock::time_point start;
  IntervallTimer() : start(std::chrono::high_resolution_clock::now()) {};
  void PrintPeriod(){
    std::cout << std::chrono::high_resolution_clock::period::den << std::endl;
  }

  int64_t getTimeInMinutes(){
    return std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::high_resolution_clock::now() - this->start).count();
  }
  int64_t getTimeInSeconds(){
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - this->start).count();
  }
  int64_t getTimeInMilliseconds(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - this->start).count();
  }
  int64_t getTimeInMicroseconds(){
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - this->start).count();
  }
  int64_t getTimeInNanoseconds(){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - this->start).count();
  }
};
