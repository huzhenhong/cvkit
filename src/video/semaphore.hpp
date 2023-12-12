/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-09-09 17:21:42
 * LastEditors  : huzhenhong
 * LastEditTime : 2021-12-27 14:25:45
 * FilePath     : \\GoodsMoveDetector\\src\\test\\Semaphore.hpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/

#pragma once
#include <condition_variable>
#include <mutex>


class Semaphore
{
  public:
    Semaphore(int threshold = 10000);
    ~Semaphore();

    void Post(int num = 1);
    void Wait();
    void SetThreshold(int threshold);
    int  GetThreshold();

  private:
    std::condition_variable m_cv;
    std::mutex              m_mtx;
    int                     m_count{0};
    int                     m_threshold;
};

Semaphore::Semaphore(int threshold)
    : m_threshold(threshold)
{
}

Semaphore::~Semaphore() {}

void Semaphore::Post(int num)
{
    std::unique_lock<std::mutex> lock(m_mtx);
    m_count += num;

    while (m_count > m_threshold)
    {
        m_cv.wait(lock);
    }

    m_cv.notify_one();
}

void Semaphore::Wait()
{
    std::unique_lock<std::mutex> lock(m_mtx);

    while (m_count == 0)
    {
        m_cv.wait(lock);
    }

    if (m_count <= m_threshold)
    {
        m_cv.notify_one();
    }

    --m_count;
}

void Semaphore::SetThreshold(int threshold)
{
    m_threshold = threshold;
}

int Semaphore::GetThreshold()
{
    return m_threshold;
}
