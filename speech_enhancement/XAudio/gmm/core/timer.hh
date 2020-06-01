/*
 * $File: timer.hh
 * $Date: Tue Dec 10 16:44:35 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#ifdef _WIN32
#include <time.h>
#else
#include "sys/time.h"
#endif
#include <string>

#ifdef _WIN32
#include <Windows.h>
#include <stdint.h>

//typedef struct timeval {
//	long tv_sec;
//	long tv_usec;
//} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
#endif

class Timer {
	public:
		long long get_time() {
			timeval tv;
			gettimeofday(&tv, 0);
			return tv.tv_sec * 1000ll + tv.tv_usec / 1000;
		}

		long long m_start;
		long long start() {
			return m_start = get_time();
		}

		long long stop() {
			return get_time() - m_start;
		}
};




class GuardedTimer {
	public:
		std::string prompt;
		Timer timer;
		bool enable;
		GuardedTimer(const std::string &prompt, bool enable = true) :
			prompt(prompt), enable(enable) {
				timer.start();
			}
		~GuardedTimer() {
			if (enable) {
				printf("%s: %.3lfs\n", prompt.c_str(), timer.stop() / 1000.0);
				fflush(stdout);
			}
		}

};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

