#ifndef SYRAH_CYCLE_TIMER_H
#define SYRAH_CYCLE_TIMER_H

#if defined(__APPLE__)
#if defined(__x86_64__)
#include <sys/sysctl.h>
#else
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif // __x86_64__ or not

#include <cstdio>  // fprintf
#include <cstdlib> // exit

#elif _WIN32
#include <time.h>
#include <windows.h>
#else
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#endif

// This uses the cycle counter of the processor.  Different
// processors in the system will have different values for this.  If
// you process moves across processors, then the delta time you
// measure will likely be incorrect.  This is mostly for fine
// grained measurements where the process is likely to be on the
// same processor.  For more global things you should use the
// Time interface.

// Also note that if you processors' speeds change (i.e. processors
// scaling) or if you are in a heterogenous environment, you will
// likely get spurious results.
class CycleTimer
{
  public:
    using SysClock = unsigned long long;

    //////////
    // Return the current CPU time, in terms of clock ticks.
    // Time zero is at some arbitrary point in the past.
    static SysClock current_ticks()
    {
#if defined(__APPLE__) && !defined(__x86_64__)
        return mach_absolute_time();
#elif defined(_WIN32)
        LARGE_INTEGER qwTime;
        QueryPerformanceCounter(&qwTime);
        return qwTime.QuadPart;
#elif defined(__x86_64__)
        unsigned int a, d;
        asm volatile("rdtsc" : "=a"(a), "=d"(d));
        return static_cast<unsigned long long>(a) | (static_cast<unsigned long long>(d) << 32);
#elif defined(__ARM_NEON__) && 0 // mrc requires superuser.
        unsigned int val;
        asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(val));
        return val;
#else
        timespec spec;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &spec);
        return CycleTimer::SysClock(static_cast<float>(spec.tv_sec) * 1e9
                                    + static_cast<float>(spec.tv_nsec));
#endif
    }

    //////////
    // Return the current CPU time, in terms of seconds.
    // This is slower than current_ticks().  Time zero is at
    // some arbitrary point in the past.
    static double current_seconds()
    {
        return static_cast<double>(current_ticks()) * seconds_per_tick();
    }

    //////////
    // Return the conversion from seconds to ticks.
    static double ticks_per_second()
    {
        return 1.0 / seconds_per_tick();
    }

    static const char *tick_units()
    {
#if defined(__APPLE__) && !defined(__x86_64__)
        return "ns";
#elif defined(__WIN32__) || defined(__x86_64__)
        return "cycles";
#else
        return "ns"; // clock_gettime
#endif
    }

    //////////
    // Return the conversion from ticks to seconds.
    static double seconds_per_tick()
    {
        static bool initialized = false;
        static double seconds_per_tick_val;
        if (initialized)
            return seconds_per_tick_val;
#if defined(__APPLE__)
#ifdef __x86_64__
        int args[] = {CTL_HW, HW_CPU_FREQ};
        unsigned int Hz;
        size_t len = sizeof(Hz);
        if (sysctl(args, 2, &Hz, &len, NULL, 0) != 0)
        {
            fprintf(stderr, "Failed to initialize seconds_per_tick_val!\n");
            exit(-1);
        }
        seconds_per_tick_val = 1.0 / (double)Hz;
#else
        mach_timebase_info_data_t time_info;
        mach_timebase_info(&time_info);

        // Scales to nanoseconds without 1e-9f
        seconds_per_tick_val
            = (1e-9 * static_cast<double>(time_info.numer)) / static_cast<double>(time_info.denom);
#endif // x86_64 or not
#elif defined(_WIN32)
        LARGE_INTEGER qwTicksPerSec;
        QueryPerformanceFrequency(&qwTicksPerSec);
        seconds_per_tick_val = 1.0 / static_cast<double>(qwTicksPerSec.QuadPart);
#else
        FILE *fp = fopen("/proc/cpuinfo", "r");
        char input[1024]; // NOLINT(modernize-avoid-c-arrays): Extensively working with C libraries.
        if (!fp)
        {
            fprintf(stderr, "%s failed: couldn't find /proc/cpuinfo.", __PRETTY_FUNCTION__);
            exit(-1);
        }
        // In case we don't find it, e.g. on the N900
        seconds_per_tick_val = 1e-9;
        while (!feof(fp) && fgets(input, 1024, fp))
        {
            // NOTE(boulos): Because reading cpuinfo depends on dynamic
            // frequency scaling it's better to read the @ sign first
            float ghz, mhz;
            if (strstr(input, "model name"))
            {
                char *at_sign = strstr(input, "@");
                if (at_sign)
                {
                    char *after_at = at_sign + 1;
                    char *ghz_str = strstr(after_at, "ghz");
                    char *mhz_str = strstr(after_at, "mhz");
                    if (ghz_str)
                    {
                        *ghz_str = '\0';
                        if (1 == sscanf(after_at, "%f", &ghz))
                        {
                            // printf("GHz = %f\n", ghz);
                            seconds_per_tick_val = 1e-9f / ghz;
                            break;
                        }
                    }
                    else if (mhz_str)
                    {
                        *mhz_str = '\0';
                        if (1 == sscanf(after_at, "%f", &mhz))
                        {
                            // printf("MHz = %f\n", mhz);
                            seconds_per_tick_val = 1e-6f / mhz;
                            break;
                        }
                    }
                }
            }
            else if (1 == sscanf(input, "cpu MHz : %f", &mhz))
            {
                // printf("MHz = %f\n", mhz);
                seconds_per_tick_val = 1e-6f / mhz;
                break;
            }
        }
        fclose(fp);
#endif

        initialized = true;
        return seconds_per_tick_val;
    }

    //////////
    // Return the conversion from ticks to milliseconds.
    static double ms_per_tick()
    {
        return seconds_per_tick() * 1000.0;
    }

    CycleTimer() = delete;
};

#endif // SYRAH_CYCLE_TIMER_H
