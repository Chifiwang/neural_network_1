#ifndef RANDOM_H
#define RANDOM_H
#include <random>
#include <chrono>

namespace Random {
    inline std::mt19937 get_rand_var() {
        std::random_device rd{};
        std::seed_seq ss{static_cast<std::seed_seq::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()),
                        rd(), rd(), rd(), rd(), rd(), rd(), rd()};

        return std::mt19937{ss};
    }

    inline std::mt19937 mt{get_rand_var()};

    template <typename T>
    inline T get_rand(T min, T max) {
        return std::uniform_real_distribution<T>{min, max}(mt);
    }
}

#endif

