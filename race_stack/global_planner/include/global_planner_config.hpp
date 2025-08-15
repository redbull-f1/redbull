#ifndef __GLOBAL_PLANNER_CONFIG_HPP__
#define __GLOBAL_PLANNER_CONFIG_HPP__
#pragma once

// STD Header
#include <string>
#include <cmath>

typedef struct {
    std::string vehicle_namespace{""};
    std::string trajectory_csv_file{"redbull_1.csv"};
    double loop_rate_hz{40.0};
} GlobalPlannerConfig;

#endif // __GLOBAL_PLANNER_CONFIG_HPP__