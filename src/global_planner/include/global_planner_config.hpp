#ifndef __GLOBAL_PLANNER_CONFIG_HPP__
#define __GLOBAL_PLANNER_CONFIG_HPP__
#pragma once

// STD Header
#include <string>
#include <cmath>

typedef struct {
    std::string vehicle_namespace{""};
    double loop_rate_hz{100.0};

} GlobalPlannerConfig;

#endif // __GLOBAL_PLANNER_CONFIG_HPP__