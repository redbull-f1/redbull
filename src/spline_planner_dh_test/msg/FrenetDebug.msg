# Frenet coordinate debugging information
std_msgs/Header header

# Object identification
int32 id
string object_type  # "ego" or "obstacle"

# Cartesian coordinates
float64 x_cartesian
float64 y_cartesian

# Frenet coordinates  
float64 s_frenet
float64 d_frenet

# Velocities (if applicable)
float64 vx_cartesian
float64 vy_cartesian
float64 vs_frenet
float64 vd_frenet

# Additional info
float64 size  # for obstacles
float64 yaw

