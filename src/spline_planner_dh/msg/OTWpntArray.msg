# Array of overtaking waypoints for local trajectory
std_msgs/Header header
OTWpnt[] waypoints
string ot_side        # "left" or "right"
string ot_line        # "left" or "right" - outside line
bool side_switch      # true if switching sides
builtin_interfaces/Time last_switch_time
