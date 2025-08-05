# redbull
- merge_bags.py : "/scan", "/car_state/odom", "/tf", "/tf_static"을 받아오는 ego의 토픽들과 "/opp/car_state/odom" 을 받아오는 opp의 topic들의 bag 파일을 시작과 끝 시점을 맞춰 주어 하나의 bag파일로 만들어 주는 코드 

- parse_bag_ros2.py : parse_bag의 코드를 참고하여 합쳐진 bag파일에서 odom의 topic의 time sync를 맞춰서 opp의 global 위치를 ego의 lidar좌표계로 변환하는 코드 