#include <rclcpp/rclcpp.hpp>
#include <crazycontroller_cpp/controller_manager.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<crazycontroller_cpp::CrazyController>();

  // Python과 동일하게: 필요한 메시지가 도착할 때까지 블록
  node->wait_for_messages();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
