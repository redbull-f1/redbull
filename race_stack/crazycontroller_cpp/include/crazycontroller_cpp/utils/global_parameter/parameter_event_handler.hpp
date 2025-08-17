// parameter_event_handler.hpp
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_event.hpp>
#include <rcl_interfaces/msg/parameter.hpp>

#include <map>
#include <vector>
#include <mutex>
#include <string>
#include <functional>
#include <optional>
#include <algorithm>

class ParameterEventHandler {
public:
  using Parameter      = rcl_interfaces::msg::Parameter;
  using ParameterEvent = rcl_interfaces::msg::ParameterEvent;

  struct ParameterCallbackHandle {
    std::string parameter_name;
    std::string node_name;
    std::function<void(const Parameter&)> callback;
    std::mutex mutex;

    ParameterCallbackHandle(std::string param, std::string node,
                            std::function<void(const Parameter&)> cb)
      : parameter_name(std::move(param)), node_name(std::move(node)), callback(std::move(cb)) {}
  };

  struct ParameterEventCallbackHandle {
    std::function<void(const ParameterEvent&)> callback;
    std::mutex mutex;

    explicit ParameterEventCallbackHandle(std::function<void(const ParameterEvent&)> cb)
      : callback(std::move(cb)) {}
  };

private:
  // Inner container that mirrors the Python nested Callbacks class
  class Callbacks {
  public:
    // key: (parameter_name, node_name)  -> FILO list of handles
    std::map<std::pair<std::string, std::string>,
             std::vector<std::shared_ptr<ParameterEventHandler::ParameterCallbackHandle>>> parameter_callbacks;
    // FILO list of event callbacks
    std::vector<std::shared_ptr<ParameterEventHandler::ParameterEventCallbackHandle>> event_callbacks;

    std::mutex mutex;

    void event_callback(const ParameterEvent::SharedPtr msg) {
      std::scoped_lock lk(mutex);
      // Parameter-specific callbacks
      for (auto &entry : parameter_callbacks) {
        const auto &key = entry.first;
        auto &list = entry.second;
        const std::string &param_name = key.first;
        const std::string &node_name  = key.second;

        if (auto opt_param = ParameterEventHandler::get_parameter_from_event(*msg, param_name, node_name)) {
          for (auto &handle : list) {
            std::scoped_lock hlock(handle->mutex);
            handle->callback(*opt_param);
          }
        }
      }
      // Event-wide callbacks
      for (auto &eh : event_callbacks) {
        std::scoped_lock hlock(eh->mutex);
        eh->callback(*msg);
      }
    }

    std::shared_ptr<ParameterCallbackHandle> add_parameter_callback(
        const std::string &parameter_name,
        const std::string &node_name,
        std::function<void(const Parameter&)> callback) {
      auto handle = std::make_shared<ParameterCallbackHandle>(parameter_name, node_name, std::move(callback));
      std::scoped_lock lk(mutex);
      auto key = std::make_pair(parameter_name, node_name);
      auto &vec = parameter_callbacks[key];
      // FILO: insert at front
      vec.insert(vec.begin(), handle);
      return handle;
    }

    void remove_parameter_callback(const std::shared_ptr<ParameterCallbackHandle> &handle) {
      std::scoped_lock lk(mutex);
      auto key = std::make_pair(handle->parameter_name, handle->node_name);
      auto it = parameter_callbacks.find(key);
      if (it == parameter_callbacks.end()) {
        throw std::runtime_error("Callback doesn't exist");
      }
      auto &vec = it->second;
      auto vit = std::find(vec.begin(), vec.end(), handle);
      if (vit == vec.end()) {
        throw std::runtime_error("Callback doesn't exist");
      }
      vec.erase(vit);
      if (vec.empty()) {
        parameter_callbacks.erase(it);
      }
    }

    std::shared_ptr<ParameterEventCallbackHandle> add_parameter_event_callback(
        std::function<void(const ParameterEvent&)> callback) {
      auto handle = std::make_shared<ParameterEventCallbackHandle>(std::move(callback));
      std::scoped_lock lk(mutex);
      // FILO: insert at front
      event_callbacks.insert(event_callbacks.begin(), handle);
      return handle;
    }

    void remove_parameter_event_callback(const std::shared_ptr<ParameterEventCallbackHandle> &handle) {
      std::scoped_lock lk(mutex);
      auto it = std::find(event_callbacks.begin(), event_callbacks.end(), handle);
      if (it == event_callbacks.end()) {
        throw std::runtime_error("Callback doesn't exist");
      }
      event_callbacks.erase(it);
    }
  };

public:
  explicit ParameterEventHandler(
      const rclcpp::Node::SharedPtr &node,
      const rclcpp::QoS &qos_profile = rclcpp::ParametersQoS(),
      const rclcpp::CallbackGroup::SharedPtr &callback_group = nullptr,
      const rclcpp::SubscriptionEventCallbacks &event_callbacks = rclcpp::SubscriptionEventCallbacks(),
      const rclcpp::QosOverridingOptions &qos_overriding_options = rclcpp::QosOverridingOptions()
      // NOTE: Python version exposes 'raw' flag; here we keep typed subscription for parity of behavior.
  )
  : node_(node), qos_profile_(qos_profile) {
    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group;
    options.event_callbacks = event_callbacks;
    options.qos_overriding_options = qos_overriding_options;

    // Subscribe to '/parameter_events' with same callback chaining behavior
    parameter_event_subscription_ = node_->create_subscription<ParameterEvent>(
        "/parameter_events",
        qos_profile_,
        std::bind(&Callbacks::event_callback, &callbacks_, std::placeholders::_1),
        options);
  }

  // Mirror: destroy()
  void destroy() {
    parameter_event_subscription_.reset();
  }

  // Static helpers (1:1 with Python API)
  static std::optional<Parameter>
  get_parameter_from_event(const ParameterEvent &event,
                           const std::string &parameter_name,
                           const std::string &node_name) {
    if (event.node == node_name) {
      for (const auto &p : event.new_parameters) {
        if (p.name == parameter_name) return p;
      }
      for (const auto &p : event.changed_parameters) {
        if (p.name == parameter_name) return p;
      }
    }
    return std::nullopt;
  }

  static std::vector<Parameter>
  get_parameters_from_event(const ParameterEvent &event) {
    std::vector<Parameter> out;
    out.reserve(event.new_parameters.size() + event.changed_parameters.size());
    out.insert(out.end(), event.new_parameters.begin(), event.new_parameters.end());
    out.insert(out.end(), event.changed_parameters.begin(), event.changed_parameters.end());
    return out;
  }

  // Public API: add/remove callbacks (names kept the same)
  std::shared_ptr<ParameterCallbackHandle>
  add_parameter_callback(const std::string &parameter_name,
                         const std::string &node_name,
                         std::function<void(const Parameter&)> callback) {
    return callbacks_.add_parameter_callback(
        parameter_name,
        _resolve_path(node_name),
        std::move(callback));
  }

  void remove_parameter_callback(const std::shared_ptr<ParameterCallbackHandle> &handle) {
    callbacks_.remove_parameter_callback(handle);
  }

  std::shared_ptr<ParameterEventCallbackHandle>
  add_parameter_event_callback(std::function<void(const ParameterEvent&)> callback) {
    return callbacks_.add_parameter_event_callback(std::move(callback));
  }

  void remove_parameter_event_callback(const std::shared_ptr<ParameterEventCallbackHandle> &handle) {
    callbacks_.remove_parameter_event_callback(handle);
  }

private:
  std::string _resolve_path(const std::string &node_path) const {
    if (node_path.empty()) {
      return node_->get_fully_qualified_name();  // e.g. "/ns/my_node"
    }
    if (!node_path.empty() && node_path.front() == '/') {
      return node_path;
    }
    std::string ns = node_->get_namespace();
    while (!ns.empty() && ns.front() == '/') ns.erase(ns.begin()); // lstrip('/')
    std::string resolved = ns.empty() ? node_path : (ns + "/" + node_path);
    if (resolved.empty() || resolved.front() == '/') return resolved;
    return "/" + resolved;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::QoS qos_profile_;
  Callbacks callbacks_;
  rclcpp::Subscription<ParameterEvent>::SharedPtr parameter_event_subscription_;
};
