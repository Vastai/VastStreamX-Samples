
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "glog/logging.h"
#include "vaststreamx/vaststreamx.h"
namespace vsx {

class CustomOpBase {
 public:
  CustomOpBase(const std::string& op_name, const std::string& elf_file,
               uint32_t device_id = 0)
      : device_id_(device_id) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";
    custom_op_ = std::make_shared<vsx::CustomOperator>(op_name, elf_file);
  }

 protected:
  std::shared_ptr<vsx::CustomOperator> custom_op_;
  uint32_t device_id_;
};
}  // namespace vsx
