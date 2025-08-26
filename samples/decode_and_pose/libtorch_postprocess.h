
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <vector>

#include "vaststreamx/vaststreamx.h"
#include "yolov8_pose.h"

std::vector<PoseResult> post_process(
    const std::vector<vsx::Tensor>& infer_output,
    const std::vector<int64_t>& model_size, uint32_t num_output,
    const std::vector<vsx::TShape>& output_shape,
    const vsx::TShape& image_shape, float conf_thres = 0.001,
    float iou_thres = 0.65);
