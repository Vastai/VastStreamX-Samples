
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <iostream>
#include <vector>

typedef struct {
  int32_t x;
  int32_t y;
  int32_t w;
  int32_t h;
  float score;
  int classId;
  std::vector<float> keyPoints;
} PoseResult;

typedef struct {
  std::vector<PoseResult> poses;
  int32_t posesNums;
  int64_t timestamp;
  int64_t channelId;
  int64_t frameId;  // frameID
} PoseResultInfo;