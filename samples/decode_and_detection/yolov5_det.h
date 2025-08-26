
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
  float x;
  float y;
  float width;
  float height;
  float score;
  int classId;
} ObjectData;

typedef struct {
  std::vector<ObjectData> objects;
  int32_t obj_nums;
  int64_t timestamp;
  int64_t channelId;
  int64_t frameId;  // frameID
} DetecResultInfo;