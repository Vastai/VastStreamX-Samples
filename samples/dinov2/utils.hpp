
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "vaststreamx/vaststreamx.h"
inline float compute_ap(const std::vector<int>& ranks, int nres) {
  int nimgranks = ranks.size();
  float ap = 0;
  float recall_step = 1.0 / nres;
  float precision_0 = 0;
  float precision_1 = 0;
  for (int j = 0; j < nimgranks; j++) {
    int rank = ranks[j];
    if (rank == 0) {
      precision_0 = 1.0;
    } else {
      precision_0 = j * 1.0 / rank;
    }
    precision_1 = (j + 1) * 1.0 / (rank + 1);
    ap += (precision_0 + precision_1) * recall_step / 2.0;
  }
  return ap;
}

inline void compute_map(
    const vsx::Tensor& ranks,
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>& gnd,
    const std::vector<int>& kappas, float& map, std::vector<float>& aps,
    std::vector<float>& pr, std::vector<std::vector<float>>& prs) {
  map = 0.0;
  int nq = gnd.size();
  aps.resize(nq, 0.f);
  pr.resize(kappas.size(), 0.f);
  prs.resize(nq, pr);
  int nempty = 0;
  int kappas_len = kappas.size();
  int height = ranks.Shape()[0];
  int width = ranks.Shape()[1];
  auto ranks_data = ranks.Data<int>();

  for (int i = 0; i < nq; i++) {
    auto qgnd = gnd[i].first;
    if (qgnd.size() == 0) {
      aps[i] = std::numeric_limits<float>::quiet_NaN();
      for (int j = 0; j < kappas_len; i++) {
        prs[i][j] = std::numeric_limits<float>::quiet_NaN();
      }
      nempty += 1;
      continue;
    }

    auto qgndj = gnd[i].second;
    std::vector<int> pos;
    std::vector<int> junk;
    for (int h = 0; h < height; h++) {
      int rank = ranks_data[h * width + i];
      for (size_t w = 0; w < qgnd.size(); w++) {
        if (rank == qgnd[w]) {
          pos.push_back(h);
          break;
        }
      }
      for (size_t w = 0; w < qgndj.size(); w++) {
        if (rank == qgndj[w]) {
          junk.push_back(h);
          break;
        }
      }
    }

    int k = 0;
    size_t ij = 0;
    if (junk.size() > 0) {
      size_t ip = 0;
      while (ip < pos.size()) {
        while (ij < junk.size() && pos[ip] > junk[ij]) {
          k += 1;
          ij += 1;
        }
        pos[ip] = pos[ip] - k;
        ip += 1;
      }
    }

    // compute ap
    auto ap = compute_ap(pos, qgnd.size());
    map = map + ap;
    aps[i] = ap;

    // compute precision @ k
    int pos_max = pos[0];
    for (size_t s = 0; s < pos.size(); s++) {
      pos[s] += 1;
      if (pos[s] > pos_max) pos_max = pos[s];
    }
    for (int j = 0; j < kappas_len; j++) {
      auto kq = std::min(pos_max, kappas[j]);
      int sum = 0;
      for (size_t s = 0; s < pos.size(); s++) {
        if (pos[s] <= kq) sum++;
      }
      prs[i][j] = sum * 1.0 / kq;
    }
    for (int j = 0; j < kappas_len; j++) {
      pr[j] += prs[i][j];
    }
  }

  map = map / (nq - nempty);
  for (int j = 0; j < kappas_len; j++) {
    pr[j] = pr[j] / (nq - nempty);
  }
}
