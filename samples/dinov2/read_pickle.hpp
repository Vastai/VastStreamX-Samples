
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>
namespace py = pybind11;

struct Gnd {
  std::vector<float> box;
  std::vector<int> easy;
  std::vector<int> hard;
  std::vector<int> junk;
};

struct PickleData {
  std::vector<Gnd> gnd;
  std::vector<std::string> qimlist;
  std::vector<std::string> imlist;
};

inline PickleData ReadPickleFile(const std::string &pickle_file) {
  py::initialize_interpreter();
  PickleData pickle_data;
  try {
    py::module_ pickle_module = py::module_::import("read_pickle");
    py::object read_pickle_func = pickle_module.attr("read_pickle_file");

    py::object data = read_pickle_func(pickle_file);

    if (py::isinstance<py::dict>(data)) {
      py::dict data_dict = data.cast<py::dict>();
      // qimlist
      auto qimlist = data_dict["qimlist"].cast<py::list>();
      for (auto &name : qimlist) {
        pickle_data.qimlist.emplace_back(name.cast<std::string>());
      }
      // imlist
      auto imlist = data_dict["imlist"].cast<py::list>();
      for (auto &name : imlist) {
        pickle_data.imlist.emplace_back(name.cast<std::string>());
      }

      // gnd
      auto gnd_list = data_dict["gnd"].cast<py::list>();
      for (auto &g : gnd_list) {
        Gnd gnd;
        auto bbx = g["bbx"].cast<py::list>();
        for (auto &b : bbx) {
          gnd.box.push_back(b.cast<float>());
        }
        auto easy = g["easy"].cast<py::list>();
        for (auto &e : easy) {
          gnd.easy.push_back(e.cast<int>());
        }
        auto hard = g["hard"].cast<py::list>();
        for (auto &e : hard) {
          gnd.hard.push_back(e.cast<int>());
        }
        auto junk = g["junk"].cast<py::list>();
        for (auto &e : junk) {
          gnd.junk.push_back(e.cast<int>());
        }
        pickle_data.gnd.push_back(std::move(gnd));
      }

    } else {
      std::cerr << "Pickle file does not contain a dictionary." << std::endl;
    }
  } catch (const py::error_already_set &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
  }

  py::finalize_interpreter();
  return pickle_data;
}
