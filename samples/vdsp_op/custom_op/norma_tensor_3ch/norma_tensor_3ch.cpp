
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "norma_tensor_3ch_op.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("elf_file", '\0', "elf file path", false,
                        "/opt/vastai/vaststreamx/data/elf/norma_tensor_3ch");
  args.add<std::string>("input_shape", '\0', "input_shape [c,h,w]", false,
                        "[3,640,640]");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);

  uint32_t device_id = args.get<uint32_t>("device_id");
  auto elf_file = args.get<std::string>("elf_file");
  auto input_shape = vsx::ParseShape(args.get<std::string>("input_shape"));

  auto argmax_op =
      vsx::NormaTensor3ChOp("norma_tensor_3ch_op", elf_file, device_id);
  auto input_tensor =
      vsx::Tensor(input_shape, vsx::Context::CPU(), vsx::TypeFlag::kInt8);

  auto output_tensor_vacc = argmax_op.Process(input_tensor);
  auto output_shape = output_tensor_vacc.Shape();
  std::cout << "output tensor shape:" << output_shape << std::endl;
  return 0;
}