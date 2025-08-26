
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_npz", '\0', "input npz file", false, "");
  args.add<std::string>("output_npz", '\0', "output npz file", false,
                        "tensor_out.npz");
  args.parse_check(argc, argv);
  return args;
}

std::vector<vsx::Tensor> ReadTensorsFromFile(const std::string& tensor_file) {
  auto tensor_map = vsx::LoadTensorMap(tensor_file);
  std::vector<vsx::Tensor> input_tensors;
  input_tensors.reserve(tensor_map.size());
  for (auto& pair : tensor_map) {
    input_tensors.push_back(pair.second);
  }
  return input_tensors;
}

int WriteTensorsToFile(const std::vector<vsx::Tensor>& tensors,
                       const std::string& tensor_file) {
  std::unordered_map<std::string, vsx::Tensor> output_map;
  int index = 0;
  for (const auto& tensor : tensors) {
    std::stringstream key;
    key << "input_" << index++;
    output_map[key.str()] = tensor;
  }
  vsx::SaveTensorMap(tensor_file, output_map);
  return 0;
}

int ChangeData(vsx::Tensor& tensor_cpu) {
  if (tensor_cpu.GetContext().dev_type != vsx::Context::kCPU) {
    LOG(ERROR) << "tensor_cpu memory is not in host.";
    return -1;
  }
  vsx::TShape shape = tensor_cpu.Shape();
  if (tensor_cpu.GetDType() == vsx::TypeFlag::kUint8) {
    uint8_t* data = tensor_cpu.MutableData<uint8_t>();
    for (int i = 0; i < 10; i++) {
      data[i] = (uint8_t)i;
    }
  } else if (tensor_cpu.GetDType() == vsx::TypeFlag::kFloat16) {
    uint16_t* data = tensor_cpu.MutableData<uint16_t>();
    for (int i = 0; i < 10; i++) {
      data[i] = static_cast<uint16_t>(i);
    }
  } else if (tensor_cpu.GetDType() == vsx::TypeFlag::kFloat32) {
    float* data = tensor_cpu.MutableData<float>();
    for (int i = 0; i < 10; i++) {
      data[i] = i;
    }
  }
  return 0;
}

std::string DTypeToString(int dtype) {
  switch (dtype) {
    case vsx::kUint8:
      return "uint8";
    case vsx::kInt8:
      return "int8";
    case vsx::kUint16:
      return "uint16";
    case vsx::kInt16:
      return "int16";
    case vsx::kUint32:
      return "uint32";
    case vsx::kInt32:
      return "int32";
    case vsx::kFloat16:
      return "float16";
    case vsx::kFloat32:
      return "float32";
    case vsx::kBfloat16:
      return "bfloat16";
    case vsx::kDTypeAny:
      return "dtype_any";
    default:
      return "unsupport data type";
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  std::string input_npz = args.get<std::string>("input_npz");
  std::string output_npz = args.get<std::string>("output_npz");
  // init env
  vsx::SetDevice(device_id);
  // read image from file
  auto input_tensors = ReadTensorsFromFile(input_npz);
  std::cout << "There are " << input_tensors.size()
            << " tensors in npz file.\n";
  for (size_t i = 0; i < input_tensors.size(); i++) {
    std::cout << "The " << i
              << "th tensor shape is: " << input_tensors[i].Shape()
              << ", data type is: "
              << DTypeToString(input_tensors[i].GetDType()) << std::endl;
  }
  // copy to device
  std::vector<vsx::Tensor> tensors_vacc;
  tensors_vacc.reserve(input_tensors.size());
  for (auto& tensor_cpu : input_tensors) {
    vsx::Tensor tensor_vacc = tensor_cpu.Clone(vsx::Context::VACC(device_id));
    tensors_vacc.push_back(tensor_vacc);
  }

  // copy to host
  std::vector<vsx::Tensor> tensors_cpu;
  tensors_vacc.reserve(input_tensors.size());
  for (auto& tensor_vacc : tensors_vacc) {
    vsx::Tensor tensor_cpu = tensor_vacc.Clone(vsx::Context::CPU());
    tensors_cpu.push_back(tensor_cpu);
  }

  // change data
  ChangeData(tensors_cpu[0]);

  // convert data type
  std::vector<vsx::Tensor> output_tensors;
  if (tensors_cpu[0].GetDType() == vsx::TypeFlag::kFloat16) {
    output_tensors = vsx::ConvertTensorFromFp16ToFp32(tensors_cpu);
  } else if (tensors_cpu[0].GetDType() == vsx::TypeFlag::kFloat32) {
    output_tensors = vsx::ConvertTensorFromFp32ToFp16(tensors_cpu);
  } else {
    output_tensors = tensors_cpu;
  }
  // write to file
  WriteTensorsToFile(output_tensors, output_npz);
  return 0;
}
