
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <thread>

#include "common/cmdline.hpp"
#include "common/model_profiler.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

class BuildInOperatorProf {
 public:
  explicit BuildInOperatorProf(const std::string& op_config,
                               uint32_t device_id = 0)
      : device_id_(device_id) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";
    ops_ = vsx::Operator::LoadOpsFromJsonFile(op_config);
    CHECK(ops_.size() == 1)
        << "Only support 1 BuildIn Op. Now it's " << ops_.size() << std::endl;

    buildin_op_ = static_cast<vsx::BuildInOperator*>(ops_[0].get());
    auto attri_keys = buildin_op_->GetAttrKeys();
    // for (auto key : attri_keys) {
    //   std::cout << "key: " << key << std::endl;
    // }
    auto op_type = buildin_op_->GetOpType();
    // std::cout << "op type: " << op_type << std::endl;
    CHECK(op_type != vsx::BuildInOperatorType::kBERT_EMBEDDING_OP)
        << "Unsupport operator BERT_EMBEDDING_OP";

    buildin_op_->GetAttribute<vsx::AttrKey::kIimageWidth>(iimage_width_);
    buildin_op_->GetAttribute<vsx::AttrKey::kIimageWidthPitch>(
        iimage_width_pitch_);
    buildin_op_->GetAttribute<vsx::AttrKey::kIimageHeight>(iimage_height_);
    buildin_op_->GetAttribute<vsx::AttrKey::kIimageHeightPitch>(
        iimage_height_pitch_);
    // get image format
    if (vsx::HasAttribute(attri_keys, "kIimageFormat")) {
      vsx::BuildInOperatorAttrImageType format;
      buildin_op_->GetAttribute<vsx::AttrKey::kIimageFormat>(format);
      iimage_format_ = vsx::ConvertToVsxFormat(format);
    }
    if (vsx::HasAttribute(attri_keys, "kOimageFormat")) {
      vsx::BuildInOperatorAttrImageType format;
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageFormat>(format);
      oimage_format_ = vsx::ConvertToVsxFormat(format);
    } else {
      oimage_format_ = iimage_format_;
    }

    if (op_type == vsx::BuildInOperatorType::kSINGLE_OP_CVT_COLOR) {
      vsx::BuildInOperatorAttrColorCvtCode cvtcolor_code;
      buildin_op_->GetAttribute<vsx::AttrKey::kColorCvtCode>(cvtcolor_code);
      vsx::ConvertToVsxFormat(cvtcolor_code, iimage_format_, oimage_format_);
    }
    oimage_count_ = 1;
    if (op_type == vsx::BuildInOperatorType::kSINGLE_OP_BATCH_CROP_RESIZE) {
      buildin_op_->GetAttribute<vsx::AttrKey::kCropNum>(oimage_count_);
      int ow, owp, oh, ohp;
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidth>(ow);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidthPitch>(owp);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeight>(oh);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeightPitch>(ohp);

      for (int i = 0; i < oimage_count_; i++) {
        oimage_width_.push_back(ow);
        oimage_width_pitch_.push_back(owp);
        oimage_height_.push_back(oh);
        oimage_height_pitch_.push_back(ohp);
      }
    } else if (op_type == vsx::BuildInOperatorType::kSINGLE_OP_SCALE) {
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageCnt>(oimage_count_);
      for (int i = 0; i < oimage_count_; i++) {
        int ow, owp, oh, ohp;
        buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidth>(ow, i);
        buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidthPitch>(owp, i);
        buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeight>(oh, i);
        buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeightPitch>(ohp, i);
        oimage_width_.push_back(ow);
        oimage_width_pitch_.push_back(owp);
        oimage_height_.push_back(oh);
        oimage_height_pitch_.push_back(ohp);
      }
    } else if (op_type ==
               vsx::BuildInOperatorType::kSINGLE_OP_COPY_MAKE_BORDER) {
      int ow, oh;
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidth>(ow);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeight>(oh);
      oimage_width_.push_back(ow);
      oimage_height_.push_back(oh);
    } else if (op_type ==
               vsx::BuildInOperatorType::
                   kFUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR_EXT) {
      int left, right, bottom, top, width, height;
      buildin_op_->GetAttribute<vsx::AttrKey::kPaddingLeft>(left);
      buildin_op_->GetAttribute<vsx::AttrKey::kPaddingRight>(right);
      buildin_op_->GetAttribute<vsx::AttrKey::kPaddingTop>(top);
      buildin_op_->GetAttribute<vsx::AttrKey::kPaddingBottom>(bottom);
      buildin_op_->GetAttribute<vsx::AttrKey::kResizeWidth>(width);
      buildin_op_->GetAttribute<vsx::AttrKey::kResizeHeight>(height);

      oimage_width_.push_back(left + right + width);
      oimage_width_pitch_.push_back(left + right + width);
      oimage_height_.push_back(top + bottom + height);
      oimage_height_pitch_.push_back(top + bottom + height);
    } else if (op_type ==
               vsx::BuildInOperatorType::kFUSION_OP_RGB_CVTCOLOR_NORM_TENSOR) {
      oimage_width_.push_back(iimage_width_);
      oimage_height_.push_back(iimage_height_);
      iimage_format_ = vsx::BGR_PLANAR;
      oimage_format_ = vsx::RGB_PLANAR;
    } else if (op_type >= vsx::BuildInOperatorType::
                              kFUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR) {
      int ow, oh;
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidth>(ow);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeight>(oh);
      oimage_width_.push_back(ow);
      oimage_height_.push_back(oh);
    } else {
      int ow, owp, oh, ohp;
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidth>(ow);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageWidthPitch>(owp);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeight>(oh);
      buildin_op_->GetAttribute<vsx::AttrKey::kOimageHeightPitch>(ohp);

      oimage_width_.push_back(ow);
      oimage_width_pitch_.push_back(owp);
      oimage_height_.push_back(oh);
      oimage_height_pitch_.push_back(ohp);
    }

    if (op_type >=
        vsx::BuildInOperatorType::kFUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR) {
      odata_type_ = vsx::TypeFlag::kFloat16;
      oimage_format_ = vsx::ImageFormat::RGB_PLANAR;
    }

    // PrintParams();
  }
  void PrintParams() {
    std::cout << "iimage_width_:" << iimage_width_ << std::endl;
    std::cout << "iimage_width_pitch_:" << iimage_width_pitch_ << std::endl;
    std::cout << "iimage_height_:" << iimage_height_ << std::endl;
    std::cout << "iimage_height_pitch_:" << iimage_height_pitch_ << std::endl;
    std::cout << "oimage_count_:" << oimage_count_ << std::endl;

    std::cout << "oimage_width_: [ ";
    for (auto& num : oimage_width_) std::cout << num << " ";
    std::cout << "]\n";

    std::cout << "oimage_width_pitch_: [ ";
    for (auto& num : oimage_width_pitch_) std::cout << num << " ";
    std::cout << "]\n";

    std::cout << "oimage_height_: [ ";
    for (auto& num : oimage_height_) std::cout << num << " ";
    std::cout << "]\n";

    std::cout << "oimage_height_pitch_: [ ";
    for (auto& num : oimage_height_pitch_) std::cout << num << " ";
    std::cout << "]\n";

    std::cout << "iimage_format_:" << vsx::ImageFormatToString(iimage_format_)
              << std::endl;
    std::cout << "oimage_format_:" << vsx::ImageFormatToString(oimage_format_)
              << std::endl;

    std::cout << "odata_type_:" << odata_type_ << std::endl;
  }
  int Process(vsx::Image& image) {
    std::vector<vsx::Image> images = {image};
    return Process(images);
  }
  int Process(std::vector<vsx::Image>& images) { return ProcessImpl(images); }

  std::vector<vsx::Image> GetTestData(
      uint32_t bsize, uint32_t dtype, const vsx::Context& context,
      const std::vector<vsx::TShape>& input_shapes) {
    const auto& input_shape = input_shapes[0];
    std::vector<vsx::Image> images;
    images.reserve(bsize);
    int width, height;
    CHECK(input_shape.ndim() >= 2);
    height = input_shape[input_shape.ndim() - 2];
    width = input_shape[input_shape.ndim() - 1];
    auto image = vsx::Image(iimage_format_, width, height, context);
    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(image);
    }

    // make device memory image for output
    for (size_t i = 0; i < images.size(); i++) {
      for (int n = 0; n < oimage_count_; n++) {
        auto image_vacc =
            vsx::Image(oimage_format_, oimage_width_[n], oimage_height_[n],
                       vsx::Context::VACC(device_id_), 0, 0, odata_type_);
        output_images_vacc_.push_back(image_vacc);
      }
    }

    return images;
  }

 protected:
  virtual int ProcessImpl(const std::vector<vsx::Image>& images) {
    std::vector<vsx::Image> input_images_vacc;

    for (const auto& image : images) {
      if (image.GetContext().dev_type == vsx::Context::kCPU) {
        auto image_vacc =
            vsx::Image(image.Format(), image.Width(), image.Height(),
                       vsx::Context::VACC(device_id_));
        image_vacc.CopyFrom(image);
        input_images_vacc.push_back(image_vacc);
      } else {
        input_images_vacc.push_back(image);
      }
    }
    // // make device memory image for output
    // std::vector<vsx::Image> output_images_vacc;
    // for (size_t i = 0; i < images.size(); i++) {
    //   for (int n = 0; n < oimage_count_; n++) {
    //     auto image_vacc =
    //         vsx::Image(oimage_format_, oimage_width_[n], oimage_height_[n],
    //                    vsx::Context::VACC(device_id_), 0, 0, odata_type_);
    //     output_images_vacc.push_back(image_vacc);
    //   }
    // }

    // buildin operator execute
    buildin_op_->Execute({input_images_vacc}, output_images_vacc_);

    return 0;
  }

 public:
  int iimage_width_, iimage_width_pitch_, iimage_height_, iimage_height_pitch_;
  int oimage_count_ = 1;
  std::vector<int> oimage_width_, oimage_width_pitch_;
  std::vector<int> oimage_height_, oimage_height_pitch_;
  vsx::ImageFormat iimage_format_ = vsx::ImageFormat::YUV_NV12, oimage_format_;
  vsx::TypeFlag odata_type_ = vsx::TypeFlag::kUint8;

 protected:
  std::vector<std::shared_ptr<vsx::Operator>> ops_;
  vsx::BuildInOperator* buildin_op_ = nullptr;
  uint32_t device_id_;
  std::vector<vsx::Image> output_images_vacc_;
};

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<std::string>("op_config", '\0', "build in op config json", true);
  args.add<uint32_t>("instance", 'i',
                     "instance number or range for each device", false, 1);
  args.add<int>("iterations", '\0', "iterations count for one profiling", false,
                10240);
  args.add<std::string>("percentiles", '\0', "percentiles of latency", false,
                        "[50, 90, 95, 99]");
  args.add<bool>("input_host", '\0', "cache input data into host memory", false,
                 0);
  args.add<uint32_t>("batch_size", '\0', "batch_size", false, 1);
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto op_config = args.get<std::string>("op_config");
  auto input_host = args.get<bool>("input_host");
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));

  auto iterations = args.get<int>("iterations");
  auto instance = args.get<uint32_t>("instance");
  auto batch_size = args.get<uint32_t>("batch_size");

  uint32_t queue_size = 0;

  std::vector<std::shared_ptr<BuildInOperatorProf>> ops;
  ops.reserve(instance);
  std::vector<vsx::Context> contexts;
  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    if (input_host) {
      contexts.push_back(vsx::Context::CPU());
    } else {
      contexts.push_back(vsx::Context::VACC(device_id));
    }
    ops.push_back(std::make_shared<BuildInOperatorProf>(op_config, device_id));
  }
  int width = ops[0]->iimage_width_;
  int height = ops[0]->iimage_height_;
  vsx::TShape input_shape({3, height, width});
  vsx::ProfilerConfig config = {instance,      iterations,  batch_size,
                                vsx::kInt32,   device_ids,  contexts,
                                {input_shape}, percentiles, queue_size};
  vsx::ModelProfiler<BuildInOperatorProf> profiler(config, ops);
  std::cout << profiler.Profiling() << std::endl;

  return 0;
}