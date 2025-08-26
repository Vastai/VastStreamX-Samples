
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/custom_op_base.hpp"
#include "opencv2/opencv.hpp"

namespace vsx {
uint32_t getInputCount(const char* op_name) { return 1; }
uint32_t getOutputCount(const char* op_name) { return 1; }

vsx::CustomOperatorCallback callback{
    getInputCount, nullptr, getOutputCount, nullptr, nullptr, nullptr, 0, 0};

enum ColorSpace_enum {
  COLOR_SPACE_BT709,
  COLOR_SPACE_BT601,
  COLOR_SPACE_BUTT  // only means end of the enum
};
typedef enum {
  INT8,
  FP16,
  FP32,  // only used for input
  BF16,  // only used for output
} NT_DTYPE;

typedef union {
  uint16_t fp16;  // datatype fp16; used when out_dtype is FP16
  uint16_t bf16;  // datatype fp16; used when out_dtype is BF16
  float fp32;     // used when out_dtype is FP32
  int8_t int8;    // used when out_dtype is INT8
} value_t;

typedef enum {
  _XI_TILE_YUV_NV12_TYPE_ = 0,
  _XI_TILE_YUV_I420_TYPE_,
  _XI_TILE_RGB888_TYPE_,
  _XI_TILE_RGB_PLANAR_TYPE_,

  _XI_TILE_BAYER_BG_TYPE_,
  _XI_TILE_BAYER_GB_TYPE_,
  _XI_TILE_BAYER_RG_TYPE_,
  _XI_TILE_BAYER_GR_TYPE_,

  _XI_TILE_GRAY_TYPE_,
} img_format_t;

/**
 * @brief common_img_obj_t struct
 * @attention pitch is in Byte unit
 *
 */
typedef struct {
  img_format_t img_type;
  uint16_t cspace;  ///< color space, ColorSpace_enum
  int32_t width;
  int32_t height;

  union {
    struct {
      uint64_t addr[3];  ///< rgb planar:  0:R, 1:G, 2:B
      int32_t pitch[3];
    } rgb_planar;

    struct {
      uint64_t addr[2];  ///< yuv_nv12:  0:Y, 1:NV
      int32_t pitch_y;
      int32_t pitch_uv;
    } yuv_nv12;

    struct {
      uint64_t addr[3];  ///< yuv_i420:  0:Y, 1:U, 2:V
      int32_t pitch_y;
      int32_t pitch_u;
      int32_t pitch_v;
    } yuv_i420;

    struct {
      uint64_t addr[1];  ///< rgb_888:  0:RGB
      int32_t pitch;
    } rgb_888;

    struct {
      uint64_t addr[1];  ///< gray:  0:Gray
      int32_t pitch;
    } gray;
  } ptr;
} common_img_obj_t;

typedef enum {
  NORMAL_EQUAL,             ///< R = R*scale_R; R/G/B is the same process as R;
  NORMAL_MINUSMEAN_DIVSTD,  ///< R = R*scale_R/std -mean_R*scale_R/std; R/G/B is
                            ///< the same process as R;
  NORMAL_DIV255_MINUSMEAN_DIVSTD,  ///< R = R*scale_R/(255*std)
                                   ///< -mean_R*scale_R/std; R/G/B is the same
                                   ///< process as R;
  NORMAL_DIV127_5_MINUSONE,  ///< R = R*scale_R/127.5 -1*scale_R; R/G/B is the
                             ///< same process as R;
  NORMAL_DIV255,  ///< R = R*scale_R/255; R/G/B is the same process as R;
} normal_type_t;

typedef struct {
  common_img_obj_t in_image;  // see resize_normal_quant_api.h for details
  uint64_t dst;               // dst address
  uint32_t out_width;         // should >= in_width
  uint32_t out_height;        // should >= in_height
  uint32_t ch_pitch;          // output channel pitch, should be a multiple of
                              // interleave_width * interleave_height

  // Supported combinations of input and output types:
  //   INT8 -> INT8: optional normalization and quantization, depending on the
  //   parameter skip_norma_quant INT8 -> FP16: always normalize FP16 -> INT8:
  //   always quantize FP16 -> FP16: no normalization or quantization FP32 ->
  //   INT8: convert to FP16 and then quantize FP32 -> FP16: convert to FP16
  //   FP32 -> BF16: convert to BF16
  NT_DTYPE in_dtype;   // Supported: INT8, FP16, FP32
  NT_DTYPE out_dtype;  // Supported: INT8, FP16, BF16

  normal_type_t norma_type;  // see resize_normal_quant_api.h for details

  // normalization parameter
  uint16_t mean[3];  // datatype fp16
  uint16_t std[3];   // datatype fp16

  // quantization parameter, required when out_dtype is INT8
  uint16_t scale[3];  // datatype fp16

  // Block size for interleave (or space-to-depth) in tensorization
  int32_t interleave_width;   // 0 means no interleave
  int32_t interleave_height;  // 0 means no interleave

  // whether to skip normalization and quantization; 0: no skip; 1: skip; only
  // used when in_dtype and out_dtype are both INT8
  int32_t skip_norma_quant;
  int32_t is_need_swap_blue;  // whether to swap blue and red channel; 0: no
                              // swap; 1: swap

  value_t
      padding;  // padding value; its type should be consistent with out_dtype
} nt_3ch_para_t;

class NormaTensor3ChOp : public CustomOpBase {
 public:
  NormaTensor3ChOp(const std::string& op_name, const std::string& elf_file,
                   uint32_t device_id = 0)
      : CustomOpBase(op_name, elf_file, device_id) {
    custom_op_->SetCallback(callback);
  }
  vsx::Tensor Process(const vsx::Tensor& tensor) {
    std::vector<vsx::Tensor> tensors = {tensor};
    return Process(tensors)[0];
  }

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Tensor>& tensors) {
    return ProcessImpl(tensors);
  }
  std::vector<vsx::Tensor> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    const auto& input_shape = input_shapes[0];
    std::vector<vsx::Tensor> images;
    images.reserve(bsize);

    auto tensor = vsx::Tensor(input_shape, context, dtype);
    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(tensor);
    }
    return images;
  }

 protected:
  virtual std::vector<vsx::Tensor> ProcessImpl(
      const std::vector<vsx::Tensor>& tensors) {
    std::vector<vsx::Tensor> results;
    for (const auto tensor : tensors) {
      auto shape = tensor.Shape();
      CHECK(shape.ndim() >= 3);
      auto width = shape[shape.ndim() - 1];
      auto height = shape[shape.ndim() - 2];
      size_t plane_offset = width * height;

      vsx::Tensor input_tensor;
      if (tensor.GetContext().dev_type == vsx::Context::kVACC) {
        input_tensor = tensor;
      } else {
        input_tensor = tensor.Clone(vsx::Context::VACC(device_id_));
      }
      vsx::Tensor output_tensor(shape, vsx::Context::VACC(device_id_),
                                vsx::TypeFlag::kFloat16);

      char* src_addr = input_tensor.MutableData<char>();
      char* dst_addr = output_tensor.MutableData<char>();
      uint64_t addr_r = reinterpret_cast<uint64_t>(src_addr);
      uint64_t addr_g = reinterpret_cast<uint64_t>(src_addr + plane_offset);
      uint64_t addr_b = reinterpret_cast<uint64_t>(src_addr + 2 * plane_offset);
      nt_3ch_para_t op_params;

      op_params.in_image.img_type = _XI_TILE_RGB_PLANAR_TYPE_;
      op_params.in_image.cspace = COLOR_SPACE_BT601;
      op_params.in_image.width = width;
      op_params.in_image.height = height;
      op_params.in_image.ptr.rgb_planar.addr[0] = addr_r;
      op_params.in_image.ptr.rgb_planar.addr[1] = addr_g;
      op_params.in_image.ptr.rgb_planar.addr[2] = addr_b;
      op_params.in_image.ptr.rgb_planar.pitch[0] = width;
      op_params.in_image.ptr.rgb_planar.pitch[1] = width;
      op_params.in_image.ptr.rgb_planar.pitch[2] = width;

      op_params.dst = reinterpret_cast<uint64_t>(dst_addr);
      op_params.out_width = width;
      op_params.out_height = height;
      op_params.ch_pitch = 4;
      op_params.in_dtype = INT8;
      op_params.out_dtype = FP16;
      op_params.norma_type = NORMAL_DIV255_MINUSMEAN_DIVSTD;
      op_params.mean[0] = 22520;
      op_params.mean[1] = 22520;
      op_params.mean[2] = 22520;
      op_params.std[0] = 15360;
      op_params.std[1] = 15360;
      op_params.std[2] = 15360;
      op_params.interleave_width = 0;
      op_params.interleave_height = 0;
      op_params.skip_norma_quant = 0;
      op_params.is_need_swap_blue = 0;
      op_params.padding.fp16 = 0;

      std::vector<vsx::Tensor> outputs{output_tensor};
      std::vector<vsx::Tensor> inputs{input_tensor};

      custom_op_->RunSync(inputs, outputs, &op_params, sizeof(nt_3ch_para_t));

      results.push_back(outputs[0]);
    }
    return results;
  }
};
}  // namespace vsx