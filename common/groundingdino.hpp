
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <tuple>

#include "cblas.h"
#include "model_cv2.hpp"
#include "model_nlp.hpp"
#include "utils.hpp"

namespace vsx {
class GroundingDinoText : public ModelNLP {
 public:
  GroundingDinoText(const std::string& model_prefix,
                    const std::string& vdsp_config, uint32_t batch_size = 1,
                    uint32_t device_id = 0, const std::string& hw_config = "")
      : ModelNLP(model_prefix, vdsp_config, batch_size, device_id, hw_config,
                 vsx::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE) {}
  std::vector<std::vector<vsx::Tensor>> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    std::vector<std::vector<vsx::Tensor>> tensors;
    tensors.reserve(bsize);
    CHECK(test_tensors_.size()) << "Test data is empty, call SetCPUTestData "
                                   "api before profile clip_text model.";
    if (context.dev_type == vsx::Context::kVACC) {
      for (uint32_t i = 0; i < bsize; i++) {
        std::vector<vsx::Tensor> tmp_vacc;
        for (auto tensor : test_tensors_) {
          tmp_vacc.push_back(std::move(tensor.Clone(context)));
        }
        tensors.push_back(tmp_vacc);
      }
    } else {
      for (uint32_t i = 0; i < bsize; i++) {
        tensors.push_back(test_tensors_);
      }
    }
    return tensors;
  }

  int SetCPUTestData(const std::vector<vsx::Tensor>& test_tensors) {
    std::vector<vsx::Tensor> empty;
    std::swap(empty, test_tensors_);
    for (const auto& tensor : test_tensors) {
      CHECK(tensor.GetContext().dev_type != vsx::Context::kVACC);
      test_tensors_.push_back(tensor);
    }
    return 0;
  }

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<std::vector<vsx::Tensor>>& tensors) {
    auto outputs = stream_->RunSync(tensors);
    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
      std::vector<vsx::Tensor> result;
      result.reserve(output.size());
      for (const auto& out : output) {
        result.push_back(out.Clone());
      }
      results.push_back(result);
    }
    return results;
  }
  std::vector<vsx::Tensor> test_tensors_;
};

class GroundingDinoImage : public ModelCV2 {
 public:
  GroundingDinoImage(const std::string& model_prefix,
                     const std::string& vdsp_config, uint32_t batch_size = 1,
                     uint32_t device_id = 0, const std::string& hw_config = "")
      : ModelCV2(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (const auto& outs : outputs) {
      std::vector<vsx::Tensor> res;
      res.reserve(outs.size());
      for (const auto& out : outs) {
        res.push_back(out.Clone());
      }
      results.push_back(res);
    }
    return results;
  }
};

class GroundingDinoDecoder {
 public:
  GroundingDinoDecoder(
      const std::string& model_prefix, uint32_t batch_size = 1,
      uint32_t device_id = 0, const std::string& hw_config = "",
      vsx::GraphOutputType output_type = vsx::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE)
      : device_id_(device_id), batch_size_(batch_size) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";

    model_ = std::make_shared<vsx::Model>(model_prefix, batch_size_, hw_config);
    model_op_ = std::make_shared<vsx::ModelOperator>(model_);
    graph_ = std::make_shared<vsx::Graph>(output_type);
    CHECK(graph_->AddOperators({model_op_}) == 0)
        << "graph AddOperators failed";
    stream_ =
        std::make_shared<vsx::Stream>(graph_, vsx::StreamBalanceMode::kBM_RUN);
    CHECK(stream_->RegisterModelOperatorOutput(model_op_) == 0)
        << "stream RegisterModelOperatorOutput failed";
    CHECK(stream_->Build() == 0) << "stream Build failed";

    model_->GetInputShapeByIndex(0, model_input_shape_);
  }

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Tensor>& img_encoded,
                                   const std::vector<vsx::Tensor>& txt_encoded,
                                   const std::vector<vsx::Tensor>& tokens,
                                   const vsx::Tensor& attention_mask,
                                   const vsx::Tensor& text_token_mask) {
    auto img_enc = vsx::Tensor(vsx::TShape({1, 22223, 256}),
                               vsx::Context::CPU(), vsx::TypeFlag::kFloat16);
    memcpy(img_enc.MutableData<char>(), img_encoded[0].Data<char>(),
           img_enc.GetDataBytes());

    int seq_len = 208;
    auto txt_enc = vsx::Tensor(vsx::TShape({1, seq_len, 256}),
                               vsx::Context::CPU(), vsx::TypeFlag::kFloat16);
    memcpy(txt_enc.MutableData<char>(), txt_encoded[0].Data<char>(),
           txt_enc.GetDataBytes());

    auto position_ids = tokens[1];

    std::vector<vsx::Tensor> inputs = {img_enc, txt_enc, attention_mask,
                                       position_ids, text_token_mask};
    std::vector<vsx::Tensor> inputs_vacc;
    for (auto& input : inputs) {
      inputs_vacc.push_back(input.Clone(vsx::Context::VACC(device_id_)));
    }

    auto outputs = stream_->RunSync({inputs_vacc})[0];
    std::vector<vsx::Tensor> outputs_cpu;

    for (auto& out : outputs) {
      outputs_cpu.push_back(out.Clone());
    }
    return outputs_cpu;
  }

 protected:
  std::shared_ptr<vsx::ModelOperator> model_op_;
  std::shared_ptr<vsx::Model> model_;
  std::shared_ptr<vsx::Graph> graph_;
  std::shared_ptr<vsx::Stream> stream_;
  uint32_t device_id_;
  uint32_t batch_size_;
  vsx::TShape model_input_shape_;
};

class GroundingDino {
 public:
  GroundingDino(const std::string& txtmod_prefix,
                const std::string& txtmod_vdsp_params,
                const std::string& imgmod_prefix,
                const std::string& imgmod_vdsp_params,
                const std::string& decmod_prefix, uint32_t batch_size = 1,
                uint32_t device_id = 0, float threshold = 0.25,
                const std::string& txtmod_hw_config = "",
                const std::string& imgmod_hw_config = "",
                const std::string& decmod_hw_config = "",
                const std::string& positive_map_file = "")
      : threshold_(threshold), device_id_(device_id) {
    text_encoder_ = std::make_shared<GroundingDinoText>(
        txtmod_prefix, txtmod_vdsp_params, batch_size, device_id,
        txtmod_hw_config);
    image_encoder_ = std::make_shared<GroundingDinoImage>(
        imgmod_prefix, imgmod_vdsp_params, batch_size, device_id,
        imgmod_hw_config);
    decoder_ = std::make_shared<GroundingDinoDecoder>(
        decmod_prefix, batch_size, device_id, decmod_hw_config);

    std::ifstream ifs(positive_map_file, std::ios::binary | std::ios::ate);
    int file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    positive_map_.resize(file_size / 4);
    ifs.read(reinterpret_cast<char*>(positive_map_.data()), file_size);
    ifs.close();
  }
  vsx::ImageFormat GetFusionOpIimageFormat() {
    return image_encoder_->GetFusionOpIimageFormat();
  }

  std::vector<vsx::Tensor> ProcessText(const std::vector<vsx::Tensor>& tokens) {
    auto outputs = text_encoder_->Process(tokens);
    std::vector<vsx::Tensor> result;
    result.reserve(outputs.size());
    for (auto& out : outputs) {
      result.push_back(out.Clone());
    }
    return result;
  }
  std::vector<vsx::Tensor> ProcessImageAndDecode(
      const std::vector<vsx::Tensor>& txt_encoded,
      const std::vector<vsx::Tensor>& tokens, const vsx::Tensor& attention_mask,
      const vsx::Tensor& text_token_mask, const vsx::Image& image) {
    auto img_encoded = image_encoder_->Process(image);
    auto dec_output = decoder_->Process(img_encoded, txt_encoded, tokens,
                                        attention_mask, text_token_mask);

    return PostProcess(dec_output, tokens[0], image.Width(), image.Height());
  }
  std::vector<vsx::Tensor> PostProcess(
      const std::vector<vsx::Tensor>& dec_output, const vsx::Tensor& input_ids,
      int image_width, int image_height, bool to_xyxy = true,
      int num_select = 300) {
    // int out_row = dec_output[0].Shape()[0];  // 912
    int out_col = dec_output[0].Shape()[1];  // 208
    int cls_num = 91;
    int real_row = 900;
    // int real_col = 195;
    int align_col = 256;
    auto logits_fp16 = dec_output[0].Clone();
    auto logits = vsx::Tensor(vsx::TShape({real_row, 256}),  // todo
                              vsx::Context::CPU(), vsx::TypeFlag::kFloat32);
    float* logits_data = logits.MutableData<float>();
    uint16_t* logits_fp16_data = logits_fp16.MutableData<uint16_t>();

    for (int i = 0; i < real_row; i++) {
      for (int j = 0; j < 195; j++) {
        logits_data[i * 256 + j] = vsx::sigmoid<float>(
            vsx::HalfToFloat(logits_fp16_data[i * out_col + j]));
      }
      for (int j = 195; j < 256; j++) {
        logits_data[i * 256 + j] = 0.0f;
      }
    }
    auto bbox = vsx::ConvertTensorFromFp16ToFp32(dec_output[1].Clone());
    auto bbox_raw_data = bbox.MutableData<float>();
    if (to_xyxy) {
      for (int i = 0; i < real_row; ++i) {
        float w = bbox_raw_data[i * 4 + 2];
        float h = bbox_raw_data[i * 4 + 3];
        bbox_raw_data[i * 4 + 0] -= (w / 2);
        bbox_raw_data[i * 4 + 1] -= (h / 2);
        bbox_raw_data[i * 4 + 2] = bbox_raw_data[i * 4 + 0] + w;
        bbox_raw_data[i * 4 + 3] = bbox_raw_data[i * 4 + 1] + h;
      }
    }

    std::vector<std::pair<int, float>> prob_to_label(real_row * cls_num);
    std::vector<float> matrix_c(real_row * cls_num);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, real_row, cls_num,
                align_col, 1.0f, logits_data, align_col, positive_map_.data(),
                align_col, 0.0f, matrix_c.data(), cls_num);
    for (size_t i = 0; i < prob_to_label.size(); ++i) {
      prob_to_label[i] = std::pair<int, float>(i, matrix_c[i]);
    }
    std::partial_sort(
        prob_to_label.begin(), prob_to_label.begin() + num_select,
        prob_to_label.end(),
        [](const std::pair<int, float> a, const std::pair<int, float> b) {
          return a.second > b.second;
        });

    int obj_num = 0;
    for (int i = 0; i < num_select; ++i) {
      if (prob_to_label[i].second >= threshold_) {
        obj_num++;
      } else {
        break;
      }
    }

    std::vector<vsx::Tensor> outputs;
    outputs.emplace_back(vsx::TShape({obj_num}), vsx::Context::CPU(),
                         vsx::TypeFlag::kFloat32);
    outputs.emplace_back(vsx::TShape({obj_num, 4}), vsx::Context::CPU(),
                         vsx::TypeFlag::kFloat32);
    outputs.emplace_back(vsx::TShape({obj_num}), vsx::Context::CPU(),
                         vsx::TypeFlag::kInt32);

    auto score_data = outputs[0].MutableData<float>();
    auto bbox_data = outputs[1].MutableData<float>();
    auto labelId_data = outputs[2].MutableData<int32_t>();

    std::vector<std::array<int, 4>> topk_boxes_idx(obj_num);
    for (int i = 0; i < obj_num; ++i) {
      auto tmp = prob_to_label[i].first / cls_num;
      topk_boxes_idx[i][0] = tmp;
      topk_boxes_idx[i][1] = tmp;
      topk_boxes_idx[i][2] = tmp;
      topk_boxes_idx[i][3] = tmp;

      score_data[i] = prob_to_label[i].second;
      labelId_data[i] = prob_to_label[i].first % cls_num;
    }
    // garther from top_boxes_idx & scale
    for (int i = 0; i < obj_num; ++i) {
      bbox_data[i * 4 + 0] =
          bbox_raw_data[topk_boxes_idx[i][0] * 4 + 0] * image_width;
      bbox_data[i * 4 + 1] =
          bbox_raw_data[topk_boxes_idx[i][1] * 4 + 1] * image_height;
      bbox_data[i * 4 + 2] =
          bbox_raw_data[topk_boxes_idx[i][2] * 4 + 2] * image_width;
      bbox_data[i * 4 + 3] =
          bbox_raw_data[topk_boxes_idx[i][3] * 4 + 3] * image_height;
    }

    return outputs;
  }

 private:
  std::shared_ptr<GroundingDinoText> text_encoder_;
  std::shared_ptr<GroundingDinoImage> image_encoder_;
  std::shared_ptr<GroundingDinoDecoder> decoder_;
  float threshold_;
  uint32_t device_id_;
  std::vector<float> positive_map_;  // shape(91, 256)
};
}  // namespace vsx
