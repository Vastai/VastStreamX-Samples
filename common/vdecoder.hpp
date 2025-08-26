
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <typeinfo>
#include <vector>

#include "common/media_decode.hpp"
#include "common/utils.hpp"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/media/media_source.h"
#include "vaststreamx/media/video_decoder.h"

namespace vsx {

#define MAX_STREAM_SIZE_4_VIDEO (4 * 1024 * 1024)

#define GET_H264_NALU_TYPE(nalu_type) \
  static_cast<vsx::NaluType>((nalu_type & 0x1F))

#define GET_HEVC_NALU_TYPE(nalu_type) \
  static_cast<vsx::NaluType>(((nalu_type & 0x7E) >> 1))

std::vector<uint8_t> nalu_prefix = {0, 0, 0, 1};
std::vector<uint8_t> nalu_prefix2 = {0, 0, 1};

inline std::vector<uint8_t> read_file_to_vector(const std::string &filename) {
  // 以二进制模式打开文件
  std::ifstream file(filename, std::ios::binary);

  std::vector<uint8_t> buffer;
  // 检查文件是否成功打开
  if (!file) {
    std::cerr << "无法打开文件" << std::endl;
  } else {
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 预留足够的空间以避免重新分配
    buffer.reserve(fileSize);

    // 读取整个文件
    buffer.assign((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());
  }

  return buffer;
}

inline bool is_nalu_start(const std::vector<uint8_t> &file_content,
                          uint32_t index) {
  if (file_content[index] == 0x00 && file_content[index + 1] == 0x00 &&
      file_content[index + 2] == 0x01) {
    return true;
  } else if (file_content[index] == 0x00 && file_content[index + 1] == 0x00 &&
             file_content[index + 2] == 0x00 &&
             file_content[index + 3] == 0x01) {
    return true;
  }
  return false;
}

inline bool is_nalu_start2(const std::vector<uint8_t> &file_content,
                           uint32_t index) {
  const uint8_t *frame = file_content.data() + index;
  if (0 == memcmp(reinterpret_cast<const void *>(frame),
                  reinterpret_cast<void *>(nalu_prefix2.data()),
                  nalu_prefix2.size())) {
    return true;
  } else if (0 == memcmp(reinterpret_cast<const void *>(frame),
                         reinterpret_cast<void *>(nalu_prefix.data()),
                         nalu_prefix.size())) {
    return true;
  }
  return false;
}

class Vdecoder : public MediaDecode {
 public:
  Vdecoder(vsx::CodecType codec_type, uint32_t device_id,
           std::string stream_list)
      : MediaDecode(codec_type, device_id),
        codec_type_(codec_type),
        stream_list_(stream_list),
        nalu_num_(0),
        index_(0) {
    vsx::SetDevice(device_id);
    video_decoder_ = std::make_unique<vsx::VideoDecoder>(codec_type);
    // if (codec_type_ == vsx::CODEC_TYPE_H264 ||
    //     codec_type_ == vsx::CODEC_TYPE_HEVC) {
    //   vector_of_nalu_ = init_nalu(stream_list, codec_type);
    // } else if (codec_type_ == vsx::CODEC_TYPE_AV1)
    {
      vsx::MediaSource media_src;
      int value = media_src.Open(stream_list, vsx::TRANSPORT_TYPE_TCP);
      if (!value) {
        LOG(ERROR) << "media_source open uri:" << stream_list
                   << " error:" << value;
        return;
      }
      while (true) {
        std::shared_ptr<vsx::DataManager> data_manager =
            std::make_shared<vsx::DataManager>(MAX_STREAM_SIZE_4_VIDEO,
                                               vsx::Context::CPU());
        bool b_missing_sps = false;
        bool b_eof = false;
        int ret = media_src.Read(data_manager, b_missing_sps, b_eof);
        auto frame_packet = std::make_shared<vsx::FramePacket>();
        ret += media_src.GetParams(*frame_packet);
        if (ret != 0) {
          if (!b_eof) {
            std::cerr << "media source uri:" << stream_list
                      << " read return:" << value << std::endl;
          }
          break;
        }
        vector_of_nalu_.push_back(data_manager);
        vector_framepacket_.push_back(frame_packet);
      }
      media_src.Close();
    }
    nalu_num_ = vector_of_nalu_.size();
  }

  std::vector<std::shared_ptr<vsx::DataManager>> init_nalu(
      std::string stream_list, vsx::CodecType codec_type) {
    std::vector<uint8_t> file_content = read_file_to_vector(stream_list);
    uint32_t file_len = file_content.size();
    uint32_t index = 0;
    std::vector<std::shared_ptr<vsx::DataManager>> vector_of_nalu;
    while (index + 4 < file_len) {
      if (!is_nalu_start(file_content, index)) {
        ++index;
        continue;
      }
      if (index != 0) {
        std::vector<uint8_t> temp;
        temp.insert(temp.begin(), file_content.begin() + index,
                    file_content.end());
        file_content = temp;
      }
      break;
    }

    if (index + 4 >= file_len) {
      std::cerr << "index:" << index << " file_len:" << file_len << std::endl;
      return vector_of_nalu;
    }

    index = 4;
    uint32_t start = 0;
    std::vector<uint8_t> nalu_buf;
    nalu_buf.resize(MAX_STREAM_SIZE_4_VIDEO);
    uint8_t *nalu_pt = nalu_buf.data();
    uint32_t nalu_len = 0;
    file_len = file_content.size();
    uint8_t *file_pt = file_content.data();
    bool get_frame = false;

    while (index + 4 < file_len) {
      if (!is_nalu_start(file_content, index)) {
        ++index;
        continue;
      }

      memcpy(reinterpret_cast<void *>(nalu_pt + nalu_len),
             reinterpret_cast<void *>(file_pt + start), index - start);
      nalu_len += (index - start);

      uint32_t nalu_type = 0;
      uint32_t nalu_header_len = 0;
      const uint8_t *frame = file_content.data() + start;

      if (0 == memcmp(reinterpret_cast<const void *>(frame),
                      reinterpret_cast<void *>(nalu_prefix.data()),
                      nalu_prefix.size())) {
        nalu_header_len = nalu_prefix.size();
      } else if (0 == memcmp(reinterpret_cast<const void *>(frame),
                             reinterpret_cast<void *>(nalu_prefix2.data()),
                             nalu_prefix2.size())) {
        nalu_header_len = nalu_prefix2.size();
      } else {
        std::cerr << "Unsupported nalu Type!" << std::endl;
        break;
      }
      if (codec_type == vsx::CODEC_TYPE_H264) {
        nalu_type = GET_H264_NALU_TYPE(frame[nalu_header_len]);
        if (nalu_type == vsx::NALU_TYPE_H264_IDR ||
            nalu_type == vsx::NALU_TYPE_H264_P) {
          get_frame = true;
        }
      } else if (codec_type == vsx::CODEC_TYPE_HEVC) {
        nalu_type = GET_HEVC_NALU_TYPE(frame[nalu_header_len]);
        if (nalu_type == NALU_TYPE_HEVC_IDR_W_RADL ||
            nalu_type == NALU_TYPE_HEVC_IDR_N_LP || nalu_type == 0x00 ||
            nalu_type == 0x01) {
          get_frame = true;
        }
      }

      if (get_frame && nalu_len > 0) {
        std::shared_ptr<vsx::DataManager> data_manager =
            std::make_shared<vsx::DataManager>(nalu_len, vsx::Context::CPU());
        memcpy(data_manager->GetDataPtr(), reinterpret_cast<void *>(nalu_pt),
               nalu_len);
        vector_of_nalu.emplace_back(data_manager);
        nalu_num_ += 1;

        nalu_len = 0;
        get_frame = false;
      }
      start = index;
      index += nalu_prefix2.size();
    }
    if (start < file_len) {
      std::vector<uint8_t> temp;
      std::shared_ptr<vsx::DataManager> data_manager =
          std::make_shared<vsx::DataManager>(file_len - start,
                                             vsx::Context::CPU());
      memcpy(data_manager->GetDataPtr(),
             reinterpret_cast<void *>(file_pt + start), file_len - start);
      vector_of_nalu.emplace_back(data_manager);
      nalu_num_ += 1;
    }

    return vector_of_nalu;
  }
  virtual bool IsKeyFrame() { return is_key_frame_; }

 protected:
  uint32_t ProcessImpl(const std::shared_ptr<vsx::DataManager> &data,
                       bool end_flag) {
    int value = 0;
    if (data) {
      value = video_decoder_->SendData(data, nullptr);
    }
    if (end_flag) {
      value = video_decoder_->StopSendData();
    }
    return value;
  }

  std::shared_ptr<vsx::DataManager> GetTestDataImpl(bool loop) {
    std::shared_ptr<vsx::DataManager> data_manager;
    std::vector<uint8_t> nalu;
    if (loop) {
      auto nalu = vector_of_nalu_[index_ % nalu_num_];
      is_key_frame_ =
          vector_framepacket_[index_ % nalu_num_]->format->is_key_frame;
      index_++;
      return nalu;
    } else if (index_ < nalu_num_) {
      auto nalu = vector_of_nalu_[index_];
      is_key_frame_ = vector_framepacket_[index_]->format->is_key_frame;
      index_++;
      return nalu;
    } else {
      is_key_frame_ = false;
      return nullptr;
    }
    is_key_frame_ = false;
    return nullptr;
  }

  bool GetResultImpl(vsx::Image &image) {
    std::shared_ptr<FrameAttr> frame_attr;
    return video_decoder_->RecvImage(image, frame_attr, true);
  }

 private:
  std::unique_ptr<vsx::VideoDecoder> video_decoder_;
  vsx::CodecType codec_type_;
  std::string stream_list_;
  uint32_t nalu_num_;
  uint32_t index_;
  std::vector<std::shared_ptr<vsx::DataManager>> vector_of_nalu_;
  std::vector<std::shared_ptr<vsx::FramePacket>> vector_framepacket_;
  bool is_key_frame_ = true;
};

}  // namespace vsx