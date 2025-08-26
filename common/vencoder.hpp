
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

#include "common/media_encode.hpp"
#include "common/utils.hpp"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/media/video_encoder.h"

namespace vsx {

typedef struct yuv_context {
  FILE* file;
  char path[MAX_PATH_LEN];
  vsx::ImageFormat format;
  off_t size;
  int pic_size;
  int luma_size;
  int chroma_cb_size;
  int chroma_cr_size;
  int stride[3];
  int width;
  int height;
  int eof;
} yuv_context;

inline void yuv_seek_to_start(yuv_context* ctx) {
  if (ctx && ctx->file) fseeko(ctx->file, 0, SEEK_SET);
}

// 打开yuv文件
int yuv_open(const char* file_name, vsx::ImageFormat fmt, int width, int height,
             int stride, yuv_context* ctx) {
  if (!file_name || !ctx) {
    printf("Invalid parameters for opening file %p, ctx %p\n", file_name, ctx);
    return -1;
  }

  if (fmt != vsx::YUV_NV12) {
    printf("Sorry, format %d is not supported yet!\n", fmt);
    return -1;
  }

  int luma_size, chroma_size_cb, chroma_size_cr;
  memset(ctx, 0, sizeof(yuv_context));
  ctx->file = fopen(file_name, "rb");
  if (ctx->file == NULL) {
    printf("Fail to open file <%s>\n", file_name);
    return -1;
  }

  fseeko(ctx->file, 0, SEEK_END);
  ctx->size = ftello(ctx->file);
  fseeko(ctx->file, 0, SEEK_SET);
  memcpy(ctx->path, file_name, strlen(file_name));
  ctx->format = fmt;
  ctx->width = width;
  ctx->height = height;
  ctx->stride[0] = stride;

  switch (ctx->format) {
    case vsx::YUV_NV12:
      luma_size = stride * height;
      ctx->stride[1] = stride;
      chroma_size_cb = ctx->stride[1] * height / 2;
      chroma_size_cr = 0;
      break;
    default:
      luma_size = 0;
      chroma_size_cb = chroma_size_cr = 0;
      break;
  }
  ctx->luma_size = luma_size;
  ctx->chroma_cb_size = chroma_size_cb;
  ctx->chroma_cr_size = chroma_size_cr;
  ctx->pic_size = luma_size + chroma_size_cb + chroma_size_cr;
  return 0;
}

// 从yuv流中读取一帧
inline int yuv_read_frame(yuv_context* ctx, vsx::Image& image) {
  if (!ctx || !image.GetDataPtr() || !image.GetDataBytes()) return -1;

  int ret = 0;
  ret = fread(image.GetDataPtr(), 1, image.GetDataBytes(), ctx->file);
  if (ret <= 0) {
    goto fail;
  }

  return ret;
fail:
  if (feof(ctx->file))
    ctx->eof = 1;
  else
    printf("Read data failed\n");
  return ret;
}

// 获取尺寸参数
inline int yuv_pic_size(const struct yuv_context* ctx, int* luma,
                        int* chroma_cb, int* chroma_cr) {
  if (!ctx || !luma || !chroma_cb || !chroma_cr) return 0;
  *luma = ctx->luma_size;
  *chroma_cb = ctx->chroma_cb_size;
  *chroma_cr = ctx->chroma_cr_size;
  return ctx->pic_size;
}

// 关闭yuv流
inline void yuv_close(yuv_context* ctx) {
  if (ctx && ctx->file) {
    fclose(ctx->file);
    memset(ctx, 0, sizeof(yuv_context));
  }
}

class Vencoder : public MediaEncode {
 public:
  Vencoder(vsx::CodecType codec_type, uint32_t device_id,
           std::string stream_list, uint32_t frame_width, uint32_t frame_height,
           vsx::ImageFormat format, uint32_t frame_rate_numerator = 30,
           uint32_t frame_rate_denominator = 1)
      : MediaEncode(codec_type, device_id),
        input_file_path_(stream_list),
        width_(frame_width),
        height_(frame_height),
        format_(format),
        codec_(codec_type) {
    vsx::VideoEncoderAdvancedConfig conf;
    conf.frame_rate_numerator = frame_rate_numerator;
    conf.frame_rate_denominator = frame_rate_denominator;
    int ret = yuv_open(input_file_path_.c_str(), format_, width_, height_,
                       width_, &yuv_ctx_);
    if (ret != 0) {
      std::cerr << "yuv_open error:" << ret << std::endl;
    }
    pic_size_ = yuv_pic_size(&yuv_ctx_, &luma_size_, &chroma_cb_size_,
                             &chroma_cr_size_);

    image_ = vsx::Image(format_, width_, height_, vsx::Context::CPU(0));
    video_encoder_ = std::make_unique<vsx::VideoEncoder>(
        codec_type, frame_width, frame_height, conf);
  }

 private:
  std::unique_ptr<vsx::VideoEncoder> video_encoder_;

  vsx::Image image_;
  struct yuv_context yuv_ctx_;
  std::string input_file_path_;
  uint32_t width_;
  uint32_t height_;
  vsx::ImageFormat format_;
  vsx::CodecType codec_;
  uint32_t pic_size_;

  int luma_size_ = 0;
  int chroma_cb_size_ = 0;
  int chroma_cr_size_ = 0;

  size_t frame_ts_ = 0;

 protected:
  uint32_t ProcessImpl(const vsx::Image& image, bool end_flag) {
    int value = 0;
    if (end_flag) {
      value = video_encoder_->StopSendImage();
      return value;
    }
    if (image.Format() == vsx::ImageFormat::YUV_NV12 && image.GetDataPtr() &&
        image.GetDataBytes()) {
      auto frame_attr = std::make_shared<vsx::FrameAttr>();
      frame_attr->frame_pts = frame_ts_;
      frame_attr->frame_dts = frame_ts_;
      frame_ts_++;
      value = video_encoder_->SendImage(image, frame_attr);
    }
    return value;
  }

  bool GetResultImpl(std::shared_ptr<vsx::DataManager>& data) {
    std::shared_ptr<vsx::FrameAttr> frame_attr;
    return video_encoder_->RecvData(data, frame_attr, 40000);
  }

  vsx::Image GetTestDataImpl(bool loop) {
    vsx::Image image;
    do {
      int len = yuv_read_frame(&yuv_ctx_, image_);
      if (len <= 0) {
        if (yuv_ctx_.eof) {
          yuv_close(&yuv_ctx_);
          if (!loop) break;

          int ret = yuv_open(input_file_path_.c_str(), format_, width_, height_,
                             width_, &yuv_ctx_);
          if (ret != 0) {
            std::cerr << "yuv_open error:" << ret << std::endl;
            break;
          }
          continue;
        } else {
          std::cerr << "yuv_read_frame error:" << len << std::endl;
          break;
        }
      } else {
        image = image_;
        break;
      }
    } while (1);
    return image;
  }
};

}  // namespace vsx