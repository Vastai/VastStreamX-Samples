
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <glob.h>

#include <fstream>

#include "glog/logging.h"
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"
namespace vsx {

inline uint32_t AsUint(const float x) {
  return *reinterpret_cast<const uint32_t *>(&x);
}

inline float AsFloat(const uint32_t x) {
  return *reinterpret_cast<const float *>(&x);
}

inline uint16_t FloatToHalf(const float x) {
  const uint32_t b = AsUint(x) + 0x00001000;
  const uint32_t e = (b & 0x7F800000) >> 23;
  const uint32_t m = b & 0x007FFFFF;
  return (b & 0x80000000) >> 16 |
         (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
         ((e < 113) & (e > 101)) *
             ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
         (e > 143) * 0x7FFF;
}

inline float HalfToFloat(const uint16_t x) {
  const uint32_t e = (x & 0x7C00) >> 10;
  const uint32_t m = (x & 0x03FF) << 13;
  const uint32_t v = AsUint(static_cast<float>(m)) >> 23;
  return AsFloat((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                 ((e == 0) & (m != 0)) *
                     ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
}

static inline uint32_t npy_halfbits_to_floatbits(uint16_t h) {
  uint16_t h_exp, h_sig;
  uint32_t f_sgn, f_exp, f_sig;

  h_exp = (h & 0x7c00u);
  f_sgn = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  switch (h_exp) {
    case 0x0000u: /* 0 or subnormal */
      h_sig = (h & 0x03ffu);
      /* Signed zero */
      if (h_sig == 0) {
        return f_sgn;
      }
      /* Subnormal */
      h_sig <<= 1;
      while ((h_sig & 0x0400u) == 0) {
        h_sig <<= 1;
        h_exp++;
      }
      f_exp = (static_cast<uint32_t>(127 - 15 - h_exp)) << 23;
      f_sig = (static_cast<uint32_t>(h_sig & 0x03ffu)) << 13;
      return f_sgn + f_exp + f_sig;
    case 0x7c00u: /* inf or NaN */
      /* All-ones exponent and a copy of the significand */
      return f_sgn + 0x7f800000u + ((static_cast<uint32_t>(h & 0x03ffu)) << 13);
    default: /* normalized */
      /* Just need to adjust the exponent and shift */
      return f_sgn + ((static_cast<uint32_t>(h & 0x7fffu) + 0x1c000u) << 13);
  }
}

inline vsx::Tensor ConvertTensorFromFp16ToFp32(const vsx::Tensor &fp16_tensor) {
  vsx::Tensor fp32_tensor;
  if (fp16_tensor.GetDType() != vsx::kFloat16) {
    LOG(ERROR) << "The data type of fp16_tensor is not kFloat16.";
    return fp32_tensor;
  }
  if (fp16_tensor.GetContext().dev_type != vsx::Context::kCPU) {
    LOG(ERROR) << "The device type of fp16_tensor is not kCPU.";
    return fp32_tensor;
  }
  fp32_tensor =
      vsx::Tensor(fp16_tensor.Shape(), vsx::Context::CPU(), vsx::kFloat32);
  size_t count = fp16_tensor.GetSize();
  uint16_t *fp16data = fp16_tensor.MutableData<uint16_t>();
  float *fp32data = fp32_tensor.MutableData<float>();
  vsx::ConvertFp16ToFp32Array(fp16data, count, fp32data);
  return fp32_tensor;
}

inline std::vector<vsx::Tensor> ConvertTensorFromFp16ToFp32(
    const std::vector<vsx::Tensor> &fp16_tensors) {
  std::vector<vsx::Tensor> fp32_tensors;
  fp32_tensors.reserve(fp16_tensors.size());
  for (const auto &ft : fp16_tensors) {
    fp32_tensors.push_back(ConvertTensorFromFp16ToFp32(ft));
  }
  return fp32_tensors;
}

inline vsx::Tensor ConvertTensorFromFp32ToFp16(const vsx::Tensor &fp32_tensor) {
  vsx::Tensor fp16_tensor;
  if (fp32_tensor.GetDType() != vsx::kFloat32) {
    LOG(ERROR) << "The data type of fp32_tensor is not kFloat32.";
    return fp16_tensor;
  }
  if (fp32_tensor.GetContext().dev_type != vsx::Context::kCPU) {
    LOG(ERROR) << "The device type of fp32_tensor is not kCPU.";
    return fp16_tensor;
  }
  fp16_tensor =
      vsx::Tensor(fp32_tensor.Shape(), vsx::Context::CPU(), vsx::kFloat16);
  size_t count = fp32_tensor.GetSize();
  float *fp32data = fp32_tensor.MutableData<float>();
  uint16_t *fp16data = fp16_tensor.MutableData<uint16_t>();
  vsx::ConvertFp32ToFp16Array(fp32data, count, fp16data);
  return fp16_tensor;
}

inline std::vector<vsx::Tensor> ConvertTensorFromFp32ToFp16(
    const std::vector<vsx::Tensor> &fp32_tensors) {
  std::vector<vsx::Tensor> fp16_tensors;
  fp16_tensors.reserve(fp32_tensors.size());
  for (const auto &ft : fp32_tensors) {
    fp16_tensors.push_back(ConvertTensorFromFp32ToFp16(ft));
  }
  return fp16_tensors;
}

template <typename T>
inline void DumpTensor(const vsx::Tensor &tensor) {
  assert(tensor.GetContext().dev_type == vsx::Context::kCPU);
  const T *tensor_data = tensor.Data<T>();
  std::cout << "Tensor Shape: " << tensor.Shape() << std::endl;
  for (int i = 0; i < tensor.GetSize(); i++) {
    std::cout << tensor_data[i] << " ";
  }
  std::cout << std::endl;
}

inline std::vector<std::string> LoadLabels(const std::string &file) {
  std::ifstream file_in(file);
  if (!file_in.good()) {
    LOG(ERROR) << "cannot open label file: " << file;
    std::exit(-1);
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file_in, line)) {
    labels.push_back(line);
  }
  file_in.close();
  return labels;
}

inline std::unordered_map<int, std::string> LoadLabelDict(
    const std::string &file) {
  std::vector<int> id_map = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
      54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
      74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
  std::ifstream file_in(file);
  if (!file_in.good()) {
    LOG(ERROR) << "cannot open label file: " << file;
    std::exit(-1);
  }
  std::unordered_map<int, std::string> label_dict;
  std::string line;
  int i = 0;
  while (std::getline(file_in, line)) {
    label_dict[id_map[i++]] = line;
  }
  file_in.close();
  return label_dict;
}

inline void StrReplace(std::string &str, const std::string &o,
                       const std::string &n) {
  auto found_pos = str.find(o);
  std::cout << "found_pos " << found_pos << std::endl;
  if (found_pos < str.size()) str.replace(found_pos, o.size(), n);
}

inline void StrErase(std::string &str, char o) {
  str.erase(std::remove(str.begin(), str.end(), o), str.end());
}

inline std::vector<std::string> Split(std::string s, std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

inline TShape ParseShape(const std::string &shape_str) {
  auto tmp_str = shape_str;
  StrErase(tmp_str, '[');
  StrErase(tmp_str, ']');
  StrErase(tmp_str, '(');
  StrErase(tmp_str, ')');
  auto str_split = Split(tmp_str, ",");
  TShape shape(str_split.size(), 0);
  for (size_t i = 0; i < str_split.size(); i++) {
    shape[i] = std::stoi(str_split[i]);
  }
  return shape;
}
inline std::vector<float> ParseVecFloat(const std::string &vec_str) {
  auto tmp_str = vec_str;
  StrErase(tmp_str, '[');
  StrErase(tmp_str, ']');
  StrErase(tmp_str, '(');
  StrErase(tmp_str, ')');
  auto str_split = Split(tmp_str, ",");
  std::vector<float> vec(str_split.size(), 0);
  for (size_t i = 0; i < str_split.size(); i++) {
    vec[i] = std::stof(str_split[i]);
  }
  return vec;
}
inline std::vector<uint32_t> ParseVecUint(const std::string &vec_str) {
  auto tmp_str = vec_str;
  StrErase(tmp_str, '[');
  StrErase(tmp_str, ']');
  StrErase(tmp_str, '(');
  StrErase(tmp_str, ')');
  auto str_split = Split(tmp_str, ",");
  std::vector<uint32_t> vec(str_split.size(), 0);
  for (size_t i = 0; i < str_split.size(); i++) {
    vec[i] = std::stoi(str_split[i]);
  }
  return vec;
}
inline std::vector<std::string> ParseVecString(const std::string &vec_str) {
  auto tmp_str = vec_str;
  StrErase(tmp_str, '[');
  StrErase(tmp_str, ']');
  StrErase(tmp_str, '(');
  StrErase(tmp_str, ')');
  return Split(tmp_str, ",");
}

template <class T>
inline std::vector<T> CalcPercentiles(
    const std::vector<T> &data, const std::vector<uint32_t> &percentiles) {
  std::vector<T> results;
  results.reserve(percentiles.size());
  auto data_copy = data;
  std::sort(data_copy.begin(), data_copy.end());
  for (const auto p : percentiles) {
    CHECK(p <= 100) << "percentile should be less or equal than 100";
    size_t index = (p / 100.0) * (data_copy.size() - 1) + 0.5;
    results.emplace_back(data_copy[index]);
  }
  return results;
}

template <class T>
inline T CalcMean(const std::vector<T> &data) {
  uint64_t sum = 0;
  for (auto x : data) {
    sum += x;
  }
  return sum / data.size();
}

inline std::vector<std::string> ReadFileList(const std::string &list_file) {
  return LoadLabels(list_file);
}

inline std::vector<std::string> GetFolderFilenameList(const std::string &folder,
                                                      std::string ext = "*") {
  glob_t glob_result;
  std::string pattern = folder + "/" + ext;
  int ret = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  CHECK(ret == 0) << "glob failed, folder: " << folder;
  std::vector<std::string> filelist;
  for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    filelist.push_back(glob_result.gl_pathv[i]);
  }
  globfree(&glob_result);
  return filelist;
}

inline int Color_I420_TO_NV12(const cv::Mat &cvI420, char **nv12_data) {
  int width = cvI420.cols;
  int height = cvI420.rows * 2 / 3;

  int nLenY = width * height;
  int nLenU = nLenY / 4;

  const char *i420bytes = reinterpret_cast<char *>(cvI420.data);
  size_t data_len = width * height * 3 / 2;
  char *nv12bytes = reinterpret_cast<char *>(malloc(data_len));
  if (nv12bytes == NULL) return -1;
  memcpy(nv12bytes, i420bytes, width * height);  // copy Y data

  for (int i = 0; i < nLenU; i++) {
    nv12bytes[nLenY + 2 * i] = i420bytes[nLenY + i];              // U
    nv12bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + nLenU + i];  // V
  }

  *nv12_data = nv12bytes;

  return 0;
}

inline int Color_NV12_TO_I420(const char *nv12_data, int width, int height,
                              cv::Mat &cvI420) {
  cv::Mat temp = cv::Mat(height * 3 / 2, width, CV_8UC1);
  char *i420bytes = reinterpret_cast<char *>(temp.data);

  memcpy(i420bytes, nv12_data, width * height);  // copy Y data
  int nLenY = width * height;
  int nLenU = nLenY / 4;

  for (int i = 0; i < nLenU; i++) {
    i420bytes[nLenY + i] = nv12_data[nLenY + 2 * i];              // U
    i420bytes[nLenY + nLenU + i] = nv12_data[nLenY + 2 * i + 1];  // V
  }
  cvI420 = temp;
  return 0;
}

inline int MakeVsxImage(cv::Mat &mat_bgr_interleave, vsx::Image &image,
                        vsx::ImageFormat format) {
  cv::Mat cvImage = mat_bgr_interleave;
  // RGB BGR Packet
  if (format == vsx::ImageFormat::BGR_INTERLEAVE ||
      format == vsx::ImageFormat::RGB_INTERLEAVE) {
    if (format == vsx::ImageFormat::RGB_INTERLEAVE) {
      cv::cvtColor(cvImage, cvImage, CV_BGR2RGB);
    }
    image = vsx::Image(format, cvImage.cols, cvImage.rows);
    size_t data_len = cvImage.cols * cvImage.rows * 3;
    void *dst_data = image.MutableData<void>();
    memcpy(dst_data, cvImage.data, data_len);
    return 0;
  }
  // NV12
  if (format == vsx::ImageFormat::YUV_NV12) {
    cv::Mat cvI420;
    cv::cvtColor(cvImage, cvI420, CV_BGR2YUV_I420);

    char *nv12data = NULL;
    if (Color_I420_TO_NV12(cvI420, &nv12data) != 0) {
      LOG(ERROR) << "Failed to call Color_I420_TO_NV12";
      return -1;
    }
    auto deleter = [](void *ptr) {
      if (ptr) {
        free(ptr);
      }
    };
    std::shared_ptr<vsx::DataManager> manager(new vsx::DataManager(
        cvImage.rows * cvImage.cols * 3 / 2, vsx::Context::CPU(0),
        reinterpret_cast<uint64_t>(nv12data), deleter));

    image = vsx::Image(vsx::ImageFormat::YUV_NV12, cvImage.cols, cvImage.rows,
                       0, 0, manager);
    return 0;
  }

  // RGBP BGRP
  if (format == vsx::ImageFormat::RGB_PLANAR ||
      format == vsx::ImageFormat::BGR_PLANAR) {
    std::vector<cv::Mat> channels;
    cv::split(cvImage, channels);
    if (format == vsx::ImageFormat::RGB_PLANAR) {
      cv::Mat temp = channels[0];
      channels[0] = channels[2];
      channels[2] = temp;
    }
    image = vsx::Image(format, cvImage.cols, cvImage.rows);
    int channel_len = cvImage.cols * cvImage.rows;
    char *dst_data = image.MutableData<char>();
    memcpy(dst_data, channels[0].data, channel_len);
    dst_data += channel_len;
    memcpy(dst_data, channels[1].data, channel_len);
    dst_data += channel_len;
    memcpy(dst_data, channels[2].data, channel_len);
    return 0;
  }

  // GRAY
  if (format == vsx::ImageFormat::GRAY) {
    image = vsx::Image(format, cvImage.cols, cvImage.rows);
    cv::Mat gray;
    cv::cvtColor(cvImage, gray, cv::COLOR_BGR2GRAY);
    int channel_len = cvImage.cols * cvImage.rows;
    char *dst_data = image.MutableData<char>();
    memcpy(dst_data, gray.data, channel_len);
    return 0;
  }
  LOG(ERROR) << "Unsupport image format: " << format;
  return -1;
}
inline int MakeVsxImage(const std::string &image_file, vsx::Image &image,
                        vsx::ImageFormat format) {
  cv::Mat mat = cv::imread(image_file);
  CHECK(!mat.empty()) << "Failed to read: " << image_file;
  return MakeVsxImage(mat, image, format);
}

inline int ConvertVsxImageToCvMatBgrPacked(const vsx::Image &image,
                                           cv::Mat &mat) {
  CHECK(image.GetContext().dev_type == vsx::Context::kCPU)
      << "Image memory should be cpu memory,not device memory";
  int width = image.Width();
  int height = image.Height();
  int width_pitch = image.WidthPitch() > 0 ? image.WidthPitch() : image.Width();
  int height_pitch =
      image.HeightPitch() > 0 ? image.HeightPitch() : image.Height();
  auto format = image.Format();
  switch (format) {
    case vsx::RGB_PLANAR:
    case vsx::BGR_PLANAR: {
      std::vector<cv::Mat> channels;
      channels.reserve(3);
      const uchar *image_data = image.Data<uchar>();
      size_t src_offset = height_pitch * width_pitch;
      for (int i = 0; i < 3; i++) {
        cv::Mat m(height, width, CV_8UC1);
        if (width == width_pitch) {
          memcpy(m.data, image_data + i * src_offset, height * width);
        } else {
          uchar *dst = m.data;
          const uchar *src = image_data + i * src_offset;
          for (int h = 0; h < height; h++) {
            memcpy(dst, src, width);
            dst += width;
            src += width_pitch;
          }
        }
        channels.push_back(m);
      }
      if (format == RGB_PLANAR) {
        cv::Mat temp = channels[0];
        channels[0] = channels[2];
        channels[2] = temp;
      }
      cv::merge(channels, mat);
      return 0;
    } break;
    case vsx::BGR_INTERLEAVE:
    case vsx::RGB_INTERLEAVE: {
      cv::Mat m(height, width, CV_8UC3);
      const uchar *image_data = image.Data<uchar>();
      if (width == width_pitch) {
        memcpy(m.data, image_data, width * height * 3);
      } else {
        uchar *dst = m.data;
        const uchar *src = image_data;
        for (int h = 0; h < height; h++) {
          memcpy(dst, src, width * 3);
          dst += width * 3;
          src += width_pitch;
        }
      }
      if (format == vsx::RGB_INTERLEAVE) {
        cv::Mat temp;
        cv::cvtColor(m, temp, cv::COLOR_BGR2RGB);
        m = temp;
      }
      mat = m;
      return 0;
    } break;
    case vsx::GRAY: {
      const uchar *image_data = image.Data<uchar>();
      mat = cv::Mat(image.Height(), image.Width(), CV_8UC1);
      if (width == width_pitch) {
        memcpy(mat.data, image_data, width * height);
      } else {
        uchar *dst = mat.data;
        const uchar *src = image_data;
        for (int h = 0; h < height; h++) {
          memcpy(dst, src, width);
          dst += width;
          src += width_pitch;
        }
      }
      cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
      return 0;
    } break;
    case vsx::YUV_NV12: {
      cv::Mat I420;
      if (width != width_pitch) {
        vsx::Image temp(vsx::YUV_NV12, width, height);
        const uchar *src = image.Data<uchar>();
        uchar *dst = temp.MutableData<uchar>();
        for (int h = 0; h < height; h++) {
          memcpy(dst, src, width);
          dst += width;
          src += width_pitch;
        }
        src += width_pitch * height_pitch;
        dst += width * height;
        for (int h = 0; h < height / 2; h++) {
          memcpy(dst, src, width);
          dst += width;
          src += width_pitch;
        }
        Color_NV12_TO_I420(temp.Data<char>(), temp.Width(), temp.Height(),
                           I420);
      } else {
        Color_NV12_TO_I420(image.Data<char>(), image.Width(), image.Height(),
                           I420);
      }
      cv::cvtColor(I420, mat, CV_YUV2BGR_I420);
      return 0;
    } break;
    default:
      LOG(ERROR) << "ERROR: Unsupport format:"
                 << vsx::ImageFormatToString(format);
      return -1;
  }
}
std::vector<int64_t> TShapeToVector(const TShape &shape) {
  std::vector<int64_t> v;
  v.reserve(shape.ndim());
  for (int i = 0; i < shape.ndim(); i++) {
    v.push_back(shape[i]);
  }
  return v;
}

template <typename T>
inline std::vector<double> softmax(const std::vector<T> &values) {
  std::vector<double> exp_values(values.size());
  double max_value = *std::max_element(values.begin(), values.end());
  double sum_exp = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    exp_values[i] = std::exp(values[i] - max_value);
    sum_exp += exp_values[i];
  }
  for (auto &value : exp_values) {
    value /= sum_exp;
  }
  return exp_values;
}

template <typename T>
inline T sigmoid(T x) {
  return 1 / (1 + exp(-x));
}

template <typename T>
inline void Normalize(T *data, size_t size) {
  double sum = 0;
  for (size_t s = 0; s < size; s++) {
    sum += data[s] * data[s];
  }
  sum = std::sqrt(sum);

  for (size_t s = 0; s < size; s++) {
    data[s] /= sum;
  }
}
inline std::vector<vsx::Tensor> ReadNpzFile(
    const std::string &npz_file, const std::string &key_prefix = "input_") {
  auto tensor_map = vsx::LoadTensorMap(npz_file);
  std::vector<vsx::Tensor> tensors;
  for (size_t i = 0; i < tensor_map.size(); i++) {
    std::stringstream key;
    key << key_prefix << i;
    tensors.push_back(tensor_map[key.str()]);
  }
  return tensors;
}

inline vsx::ImageFormat ConvertToVsxFormat(
    vsx::BuildInOperatorAttrImageType image_type) {
  switch (image_type) {
    case vsx::BuildInOperatorAttrImageType::kYUV_NV12:
      return vsx::ImageFormat::YUV_NV12;
    case vsx::BuildInOperatorAttrImageType::kRGB_PLANAR:
      return vsx::ImageFormat::RGB_PLANAR;
    case vsx::BuildInOperatorAttrImageType::kBGR_PLANAR:
      return vsx::ImageFormat::BGR_PLANAR;
    case vsx::BuildInOperatorAttrImageType::kRGB888:
      return vsx::ImageFormat::RGB_INTERLEAVE;
    case vsx::BuildInOperatorAttrImageType::kBGR888:
      return vsx::ImageFormat::BGR_INTERLEAVE;
    case vsx::BuildInOperatorAttrImageType::kGRAY:
      return vsx::ImageFormat::GRAY;
    default:
      break;
  }
  CHECK(false) << "Unrecognize image type: " << image_type;
  return vsx::ImageFormat::YUV_NV12;
}

inline bool HasAttribute(const std::vector<std::string> &keys,
                         const std::string &attr) {
  for (const auto &key : keys) {
    if (attr == key) {
      return true;
    }
  }
  return false;
}

inline void ConvertToVsxFormat(vsx::BuildInOperatorAttrColorCvtCode cvtcode,
                               vsx::ImageFormat &iimage_format,
                               vsx::ImageFormat &oimage_format) {
  switch (cvtcode) {
    case vsx::BuildInOperatorAttrColorCvtCode::kYUV2RGB_NV12:
      iimage_format = vsx::ImageFormat::YUV_NV12;
      oimage_format = vsx::ImageFormat::RGB_PLANAR;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kYUV2BGR_NV12:
      iimage_format = vsx::ImageFormat::YUV_NV12;
      oimage_format = vsx::ImageFormat::BGR_PLANAR;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kBGR2RGB:
      iimage_format = vsx::ImageFormat::BGR_PLANAR;
      oimage_format = vsx::ImageFormat::RGB_PLANAR;
      break;
    // case vsx::BuildInOperatorAttrColorCvtCode::kRGB2BGR:
    //   iimage_format = vsx::ImageFormat::RGB_PLANAR;
    //   oimage_format = vsx::ImageFormat::BGR_PLANAR;
    case vsx::BuildInOperatorAttrColorCvtCode::kBGR2RGB_INTERLEAVE2PLANAR:
      iimage_format = vsx::ImageFormat::BGR_INTERLEAVE;
      oimage_format = vsx::ImageFormat::RGB_PLANAR;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kRGB2BGR_INTERLEAVE2PLANAR:
      iimage_format = vsx::ImageFormat::RGB_INTERLEAVE;
      oimage_format = vsx::ImageFormat::BGR_PLANAR;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kBGR2BGR_INTERLEAVE2PLANAR:
      iimage_format = vsx::ImageFormat::BGR_INTERLEAVE;
      oimage_format = vsx::ImageFormat::BGR_PLANAR;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kRGB2RGB_INTERLEAVE2PLANAR:
      iimage_format = vsx::ImageFormat::RGB_INTERLEAVE;
      oimage_format = vsx::ImageFormat::RGB_PLANAR;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kYUV2GRAY_NV12:
      iimage_format = vsx::ImageFormat::YUV_NV12;
      oimage_format = vsx::ImageFormat::GRAY;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kBGR2GRAY_INTERLEAVE:
      iimage_format = vsx::ImageFormat::BGR_INTERLEAVE;
      oimage_format = vsx::ImageFormat::GRAY;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kBGR2GRAY_PLANAR:
      iimage_format = vsx::ImageFormat::BGR_PLANAR;
      oimage_format = vsx::ImageFormat::GRAY;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kRGB2GRAY_INTERLEAVE:
      iimage_format = vsx::ImageFormat::RGB_INTERLEAVE;
      oimage_format = vsx::ImageFormat::GRAY;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kRGB2GRAY_PLANAR:
      iimage_format = vsx::ImageFormat::RGB_PLANAR;
      oimage_format = vsx::ImageFormat::GRAY;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kRGB2YUV_NV12_PLANAR:
      iimage_format = vsx::ImageFormat::RGB_PLANAR;
      oimage_format = vsx::ImageFormat::YUV_NV12;
      break;
    case vsx::BuildInOperatorAttrColorCvtCode::kBGR2YUV_NV12_PLANAR:
      iimage_format = vsx::ImageFormat::BGR_PLANAR;
      oimage_format = vsx::ImageFormat::YUV_NV12;
      break;
    default:
      CHECK(false) << "Unrecognize cvtcolor code: " << cvtcode;
      break;
  }
}

inline int ReadBinaryFile(const std::string &filename,
                          std::shared_ptr<vsx::DataManager> &data) {
  std::ifstream infile;
  infile.open(filename, std::ios::binary);
  CHECK(infile.is_open()) << "Failed to open: " << filename << std::endl;
  infile.seekg(0, std::ios::end);
  size_t file_length = infile.tellg();
  char *content = new char[file_length];
  infile.seekg(0, std::ios::beg);
  infile.read(content, file_length);
  infile.close();

  std::shared_ptr<vsx::DataManager> manager(
      new vsx::DataManager(file_length, vsx::Context::CPU(),
                           reinterpret_cast<uint64_t>(content), [](void *ptr) {
                             if (ptr) delete[] reinterpret_cast<char *>(ptr);
                           }));
  data = manager;
  return 0;
}

inline int WriteBinaryFile(const std::string &filename, const void *data,
                           size_t data_size) {
  std::ofstream outfile;
  outfile.open(filename, std::ios::binary);
  CHECK(outfile.is_open()) << "Failed to open: " << filename << std::endl;
  outfile.write(reinterpret_cast<const char *>(data), data_size);
  outfile.close();
  return 0;
}

inline vsx::Tensor bert_get_activation_fp16_A(const vsx::Tensor &activation) {
  int N, M, K;
  auto shape = activation.Shape();
  if (shape.ndim() == 2) {
    M = shape[0];
    K = shape[1];
    N = 1;
  } else {
    N = shape[0];
    M = shape[1];
    K = shape[2];
  }
  int m_group = 16, k_group = 16;
  int pad_M = M, pad_K = K;
  if (M % m_group != 0) {
    int pad_m = m_group - M % m_group;
    pad_M += pad_m;
  }
  if (K % k_group != 0) {
    int pad_k = k_group - K % k_group;
    pad_K += pad_k;
  }

  int n_num = N;
  int m_num = pad_M / m_group;
  int k_num = pad_K / k_group;
  int block_size = m_group * k_group;

  vsx::Tensor result({n_num, m_num, k_num, block_size}, vsx::Context::CPU(),
                     vsx::TypeFlag::kFloat16);
  uint16_t *dst = result.MutableData<uint16_t>();
  memset(dst, 0, result.GetDataBytes());

  const uint16_t *src = activation.Data<uint16_t>();
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        int addr = (m % m_group) * k_group + (k % k_group);
        int dst_offset = n * (m_num * k_num * block_size) +
                         (m / m_group) * k_num * block_size +
                         (k / k_group) * block_size + addr;
        int src_offset = n * M * K + m * K + k;
        dst[dst_offset] = src[src_offset];
      }
    }
  }
  return result;
}
inline nlohmann::json single_encode(const vsx::Tensor &mask, int n) {
  int height = mask.Shape()[1];
  int width = mask.Shape()[2];

  nlohmann::json rle;
  auto size = nlohmann::json::array();
  size.push_back(height);
  size.push_back(width);
  rle["size"] = size;

  std::vector<uint32_t> count_vec;
  uint32_t count = 0;
  uint8_t previous = 0;

  const uint8_t *data = mask.Data<uint8_t>() + n * width * height;
  for (int w = 0; w < width; w++) {
    for (int h = 0; h < height; h++) {
      auto value = data[h * width + w];
      if (value != previous) {
        count_vec.push_back(count);
        count = 0;
        previous = value;
      }
      ++count;
    }
  }

  count_vec.push_back(count);

  char *s = new char[count_vec.size() * 6 + 1];
  size_t p = 0;
  for (size_t i = 0; i < count_vec.size(); i++) {
    int64_t x = static_cast<int64_t>(count_vec[i]);
    if (i > 2) x -= static_cast<int64_t>(count_vec[i - 2]);
    bool more = true;
    while (more) {
      char c = x & 0x1f;
      x >>= 5;
      more = (c & 0x10) ? x != -1 : x != 0;
      if (more) c |= 0x20;
      c += 48;
      s[p++] = c;
    }
  }

  s[p] = 0;
  std::string result = s;
  rle["counts"] = result;
  delete[] s;

  return rle;
}

inline uint32_t coco80_to_coco91_class(uint32_t class_id) {
  static std::vector<uint32_t> x = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
      54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
      74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
  CHECK(class_id < x.size())
      << "class_id(" << class_id << ") must be smaller than " << x.size();
  return x[class_id];
}

template <typename T>
inline void print_vsx_tensor(const vsx::Tensor &tensor) {
  auto shape = tensor.Shape();
  std::cout << "vsx tensor shape: [";
  for (auto s : shape) {
    std::cout << s << ",";
  }
  std::cout << "]\n";

  if (shape.ndim() > 2) {
    std::cout << "print_vsx_tensor only support dim <= 2 trensor.\n";
    return;
  }
  const T *data = tensor.Data<T>();

  if (shape.ndim() == 1) {
    std::cout << "[";
    for (int i = 0; i < tensor.GetSize(); i++) {
      if (i >= 3 && tensor.GetSize() - i > 3) {
        if (i == 3) std::cout << "... ";
        continue;
      }
      std::cout << data[i] << " ";
    }
    std::cout << "]\n";
    return;
  }

  int height = shape[0];
  int width = shape[1];
  std::cout << "[";
  for (int h = 0; h < height; h++) {
    if (h >= 3 && height - h > 3) {
      if (h == 3) std::cout << "... \n";
      continue;
    }
    std::cout << "[";
    for (int w = 0; w < width; w++) {
      if (w >= 3 && width - w > 3) {
        if (w == 3) std::cout << "... ";
        continue;
      }
      std::cout << data[h * width + w] << " ";
    }
    std::cout << "]\n";
  }
  std::cout << "]\n";
}

template <>
inline void print_vsx_tensor<uint16_t>(const vsx::Tensor &tensor) {
  auto shape = tensor.Shape();
  std::cout << "vsx tensor shape: [";
  for (auto s : shape) {
    std::cout << s << ",";
  }
  std::cout << "]\n";

  if (shape.ndim() > 2) {
    std::cout << "print_vsx_tensor only support dim <= 2 trensor.\n";
    return;
  }
  const uint16_t *data = tensor.Data<uint16_t>();

  if (shape.ndim() == 1) {
    std::cout << "[";
    for (size_t i = 0; i < tensor.GetSize(); i++) {
      if (i >= 3 && tensor.GetSize() - i > 3) {
        if (i == 3) std::cout << "... ";
        continue;
      }
      std::cout << vsx::HalfToFloat(data[i]) << " ";
    }
    std::cout << "]\n";
    return;
  }

  int height = shape[0];
  int width = shape[1];
  std::cout << "[";
  for (int h = 0; h < height; h++) {
    if (h >= 3 && height - h > 3) {
      if (h == 3) std::cout << "... \n";
      continue;
    }
    std::cout << "[";
    for (int w = 0; w < width; w++) {
      if (w >= 3 && width - w > 3) {
        if (w == 3) std::cout << "... ";
        continue;
      }
      std::cout << vsx::HalfToFloat(data[h * width + w]) << " ";
    }
    std::cout << "]\n";
  }
  std::cout << "]\n";
}

}  // namespace vsx