#include <opencv2/opencv.hpp>
#include <string>

#include "common/utils.hpp"
#include "vdsp_ops/rotate_op.hpp"
int main(int argc, char* argv[]) {
  int device_id = 0;
  int rotate_type = atoi(argv[1]);

  vsx::SetDevice(device_id);

  std::string img_path = "../data/images/cycling.jpg";
  cv::Mat img = cv::imread(img_path);

  vsx::RotateOp rotate_op("../samples/ocr_e2e/simple_rotate_debug", device_id);

  vsx::Image vsx_image;
  vsx::MakeVsxImage(img, vsx_image, vsx::ImageFormat::RGB_PLANAR);

  auto result = rotate_op.Process(vsx_image, vsx::rotate_degree_e(rotate_type));

  cv::Mat bgr888;
  vsx::ConvertVsxImageToCvMatBgrPacked(result.Clone(), bgr888);

  cv::imwrite("img_rotated.jpg", bgr888);
  std::cout << "Write result to img_rotated.jpg.\n";
  return 0;
}
