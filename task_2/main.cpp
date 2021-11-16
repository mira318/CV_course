#include <iostream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <time.h>

static const int MAX_UINT8_T = 255;

class Huang_hist {
  private:
  static const int bins_num = MAX_UINT8_T;
  int bins[bins_num];
  int before_med_sum = 0;
  uint8_t current_med = 0;
  int window_size;

  public:
  Huang_hist(int sz){
    window_size = sz;
    for(int bin: bins){
      bin = 0;
    }
  }

  ~Huang_hist() = default;

  uint8_t find_med_first_time(const uint8_t* window){
    for(int bin: bins){
      bin = 0;
    }
    for(int i = 0; i < window_size; ++i){
      bins[window[i]]++;
    }
    int cur_less_sum = 0;
    for(int i = 0; i < bins_num; ++i){

    }
  }
};














uint8_t find_med_by_sort(uint8_t* window, int filter_r) {
  // Написать в комментарий про std::nth_element
  int size = (2 * filter_r + 1) * (2 * filter_r + 1);
  if (size % 2 == 0) {
    // в данном случае медиана должна считаться по двум средним элементам
    std::nth_element(window, window + size / 2, window + size);
    std::nth_element(window, window + (size - 1) / 2, window + size);

    // математическое округление в делении
    return ((window[(size - 1) / 2] + window[size / 2]) + 1) / 2;
  } else {
    std::nth_element(window, window + size / 2, window + size);
    return window[size / 2];
  }
}

cv::Mat naive_filter(const cv::Mat& original_image, int filter_r){
  int n, m;
  cv::Size sz = original_image.size();
  n = sz.height;
  m = sz.width;
  uint8_t* origin_pixels = reinterpret_cast<uint8_t*>(original_image.data);
  uint8_t* new_pixels = new uint8_t[n * m];
  uint8_t filter_window[(2 * filter_r + 1) * (2 * filter_r + 1)];

  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      new_pixels[i * m + j] = 0;
    }
  }
  for(int i = 0; i < n - (2 * filter_r + 1); ++i){
    for(int j = 0; j < m - (2 * filter_r + 1); ++j){
      for(int x = 0; x < 2 * filter_r + 1; ++x){
        for(int y = 0; y < 2 * filter_r + 1; ++y){
          filter_window[x * (2 * filter_r + 1) + y] = origin_pixels[(i + x) * m + (j + y)];
          new_pixels[(i + filter_r + 1) * m + (j + filter_r + 1)] = find_med_by_sort(filter_window, filter_r);
        }
      }
    }
  }
  return cv::Mat(n, m, CV_8UC1, new_pixels);
}

int main() {

  std::string input_path = "../images/trees.jpg";
  cv::Mat origin = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
  if(origin.empty())
  {
    std::cout << "Could not read the image: " << input_path << std::endl;
    return 1;
  }

  std::cout << "channels = " << origin.channels() << std::endl;
  std::cout << "type = " << origin.type() << std::endl;

  cv::Mat naive_filtered = naive_filter(origin, 1);

  cv::namedWindow("origin image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("origin image", origin);
  cv::namedWindow("naive filtered image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("naive filtered image", naive_filtered);

  cv::waitKey(0);


  return 0;
}
