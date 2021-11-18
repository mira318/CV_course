#include <iostream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <time.h>
#include <algorithm> // will not need this

static const int MAX_UINT8_T = 255;

class Huang_hist {
  private:
  static const int bins_num = MAX_UINT8_T + 1;
  int bins[bins_num];
  int before_med_sum = 0;
  uint8_t current_med = 0;
  int window_size, r;

  public:
  Huang_hist(const int sz, const int filter_r){
    window_size = sz;
    r = filter_r;
    for(int i = 0; i < bins_num; ++i){
      bins[i] = 0;
    }
  }

  ~Huang_hist() = default;

  uint8_t find_med_first_time(const uint8_t* window){
    for(int i = 0; i < bins_num; ++i){
      bins[i] = 0;
    }
    for(int i = 0; i < window_size; ++i){
      bins[static_cast<int>(window[i])]++;
    }
    before_med_sum = 0;
    for(int i = 0; i < bins_num; ++i){
      if(before_med_sum + bins[i] >= (window_size + 1) / 2){
        current_med = static_cast<uint8_t>(i);
        break;
      }
      before_med_sum += bins[i];
    }
    return current_med;
  }

  uint8_t find_med_in_row(const uint8_t* output, const uint8_t* input){
    for(int i = 0; i < 2 * r + 1; ++i){
      bins[static_cast<int>(output[i])]--;
      if(output[i] < current_med){
        before_med_sum--;
      }
      bins[static_cast<int>(input[i])]++;
      if(input[i] < current_med){
        before_med_sum++;
      }
    }

    // Если оказалось слишком мало -- идём вправо и прибавляем
    if(before_med_sum > window_size / 2){
      while(before_med_sum > window_size / 2){
        current_med--;
        before_med_sum -= bins[static_cast<int>(current_med)];
      }
      return current_med;
    }

    // Если оказалось слишком много -- идём влево и вычитаем
    if(before_med_sum + bins[static_cast<int>(current_med)] <= window_size / 2){
      while(before_med_sum + bins[static_cast<int>(current_med)] <= window_size / 2){
        before_med_sum += bins[static_cast<int>(current_med)];
        current_med++;
      }
      return current_med;
    }

    //могли не пойти никуда
    return current_med;
  }
};

class Perrault_hist{
private:
  static const int bins_num = MAX_UINT8_T + 1;
  uint8_t current_med = 0;
  int window_size, r, m;


};

uint8_t find_med_by_sort(uint8_t* window, const int filter_r) {
  // Написать в комментарий про std::nth_element
  // Пока у фильтра есть радиус, в нём будет нечётное число элементов => не надо ничего усреднять
  int size = (2 * filter_r + 1) * (2 * filter_r + 1);
  std::nth_element(window, window + size / 2, window + size);
  return window[size / 2];
}

cv::Mat huang_filter(cv::Mat original_image, const int filter_r) {
  int n, m;
  cv::Size sz = original_image.size();
  n = sz.height;
  m = sz.width;
  uint8_t *original_pixels = reinterpret_cast<uint8_t *>(original_image.data);
  uint8_t *new_pixels = new uint8_t[n * m];
  int window_size = (2 * filter_r + 1) * (2 * filter_r + 1);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      new_pixels[i * m + j] = 0;
    }
  }

  Huang_hist hist = Huang_hist(window_size, filter_r);
  uint8_t *first_time_input = new uint8_t[window_size];
  uint8_t *row_input = new uint8_t[2 * filter_r + 1];
  uint8_t *row_output = new uint8_t[2 * filter_r + 1];

  for (int i = 0; i < n - (2 * filter_r + 1); ++i) {
    for (int j = 0; j < m - (2 * filter_r + 1); ++j) {
      if (j == 0) {
        for (int x = 0; x < 2 * filter_r + 1; ++x) {
          for (int y = 0; y < 2 * filter_r + 1; ++y) {
            first_time_input[x * (2 * filter_r + 1) + y] = original_pixels[(i + x) * m + (j + y)];
          }
        }
        new_pixels[(i + filter_r + 1) * m + (j + filter_r + 1)] = hist.find_med_first_time(first_time_input);
      } else {
        for (int x = 0; x < 2 * filter_r + 1; ++x) {
          row_output[x] = original_pixels[(i + x) * m + (j - 1)];
          row_input[x] = original_pixels[(i + x) * m + (j + (2 * filter_r))];
        }
        new_pixels[(i + filter_r + 1) * m + (j + filter_r + 1)] = hist.find_med_in_row(row_output, row_input);
      }
    }
  }
  return cv::Mat(n, m, CV_8UC1, new_pixels);
}



/* void test_med(int size = 9) {
  uint8_t first_input[9] = {255, 255, 255, 255, 255, 255, 255, 255, 255};
  std::sort(first_input, first_input + size);
  std::cout << "first_input = " << std::endl;
  int filter_r = 1;
  Huang_hist hist = Huang_hist(size, filter_r);

  for(int i = 0; i < size; ++i){
    std::cout << static_cast<int>(first_input[i]) << " ";
  }
  std::cout << std::endl;

  uint8_t first_med = hist.find_med_first_time(first_input);
  std::cout << "first med = " << static_cast<int>(first_med) << std::endl;

  uint8_t* input = new uint8_t[2 * filter_r + 1];
  uint8_t* output = new uint8_t[2 * filter_r + 1];

  std::cout << "output:" << std::endl;
  for(int i = 0; i < 2 * filter_r + 1; ++i){
    output[i] = 255;
    std::cout << static_cast<int>(output[i]) << " ";
    input[i] = 255;
  }

  std::cout << std::endl << "input:" << std::endl;
  for(int i = 0; i < 2 * filter_r + 1; ++i){
    std::cout << static_cast<int>(input[i]) << " ";
  }
  std::cout << std::endl;

  int second_med = hist.find_med_in_row(output, input);
  std::cout << "second med = " << static_cast<int>(second_med) << std::endl;
}*/

//cv::Mat constant_time_filter(const cv::Mat& original_image, const int filter_r){
//}










cv::Mat naive_filter(const cv::Mat& original_image, const int filter_r){
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

  std::string input_path = "../images/stones.jpg";
  cv::Mat origin = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
  if(origin.empty())
  {
    std::cout << "Could not read the image: " << input_path << std::endl;
    return 1;
  }

  std::cout << "channels = " << origin.channels() << std::endl;
  std::cout << "type = " << origin.type() << std::endl;

  cv::Mat naive_filtered = naive_filter(origin, 1);
  cv::Mat huang_filtered = huang_filter(origin, 1);

  cv::namedWindow("origin image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("origin image", origin);

  cv::namedWindow("naive filtered image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("naive filtered image", naive_filtered);

  cv::namedWindow("huang filtered image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("huang filtered image", huang_filtered);


  cv::waitKey(0);

  return 0;
}
