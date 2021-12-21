#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

const int MAX_UINT8_T = 255;
const int MAX_2POW_SIZE = 30;

int* fht_merge(const int* acc0, const int* acc1, int n0, int m0){
  int n = n0;
  int m = m0 * 2;
  int* accumulator = new int[n * m];
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      accumulator[m * i + j] = 0;
    }
  }

  int prev_strip;
  int shift_left;
  int shifted_row;
  for(int strip = 0; strip < m; ++strip){
    prev_strip = strip / 2;
    shift_left = strip - prev_strip;
    for(int row = 0; row < n; ++row){
      shifted_row = (row + shift_left) % n;
      accumulator[m * row + strip] = acc0[m0 * row + prev_strip] + acc1[m0 * shifted_row + prev_strip];
    }
  }

  if(m > 2){
    delete[] acc0;
    delete[] acc1;
  }
  return accumulator;
}

int* fht(int* pixels_int, int n, int m){
  if(m < 2){
    return pixels_int;
  } else {
    int mid = m / 2;
    int mid_left = m - mid;
    int* left_half = new int[n * mid];
    int* right_half = new int[n * mid_left];

    for(int i = 0; i < n; ++i){
      for(int j = 0; j < mid; ++j){
        left_half[i * mid + j] = pixels_int[i * m + j];
      }
    }

    for(int i = 0; i < n; ++i){
      for(int j = 0; j < mid_left; ++j){
        right_half[i * mid_left + j] = pixels_int[i * m + (j + mid)];
      }
    }

    int* current_res = fht_merge(fht(left_half, n, mid),
                     fht(right_half, n, mid_left), n, mid);
    delete[] left_half;
    delete[] right_half;
    return current_res;
  }
}

uint8_t* padding(const uint8_t* input, int* n_ptr, int* m_ptr){
  // Находим степени 2, сделаем это 1 раз
  int pows[MAX_2POW_SIZE + 1];
  pows[0] = 1;
  for(int i = 1; i <= MAX_2POW_SIZE; ++i){
    pows[i] = pows[i - 1] * 2;
  }

  // Находим, до каких границ сделать паддинг
  int old_n = *n_ptr;
  int old_m = *m_ptr;
  int j = 0;
  while(old_n > pows[j]){
    j++;
  }
  int new_n = pows[j];
  j = 0;
  while(old_m > pows[j]){
    j++;
  }
  int new_m = pows[j];

  // Заполняем известные значения
  uint8_t* padded = new uint8_t[new_n * new_m];
  for(int i = 0; i < old_n; ++i){
    for(int j = 0; j < old_m; ++j){
      padded[i * new_m + j] = input[i * old_m + j];
    }
  }

  // Заполняем края нулями
  for(int i = old_n; i < new_n; ++i){
    for(int j = 0; j < new_m; ++j){
      padded[i * new_m + j] = 0;
    }
  }
  for(int i = 0; i < old_n; ++i){
    for(int j = old_m; j < new_m; ++j){
      padded[i * new_m + j] = 0;
    }
  }

  *n_ptr = new_n;
  *m_ptr = new_m;
  return padded;
}

uint8_t* rotate_left(const uint8_t* input, int n, int m){
  uint8_t* res = new uint8_t[m * n];
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      res[i * n + j] = input[j * m + (m - i - 1)];
    }
  }
  return res;
}

uint8_t* rotate_right(const uint8_t* input, int n, int m){
  uint8_t* res = new uint8_t[m * n];
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      res[i * n + j] = input[(n - j - 1) * m + i];
    }
  }
  return res;
}

uint8_t* rotate_mirror(const uint8_t* input, int n, int m){
  uint8_t* res = new uint8_t[n * m];
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      res[i * m + j] = input[(n - i - 1) * m + (m - j - 1)];
    }
  }
  return res;
}

int* fht(cv::Mat input_image){
  cv::Size sz = input_image.size();
  int n = sz.width;
  int m = sz.height;
  int st = 1;
  uint8_t* pixels = reinterpret_cast<uint8_t*>(input_image.data);
  int* pixels_int = new int[n * m];
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      pixels_int[i * m + j] = static_cast<int>(pixels[i * m + j]);
    }
  }
  return fht(pixels_int, n, m);
}

cv::Mat visualize_fht(int* fht_res, int n, int m){
  long double cur_max = 0;
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      if(fht_res[i * m + j] > cur_max){
        cur_max = fht_res[i * m + j];
      }
    }
  }

  uint8_t* res_image = new uint8_t[n * m];
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      res_image[i * m + j] = round(MAX_UINT8_T * (fht_res[i * m + j] / cur_max));
    }
  }
  return cv::Mat(n, m, CV_8UC1, res_image);
}

int main() {
  cv::Mat input_image = cv::imread("../images/1.jpg", cv::IMREAD_GRAYSCALE);
  cv::Size sz = input_image.size();
  int n = sz.height;
  int m = sz.width;
  uint8_t* new_pixels = reinterpret_cast<uint8_t*>(input_image.data);
  uint8_t* padded = padding(new_pixels, &n, &m);
  cv::Mat created = cv::Mat(n, m, CV_8UC1, padded);
  std::string window_name = "created image";
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::imshow(window_name, created);

  uint8_t* left_res = rotate_left(padded, n, m);
  cv::Mat rotated_left = cv::Mat(m, n, CV_8UC1, left_res);
  std::string window_left_name = "left rot image";
  cv::namedWindow(window_left_name, cv::WINDOW_NORMAL);
  cv::imshow(window_left_name, rotated_left);

  uint8_t* right_res = rotate_right(padded, n, m);
  cv::Mat rotated_right = cv::Mat(m, n, CV_8UC1, right_res);
  std::string window_right_name = "right rot image";
  cv::namedWindow(window_right_name, cv::WINDOW_NORMAL);
  cv::imshow(window_right_name, rotated_right);

  uint8_t* mirror_res = rotate_mirror(padded, n, m);
  cv::Mat rotated_mirror = cv::Mat(n, m, CV_8UC1, mirror_res);
  std::string window_mirror_name = "twice rot image";
  cv::namedWindow(window_mirror_name, cv::WINDOW_NORMAL);
  cv::imshow(window_mirror_name, rotated_mirror);

  cv::waitKey(0);
  cv::destroyWindow(window_name);
  cv::destroyWindow(window_left_name);
  cv::destroyWindow(window_right_name);
  cv::destroyWindow(window_mirror_name);
  
  int* res_straight = fht(created);
  int* res_left = fht(rotated_left);
  int* res_right = fht(rotated_right);
  int* res_mirror = fht(rotated_mirror);

  cv::Mat visual_fht = visualize_fht(res_straight, n, m);
  std::string window_fht_name = "fht image";
  cv::namedWindow(window_fht_name, cv::WINDOW_NORMAL);
  cv::imshow(window_fht_name, visual_fht);

  cv::Mat visual_fht_left = visualize_fht(res_left, m, n);
  std::string window_left_fht_name = "fht image left rotated";
  cv::namedWindow(window_left_fht_name, cv::WINDOW_NORMAL);
  cv::imshow(window_left_fht_name, visual_fht_left);

  cv::Mat visual_fht_right = visualize_fht(res_right, m, n);
  std::string window_right_fht_name = "fht image right rotated";
  cv::namedWindow(window_right_fht_name, cv::WINDOW_NORMAL);
  cv::imshow(window_right_fht_name, visual_fht_right);

  cv::Mat visual_fht_mirror = visualize_fht(res_mirror, n, m);
  std::string window_mirror_fht_name = "fht image twice rotated";
  cv::namedWindow(window_mirror_fht_name, cv::WINDOW_NORMAL);
  cv::imshow(window_mirror_fht_name, visual_fht_mirror);

  cv::waitKey(0);
  cv::destroyWindow(window_fht_name);
  cv::destroyWindow(window_left_fht_name);
  cv::destroyWindow(window_right_fht_name);
  cv::destroyWindow(window_mirror_fht_name);


  created.release();
  rotated_left.release();
  rotated_right.release();
  rotated_mirror.release();
  visual_fht.release();
  visual_fht_left.release();
  visual_fht_right.release();
  visual_fht_mirror.release();

  delete[] res_straight;
  delete[] res_left;
  delete[] res_right;
  delete[] res_mirror;
  return 0;
}
