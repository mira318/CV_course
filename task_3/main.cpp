#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

int* fht_merge(int* acc0, int* acc1, int n0, int m0){
  std::cout << "started fht_merge, n0 = " << n0 << ", m0 = " << m0 <<std::endl;
  std::cout << "acc0 = " << std::endl;
  for(int i = 0; i < n0; ++i){
    for(int j = 0; j < m0; ++j){
      std::cout << acc0[i * m0 + j] << " ";
    }
    std::cout << std::endl;
  }

  for(int i = 0; i < n0; ++i){
    for(int j = 0; j < m0; ++j){
      std::cout << acc1[i * m0 + j] << " ";
    }
    std::cout << std::endl;
  }

  int n = n0;
  int m = m0 * 2;
  int* accumulator = new int[n * m];
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      accumulator[n * i + j] = 0;
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
      accumulator[m * row + strip] = acc0[m * row + prev_strip] + acc1[m * shifted_row + prev_strip];
    }
  }

  if(m > 2){
    delete[] acc0;
    delete[] acc1;
  }
  return accumulator;
}

int* fht(uint8_t* pixels, int n, int m){
  std::cout << "started fht, n = "<< n << ", m = " << m << std::endl;
  if(m < 2){
    int* returned = new int[n];
    std::cout << "returned = " << std::endl;
    for(int i = 0; i < n; ++i){
      // Нет, я не могу скастовать указатель
      returned[i] = static_cast<int>(pixels[i]);
      std::cout << returned[i] << " ";
    }
    std::cout << std::endl;
    return returned;
  } else {
    int mid = m / 2;
    std::cout << "mid = " << mid << std::endl;
    int mid_left = m - mid;
    uint8_t left_half[n * mid];
    uint8_t right_half[n * mid_left];

    for(int i = 0; i < n; ++i){
      for(int j = 0; j < mid; ++j){
        left_half[i * mid + j] = pixels[i * m + j];
      }
    }

    for(int i = 0; i < n; ++i){
      for(int j = 0; j < mid; ++j){
        right_half[i * mid_left + j] = pixels[i * m + (j + mid)];
      }
    }

    std::cout << "left_half = " << std::endl;
    for(int i = 0; i < n; ++i){
      for(int j = 0; j < mid; ++j){
        std::cout << static_cast<int>(left_half[i * mid + j]) << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "right_half = " << std::endl;
    for(int i = 0; i < n; ++i){
      for(int j = 0; j < mid_left; ++j){
        std::cout << static_cast<int>(right_half[i * mid_left + j]) << " ";
      }
      std::cout << std::endl;
    }

    return fht_merge(fht(left_half, n, mid),
                     fht(right_half, n, mid_left), n, mid);
  }
}

int* fht(cv::Mat input_image){
  cv::Size sz = input_image.size();
  int n = sz.width;
  int m = sz.height;
  int st = 1;
  uint8_t* pixels = reinterpret_cast<uint8_t*>(input_image.data);
  int* pixels_int = new int[n * m];
  /*for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      pixels_int[i * m + j] = static_cast<int>(pixels[i * m + j]);
    }
  }*/
  return fht(pixels_int, n, m);
}

int main() {
  int n = 8;
  int m = 8;
  uint8_t* new_pixels = new uint8_t[n * m];
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      if(i == j){
        new_pixels[i * n + j] = 255;
      } else {
        new_pixels[i * n + j] = 0;
      }
    }
  }
  cv::Mat created = cv::Mat(n, m, CV_8UC1, new_pixels);
  std::string window_name = "created image";
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::imshow(window_name, created);
  cv::waitKey(0);
  int* res = fht(created);
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      std::cout << res[i * m + j] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
