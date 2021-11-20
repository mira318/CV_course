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
  int r, window_size;

  public:
  Huang_hist(const int filter_r){
    r = filter_r;
    window_size = (2 * r + 1) * (2 * r + 1);
  }

  ~Huang_hist() = default;

  void to_string(){
    std::cout << "bins:" << std::endl;
    for(int i = 0; i < bins_num; ++i){
      if(bins[i] != 0){
        for(int t = 0; t < bins[i]; ++t) {
          std::cout << i << " ";
        }
      }
    }
    std::cout << std::endl;
    std::cout << "med = " << static_cast<int>(current_med) << std::endl;
    std::cout << "before_med_sum = " << before_med_sum << std::endl;
  }

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

cv::Mat huang_filter(cv::Mat& original_image, const int filter_r) {
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

  Huang_hist hist = Huang_hist(filter_r);
  uint8_t *first_time_input = new uint8_t[window_size];
  uint8_t *row_input = new uint8_t[2 * filter_r + 1];
  uint8_t *row_output = new uint8_t[2 * filter_r + 1];

  for (int i = 0; i < n - (2 * filter_r); ++i) {
    for (int j = 0; j < m - (2 * filter_r); ++j) {
      if (j == 0) {
        for (int x = 0; x < 2 * filter_r + 1; ++x) {
          for (int y = 0; y < 2 * filter_r + 1; ++y) {
            first_time_input[x * (2 * filter_r + 1) + y] = original_pixels[(i + x) * m + (j + y)];
          }
        }
        new_pixels[(i + filter_r) * m + (j + filter_r)] = hist.find_med_first_time(first_time_input);
      } else {
        for (int x = 0; x < 2 * filter_r + 1; ++x) {
          row_output[x] = original_pixels[(i + x) * m + (j - 1)];
          row_input[x] = original_pixels[(i + x) * m + (j + (2 * filter_r))];
        }
        new_pixels[(i + filter_r) * m + (j + filter_r)] = hist.find_med_in_row(row_output, row_input);
      }
    }
  }
  delete[] first_time_input;
  delete[] row_input;
  delete[] row_output;
  return cv::Mat(n, m, CV_8UC1, new_pixels);
}


class Perrault_hist{
private:
  static const int bins_num = MAX_UINT8_T + 1;
  int current_hist[bins_num];
  uint8_t current_med = 0;
  int before_med_sum = 0;
  int n, m, r, window_size;

  int** rows_hists;
  int row_ind = 0;
  int string_ind = 0;
  uint8_t* pixels;

public:
  Perrault_hist(const int filter_r, const cv::Mat& image){
    r = filter_r;
    window_size = (2 * r + 1) * (2 * r + 1);
    cv::Size sz = image.size();
    n = sz.height;
    m = sz.width;
    pixels = reinterpret_cast<uint8_t*>(image.data);

    // Создаём гистограммы столбцов
    rows_hists = new int*[m];
    for(int j = 0; j < m; ++j){
      rows_hists[j] = new int[bins_num];
      for(int i = 0; i < bins_num; ++i){
        rows_hists[j][i] = 0;
      }
      for(int i = 0; i < 2 * r + 1; ++i){
        rows_hists[j][pixels[i * m + j]]++;
      }
    }
  }

  ~Perrault_hist(){
    // нужно очистить память, которую выделили через new
    for(int i = 0; i < m; ++i){
      delete[] rows_hists[i];
    }
    delete[] rows_hists;
  }

  void to_string(){
    std::cout << "current_hist:" << std::endl;
    for(int i = 0; i < bins_num; ++i){
      if(current_hist[i] != 0){
        for(int t = 0; t < current_hist[i]; ++t) {
          std::cout << i << " ";
        }
      }
    }
    std::cout << std::endl;
    std::cout << "med = " << static_cast<int>(current_med) << std::endl;
    std::cout << "before_med_sum = " << before_med_sum << std::endl;
  }

  uint8_t first_med_in_row(){
    before_med_sum = 0;
    for(int i = 0; i < bins_num; ++i){
      current_hist[i] = 0;
    }

    for(int j = 0; j < 2 * r + 1; ++j){
      for(int i = 0; i < bins_num; ++i){
        current_hist[i] += rows_hists[j][i];
      }
    }

    for(int i = 0; i < bins_num; ++i){
      if(before_med_sum + current_hist[i] >= (window_size + 1) / 2){
        current_med = static_cast<uint8_t>(i);
        break;
      }
      before_med_sum += current_hist[i];
    }
    return current_med;
  }

  uint8_t next_median(){
    if(row_ind == 0){
      // началась новая строка
      if(string_ind != 0){
        // надо "опустить" первые 2r + 1 столбцов
        for(int j = 0; j < 2 * r + 1; ++j){
          rows_hists[j][pixels[(string_ind - 1) * m + j]]--;
          rows_hists[j][pixels[(string_ind + 2 * r) * m + j]]++;
        }
      }
      row_ind++;
      return first_med_in_row();
    }

    // переход в строке
    // перемещаем один пиксель в гистограмме столбца
    int outcoming_row = row_ind - 1;
    int incoming_row = row_ind + 2 * r;
    if(string_ind != 0){
      rows_hists[incoming_row][pixels[(string_ind - 1) * m + incoming_row]]--;
      rows_hists[incoming_row][pixels[(string_ind + 2 * r) * m + incoming_row]]++;
    }

    // Добавляем и удаляем столбцы в гистограмме. Тут же пересчитываем сумму перед медианой
    for(int i = 0; i < bins_num; ++i){
      current_hist[i] += rows_hists[incoming_row][i];
      current_hist[i] -= rows_hists[outcoming_row][i];
      if(i < static_cast<int>(current_med)){
        before_med_sum += rows_hists[incoming_row][i];
        before_med_sum -= rows_hists[outcoming_row][i];
      }
    }

    row_ind++;
    if(row_ind == m - (2 * r)){
      row_ind = 0;
      string_ind++;
    }

    // Обновляем сумму перед медианой и саму медиану как в алгоритме Huang
    if(before_med_sum > window_size / 2){
      while(before_med_sum > window_size / 2){
        current_med--;
        before_med_sum -= current_hist[static_cast<int>(current_med)];
      }
      return current_med;
    }

    if(before_med_sum + current_hist[static_cast<int>(current_med)] <= window_size / 2){
      while(before_med_sum + current_hist[static_cast<int>(current_med)] <= window_size / 2){
        before_med_sum += current_hist[static_cast<int>(current_med)];
        current_med++;
      }
      return current_med;
    }

    return current_med;
  }
};

cv::Mat perrault_filter(cv::Mat& original_image, const int filter_r) {
  int n, m;
  cv::Size sz = original_image.size();
  n = sz.height;
  m = sz.width;
  uint8_t *new_pixels = new uint8_t[n * m];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      new_pixels[i * m + j] = 0;
    }
  }

  Perrault_hist phist = Perrault_hist(filter_r, original_image);
  for (int i = 0; i < n - (2 * filter_r); ++i) {
    for (int j = 0; j < m - (2 * filter_r); ++j) {
      new_pixels[(i + filter_r) * m + (j + filter_r)] = phist.next_median();
    }
  }

  return cv::Mat(n, m, CV_8UC1, new_pixels);
}

uint8_t find_med_by_sort(uint8_t* window, const int filter_r) {
  // Написать в комментарий про std::nth_element
  // Пока у фильтра есть радиус, в нём будет нечётное число элементов => не надо ничего усреднять
  int size = (2 * filter_r + 1) * (2 * filter_r + 1);
  std::nth_element(window, window + size / 2, window + size);
  return window[size / 2];
}

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
  for(int i = 0; i < n - (2 * filter_r); ++i){
    for(int j = 0; j < m - (2 * filter_r); ++j){

      for(int x = 0; x < 2 * filter_r + 1; ++x){
        for(int y = 0; y < 2 * filter_r + 1; ++y){
          filter_window[x * (2 * filter_r + 1) + y] = origin_pixels[(i + x) * m + (j + y)];
        }
      }

      new_pixels[(i + filter_r) * m + (j + filter_r)] = find_med_by_sort(filter_window, filter_r);
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


  cv::Mat naive_filtered = naive_filter(origin, 5);
  cv::Mat huang_filtered = huang_filter(origin, 5);
  cv::Mat perrault_filtered = perrault_filter(origin, 5);

  cv::namedWindow("origin image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("origin image", origin);

  cv::namedWindow("naive filtered image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("naive filtered image", naive_filtered);

  cv::namedWindow("huang filtered image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("huang filtered image", huang_filtered);

  cv::namedWindow("perrault filtered image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE
  imshow("perrault filtered image", perrault_filtered);

  cv::waitKey(0);

  uint8_t* naive_pixel = reinterpret_cast<uint8_t*>(naive_filtered.data);
  uint8_t* huang_pixel = reinterpret_cast<uint8_t*>(huang_filtered.data);
  uint8_t* perrault_pixel = reinterpret_cast<uint8_t*>(perrault_filtered.data);
  cv::Size sz = naive_filtered.size();
  int n = sz.height;
  int m = sz.width;
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      if(huang_pixel[i * m + j] != naive_pixel[i * m + j]){
        std::cout << "ALERT in huang with naive, i = " << i << ", j = " << j << std::endl;
        std::cout << "huang value = " << static_cast<int>(huang_pixel[i * m + j]) << std::endl;
        std::cout << "naive value = " << static_cast<int>(naive_pixel[i * m + j]) << std::endl;

      }
      if(perrault_pixel[i * m + j] != naive_pixel[i * m + j]){
        std::cout << "ALERT in perrault with naive i = " << i << ", j = " << j << std::endl;
        std::cout << "perrault value = " << static_cast<int>(huang_pixel[i * m + j]) << std::endl;
        std::cout << "naive value = " << static_cast<int>(naive_pixel[i * m + j]) << std::endl;
      }
    }
  }

  return 0;
}
