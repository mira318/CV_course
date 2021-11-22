#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <ctime>

static const int MAX_UINT8_T = 255;

uint8_t* add_borders(const uint8_t* originals, const int n, const int m, const int r){
  /* Функция, чтобы добавить края: нужно добавить по r пикселей с каждой стороны. Края монотонные: дублируем последний
   * известный пиксель в заданном направлении. Это необходимо, чтобы результат можно было сравнить с результатом
   * MedianBlur от opencv.
   * */
  int new_n = n + 2 * r;
  int new_m = m + 2 * r;
  uint8_t* new_pixels = new uint8_t[new_n * new_m];

  // Копируем известные значения
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      new_pixels[(i + r) * new_m + (j + r)] = originals[i * m + j];
    }
  }

  // Растягиваем столбцы
  for(int j = 0; j < m; ++j){
    for(int t = 0; t < r; ++t){
      new_pixels[t * new_m + (j + r)] = new_pixels[r * new_m + (j + r)];
      new_pixels[(new_n - t - 1) * new_m  + (j + r)] = new_pixels[(new_n - r - 1) * new_m + (j + r)];
    }
  }

  // Растягиваем строки, в том числе новые
  for(int i = 0; i < new_n; ++i){
    for(int t = 0; t < r; ++t){
      new_pixels[i * new_m + t] = new_pixels[i * new_m + r];
      new_pixels[i * new_m + (new_m - t - 1)] = new_pixels[i * new_m + (new_m - r - 1)];
    }
  }
  return new_pixels;
}


class Huang_hist {
  /*
   * Класс, чтобы поддерживать гистограмму в алгоритме Huang
   */
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
    // Выводящая функция для дебага
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

  uint8_t first_med_in_row(const uint8_t* window){
    // Строим новую гистограмму
    for(int i = 0; i < bins_num; ++i){
      bins[i] = 0;
    }
    for(int i = 0; i < window_size; ++i){
      bins[static_cast<int>(window[i])]++;
    }
    before_med_sum = 0;

    // Пересчитываем медиану и сумму перед ней "вручную"
    for(int i = 0; i < bins_num; ++i){
      if(before_med_sum + bins[i] >= (window_size + 1) / 2){
        current_med = static_cast<uint8_t>(i);
        break;
      }
      before_med_sum += bins[i];
    }
    return current_med;
  }

  uint8_t next_med_in_row(const uint8_t* output, const uint8_t* input){
    // Добавляем новый и удаляем старый столбцы по пикселям
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

    // Пересчитываем медиану и сумму перед ней.
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

cv::Mat huang_filter(cv::Mat& original_image, const int filter_r, long double* timer_place = nullptr) {
  /*
   * Функция, реализующая медианную фильтрацию алгоритмом Huang. Сам поиск медианы и хранение гистограммы осуществляет
   * hist из класса Huang_hist
   */
  int n, m;
  cv::Size sz = original_image.size();
  n = sz.height;
  m = sz.width;
  int window_size = (2 * filter_r + 1) * (2 * filter_r + 1);
  uint8_t* original_pixels = reinterpret_cast<uint8_t*>(original_image.data);
  uint8_t* new_pixels = new uint8_t[n * m];

  int new_n = n + 2 * filter_r;
  int new_m = m + 2 * filter_r;
  uint8_t* bordered_pixels = add_borders(original_pixels, n, m, filter_r);

  Huang_hist hist = Huang_hist(filter_r);
  uint8_t* first_time_input = new uint8_t[window_size]; // чтобы передавать элементы первой гистограммы в строке
  uint8_t* row_input = new uint8_t[2 * filter_r + 1]; // чтобы передавать список добавляемых элементов
  uint8_t* row_output = new uint8_t[2 * filter_r + 1]; // чтобы передавать список удаляемых элементов

  time_t start = clock();
  for (int i = 0; i < new_n - (2 * filter_r); ++i) {
    for (int j = 0; j < new_m - (2 * filter_r); ++j) {
      if (j == 0) {
        // строка только началась
        for (int x = 0; x < 2 * filter_r + 1; ++x) {
          for (int y = 0; y < 2 * filter_r + 1; ++y) {
            first_time_input[x * (2 * filter_r + 1) + y] = bordered_pixels[(i + x) * new_m + (j + y)];
          }
        }
        new_pixels[i * m + j] = hist.first_med_in_row(first_time_input);
      } else {
        // номер столбца j != 0
        for (int x = 0; x < 2 * filter_r + 1; ++x) {
          row_output[x] = bordered_pixels[(i + x) * new_m + (j - 1)];
          row_input[x] = bordered_pixels[(i + x) * new_m + (j + (2 * filter_r))];
        }
        new_pixels[i * m + j] = hist.next_med_in_row(row_output, row_input);
      }
    }
  }
  time_t finish = clock();

  if(timer_place != nullptr){
    *timer_place = static_cast<long double>(finish - start) / CLOCKS_PER_SEC;
  }

  delete[] bordered_pixels;
  bordered_pixels = nullptr;
  delete[] first_time_input;
  delete[] row_input;
  delete[] row_output;
  return cv::Mat(n, m, CV_8UC1, new_pixels);
}


class Perreault_hist{
  /*
   * Класс, чтобы поддерживать гистограмму в алгоритме за O(1).
   */
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
  Perreault_hist(uint8_t* original_pixels, const int orig_n, const int orig_m, const int filter_r){
    /* В конструкторе придётся принять все пиксели сразу, потому что необходимо считать и обновлять гистограммы
    ** по столбцам. В итоге основная работа с изображением почти полностью происходит внутри класса.
    */
    pixels = original_pixels;
    n = orig_n;
    m = orig_m;
    r = filter_r;
    window_size = (2 * r + 1) * (2 * r + 1);

    // Создаём и инициализируем гистограммы столбцов
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

  ~Perreault_hist(){
    // нужно очистить память, которую выделили через new под гистограммы столбцов.
    for(int i = 0; i < m; ++i){
      delete[] rows_hists[i];
    }
    delete[] rows_hists;
  }

  void to_string(){
    // Выводящая функция для дебага.
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
    // Первую гистограмму в строке всё ещё надо пересчитывать заново, но теперь будем складывать сразу столбцы,
    // а не по пикселям.

    before_med_sum = 0;
    for(int i = 0; i < bins_num; ++i){
      current_hist[i] = 0;
    }
    for(int j = 0; j < 2 * r + 1; ++j){
      for(int i = 0; i < bins_num; ++i){
        current_hist[i] += rows_hists[j][i];
      }
    }

    // Вычисляем медиану и before_med_sum простым проходом по гистограмме
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
      // началась новая строка, будем считать медиану через first_med_in_row,
      // но сначала надо "опустить" первые 2r + 1 столбцов
      if(string_ind != 0){
        for(int j = 0; j < 2 * r + 1; ++j){
          rows_hists[j][pixels[(string_ind - 1) * m + j]]--;
          rows_hists[j][pixels[(string_ind + 2 * r) * m + j]]++;
        }
      }
      row_ind++;
      return first_med_in_row();
    }

    /* переход внутри строки */
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

    // Класс хранит и поддерживает собственные индексы. Это нужно, чтобы вызывать first_med_in_row в начале строки
    // и следить за добавляемыми/удаляемыми столбцами.
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

cv::Mat perreault_filter(cv::Mat& original_image, const int filter_r, long double* timer_place = nullptr) {
  /*
   * Функция, которая реализует медианную фильтрацию за O(1) на пиксель. (Perrault - фамилия первого из авторов статьи)
   * Основная работа: подсчёт гистограмм по столбцам, их сложение и нахождение всех медиан - происходит внутри
   * гистограммы phist из класса Perrault_hist
   */
  int n, m;
  cv::Size sz = original_image.size();
  n = sz.height;
  m = sz.width;
  uint8_t* original_pixels = reinterpret_cast<uint8_t*>(original_image.data);
  uint8_t* new_pixels = new uint8_t[n * m];

  int new_n = n + 2 * filter_r;
  int new_m = m + 2 * filter_r;
  uint8_t* bordered_pixels = add_borders(original_pixels, n, m, filter_r);

  time_t start = clock();
  Perreault_hist phist = Perreault_hist(bordered_pixels, new_n, new_m, filter_r);
  for (int i = 0; i < new_n - (2 * filter_r); ++i) {
    for (int j = 0; j < new_m - (2 * filter_r); ++j) {
      new_pixels[i * m + j] = phist.next_median();
    }
  }
  time_t finish = clock();

  if(timer_place != nullptr){
    *timer_place = static_cast<long double>(finish - start) / CLOCKS_PER_SEC;
  }

  delete[] bordered_pixels;
  bordered_pixels = nullptr;
  return cv::Mat(n, m, CV_8UC1, new_pixels);
}

uint8_t find_med_by_sort(uint8_t* window, const int filter_r) {
  /*
   * Функция для поиска медианы в алгоритме с простой сортировкой.
   * Для подсчёта самой медианы использую nth_element. Эта функция ищет порядковую статистику
   * с заданным номером и разделяет массив относительно неё. Она несколько быстрее чем сортировка, и в среднем работает
   * за O(n), но в худшем случае асимптотика остаётся равной O(nlogn).
   * */
  int window_size = (2 * filter_r + 1) * (2 * filter_r + 1);
  // Пока у фильтра есть радиус, в нём будет нечётное число элементов => в пересчёте медианы не придётся усреднять 2
  // соседних элемента массива из середины.
  std::nth_element(window, window + window_size / 2, window + window_size);
  return window[window_size / 2];
}

cv::Mat naive_filter(const cv::Mat& original_image, const int filter_r, long double* timer_place = nullptr){
  /*
   * Функция, которая реализует медианную фильтрацию с помощью простой сортировки. Сама медиана считается в функции
   * find_med_by_sort.
   */
  cv::Size sz = original_image.size();
  int n = sz.height;
  int m = sz.width;
  uint8_t filter_window[(2 * filter_r + 1) * (2 * filter_r + 1)];
  uint8_t* original_pixels = reinterpret_cast<uint8_t*>(original_image.data);
  uint8_t* new_pixels = new uint8_t[n * m];

  int new_n = n + 2 * filter_r;
  int new_m = m + 2 * filter_r;
  uint8_t* bordered_pixels = add_borders(original_pixels, n, m, filter_r);

  time_t start = clock();
  for(int i = 0; i < new_n - (2 * filter_r); ++i){
    for(int j = 0; j < new_m - (2 * filter_r); ++j){

      // Складываем в массив текущие элементы фильтра. Не вспоминаем о том, что текущий фильтр похож на предыдущий.
      for(int x = 0; x < 2 * filter_r + 1; ++x){
        for(int y = 0; y < 2 * filter_r + 1; ++y){
          filter_window[x * (2 * filter_r + 1) + y] = bordered_pixels[(i + x) * new_m + (j + y)];
        }
      }

      new_pixels[i * m + j] = find_med_by_sort(filter_window, filter_r);
    }
  }
  time_t finish = clock();
  if(timer_place != nullptr){
    *timer_place = static_cast<long double>(finish - start) / CLOCKS_PER_SEC;
  }
  // Выделяли память, когда добавляли границы.
  delete[] bordered_pixels;
  bordered_pixels = nullptr;
  return cv::Mat(n, m, CV_8UC1, new_pixels);
}

bool check_similarity(cv::Mat& original_image, const int filter_r, bool save = false, std::string opencv_file = "\0",
                      std::string naive_file = "\0", std::string huang_file = "\0", std::string perreault_file = "\0"){
  /*
   * Функция для проверки правильности алгоритмов. Запускает все 4 (вместе с cv::medianBlur) алгоритма для фиксированных
   * картинки и радиуса и сравнивает результаты.
   * */
  cv::Size sz = original_image.size();
  int n = sz.height;
  int m = sz.width;
  bool ans = true;
  long double timer_place;
  cv::Mat opencv_filtered = cv::Mat(sz.height, sz.width, CV_8UC1);
  time_t start = clock();
  // В официальной документации написано, что medianBlur принимает ksize. Это размер квадратной матрицы с фильтром.
  // Т. е., если я хочу работать с радиусом, надо пересчитать.
  cv::medianBlur(original_image, opencv_filtered, 2 * filter_r + 1);
  time_t finish = clock();
  std::cout << "opencv time in seconds = " << static_cast<long double>(finish - start) / CLOCKS_PER_SEC << std::endl;

  cv::Mat naive_filtered = naive_filter(original_image, filter_r, &timer_place);
  std::cout << "sorting algorithm time in seconds = " << timer_place << std::endl;

  cv::Mat huang_filtered = huang_filter(original_image, filter_r, &timer_place);
  std::cout << "huang algorithm time in seconds = " << timer_place << std::endl;

  cv::Mat perreault_filtered = perreault_filter(original_image, filter_r, &timer_place);
  std::cout << "perreault (O(1)) algorithm time in seconds = " << timer_place << std::endl;

  uint8_t* naive_pixel = reinterpret_cast<uint8_t*>(naive_filtered.data);
  uint8_t* huang_pixel = reinterpret_cast<uint8_t*>(huang_filtered.data);
  uint8_t* perreault_pixel = reinterpret_cast<uint8_t*>(perreault_filtered.data);
  uint8_t* opencv_pixel = reinterpret_cast<uint8_t*>(opencv_filtered.data);

  // Возможно, стоило использовать assert, но мне хотелось видеть, где именно ошибка.
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      if(huang_pixel[i * m + j] != naive_pixel[i * m + j]){
        std::cout << "ALERT in Huang with naive i = " << i << ", j = " << j << std::endl;
        std::cout << "huang value = " << static_cast<int>(huang_pixel[i * m + j]) << std::endl;
        std::cout << "naive value = " << static_cast<int>(naive_pixel[i * m + j]) << std::endl;
        ans = false;
      }
      if(perreault_pixel[i * m + j] != naive_pixel[i * m + j]){
        std::cout << "ALERT in perrault with naive i = " << i << ", j = " << j << std::endl;
        std::cout << "perrault value = " << static_cast<int>(huang_pixel[i * m + j]) << std::endl;
        std::cout << "naive value = " << static_cast<int>(naive_pixel[i * m + j]) << std::endl;
        ans = false;
      }
      if(naive_pixel[i * m + j] != opencv_pixel[i * m + j]){
        std::cout << "ALERT in naive with opencv, i = " << i << ", j = " << j << std::endl;
        std::cout << "naive value = " << static_cast<int>(naive_pixel[i * m + j]) << std::endl;
        std::cout << "opencv value = " << static_cast<int>(opencv_pixel[i * m + j]) << std::endl;
        ans = false;
      }
    }
  }

  if(save){
    try{
      cv::imwrite(opencv_file, opencv_filtered);
    }
    catch (cv::Exception ex){
      std::cout << "Exception while writing first image (opencv) to the file: " << ex.what() << std::endl;
    }

    try{
      cv::imwrite(naive_file, naive_filtered);
    }
    catch (cv::Exception ex){
      std::cout << "Exception while writing second image (naive) to the file: " << ex.what() << std::endl;
    }

    try{
      cv::imwrite(huang_file, huang_filtered);
    }
    catch (cv::Exception ex){
      std::cout << "Exception while writing third image (Huang) to the file: " << ex.what() << std::endl;
    }

    try{
      cv::imwrite(perreault_file, perreault_filtered);
    }
    catch (cv::Exception ex){
      std::cout << "Exception while writing forth image (Perrault) to the file: " << ex.what() << std::endl;
    }
  }

  return ans;
}

void time_register(cv::Mat& test_image, char way){
  long double timer;

  switch(way){
    case 'c': {
      std::cout << "Times for opencv medianBlur:" << std::endl;
      cv::Size sz = test_image.size();
      int n = sz.height;
      int m = sz.width;
      cv::Mat opencv_filtered = cv::Mat(n, m, CV_8UC1);
      for (int r = 1; r < 100; ++r) {
        time_t start = clock();
        cv::medianBlur(test_image, opencv_filtered, 2 * r + 1);
        time_t finish = clock();
        timer = static_cast<long double>(finish - start) / CLOCKS_PER_SEC;
        std::cout << timer << " ";
        fflush(stdout);
      }
      break;
    }
    case 'n': {
      cv::Mat res;
      std::cout << "Times for naive median filter with assortment:" << std::endl;
      for (int r = 1; r < 100; ++r) {
        res = naive_filter(test_image, 2 * r + 1, &timer);
        std::cout << timer << " ";
        fflush(stdout);
      }
      break;
    }
    case 'h': {
      cv::Mat res;
      std::cout << "Times for Huang median filter:" << std::endl;
      for (int r = 1; r < 100; ++r) {
        huang_filter(test_image, 2 * r + 1, &timer);
        std::cout << timer << " ";
        fflush(stdout);
      }
      break;
    }
    case 'p': {
      cv::Mat res;
      std::cout << "Times for Perrault median filter that claimed to be O(1):" << std::endl;
      for (int r = 1; r < 100; ++r) {
        perreault_filter(test_image, 2 * r + 1, &timer);
        std::cout << timer << " ";
        fflush(stdout);
      }
      break;
    }
    default: {
      break;
    }
  }
}

bool check_file(const cv::Mat& get_image){
  /*
   * Проверяем, что дали изображение формата CV_8UC1.
   */
  if(get_image.channels() != 1){
    std::cout << "Wrong number of channels. Should be 3." << std::endl;
    return false;
  }
  if(get_image.type() != 0){
    std::cout << "Wrong pixel values. Should be from 0 to 255." << std::endl;
    return false;
  }
  return true;
}

void helper_writer(){
  std::cout << "Unknown command. ways of usage: " << std::endl;
  std::cout << "    <filter> <inputFilePath> <outputFilePath> <r>: use <filter> with radius r on the image from "
               "<inputFilePath> and write the result to the <outputFilePath>" << std::endl;
  std::cout << "    <filter> <inputFilePath>: use <filter> with radius from 1 to 99 and write down the time in "
               "seconds. This function may take some time, since not all algorithms are fast." << std::endl;
  std::cout << "    compare <inputFilePath> <r>: use all 4 filters on the image from <inputFilePath> with radius r. "
               "Check if the results are similar and report weather mistakes occurred." << std::endl;
  std::cout << "    compare <inputFilePath> <r> <outputFilePathCV> <outputFilePathSort> <outputFilePathHuang> "
               "<outputFilePathPerreault>: use all 4 filters on the image from <inputFilePath> with radius r."
            << std::endl << "    Check if the results are similar and report weather mistakes occurred. "
                            "Write the results into given files in this order:" << std::endl;
  std::cout << "        first - opencv" << std::endl;
  std::cout << "        second - algorithm with simple sort" << std::endl;
  std::cout << "        third - Huang algorithm" << std::endl;
  std::cout << "        fourth - Perrault algorithm, which claimed to be O(1) por pixel." << std::endl;
  std::cout << "<filter> is one of following:" << std::endl;
  std::cout << "-opencv - for standard medianBlur filter from opencv." << std::endl;
  std::cout << "-sort - for naive algorithm, that finds a median by sorting the elements." << std::endl;
  std::cout << "-huang - for Huang algorithm, that should be O(n) por pixel." << std::endl;
  std::cout << "-perreault - for the algorithm, that described in article by Simon Perreault and Patrick Hébert. "
               "Claimed to be O(1) por pixel." << std::endl;
}
int main(int argc, char** argv) {
  if((argc < 2) || (argc == 6) || (argc == 7) || (argc > 9)){
    helper_writer();
    return 0;
  }
  std::string arg1 = std::string(argv[1]);

  if(argc == 5){
    std::string input_path = std::string(argv[2]);
    std::string output_path = std::string(argv[3]);
    int filter_r = std::stoi(std::string(argv[4]));
    cv::Mat original_image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if(original_image.empty())
    {
      std::cout << "Could not read the image: " << input_path << std::endl;
      return -1;
    }
    if(check_file(original_image)){
      cv::Size sz = original_image.size();
      int n = sz.height;
      int m = sz.width;
      cv::Mat filtered = cv::Mat(n, m, CV_8UC1);

      if(arg1 == "-opencv"){
        cv::medianBlur(original_image, filtered, 2 * filter_r + 1);
      } else {
        if(arg1 == "-sort"){
          filtered = naive_filter(original_image, filter_r);
        } else {
          if(arg1 == "-huang"){
            filtered = huang_filter(original_image, filter_r);
          } else {
            if(arg1 == "-perreault"){
              filtered = perreault_filter(original_image, filter_r);
            } else {
              std::cout << "No such filter" << std::endl;
            }
          }
        }
      }
      try {
        cv::imwrite(output_path, filtered);
      }
      catch (cv::Exception& ex){
        std::cout << "Exception while writing to the output file " << ex.what() << std::endl;
      }
      return 0;
    } else {
      return -1;
    }
  }

  if(argc == 3){
    std::string input_path = std::string(argv[2]);
    cv::Mat original_image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if(original_image.empty())
    {
      std::cout << "Could not read the image: " << input_path << std::endl;
      return -1;
    }
    if(check_file(original_image)){
      if(arg1 == "-opencv"){
        time_register(original_image, 'c');
        return 0;
      } else {
        if(arg1 == "-sort"){
          time_register(original_image, 'n');
          return 0;
        } else {
          if(arg1 == "-huang"){
            time_register(original_image, 'h');
            return 0;
          } else {
            if(arg1 == "-perreault"){
              time_register(original_image, 'p');
              return 0;
            } else {
              std::cout << "No such filter" << std::endl;
              return 0;
            }
          }
        }
      }
    } else {
      return -1;
    }
  }

  if (arg1 == "-compare") {
    std::string input_path = std::string(argv[2]);
    int filter_r = std::stoi(std::string(argv[3]));
    cv::Mat original_image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (original_image.empty()) {
      std::cout << "Could not read the image: " << input_path << std::endl;
      return -1;
    }
    if (check_file(original_image)) {
      if (argc == 4) {
        if (check_similarity(original_image, filter_r)) {
          std::cout << "All 4 results are similar." << std::endl;
          return 0;
        }
      }
      if (argc == 8) {
        std::string opencv_file = std::string(argv[4]);
        std::string naive_file = std::string(argv[5]);
        std::string huang_file = std::string(argv[6]);
        std::string perreault_file = std::string(argv[7]);
        if (check_similarity(original_image, filter_r, true, opencv_file, naive_file,
                             huang_file, perreault_file)) {
          std::cout << "All 4 results are similar and written to the corresponding files" << std::endl;
          return 0;
        }
      }
    } else {
      return -1;
    }
  }
  helper_writer();
  return 0;
}
