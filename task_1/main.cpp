#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <time.h>

static const int MAX_UINT8_T = 255;
static const long double BRIGTNESS_RED = 0.299;
static const long double BRIGTNESS_GREEN = 0.587;
static const long double BRIGTNESS_BLUE = 0.114;

uint8_t green_for_red_blue(const int r, const int ru, const int rl, const int rr, const int rd,
                           const int gu, const int gl, const int gr, const int gd){
  // Функция с шага 1, вычисляет красный (или синий) при известном зелёном

  int north = 2 * std::abs(r - ru) + std::abs(gd - gu);
  int east = 2 * std::abs(r - rr) + std::abs(gl - gr);
  int west = 2 * std::abs(r - rl) + std::abs(gr - gl);
  int south = 2 * std::abs(r - rd) + std::abs(gu - gd);
  int min_grad = std::min(north, std::min(east, std::min(west, south)));
  int returned;

  if(min_grad == north){
    returned = (3 * gu + gd + r - ru) / 4;
  }

  if(min_grad == east){
    returned = (3 * gr + gl + r - rr) / 4;
  }

  if(min_grad == west){
    returned = (3 * gl + gr + r - rl) / 4;
  }

  if(min_grad == south){
    returned = (3 * gd + gu + r - gd) / 4;
  }

  return static_cast<uint8_t>(std::min(std::max(0, returned), MAX_UINT8_T));
}

uint8_t hue_transit(const int L1, const int L2, const int L3, const int V1, const int V3){
  // Функция применяется в шагах 2 и 3. Восстанавливает красный или синий по известному изменению зелёного
  int returned;
  if(((L1 < L2) && (L2 < L3)) || ((L1 > L2) && (L2 > L3)))
    returned = V1 + ((V3 - V1) * (L2 - L1) / (L3 - L1));
  else
    returned = ((V1 + V3) / 2) + ((L2 - ((L1 + L3) / 2)) / 2);

  return static_cast<uint8_t>(std::min(std::max(0, returned), MAX_UINT8_T));
}

uint8_t blue_for_red(const int bul, const int bur, const int bdl, const int bdr,
                     const int gul, const int gur, const int g, const int gdl, const int gdr,
                     const int rul, const int rur, const int r, const int rdl, const int rdr){
  // Функция с шага 3. Восстанавливает синий для пикселя, который изначально был красным

  int northeast = std::abs(bur - bdl) + std::abs(rur - r) + std::abs(r - rdl) +
      std::abs(gur - g) + std::abs(g - gdl);
  int northwest = std::abs(bul - bdr) + std::abs(rul - r) + std::abs(r - rdr) +
      std::abs(gul - g) + std::abs(g - gdr);

  if(northeast < northwest){
    return hue_transit(gur, g, gdl, bur, bdl);
  }
  return hue_transit(gul, g, gdr, bul, bdr);
}

uint8_t red_for_blue(const int bul, const int bur, const int b, const int bdl, const int bdr,
                     const int gul, const int gur, const int g, const int gdl, const int gdr,
                     const int rul, const int rur, const int rdl, const int rdr){
  // Функция с шага 3. Восстанавливает красный для пикселя, который изначально был синим

  int northeast = std::abs(rur - rdl) + std::abs(bur - b) + std::abs(b - bdl) +
      std::abs(gur - g) + std::abs(g - gdr);
  int northwest = std::abs(rul - rdr) + std::abs(bul - b) + std::abs(b - bdr) +
      std::abs(gul - g) + std::abs(g - gdr);

  if(northeast < northwest){
    return hue_transit(gur, g, gdl, rur, rdl);
  }
  return hue_transit(gul, g, gdr, rul, rdl);
}

uint8_t* get_borders(const uint8_t* origin_pixels, const int n, const int m, const int channels){
  /*
   * Отражаем края, так, чтобы сохранялась сетка фильтра Байера. Для этого надодобавить по 2 строки и 2 столбца
   * с каждой стороны.
  */

  int new_n = n + 4;
  int new_m = m + 4;
  uint8_t* new_pixels = new uint8_t[channels * new_n * new_m];

  // Копируем известные значения
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      new_pixels[channels * ((i + 2) * new_m + (j + 2))] = origin_pixels[channels * (i * m + j)];
      new_pixels[channels * ((i + 2) * new_m + (j + 2)) + 1] = origin_pixels[channels * (i * m + j) + 1];
      new_pixels[channels * ((i + 2) * new_m + (j + 2)) + 2] = origin_pixels[channels * (i * m + j) + 2];
    }
  }

  // Отражаем столбцы
  for(int i = 0; i < n; ++i){
    new_pixels[channels * ((i + 2) * new_m + 0)] = origin_pixels[channels * (i * m + 2)];
    new_pixels[channels * ((i + 2) * new_m + 0) + 1] = origin_pixels[channels * (i * m + 2) + 1];
    new_pixels[channels * ((i + 2) * new_m + 0) + 2] = origin_pixels[channels * (i * m + 2) + 2];

    new_pixels[channels * ((i + 2) * new_m + 1)] = origin_pixels[channels * (i * m + 1)];
    new_pixels[channels * ((i + 2) * new_m + 1) + 1] = origin_pixels[channels * (i * m + 1) + 1];
    new_pixels[channels * ((i + 2) * new_m + 1) + 2] = origin_pixels[channels * (i * m + 1) + 2];

    new_pixels[channels * ((i + 2) * new_m + (new_m - 1))] = origin_pixels[channels * (i * m + (m - 3))];
    new_pixels[channels * ((i + 2) * new_m + (new_m - 1)) + 1] = origin_pixels[channels * (i * m + (m - 3)) + 1];
    new_pixels[channels * ((i + 2) * new_m + (new_m - 1)) + 2] = origin_pixels[channels * (i * m + (m - 3)) + 2];

    new_pixels[channels * ((i + 2) * new_m + (new_m - 2))] = origin_pixels[channels * (i * m + (m - 2))];
    new_pixels[channels * ((i + 2) * new_m + (new_m - 2)) + 1] = origin_pixels[channels * (i * m + (m - 2)) + 1];
    new_pixels[channels * ((i + 2) * new_m + (new_m - 2)) + 2] = origin_pixels[channels * (i * m + (m - 2)) + 2];
  }

  // Отражаем строки
  for(int j = 0; j < m; ++j){
    new_pixels[channels * (0 * new_m + (j + 2))] = origin_pixels[channels * (2 * m + j)];
    new_pixels[channels * (0 * new_m + (j + 2)) + 1] = origin_pixels[channels * (2 * m + j) + 1];
    new_pixels[channels * (0 * new_m + (j + 2)) + 2] = origin_pixels[channels * (2 * m + j) + 2];

    new_pixels[channels * (1 * new_m + (j + 2))] = origin_pixels[channels * (1 * m + j)];
    new_pixels[channels * (1 * new_m + (j + 2)) + 1] = origin_pixels[channels * (1 * m + j) + 1];
    new_pixels[channels * (1 * new_m + (j + 2)) + 2] = origin_pixels[channels * (1 * m + j) + 2];

    new_pixels[channels * ((new_n - 1) * new_m + (j + 2))] = origin_pixels[channels * ((n - 3) * m + j)];
    new_pixels[channels * ((new_n - 1) * new_m + (j + 2)) + 1] = origin_pixels[channels * ((n - 3) * m + j) + 1];
    new_pixels[channels * ((new_n - 1) * new_m + (j + 2)) + 2] = origin_pixels[channels * ((n - 3) * m + j) + 2];

    new_pixels[channels * ((new_n - 2) * new_m + (j + 2))] = origin_pixels[channels * ((n - 2) * m + j)];
    new_pixels[channels * ((new_n - 2) * new_m + (j + 2)) + 1] = origin_pixels[channels * ((n - 2) * m + j) + 1];
    new_pixels[channels * ((new_n - 2) * new_m + (j + 2)) + 2] = origin_pixels[channels * ((n - 2) * m + j) + 2];

  }

  // для оставшихся по краям квадратов делам отражение по столбцам
  // левый верхний квадрат
  new_pixels[channels * (0 * new_m + 0)] = new_pixels[channels * (0 * new_m + 4)];
  new_pixels[channels * (0 * new_m + 0) + 1] = new_pixels[channels * (0 * new_m + 4) + 1];
  new_pixels[channels * (0 * new_m + 0) + 2] = new_pixels[channels * (0 * new_m + 4) + 2];

  new_pixels[channels * (0 * new_m + 1)] = new_pixels[channels * (0 * new_m + 3)];
  new_pixels[channels * (0 * new_m + 1) + 1] = new_pixels[channels * (0 * new_m + 3) + 1];
  new_pixels[channels * (0 * new_m + 1) + 2] = new_pixels[channels * (0 * new_m + 3) + 2];

  new_pixels[channels * (1 * new_m + 0)] = new_pixels[channels * (1 * new_m + 4)];
  new_pixels[channels * (1 * new_m + 0) + 1] = new_pixels[channels * (1 * new_m + 4) + 1];
  new_pixels[channels * (1 * new_m + 0) + 2] = new_pixels[channels * (1 * new_m + 4) + 2];

  new_pixels[channels * (1 * new_m + 1)] = new_pixels[channels * (1 * new_m + 3)];
  new_pixels[channels * (1 * new_m + 1) + 1] = new_pixels[channels * (1 * new_m + 3) + 1];
  new_pixels[channels * (1 * new_m + 1) + 2] = new_pixels[channels * (1 * new_m + 3) + 2];

  //правый верхний квадрат
  new_pixels[channels * (0 * new_m + (new_m - 1))] = new_pixels[channels * (0 * new_m + (new_m - 5))];
  new_pixels[channels * (0 * new_m + (new_m - 1)) + 1] = new_pixels[channels * (0 * new_m + (new_m - 5)) + 1];
  new_pixels[channels * (0 * new_m + (new_m - 1)) + 2] = new_pixels[channels * (0 * new_m + (new_m - 5)) + 2];

  new_pixels[channels * (0 * new_m + (new_m - 2))] = new_pixels[channels * (0 * new_m + (new_m - 4))];
  new_pixels[channels * (0 * new_m + (new_m - 2)) + 1] = new_pixels[channels * (0 * new_m + (new_m - 4)) + 1];
  new_pixels[channels * (0 * new_m + (new_m - 2)) + 2] = new_pixels[channels * (0 * new_m + (new_m - 4)) + 2];

  new_pixels[channels * (1 * new_m + (new_m - 1))] = new_pixels[channels * (1 * new_m + (new_m - 5))];
  new_pixels[channels * (1 * new_m + (new_m - 1)) + 1] = new_pixels[channels * (1 * new_m + (new_m - 5)) + 1];
  new_pixels[channels * (1 * new_m + (new_m - 1)) + 2] = new_pixels[channels * (1 * new_m + (new_m - 5)) + 2];

  new_pixels[channels * (1 * new_m + (new_m - 2))] = new_pixels[channels * (1 * new_m + (new_m - 4))];
  new_pixels[channels * (1 * new_m + (new_m - 2)) + 1] = new_pixels[channels * (1 * new_m + (new_m - 4)) + 1];
  new_pixels[channels * (1 * new_m + (new_m - 2)) + 2] = new_pixels[channels * (1 * new_m + (new_m - 4)) + 2];

  // левый нижниий квадрат
  new_pixels[channels * ((new_n - 1) * new_m + 0)] = new_pixels[channels * ((new_n - 1) * new_m + 4)];
  new_pixels[channels * ((new_n - 1) * new_m + 0) + 1] = new_pixels[channels * ((new_n - 1) * new_m + 4) + 1];
  new_pixels[channels * ((new_n - 1) * new_m + 0) + 2] = new_pixels[channels * ((new_n - 1) * new_m + 4) + 2];

  new_pixels[channels * ((new_n - 1) * new_m + 1)] = new_pixels[channels * ((new_n - 1) * new_m + 3)];
  new_pixels[channels * ((new_n - 1) * new_m + 1) + 1] = new_pixels[channels * ((new_n - 1) * new_m + 3) + 1];
  new_pixels[channels * ((new_n - 1) * new_m + 1) + 2] = new_pixels[channels * ((new_n - 1) * new_m + 3) + 2];

  new_pixels[channels * ((new_n - 2) * new_m + 0)] = new_pixels[channels * ((new_n - 2) * new_m + 4)];
  new_pixels[channels * ((new_n - 2) * new_m + 0) + 1] = new_pixels[channels * ((new_n - 2) * new_m + 4) + 1];
  new_pixels[channels * ((new_n - 2) * new_m + 0) + 2] = new_pixels[channels * ((new_n - 2) * new_m + 4) + 2];

  new_pixels[channels * ((new_n - 2) * new_m + 1)] = new_pixels[channels * ((new_n - 2) * new_m + 3)];
  new_pixels[channels * ((new_n - 2) * new_m + 1) + 1] = new_pixels[channels * ((new_n - 2) * new_m + 3) + 1];
  new_pixels[channels * ((new_n - 2) * new_m + 1) + 2] = new_pixels[channels * ((new_n - 2) * new_m + 3) + 2];

  //правый нижний квадрат
  new_pixels[channels * ((new_n - 1) * new_m + (new_m - 1))] =
      new_pixels[channels * ((new_n - 1) * new_m + (new_m - 5))];
  new_pixels[channels * ((new_n - 1) * new_m + (new_m - 1)) + 1] =
      new_pixels[channels * ((new_n - 1) * new_m + (new_m - 5)) + 1];
  new_pixels[channels * ((new_n - 1) * new_m + (new_m - 1)) + 2] =
      new_pixels[channels * ((new_n - 1) * new_m + (new_m - 5)) + 2];

  new_pixels[channels * ((new_n - 1) * new_m + (new_m - 2))] =
      new_pixels[channels * ((new_n - 1) * new_m + (new_m - 4))];
  new_pixels[channels * ((new_n - 1) * new_m + (new_m - 2)) + 1] =
      new_pixels[channels * ((new_n - 1) * new_m + (new_m - 4)) + 1];
  new_pixels[channels * ((new_n - 1) * new_m + (new_m - 2)) + 2] =
      new_pixels[channels * ((new_n - 1) * new_m + (new_m - 4)) + 2];

  new_pixels[channels * ((new_n - 2) * new_m + (new_m - 1))] =
      new_pixels[channels * ((new_n - 2) * new_m + (new_m - 5))];
  new_pixels[channels * ((new_n - 2) * new_m + (new_m - 1)) + 1] =
      new_pixels[channels * ((new_n - 2) * new_m + (new_m - 5)) + 1];
  new_pixels[channels * ((new_n - 2) * new_m + (new_m - 1)) + 2] =
      new_pixels[channels * ((new_n - 2) * new_m + (new_m - 5)) + 2];

  new_pixels[channels * ((new_n - 2) * new_m + (new_m - 2))] =
      new_pixels[channels * ((new_n - 2) * new_m + (new_m - 4))];
  new_pixels[channels * ((new_n - 2) * new_m + (new_m - 2)) + 1] =
      new_pixels[channels * ((new_n - 2) * new_m + (new_m - 4)) + 1];
  new_pixels[channels * ((new_n - 2) * new_m + (new_m - 2)) + 2] =
      new_pixels[channels * ((new_n - 2) * new_m + (new_m - 4)) + 2];

  return new_pixels;
}

uint8_t* cut_borders(const uint8_t* pixels_computed, const int n, const int m, const int channels){
  /*
   * Функция обрезки краёв. После того как восстановили цвета "сдуваем" изображение обратно.
   */
  int new_n = n - 4;
  int new_m = m - 4;
  uint8_t* final_pixels = new uint8_t[channels * new_n * new_m];
  for(int i = 0; i < new_n; ++i){
    for(int j = 0; j < new_m; ++j){
      final_pixels[channels * (i * new_m + j)] = pixels_computed[channels * ((i + 2) * m + (j + 2))];
      final_pixels[channels * (i * new_m + j) + 1] = pixels_computed[channels * ((i + 2) * m + (j + 2)) + 1];
      final_pixels[channels * (i * new_m + j) + 2] = pixels_computed[channels * ((i + 2) * m + (j + 2)) + 2];
    }
  }
  return final_pixels;
}


cv::Mat transform(const cv::Mat& origin_image){
  /*
   * Основная функция. По шагам реализует демозаикинг
   */
  int n, m;
  //n строк и m столбцов
  cv::Size sz = origin_image.size();
  n = sz.height;
  m = sz.width;
  int channels = 3;
  uint8_t* origin_pixels = reinterpret_cast<uint8_t*>(origin_image.data);
  uint8_t* new_pixels = get_borders(origin_pixels, n, m, channels);

  // Раздули изображение, чтобы не потерять качество: стало n + 4 строк и m + 4 столбца
  n += 4;
  m += 4;

  time_t start = clock();
  // Первый шаг: восстанавливаем зелёный для синиего и красного
  for(int i = 2; i < n - 2; ++i){
    for(int j = 2; j < m - 2; ++j){
      if((i + j) % 2 == 0){
        if(i % 2 == 0){
          // для красных пикселей
          new_pixels[channels * (i * m + j) + 1] = green_for_red_blue(
              new_pixels[channels * (i * m + j) + 2],
              new_pixels[channels * ((i - 2) * m + j) + 2],new_pixels[channels * (i * m + (j - 2)) + 2],
              new_pixels[channels * (i * m + (j + 2)) + 2],new_pixels[channels * ((i + 2) * m + j) + 2],
              new_pixels[channels * ((i - 1) * m + j) + 1],new_pixels[channels * (i * m + (j - 1)) + 1],
              new_pixels[channels * (i * m + (j + 1)) + 1],new_pixels[channels * ((i + 1) * m + j) + 1]
              );
        } else {
          // для синих пикселей
          new_pixels[channels * (i * m + j) + 1] = green_for_red_blue(
              new_pixels[channels * (i * m + j)],
              new_pixels[channels * ((i - 2) * m + j)],new_pixels[channels * (i * m + (j - 2))],
              new_pixels[channels * (i * m + (j + 2))],new_pixels[channels * ((i + 2) * m + j)],
              new_pixels[channels * ((i - 1) * m + j) + 1],new_pixels[channels * (i * m + (j - 1)) + 1],
              new_pixels[channels * (i * m + (j + 1)) + 1],new_pixels[channels * ((i + 1) * m + j) + 1]
          );
        }
      }
    }
  }

  // Второй шаг: вычисляем красный и синий для зелёных пикселей
  for(int i = 1; i < n - 1; ++i) {
    for(int j = 1; j < m - 1; ++j) {
      if((i + j) % 2 == 1) {
        if(i % 2 == 0) {
          // В строке с красным, значит синий по вертикали, красный по горизонтали
          new_pixels[channels * (i * m + j)] = hue_transit(
              new_pixels[channels * (i * m + (j - 1)) + 1],
              new_pixels[channels * (i * m + j) + 1],
              new_pixels[channels * (i * m + (j + 1)) + 1],
              new_pixels[channels * ((i - 1) * m + j)], new_pixels[channels * ((i + 1) * m + j)]
          );
          new_pixels[channels * (i * m + j) + 2] = hue_transit(
              new_pixels[channels * ((i - 1) * m + j) + 1],
              new_pixels[channels * (i * m + j) + 1],
              new_pixels[channels * ((i + 1) * m + j) + 1],
              new_pixels[channels * (i * m + (j - 1)) + 2], new_pixels[channels * (i * m + (j + 1)) + 2]
          );
        } else {
          // В строке с синим, значит синий по горизонтали, красный по вертикали
          new_pixels[channels * (i * m + j)] = hue_transit(
              new_pixels[channels * (i * m + (j - 1)) + 1],
              new_pixels[channels * (i * m + j) + 1],
              new_pixels[channels * (i * m + (j + 1)) + 1],
              new_pixels[channels * (i * m + (j - 1))], new_pixels[channels * (i * m + (j + 1))]
          );
          new_pixels[channels * (i * m + j) + 2] = hue_transit(
              new_pixels[channels * ((i - 1) * m + j) + 1],
              new_pixels[channels * (i * m + j) + 1],
              new_pixels[channels * ((i + 1) * m + j) + 1],
              new_pixels[channels * ((i - 1) * m + j) + 2], new_pixels[channels * ((i + 1) * m + j) + 2]
          );
        }
      }
    }
  }

  // Третий шаг: вычисляем синий для красного и красный для синего
  for(int i = 2; i < n - 2; ++i) {
    for (int j = 2; j < m - 2; ++j) {
      if((i + j) % 2 == 0){
        if(i % 2 == 0){
          // синий для красного
          new_pixels[channels * (i * m + j)] = blue_for_red(
              new_pixels[channels * ((i - 1) * m + (j - 1))],
              new_pixels[channels * ((i - 1) * m + (j + 1))],
              new_pixels[channels * ((i + 1) * m + (j - 1))],
              new_pixels[channels * ((i + 1) * m + (j + 1))],

              new_pixels[channels * ((i - 1) * m + (j - 1)) + 1],
              new_pixels[channels * ((i - 1) * m + (j + 1)) + 1],
              new_pixels[channels * (i * m + j) + 1],
              new_pixels[channels * ((i + 1) * m + (j - 1)) + 1],
              new_pixels[channels * ((i + 1) * m + (j + 1)) + 1],

              new_pixels[channels * ((i - 2) * m + (j - 2)) + 2],
              new_pixels[channels * ((i - 2) * m + (j + 2)) + 2],
              new_pixels[channels * (i * m + j) + 2],
              new_pixels[channels * ((i + 2) * m + (j - 2)) + 2],
              new_pixels[channels * ((i + 2) * m + (j + 2)) + 2]
          );
        } else {
          // красный для синего
          new_pixels[channels * (i * m + j) + 2] = red_for_blue(
              new_pixels[channels * ((i - 2) * m + (j - 2))],
              new_pixels[channels * ((i - 2) * m + (j + 2))],
              new_pixels[channels * (i * m + j)],
              new_pixels[channels * ((i + 2) * m + (j - 2))],
              new_pixels[channels * ((i + 2) * m + (j + 2))],

              new_pixels[channels * ((i - 1) * m + (j - 1)) + 1],
              new_pixels[channels * ((i - 1) * m + (j + 1)) + 1],
              new_pixels[channels * (i * m + j) + 1],
              new_pixels[channels * ((i + 1) * m + (j - 1)) + 1],
              new_pixels[channels * ((i + 1) * m + (j + 1)) + 1],

              new_pixels[channels * ((i - 1) * m + (j - 1)) + 2],
              new_pixels[channels * ((i - 1) * m + (j + 1)) + 2],
              new_pixels[channels * ((i + 1) * m + (j - 1)) + 2],
              new_pixels[channels * ((i + 1) * m + (j + 1)) + 2]
          );
        }
      }
    }
  }
  time_t finish = clock();
  std::cout << "Время работы алгоритма в секундах: " << static_cast<double>(finish - start) / CLOCKS_PER_SEC <<
  std::endl;

  // Обрезаем края, чтобы выдавать картинку того же размера.
  uint8_t* final_pixels = cut_borders(new_pixels, n, m, channels);
  // Завели через new, когда расширяли границы.
  delete[] new_pixels;
  return cv::Mat(n - 4, m - 4, CV_8UC3, final_pixels);
}

bool check(const cv::Mat& get_image){
  /*
   * Проверяем, что дали изображение формата CV_8UC3.
   */
  if(get_image.channels() != 3){
    std::cout << "Wrong number of channels. Should be 3." << std::endl;
    return false;
  }
  if(get_image.type() != 16){
    std::cout << "Wrong pixel values. Should be from 0 to 255." << std::endl;
    return false;
  }
  return true;
}

void compare(const cv::Mat& transformed, const cv::Mat& original){
  /*
   * Функция, которая считает PSNR и выводит ответ.
   */
  int n = transformed.size().height;
  int m = transformed.size().width;
  int channels = 3;
  if((n != original.size().height) || (m != original.size().width)) {
    std::cout << "Different image sizes. Will not count PSNR for this. Use padding to make them similar size."
    << std::endl;
  }

  uint8_t* transformed_pixels = reinterpret_cast<uint8_t*>(transformed.data);
  uint8_t* original_pixels = reinterpret_cast<uint8_t*>(original.data);
  long double y, y_ref;
  long double MSE = 0;
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
       y = BRIGTNESS_BLUE * original_pixels[channels * (i * m + j)] +
           BRIGTNESS_GREEN * original_pixels[channels * (i * m + j) + 1] +
           BRIGTNESS_RED * original_pixels[channels * (i * m + j) + 2];

      y_ref = BRIGTNESS_BLUE * transformed_pixels[channels * (i * m + j)] +
          BRIGTNESS_GREEN * transformed_pixels[channels * (i * m + j) + 1] +
          BRIGTNESS_RED * transformed_pixels[channels * (i * m + j) + 2];
      MSE += (y - y_ref) * (y - y_ref);
    }
  }
  MSE /= n * m;
  long double PSNR = 10 * std::log10(MAX_UINT8_T * MAX_UINT8_T / MSE);
  std::cout << "PSNR = " << PSNR << std::endl;
}

int main(int argc, char** argv) {
  if(argc != 3 && argc != 4){
    std::cout << "Unknown command. Ways of usage:" << std::endl;
    std::cout << "    <inputFile> <outputFile>: to do demosaicing for <inputFile> and write the result to <outputFile>"
    << std::endl;
    std::cout << "    <inputFile> <outputFile> <originalFile>: to do demosaicing <inputFile>, write the result to "
    << "<outputFile> and compare (count PSNR) with <originalFile>" << std::endl;
    return 0;
  }

  std::string inputFile = argv[1];
  std::string outputFile = argv[2];
  std::string originalFile;
  cv::Mat image;
  cv::Mat transformed_image;
  cv::Mat original_image;

  image = cv::imread(inputFile, cv::IMREAD_COLOR);
  if(image.empty()) {
    std::cout <<  "Image not found or unable to open" << std::endl ;
    return -1;
  }
  if(argc == 4){
    originalFile = argv[3];
    original_image = cv::imread(originalFile, cv::IMREAD_COLOR);
    if(!check(original_image)){
      std::cout << "Check the original file" << std::endl;
      return -1;
    }
  }

  if(check(image)){
    transformed_image = transform(image);
    try {
      cv::imwrite(outputFile, transformed_image);
    }
    catch (cv::Exception& ex){
      std::cout << "Exception while writing to the outputFile: " << ex.what() << std::endl;
    }
  } else {
    std::cout << "Check the input file" << std::endl;
    return -1;
  }

  if(argc == 4){
    compare(transformed_image, original_image);
    original_image.release();
  }
  image.release();
  transformed_image.release();
  return 0;
}
