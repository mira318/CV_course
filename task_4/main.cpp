#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <random>
#include <cmath>
#include <filesystem>

static const int MAX_UINT8_T = 255;

uint8_t uint8_t_round(double value){
    // Функция для безопасного округления в uint8_t
    int val = std::round(value);
    return static_cast<uint8_t>(std::min(std::max(0, val), MAX_UINT8_T));
}

double round_light(double val){
    // Округление уровня освещённости в приемлимые границы
    return std::min(0.8, std::max(0.1, val));
}

cv::Mat add_paper_noise(const cv::Mat& input_image, double diffect_prob = 0.3){
    // Функция для имитации структуры бумаги.
    // Генерируем диффект из мелкодисперсного нормального распределения.
    // Генерируем индикатор события "появился диффект" из распределения Бернулли.
    // Вероятность диффекта может меняться, т к качество бумаги бывает разным.

    cv::Size sz = input_image.size();
    int n = sz.height;
    int m = sz.width;
    int channels = 3;

    std::default_random_engine gen(time(0));
    std::normal_distribution<> normal(1, 0.2);
    std::bernoulli_distribution bern(diffect_prob);

    uint8_t* new_pixels = new uint8_t[channels * n * m];
    uint8_t* had_pixels = reinterpret_cast<uint8_t*>(input_image.data);

    double val;
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m; ++j){
            new_pixels[channels * (i * m + j)] = had_pixels[channels * (i * m + j)];
            new_pixels[channels * (i * m + j) + 1] = had_pixels[channels * (i * m + j) + 1];
            new_pixels[channels * (i * m + j) + 2] = had_pixels[channels * (i * m + j) + 2];

            // Величина диффекта
            val = normal(gen) * bern(gen);

            // Добавляем сгенерированный диффект ко всем каналам
            new_pixels[channels * (i * m + j)] = uint8_t_round(
                    had_pixels[channels * (i * m + j)] * (1 + val));
            new_pixels[channels * (i * m + j) + 1] = uint8_t_round(
                    had_pixels[channels * (i * m + j) + 1] * (1 + val));
            new_pixels[channels * (i * m + j) + 2] = uint8_t_round(
                    had_pixels[channels * (i * m + j) + 2] * (1 + val));
        }
    }

    cv:: Mat get = cv::Mat(n, m, CV_8UC3, new_pixels);
    // Размываем диффекты фильтром с маленьким ядром, чтобы они не выглядели как аккуратные точки
    cv::GaussianBlur(get, get, cv::Size(1, 1), 1, 1);
    return get;
}

cv::Mat make_projection(const cv::Mat& input_image,
                        float x0_shift = -35.55, float x1_shift = 37.72,
                        float x2_shift = 112.87, float x3_shift = -114.22,
                        float y0_shift = -49.55, float y1_shift = -50.76,
                        float y2_shift = 80.76, float y3_shift = 79.23){

    // Делаем проективное преобразование
    cv::Size sz = input_image.size();
    int n = sz.width;
    int m = sz.height;
    cv::Point2f input_corners[4];
    cv::Point2f output_corners[4];

    // Те сдвиги, которые хотим получить после преобразования
    input_corners[0] = cv::Point2f( x0_shift, y0_shift );
    input_corners[1] = cv::Point2f( n + x1_shift, y1_shift);
    input_corners[2] = cv::Point2f( n + x2_shift, m + y2_shift);
    input_corners[3] = cv::Point2f( x3_shift, m + y3_shift);

    // Те границы, в которых работаем
    output_corners[0] = cv::Point2f( 0, 0);
    output_corners[1] = cv::Point2f( n - 1, 0);
    output_corners[2] = cv::Point2f( n - 1, m - 1);
    output_corners[3] = cv::Point2f( 0, m - 1);

    cv::Mat output_image = input_image.clone();
    cv::Mat lambda = cv::Mat::zeros( n, m, CV_8UC3);

    // Хотим получить матрицу перехода.
    // Немного обманываем библиотеку, делая вид, что пытаемся скорректировать изображение, поэтому всё так запутано.
    lambda = cv::getPerspectiveTransform( input_corners, output_corners );
    warpPerspective(input_image, output_image, lambda, sz);
    return output_image;
}

cv::Mat contrast_correction(const cv::Mat& input_image, double alpha = 0.75, double beta = 30){
    // alpha-beta коррекция яркости и контрасности изображения
    cv::Mat output_image = cv::Mat::zeros(input_image.size().width, input_image.size().height, CV_8UC3);
    input_image.convertTo(output_image, -1, alpha, beta);
    return output_image;
}

cv::Mat add_light(const cv::Mat& input_image, int ratio = 250, double light_prob = 0.15,
                  double light_blue = 0.5, double light_green = 0.6, double light_red = 0.7,
                  cv::Size final_blur_kernel = cv::Size(5, 5)){
    // Функция, чтобы сымитировать источники света
    std::default_random_engine gen(time(0));
    std::normal_distribution<> normal(1, 0.2);
    std::bernoulli_distribution bern(light_prob);

    cv::Size sz = input_image.size();
    int n = sz.height;
    int m = sz.width;
    int light_n = n / ratio;
    int light_m = m / ratio;
    int channels = 3;

    // Генерируем белые точки на тёмном изображении, которое в ratio раз меньше исходного
    double val;
    uint8_t* light_pixels = new uint8_t[light_n * light_m];
    for(int i = 0; i < light_n; ++i){
        for(int j = 0; j < light_m; ++j){
            val = normal(gen) * bern(gen);
            light_pixels[i * light_m + j] = uint8_t_round(val * MAX_UINT8_T);
        }
    }

    // Растягиваем полученное изображение до исходных размеров, точки сглаживаем двумя последовательными фильтами
    cv::Mat light = cv::Mat(light_n, light_m, CV_8UC1, light_pixels);
    cv::resize(light, light, sz, cv::INTER_AREA);

    cv::blur(light, light, cv::Size(3, 3));
    cv::GaussianBlur(light, light, cv::Size(15, 15), 0, 0);

    light_pixels = reinterpret_cast<uint8_t*>(light.data);
    uint8_t* had_pixels = reinterpret_cast<uint8_t*>(input_image.data);
    uint8_t* lighted_pixels = new uint8_t[channels * n * m];

    // Добавляем полученные световые пятна на исходное изображение.
    // Считаем, что у свет может имееть разные соотношения касного, зелёного и синего
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m; ++j){
            lighted_pixels[channels * (i * m + j)] = uint8_t_round(
                    had_pixels[channels * (i * m + j)] + light_blue * light_pixels[i * m + j]);
            lighted_pixels[channels * (i * m + j) + 1] = uint8_t_round(
                    had_pixels[channels * (i * m + j) + 1] + light_green * light_pixels[i * m + j]);
            lighted_pixels[channels * (i * m + j) + 2] = uint8_t_round(
                    had_pixels[channels * (i * m + j) + 2] + light_red * light_pixels[i * m + j]);
        }
    }

    light.release();
    cv::Mat lighted = cv::Mat(n, m, CV_8UC3, lighted_pixels);

    // Добавляем ещё один фильтер, чтобы сгладить световые пятна светильников
    // и сымитировать размытие камерой всего изображения
    cv::GaussianBlur(lighted, lighted, final_blur_kernel, 0, 0);
    return lighted;
}

void generate_from_file(cv::Mat input_image, int num, const std::string& output_file_path){
    // Функция, которая по одному изображению генерирует num искажённых и записывает их в выходные файлы
    cv::Size sz = input_image.size();
    int n = sz.width;
    int m = sz.height;
    std::string next_output_file;

    // Чтобы результаты получились разными, параметры необходимо варировать.
    // Для этого служат представленные распределения.
    std::default_random_engine gen(time(0));
    std::uniform_real_distribution<double> uniform_diffect_prob(0.003,0.3);
    std::uniform_real_distribution<float> uniform_x_shift(-n / 10.0, n / 10.0);
    std::uniform_real_distribution<float> uniform_y_shift(-m / 15.0, m / 15.0);
    std::normal_distribution<float> norm_x_diff(0, n / 20.0);
    std::normal_distribution<float> norm_y_diff(0, m / 60.0);
    std::uniform_real_distribution<double> uniform_alpha(0.55, 0.85);
    std::uniform_int_distribution<int> uniform_ratio(120, 600);
    std::uniform_real_distribution<double> uniform_light_common(0.2, 0.7);
    std::normal_distribution<double> norm_light_diff(0, 0.1);
    std::uniform_int_distribution<int> uniform_kernel_half(0, 4);
    std::uniform_real_distribution<double> uniform_rotation(0, 1);

    // Заранее выделяем память, чтобы не освбождать/выделять в цикле.
    cv::Mat noised = input_image.clone();
    cv::Mat projected = input_image.clone();
    cv::Mat contrast_corrected = input_image.clone();
    cv::Mat lighted = input_image.clone();
    for(int i = 0; i < num; ++i){
        // Имитируем бумагу разного качества.
        noised = add_paper_noise(input_image, uniform_diffect_prob(gen));

        // Параметры для проективного преобразования.
        // Считаем, что в кадр могли попасть или не попасть лишь небольшие по отношению ко всему документу детали.
        // Считаем, что фотографирует человек и он не допустит сильных искажений видимых прямых линий.
        float x0_shift = uniform_x_shift(gen);
        float y0_shift = uniform_y_shift(gen);
        float x1_shift = uniform_x_shift(gen);
        float y1_shift = y0_shift + norm_y_diff(gen);
        float x2_shift = x1_shift + norm_x_diff(gen);
        float y2_shift = uniform_y_shift(gen);
        float x3_shift = x0_shift + norm_x_diff(gen);
        float y3_shift = y2_shift + norm_y_diff(gen);
        projected = make_projection(noised,
                                    x0_shift, x1_shift, x2_shift, x3_shift,
                                    y0_shift, y1_shift, y2_shift, y3_shift);

        // alpha-beta коррекция. Варируем alpha < 1 -- уменьшаем контрастность изображения.
        contrast_corrected = contrast_correction(projected, uniform_alpha(gen), 30);

        // Параметры для источников света.
        // Генерируем размер и вероятность появления.
        int ratio = uniform_ratio(gen);
        double light_prob = 0.06 * (ratio / 120.0);

        // Генерируем соотношение цветов в свете источника.
        double green_light = uniform_light_common(gen);
        double blue_light = round_light(green_light + norm_light_diff(gen));
        double red_light = round_light(green_light + norm_light_diff(gen));

        // Ядро в финальном фильтре Гаусса - так меняем "качество" камеры.
        int kernel_width = 2 * uniform_kernel_half(gen) + 1;
        int kernel_height = 2 * uniform_kernel_half(gen) + 1;
        lighted = add_light(contrast_corrected, ratio, light_prob,
                                    blue_light, green_light, red_light,
                                    cv::Size(kernel_width,kernel_height));

        // Делаем повороты достаточно редко
        // т к почти все современные устройства фотографирования способны развернуть изображение правильно.
        double rotate = uniform_rotation(gen);
        if(rotate <= 0.2){
            cv::rotate(lighted, lighted, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        if(rotate >= 0.8){
            cv::rotate(lighted, lighted, cv::ROTATE_90_CLOCKWISE);
        }

        next_output_file = output_file_path + "_" + std::to_string(i) + ".jpg";
        try{
            cv::imwrite(next_output_file, lighted);
        } catch (cv::Exception& ex){
            std::cout << "Can't write to file " << next_output_file << " -- skip" << std::endl << ex.what() << std::endl;
        }
    }
    // Освобождение памяти.
    input_image.release();
    noised.release();
    projected.release();
    contrast_corrected.release();
    lighted.release();
}

int main(){
    std::string input_directory_name;
    std::string output_directory_name;
    int num;
    std::cout << "write directory with input files:" << std::endl;
    std::cin >> input_directory_name;
    std::cout << "write directory for output files. Cautious: old files maybe erased." << std::endl;
    std::cin >> output_directory_name;
    std::cout << "how many new samples from one file?" << std::endl;
    std::cin >> num;

    // Рекурсивный обход директории и запуск генератора.
    for (const auto& file : std::filesystem::recursive_directory_iterator(input_directory_name)){
        if(!file.is_directory()) {
            cv::Mat input_image = cv::imread(file.path().string(), cv::IMREAD_COLOR);
            if(input_image.empty()){
                std::cout << "Can't read " << file.path() << " as image -- skip" << std::endl;
            } else {
                generate_from_file(input_image, num, output_directory_name + "/" +
                file.path().filename().replace_extension("").string());
            }
        }
    }
    return 0;
}