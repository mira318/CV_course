#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <ctime>
#include <filesystem>

std::vector<int> count_matches(const cv::Mat& descriptors1, const cv::Mat& descriptors2, const int code){
    cv::BFMatcher matcher;
    if(code == 1){
        cv::BFMatcher matcher(cv::NORM_HAMMING2, true);
    } else {
        cv::BFMatcher matcher(cv::NORM_L2, true);
    }

    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<int> cleared_matches;
    matches.clear();
    cleared_matches.clear();
    matcher.knnMatch(descriptors1, descriptors2, matches, 1);
    int threshold;
    switch(code){
        case 0:
            threshold = 95;
            break;
        case 1:
            threshold = 775;
            break;
        case 2:
            threshold = 87;
            break;
        default:
            break;
    }

    for(int i = 0; i < matches.size(); ++i){
        if(matches[i][0].distance < threshold){
            cleared_matches.push_back(matches[i][0].queryIdx);
        }
    }
    matches.clear();
    return cleared_matches;
}

cv::Mat run_sift_detector(cv::Mat& input_image, std::vector<cv::KeyPoint>& points, long double* timer){
    cv::Mat descriptors;
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create(
            0,9,0.11, 5, 1.6);

    time_t start = clock();
    detector->detect(input_image, points);
    time_t finish = clock();
    time_t time_detecting = finish - start;
    *timer = static_cast<long double>(time_detecting) / CLOCKS_PER_SEC;

    detector->compute(input_image, points, descriptors);
    detector->clear();
    detector.release();
    return descriptors;

}

cv::Mat run_brisk_detector(cv::Mat& input_image, std::vector<cv::KeyPoint>& points, long double* timer){
    cv::Mat descriptors;
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create(90, 8, 2.3);

    time_t start = clock();
    detector->detect(input_image, points);
    time_t finish = clock();
    time_t time_detecting = finish - start;
    *timer = static_cast<long double>(time_detecting) / CLOCKS_PER_SEC;

    detector->compute(input_image, points, descriptors);
    detector->clear();
    detector.release();
    return descriptors;
}

cv::Mat run_harris_detector(cv::Mat& input_image, std::vector<cv::KeyPoint>& points, long double* timer){
    cv::Mat descriptors;
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    cv::Mat dst;

    time_t start = clock();
    cv::cornerHarris(input_image, dst, 2, 3, 0.04);
    time_t finish = clock();
    time_t time_detecting = finish - start;
    *timer = static_cast<long double>(time_detecting) / CLOCKS_PER_SEC;

    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst, dst);
    int n = dst.size().height;
    int m = dst.size().width;
    uint8_t* dst_pixels = reinterpret_cast<uint8_t*>(dst.data);

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m; ++j){
            if(dst_pixels[i * m + j] > 87){
                points.push_back(cv::KeyPoint(j, i, 4));
            }
        }
    }
    dst.release();

    detector->compute(input_image, points, descriptors);
    detector->clear();
    detector.release();
    return descriptors;
}

/*
 * 0 - SIFT
 * 1 - BRISK
 * 2 - Harris
 * */
std::vector<float> repeatability_in_directory(std::string directory_name, const int code,
                                              std::vector<long double>& avg_times){
    std::vector<std::filesystem::path> files;
    std::copy(std::filesystem::directory_iterator(directory_name),  std::filesystem::directory_iterator(),
              std::back_inserter(files));
    std::sort(files.begin(), files.end());

    std::vector<cv::KeyPoint> all_points;
    std::vector<cv::KeyPoint> curr_points;
    cv::Mat all_descriptors;
    cv::Mat curr_descriptors;

    cv::Mat curr_image;
    std::vector<int> curr_matches;
    std::vector<float> points_nums;
    long double timer;

    all_points.clear();
    for(int i = 0; i < files.size(); ++i){
        curr_image = imread(files[i], cv::IMREAD_GRAYSCALE);
        curr_points.clear();
        switch(code){
            case 0:
                curr_descriptors = run_sift_detector(curr_image, curr_points, &timer);
                break;
            case 1:
                curr_descriptors = run_brisk_detector(curr_image, curr_points, &timer);
                break;
            case 2:
                curr_descriptors = run_harris_detector(curr_image, curr_points, &timer);
                break;
            default:
                throw std::invalid_argument("Incorrect detector code");
        }
        points_nums.push_back(curr_points.size());
        avg_times.push_back(timer / curr_points.size());

        if(i != 0){
            curr_matches = count_matches(curr_descriptors, all_descriptors, code);
            int t = 0;
            for(int j = 0; j < curr_points.size(); ++j){
                if(t < curr_matches.size() && j == curr_matches[t]){
                    t++;
                } else {
                    all_points.push_back(curr_points[j]);
                    all_descriptors.push_back(curr_descriptors.row(j));
                }
            }
        } else {
            all_descriptors = curr_descriptors;
            all_points = curr_points;
        }

    }
    for(int i = 0; i < points_nums.size(); ++i){
        points_nums[i] = points_nums[i] / all_points.size();
    }

    all_descriptors.release();
    curr_descriptors.release();
    all_points.clear();
    curr_points.clear();

    files.clear();
    curr_image.release();
    curr_matches.clear();
    return points_nums;
}

int main() {
    std::vector<float> repeatabilities;
    std::vector<long double> avg_times;
    for(int code = 0; code < 3; ++code) {
        avg_times.clear();
        try {
            repeatabilities = repeatability_in_directory("./../images", code, avg_times);
            std::cout  << "code = " << code << ", repeatabilites:" << std::endl;
            for(int i = 0; i < repeatabilities.size(); ++i){
                std::cout << repeatabilities[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "time per point on image in seconds: " << std::endl;
            for(int i = 0; i < avg_times.size(); ++i){
                std::cout << avg_times[i] << " ";
            }
            std::cout << std::endl;
        } catch (std::invalid_argument &e){
            std::cout << "Exception: " << e.what() << std::endl;
            return 0;
        }
    }
    return 0;
}
