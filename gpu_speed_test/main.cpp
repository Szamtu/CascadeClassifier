#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <chrono>

int main() {

    cv::dnn::Net net = cv::dnn::readNet("yolov4p6.weights", "yolov4p6.cfg");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


    cv::VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Nie można otworzyć pliku wideo." << std::endl;
        return -1;
    }

    int frameCount = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (frameCount < 1000) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        cv::Mat detection = net.forward();
        frameCount++;
    }

    // Obliczenie FPS
    auto end = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    double fps = frameCount / elapsedSeconds;

    std::cout << "Przetworzono " << frameCount << " klatek w " << elapsedSeconds << " sekund." << std::endl;
    std::cout << "Szybkość przetwarzania: " << fps << " FPS" << std::endl;

    return 0;
}