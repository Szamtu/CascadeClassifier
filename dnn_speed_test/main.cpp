#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>

int main(int argc, char *argv[])
{
    if(argc < 7){
        std::cout << "Missing args" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./dnn_speed_test cfg_file weights_file inputW inputH showVideo video_file " << std::endl;
        return -1;
    }

    auto cfg = std::string(argv[1]);
    auto weights = std::string(argv[2]);
    auto width = std::stoi(argv[3]);
    auto height = std::stoi(argv[4]);
    auto show = std::stoi(argv[5]) > 0;
    auto video = std::string(argv[6]);

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) {
        std::cout << "Failed to open video file." << std::endl;
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg, weights);
    net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);

    if (net.empty()) {
        std::cout << "Failed to load DNN" << std::endl;
        return -1;
    }

    if(show){
        cv::namedWindow("Wideo", cv::WINDOW_NORMAL);
    }

    cv::Mat frame;
    cv::Mat blob;
    cv::Mat detections;

    // warmup - lazy net init
    cap.read(frame);
    blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    detections = net.forward();

    double totalTime{0};
    int totalFrames{0};

    while (cap.read(frame)) {
        blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false);

        auto start = std::chrono::high_resolution_clock::now();
        net.setInput(blob);
        detections = net.forward();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        totalTime += duration / 1000.0;
        totalFrames += 1;

        std::cout << "CURRENT FPS: " << 1000000.0 / duration << " AVG_FPS:" << totalFrames / (totalTime / 1000.0) << std::endl;

        if(show){
            for (int i = 0; i < detections.rows; ++i) {
                float confidence = detections.at<float>(i, 4);

                if (confidence > 0.5) {
                    int x = static_cast<int>(detections.at<float>(i, 0) * frame.cols);
                    int y = static_cast<int>(detections.at<float>(i, 1) * frame.rows);
                    int width = static_cast<int>(detections.at<float>(i, 2) * frame.cols);
                    int height = static_cast<int>(detections.at<float>(i, 3) * frame.rows);

                    cv::rectangle(frame, cv::Rect(x - (width / 2), y - (height / 2), width, height), cv::Scalar(0, 255, 0), 2);
                }
            }

            cv::imshow("Wideo", frame);
            cv::waitKey(1);
        }
    }
}
