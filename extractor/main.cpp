#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <vector>

std::vector<std::string> getOutputsNames(cv::dnn::Net const &a_net)
{
  std::vector<std::string> names{};
  auto outLayers = a_net.getUnconnectedOutLayers();
  auto layersNames = a_net.getLayerNames();

  names.resize(outLayers.size());
  for (size_t i = 0; i < outLayers.size(); ++i) {
    names[i] = layersNames[static_cast<size_t>(outLayers[i]) - 1];
  }

  return names;
}

typedef struct DNNRect {
  cv::Rect boundingBox{};
  int classId{};
  float confidence{};
} DNNRect;

std::vector<DNNRect> thresholdDetections(std::vector<cv::Mat> const &a_detections,
                                      int const a_width, int const a_height,
                                      double const a_confThreshold, double const a_nmsThreshold){
    std::vector<DNNRect> thresholdedDetections{};

    if (a_detections.size() && a_width > 0 && a_height > 0) {
      std::vector<int> classIds;
      std::vector<float> confidences;
      std::vector<cv::Rect> boxes;

      for (size_t i = 0; i < a_detections.size(); ++i) {
        auto data = reinterpret_cast<float *>(a_detections[i].data);
        for (int j = 0; j < a_detections[i].rows; ++j, data += a_detections[i].cols) {
          cv::Mat scores = a_detections[i].row(j).colRange(5, a_detections[i].cols);

          cv::Point classIdPoint;
          double confidence;

          minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
          if (confidence > a_confThreshold) {
            int centerX = static_cast<int>((data[0] * a_width));
            int centerY = static_cast<int>((data[1] * a_height));
            int width = static_cast<int>((data[2] * a_width));
            int height = static_cast<int>((data[3] * a_height));
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            if(left < 0) left = 0;
            if(left + width >= a_width) width = a_width - left;
            if(top < 0) top = 0;
            if(top + height >= a_height) height = a_height - top;

            classIds.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(confidence));
            boxes.push_back(cv::Rect(left, top, width, height));
          }
        }
      }

      std::vector<int> indices;
      cv::dnn::NMSBoxes(boxes, confidences, static_cast<float>(a_confThreshold), static_cast<float>(a_nmsThreshold),
                        indices);
      for (size_t i = 0; i < indices.size(); ++i) {
        auto idx = static_cast<size_t>(indices[i]);
        thresholdedDetections.push_back({ boxes[idx], classIds[idx], confidences[idx] });
      }
    }

    return thresholdedDetections;
}

int main(int argc, char *argv[])
{
    auto cfg = std::string("yolov4-tiny_640.cfg");
    auto weights = std::string("yolov4-tiny.weights");
    auto width = 640;
    auto height = 640;
    auto video = std::string("/mnt/c/Users/Adams/Desktop/raw_vid/20230819_141924.mp4");
    auto confThreshold = 0.5;
    auto nmsThreshold = 0.5;
    auto minWidth = 100;
    auto minHeight = 100;
    auto classID = 2;
    auto dstDir = std::string("/mnt/c/Users/Adams/Desktop/stage_3/raw_extract/");
    auto dstPrefix = std::string("l1");
    auto nSkipFrames = 12;

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg, weights);
    net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    auto outputsNames = getOutputsNames(net);

    if (net.empty()) {
        std::cout << "Failed to load DNN" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) {
        std::cout << "Failed to open video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat blob;
    std::vector<cv::Mat> detections;

    double totalTime{0};
    int totalFrames{0};
    int detectIndex{0};
    auto statsTimer = std::chrono::high_resolution_clock::now();
    auto frameCount{0};

    cap.read(frame);
    blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    net.forward(detections, outputsNames);

    while (cap.read(frame)) {
        frameCount++;
        if(frameCount < nSkipFrames) continue;
        frameCount = 0;

        blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false);
        auto start = std::chrono::high_resolution_clock::now();
        net.setInput(blob);
        net.forward(detections, outputsNames);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        totalTime += duration / 1000.0;
        totalFrames += 1;

        auto statsDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - statsTimer).count();
        if(statsDuration >= 1000000.0){
            std::cout << "CURRENT FPS: " << 1000000.0 / duration << " AVG_FPS:" << totalFrames / (totalTime / 1000.0) << std::endl;
            statsTimer = std::chrono::high_resolution_clock::now();
        }

        auto thresholdedDetections = thresholdDetections(detections, frame.cols, frame.rows, confThreshold, nmsThreshold);
        for(auto &item : thresholdedDetections){
            if(item.classId == classID && item.boundingBox.width >= minWidth && item.boundingBox.height >= minHeight){
                auto dstFilename = dstDir;
                dstFilename += std::to_string(detectIndex);
                dstFilename += "_";
                dstFilename += dstPrefix;
                dstFilename += ".jpg";
                cv::imwrite(dstFilename, frame(item.boundingBox));
                detectIndex++;
            }
        }
    }
}
