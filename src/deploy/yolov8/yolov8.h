#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov8_utils.h"


class Yolov8
{
  public:
    Yolov8() {}

    ~Yolov8() {}

    bool                     ReadModel(cv::dnn::Net& net,
                                       std::string&  netPath,
                                       bool          isCuda);

    bool                     Detect(cv::Mat&                srcImg,
                                    cv::dnn::Net&           net,
                                    std::vector<OutputSeg>& output);

    int                      _netWidth  = 640;  // ONNX图片输入宽度
    int                      _netHeight = 640;  // ONNX图片输入高度


    // 类别名，自己的模型需要修改此项
    std::vector<std::string> _className = {"person",
                                           "bicycle",
                                           "car",
                                           "motorcycle"};

  private:
    float _classThreshold = 0.25;
    float _nmsThreshold   = 0.45;
};
