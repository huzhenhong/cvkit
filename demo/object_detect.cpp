/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-09-13 06:26:46
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-19 02:04:22
 * FilePath     : \\goodsmovedetector\\test\\TestVideoV1.0.0.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "cmdline.h"
#include "spdlog/spdlog.h"
#include "camera_wrapper.hpp"
#include "yolov8/yolov8.h"
#include "yolov8/yolov8_onnx.h"


int main(int argc, char* argv[])
{
    SPDLOG_INFO("=================== parse command line begin ===================");
    cmdline::parser argParser;
    argParser.add<std::string>("src_path", 's', "where to read video", true, "");
    argParser.add<std::string>("model_path", 'm', "where to load model", true, "");
    argParser.add("debug", '\0', "is debug mode");
    argParser.parse_check(argc, argv);

    auto srcPath    = argParser.get<std::string>("src_path");
    auto g_modePath = argParser.get<std::string>("model_path");
    auto g_bDebug   = argParser.exist("debug") ? true : false;

    SPDLOG_INFO("=================== do something else ===================");
    fmt::print("src_path: {}, debug: {}", srcPath, g_bDebug);


    Yolov8       detecter_by_opencv;
    Yolov8Onnx   detecter_by_onnx;

    cv::dnn::Net net;
    if (detecter_by_opencv.ReadModel(net, g_modePath, false))
    {
        std::cout << "read net ok!" << std::endl;
    }
    else
    {
        return -1;
    }

    if (detecter_by_onnx.ReadModel(g_modePath, false))
    {
        std::cout << "read net ok!" << std::endl;
    }
    else
    {
        return -2;
    }

    // 生成随机颜色
    std::vector<cv::Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++)
    {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }

    auto CameraCB = [&](const cv::Mat& frame, uint64_t frameIdx)
    {
        fmt::print("detecting");
        auto img = const_cast<cv::Mat&>(frame);

        // if (false)
        if (true)
        {
            std::vector<OutputSeg> result;
            if (detecter_by_opencv.Detect(img, net, result))
            {
                DrawPred(img, result, detecter_by_opencv._className, color);
            }
            else
            {
                std::cout << "Detect Failed!" << std::endl;
            }
        }
        else
        {
            std::vector<OutputSeg> result;
            if (detecter_by_onnx.OnnxDetect(img, result))
            {
                DrawPred(img, result, detecter_by_onnx._className, color);
            }
            else
            {
                std::cout << "Detect Failed!" << std::endl;
            }
        }
    };

    cvkit::WrapperParam<cv::Mat> wrapperParam;
    wrapperParam.bAsync          = false;
    wrapperParam.bShow           = g_bDebug;
    wrapperParam.cacheSize       = 0;
    wrapperParam.extractInterval = 25;
    wrapperParam.winname         = "show";
    wrapperParam.source          = srcPath;
    wrapperParam.frameCallBack   = CameraCB;

    auto camera = cvkit::CameraWrapper<cv::Mat>(wrapperParam);
    camera.Execute();

    return 0;
}
