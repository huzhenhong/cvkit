/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-25 16:45:30
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-19 03:38:00
 * FilePath     : \\goodsmovedetector\\3rdparty\\cvkit\\img\\ImgProc.h
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>


namespace cvkit
{
    float                        IOU(cv::Rect rect1, cv::Rect rect2);

    int                          MinDistanceBetweenRectangles(const cv::Rect& rect1, const cv::Rect& rect2);

    cv::Mat                      CombineContour(cv::Mat contour1, cv::Mat contour2);

    void                         CombineRect(std::vector<cv::Rect>& rectVec, int th = 10.0);

    void                         CombineRect(std::vector<cv::Rect>& rectVec, std::vector<cv::Mat>& contourVec, int th = 10.0);

    cv::Mat                      RegionGrow(cv::Mat src, cv::Point2i pt, int th, int seedValue);

    std::pair<cv::Mat, double>   GetMaxContour(const std::vector<cv::Mat>& contourVec);

    double                       SsimDetect(cv::Mat image_ref, cv::Mat image_obj);

    double                       ComputeSsim(cv::Mat& standard, cv::Mat& frame);

    // 抠图
    std::pair<cv::Mat, cv::Rect> GrabRoi(const cv::Mat& img, const cv::Rect& roiRect, double borderRate = 0.15);

    double                       TemplateMatch(const cv::Mat& search, const cv::Mat& temp, int matchMethod, cv::Rect& matchRect);

    // void CvShow(const std::string& winname, const cv::Mat& img, bool bWait = false);

    cv::Rect                     ResizeROI(const cv::Mat& img, const cv::Rect& roiRect, double borderRate = 0.1);

    double                       ComputeTargetAndBorderColorSimilarity(const cv::Mat& img, const cv::Mat& contour, const cv::Rect& roiRect);

    struct Border
    {
        float scaleRatio{0.0f};
        int   top{0};
        int   bottom{0};
        int   left{0};
        int   right{0};
    };

    Border LetterBoxResize(const cv::Mat&         src,
                           cv::Mat&               dst,
                           cv::InterpolationFlags interpolationMode,
                           const cv::Scalar&      fillColor);

    void   DrawDottedCircle(cv::Mat img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color, int thickness);

    void   DrawDottedLine(cv::Mat img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color, int thickness);

    void   DrawDottedRect(cv::Mat img, cv::Rect rect, cv::Scalar color, int thickness, int type = 0);

}  // namespace cvkit