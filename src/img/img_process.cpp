/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-25 17:21:12
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-19 04:05:22
 * FilePath     : \\goodsmovedetector\\3rdparty\\cvkit\\img\\ImgProc.cpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/

#include "img_process.h"


namespace cvkit
{
    float IOU(cv::Rect rect1, cv::Rect rect2)
    {
        auto Intersection = [](const cv::Rect& rect1, const cv::Rect& rect2)
        {
            int x1 = std::max(rect1.x, rect2.x);
            int y1 = std::max(rect1.y, rect2.y);

            int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
            int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

            if (x2 < x1 || y2 < y1)
            {
                return -1;
            }

            return (x2 - x1) * (y2 - y1);
        };

        auto Union = [](const cv::Rect& rect1, const cv::Rect& rect2)
        {
            int x1 = std::min(rect1.x, rect2.x);
            int y1 = std::min(rect1.y, rect2.y);

            int x2 = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
            int y2 = std::max(rect1.y + rect1.height, rect2.y + rect2.height);

            return (x2 - x1) * (y2 - y1);
        };

        return 1.0f * Intersection(rect1, rect2) / Union(rect1, rect2);
    }

    int MinDistanceBetweenRectangles(const cv::Rect& rect1, const cv::Rect& rect2)
    {
        int       minDist;

        cv::Point C1, C2;
        C1.x = rect1.x + (rect1.width / 2);
        C1.y = rect1.y + (rect1.height / 2);
        C2.x = rect2.x + (rect2.width / 2);
        C2.y = rect2.y + (rect2.height / 2);

        int Dx, Dy;
        Dx = abs(C2.x - C1.x);
        Dy = abs(C2.y - C1.y);

        if ((Dx < ((rect1.width + rect2.width) / 2)) && (Dy >= ((rect1.height + rect2.height) / 2)))
        {
            minDist = Dy - ((rect1.height + rect2.height) / 2);
        }
        else if ((Dx >= ((rect1.width + rect2.width) / 2)) && (Dy < ((rect1.height + rect2.height) / 2)))
        {
            minDist = Dx - ((rect1.width + rect2.width) / 2);
        }
        else if ((Dx >= ((rect1.width + rect2.width) / 2)) && (Dy >= ((rect1.height + rect2.height) / 2)))
        {
            int delta_x = Dx - ((rect1.width + rect2.width) / 2);
            int delta_y = Dy - ((rect1.height + rect2.height) / 2);
            minDist     = static_cast<int>(sqrt(delta_x * delta_x + delta_y * delta_y));
        }
        else
        {
            minDist = -1;
        }

        return minDist;
    }

    cv::Mat CombineContour(cv::Mat contour1, cv::Mat contour2)
    {
        cv::Mat coutour(contour1.rows + contour2.rows, 1, contour1.type());
        contour1.copyTo(coutour.rowRange(0, contour1.rows));
        contour2.copyTo(coutour.rowRange(contour1.rows, coutour.rows));

        return coutour;
    }

    void CombineRect(std::vector<cv::Rect>& rectVec, std::vector<cv::Mat>& contourVec, int th)
    {
        int combinePairNum{0};

        do
        {
            combinePairNum = 0;

            for (size_t i = 0; i < rectVec.size(); i++)
            {
                for (size_t j = 0; j < rectVec.size(); j++)
                {
                    if (i != j && MinDistanceBetweenRectangles(rectVec[i], rectVec[j]) < th)
                    {
                        rectVec[std::min(i, j)]    = rectVec[i] | rectVec[j];
                        contourVec[std::min(i, j)] = CombineContour(contourVec[i], contourVec[j]);
                        i                          = std::min(i, j);
                        rectVec.erase(rectVec.begin() + j);
                        contourVec.erase(contourVec.begin() + j);

                        j--;
                        combinePairNum++;
                    }
                }
            }

        } while (combinePairNum > 0);
    }

    void CombineRect(std::vector<cv::Rect>& rectVec, int th)
    {
        int combinePairNum{0};

        do
        {
            combinePairNum = 0;

            for (size_t i = 0; i < rectVec.size(); i++)
            {
                for (size_t j = 0; j < rectVec.size(); j++)
                {
                    if (i != j && MinDistanceBetweenRectangles(rectVec[i], rectVec[j]) < th)
                    {
                        rectVec[i] = rectVec[i] | rectVec[j];
                        rectVec.erase(rectVec.begin() + j);  // ��ʱ�ڴ��ؽ���
                        j--;                                 // ��ֹ������ǰ
                        combinePairNum++;
                    }
                }
            }

        } while (combinePairNum > 0);
    }

    /***************************************************************************************
    Function:  ���������㷨
    Input:     src ������ԭͼ�� pt ��ʼ������ th ��������ֵ����
    Output:    ��ʵ�ʵ����ڵ����� ʵ�����ǰ�ɫ�����������Ǻ�ɫ
    Description: �������������Ϊ��ɫ(255),����ɫΪ��ɫ(0)
    Return:    cv::Mat
    Others:    NULL
    ***************************************************************************************/
    cv::Mat RegionGrow(cv::Mat src, cv::Point2i pt, int th, int seedValue)
    {
        cv::Point2i              ptGrowing;                                         // ��������λ��
        int                      nGrowLable = 0;                                    // ����Ƿ�������
        int                      nSrcValue  = seedValue;                            // �������Ҷ�ֵ
        int                      nCurValue  = 0;                                    // ��ǰ������Ҷ�ֵ
        cv::Mat                  matDst     = cv::Mat::zeros(src.size(), CV_8UC1);  // ����һ���հ��������Ϊ��ɫ
        // ��������˳������
        int                      DIR[8][2]  = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
        std::vector<cv::Point2i> vcGrowPt;                         // ������ջ
        vcGrowPt.push_back(pt);                                    // ��������ѹ��ջ��
        matDst.at<uchar>(pt.y, pt.x) = 255;                        // ���������
        nSrcValue                    = src.at<uchar>(pt.y, pt.x);  // ��¼������ĻҶ�ֵ

        while (!vcGrowPt.empty())  // ����ջ��Ϊ��������
        {
            pt = vcGrowPt.back();  // ȡ��һ��������
            vcGrowPt.pop_back();

            // �ֱ�԰˸������ϵĵ��������
            for (int i = 0; i < 9; ++i)
            {
                ptGrowing.x = pt.x + DIR[i][0];
                ptGrowing.y = pt.y + DIR[i][1];
                // ����Ƿ��Ǳ�Ե��
                if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x > (src.cols - 1) || (ptGrowing.y > src.rows - 1))
                    continue;

                nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);  // ��ǰ��������ĻҶ�ֵ

                if (nGrowLable == 0)  // �����ǵ㻹û�б�����
                {
                    nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
                    if (abs(nSrcValue - nCurValue) < th)  // ����ֵ��Χ��������
                    {
                        matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;  // ���Ϊ��ɫ
                        vcGrowPt.push_back(ptGrowing);                     // ����һ��������ѹ��ջ��
                    }
                }
            }
        }
        return matDst.clone();
    }

    std::pair<cv::Mat, double> GetMaxContour(const std::vector<cv::Mat>& contourVec)
    {
        double  maxArea = 0.0;
        cv::Mat maxContour;
        for (const auto& contour : contourVec)
        {
            double curAera = contourArea(contour);
            if (curAera > maxArea)
            {
                maxArea    = curAera;
                maxContour = contour;
            }
        }

        return std::pair<cv::Mat, double>(maxContour, maxArea);
    };

    double SsimDetect(cv::Mat image_ref, cv::Mat image_obj)
    {
        double C1 = 6.5025, C2 = 58.5225;

        int    width    = image_ref.cols;
        int    height   = image_ref.rows;
        int    width2   = image_obj.cols;
        int    height2  = image_obj.rows;
        double mean_x   = 0;
        double mean_y   = 0;
        double sigma_x  = 0;
        double sigma_y  = 0;
        double sigma_xy = 0;

        for (int v = 0; v < height; v++)
        {
            for (int u = 0; u < width; u++)
            {
                mean_x += image_ref.at<uchar>(v, u);
                mean_y += image_obj.at<uchar>(v, u);
            }
        }

        mean_x = mean_x / width / height;
        mean_y = mean_y / width / height;

        for (int v = 0; v < height; v++)
        {
            for (int u = 0; u < width; u++)
            {
                sigma_x += (image_ref.at<uchar>(v, u) - mean_x) * (image_ref.at<uchar>(v, u) - mean_x);
                sigma_y += (image_obj.at<uchar>(v, u) - mean_y) * (image_obj.at<uchar>(v, u) - mean_y);
                sigma_xy += abs((image_ref.at<uchar>(v, u) - mean_x) * (image_obj.at<uchar>(v, u) - mean_y));
            }
        }

        sigma_x  = sigma_x / (width * height - 1);
        sigma_y  = sigma_y / (width * height - 1);
        sigma_xy = sigma_xy / (width * height - 1);

        double fenzi = (2 * mean_x * mean_y + C1) * (2 * sigma_xy + C2);
        double fenmu = (mean_x * mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
        double ssim  = fenzi / fenmu;

        return ssim;
    }

    double ComputeSsim(cv::Mat& standard, cv::Mat& frame)
    {
        assert(standard.type() == frame.type());

        if (CV_8UC3 == standard.type())
        {
            cv::cvtColor(standard, standard, cv::COLOR_BGR2GRAY);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        }

        // cv::namedWindow("standard", cv::WINDOW_NORMAL);
        // cv::imshow("standard", standard);
        // cv::namedWindow("frame", cv::WINDOW_NORMAL);
        // cv::imshow("frame", frame);

        const double C1 = 6.5025;
        const double C2 = 58.5225;
        cv::Size     win_size(3, 3);
        // cv::Size     win_size(7, 7);
        int          ndim = standard.dims;
        cv::Mat      X(standard);
        cv::Mat      Y(frame);
        double       NP       = std::pow(win_size.height, ndim);
        double       cov_norm = NP / (NP - 1);

        cv::Mat      ux, uy, uxx, uyy, uxy;
        cv::blur(X, ux, win_size);
        cv::blur(Y, uy, win_size);
        cv::blur(X.mul(X), uxx, win_size);
        cv::blur(Y.mul(Y), uyy, win_size);
        cv::blur(X.mul(Y), uxy, win_size);

        cv::Mat   temp_vxx(ux.mul(ux));
        cv::Mat   temp_vyy(uy.mul(uy));
        cv::Mat   temp_vxy(ux.mul(uy));

        cv::Mat   vx(cov_norm * (uxx - temp_vxx));
        cv::Mat   vy(cov_norm * (uyy - temp_vyy));
        cv::Mat   vxy(cov_norm * (uxy - temp_vxy));

        cv::Mat   a1(2 * temp_vxy + C1);
        cv::Mat   a2(2 * vxy + C2);
        cv::Mat   b1(temp_vxx + temp_vyy + C1);
        cv::Mat   b2(vx + vy + C2);

        cv::Mat   d(b1.mul(b2));
        cv::Mat   s((a1.mul(a2)) / d);
        const int pad  = 3;
        const int dpad = 6;
        s              = s(cv::Rect(pad, pad, s.cols - dpad, s.rows - dpad));
        double ssim    = cv::mean(s)[0];
        return ssim;
    }

    // 抠图
    std::pair<cv::Mat, cv::Rect> GrabRoi(const cv::Mat& img, const cv::Rect& roiRect, double borderRate)
    {
        int      horizontalBorder = int(roiRect.width * borderRate);
        int      verticalBorder   = int(roiRect.height * borderRate);
        int      x1               = std::max(roiRect.x - horizontalBorder, 0);
        int      y1               = std::max(roiRect.y - verticalBorder, 0);
        int      x2               = std::min(roiRect.x + roiRect.width + horizontalBorder, img.cols);
        int      y2               = std::min(roiRect.y + roiRect.height + verticalBorder, img.rows);

        cv::Rect roi = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        return std::pair<cv::Mat, cv::Rect>(img(roi), roi);
    }

    double TemplateMatch(const cv::Mat& search, const cv::Mat& temp, int matchMethod, cv::Rect& matchRect)
    {
        if (search.empty() || temp.empty())
        {
            return 0.0;
        }

        cv::Mat matchResult;
        cv::matchTemplate(search, temp, matchResult, matchMethod);

        double    minValue, maxValue, score;
        cv::Point minLocation, maxLocation, matchLocation;
        minMaxLoc(matchResult, &minValue, &maxValue, &minLocation, &maxLocation);

        if (cv::TM_SQDIFF == matchMethod || cv::TM_SQDIFF_NORMED == matchMethod)
        {
            matchLocation = minLocation;
            score         = 1 - minValue;
        }
        else
        {
            matchLocation = maxLocation;
            score         = maxValue;
        }

        matchRect.x      = matchLocation.x;
        matchRect.y      = matchLocation.y;
        matchRect.width  = temp.cols;
        matchRect.height = temp.rows;

        return score;
    }

    // void CvShow(const std::string& winname, const cv::Mat& img, bool bWait)
    // {
    //     cv::namedWindow(winname, cv::WINDOW_NORMAL);
    //     cv::imshow(winname, img);

    //     if (bWait)
    //     {
    //         cv::waitKey();
    //     }
    // }

    double ComputeTargetAndBorderColorSimilarity(const cv::Mat& img, const cv::Mat& contour, const cv::Rect& roiRect)
    {
        // 获取目标
        cv::Mat targetMsk = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::fillPoly(targetMsk, contour, cv::Scalar(255));
        // // 防止轮廓空洞
        // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        // cv::morphologyEx(objMsk, objMsk, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
        cv::Mat target;
        cv::bitwise_and(img, img, target, targetMsk);

        // 获取目标边框
        cv::Mat borderMsk = cv::Mat::zeros(img.size(), CV_8UC1);
        borderMsk(roiRect).setTo(255);
        cv::bitwise_xor(borderMsk, targetMsk, borderMsk, borderMsk);

        cv::Mat border;
        cv::bitwise_and(img, img, border, borderMsk);

        // 颜色范数计算，理论上目标和边框颜色差别很大
        cv::Mat targetGray;
        cv::cvtColor(target, targetGray, cv::COLOR_BGR2GRAY);
        int        targetElemNum  = cv::countNonZero(targetGray);
        cv::Scalar targetSum      = cv::sum(target);
        cv::Scalar targetMean     = targetSum / targetElemNum;
        auto       targetMeanNorm = cv::norm(targetMean, cv::NORM_L1);

        cv::Mat    borderGray;
        cv::cvtColor(border, borderGray, cv::COLOR_BGR2GRAY);
        int        borderElemNum  = cv::countNonZero(borderGray);
        cv::Scalar borderSum      = cv::sum(border);
        cv::Scalar borderMean     = borderSum / borderElemNum;
        auto       borderMeanNorm = cv::norm(borderMean, cv::NORM_L1);

        auto       rate = std::min(targetMeanNorm, borderMeanNorm) / std::max(targetMeanNorm, borderMeanNorm);
        return rate;
    }

    cv::Rect ResizeROI(const cv::Mat& img, const cv::Rect& roiRect, double borderRate)
    {
        int      horizontalBorder = int(roiRect.width * borderRate);
        int      verticalBorder   = int(roiRect.height * borderRate);

        int      topBorder    = std::min(verticalBorder, roiRect.y);
        int      bottomBorder = std::min(verticalBorder, img.rows - roiRect.y - roiRect.height);
        int      leftBorder   = std::min(horizontalBorder, roiRect.x);
        int      rightBorder  = std::min(horizontalBorder, img.cols - roiRect.x - roiRect.width);

        // 目标外扩
        int      x1 = roiRect.x - leftBorder;
        int      y1 = roiRect.y - topBorder;
        int      x2 = roiRect.x + roiRect.width + rightBorder;
        int      y2 = roiRect.y + roiRect.height + bottomBorder;

        cv::Rect borderRoi = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        return borderRoi;
    }

    Border LetterBoxResize(const cv::Mat&         src,
                           cv::Mat&               dst,
                           cv::InterpolationFlags interpolationMode,
                           const cv::Scalar&      fillColor)
    {
        assert(!src.empty() && !dst.empty());

        int     srcHeight = src.rows;
        int     srcWidth  = src.cols;
        int     dstHeight = dst.rows;
        int     dstWidth  = dst.cols;

        float   scaleRatioHeight = float(dstHeight) / srcHeight;
        float   scaleRatioWidth  = float(dstWidth) / srcWidth;

        float   scaleRatio = scaleRatioHeight < scaleRatioWidth ? scaleRatioHeight : scaleRatioWidth;

        int     resizedHeight = int(srcHeight * scaleRatio);
        int     resizedWidth  = int(srcWidth * scaleRatio);

        cv::Mat resized;
        cv::resize(src, resized, {resizedWidth, resizedHeight}, 0, 0, interpolationMode);

        // border
        int top    = (dstHeight - resizedHeight) / 2;
        int bottom = dstHeight - top - resizedHeight;
        int left   = (dstWidth - resizedWidth) / 2;
        int right  = dstWidth - left - resizedWidth;
        cv::copyMakeBorder(resized, dst, top, bottom, left, right, cv::BORDER_CONSTANT, fillColor);

        Border res;
        res.scaleRatio = scaleRatio;
        res.top        = top;
        res.bottom     = bottom;
        res.left       = left;
        res.right      = right;

        return res;
    }

    void DrawDottedCircle(cv::Mat img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color, int thickness)
    {
        float n = 15;  // 虚点间隔
        float w = p2.x - p1.x, h = p2.y - p1.y;
        float l = sqrtf(w * w + h * h);
        int   m = l / n;
        n       = l / m;  // 矫正虚点间隔，使虚点数为整数

        cv::circle(img, p1, 1, color, thickness);  // 画起点
        cv::circle(img, p2, 1, color, thickness);  // 画终点
        // 画中间点
        if (p1.y == p2.y)  // 水平线：y = m
        {
            float x1 = std::min(p1.x, p2.x);
            float x2 = std::max(p1.x, p2.x);
            for (float x = x1 + n; x < x2; x = x + n)
                cv::circle(img, cv::Point2f(x, p1.y), 1, color, thickness);
        }
        else if (p1.x == p2.x)  // 垂直线, x = m
        {
            float y1 = std::min(p1.y, p2.y);
            float y2 = std::max(p1.y, p2.y);
            for (float y = y1 + n; y < y2; y = y + n)
                cv::circle(img, cv::Point2f(p1.x, y), 1, color, thickness);
        }
        else  // 倾斜线，与x轴、y轴都不垂直或平行
        {
            // 直线方程的两点式：(y-y1)/(y2-y1)=(x-x1)/(x2-x1) -> y = (y2-y1)*(x-x1)/(x2-x1)+y1
            float m  = n * abs(w) / l;
            float k  = h / w;
            float x1 = std::min(p1.x, p2.x);
            float x2 = std::max(p1.x, p2.x);
            for (float x = x1 + m; x < x2; x = x + m)
                cv::circle(img, cv::Point2f(x, k * (x - p1.x) + p1.y), 1, color, thickness);
        }
    }

    void DrawDottedLine(cv::Mat img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color, int thickness)
    {
        float n = 5;  // 线长度
        float w = p2.x - p1.x, h = p2.y - p1.y;
        float l = sqrtf(w * w + h * h);
        // 矫正线长度，使线个数为奇数
        int   m = l / n;
        m       = m % 2 ? m : m + 1;
        n       = l / m;

        circle(img, p1, 1, color, thickness);  // 画起点
        circle(img, p2, 1, color, thickness);  // 画终点
        // 画中间点
        if (p1.y == p2.y)  // 水平线：y = m
        {
            float x1 = std::min(p1.x, p2.x);
            float x2 = std::max(p1.x, p2.x);
            for (float x = x1, n1 = 2 * n; x < x2; x = x + n1)
                cv::line(img, cv::Point2f(x, p1.y), cv::Point2f(x + n, p1.y), color, thickness);
        }
        else if (p1.x == p2.x)  // 垂直线, x = m
        {
            float y1 = std::min(p1.y, p2.y);
            float y2 = std::max(p1.y, p2.y);
            for (float y = y1, n1 = 2 * n; y < y2; y = y + n1)
                cv::line(img, cv::Point2f(p1.x, y), cv::Point2f(p1.x, y + n), color, thickness);
        }
        else  // 倾斜线，与x轴、y轴都不垂直或平行
        {
            // 直线方程的两点式：(y-y1)/(y2-y1)=(x-x1)/(x2-x1) -> y = (y2-y1)*(x-x1)/(x2-x1)+y1
            float n1 = n * abs(w) / l;
            float k  = h / w;
            float x1 = std::min(p1.x, p2.x);
            float x2 = std::max(p1.x, p2.x);
            for (float x = x1, n2 = 2 * n1; x < x2; x = x + n2)
            {
                cv::Point p3 = cv::Point2f(x, k * (x - p1.x) + p1.y);
                cv::Point p4 = cv::Point2f(x + n1, k * (x + n1 - p1.x) + p1.y);
                cv::line(img, p3, p4, color, thickness);
            }
        }
    }

    void DrawDottedRect(cv::Mat img, cv::Rect rect, cv::Scalar color, int thickness, int type)
    {
        auto point1 = rect.tl();
        auto point2 = point1;
        point2.x += rect.width;
        auto point3 = rect.br();
        auto point4 = point3;
        point4.x -= rect.width;

        if (0 == type)
        {
            DrawDottedLine(img, point1, point2, color, thickness);
            DrawDottedLine(img, point2, point3, color, thickness);
            DrawDottedLine(img, point3, point4, color, thickness);
            DrawDottedLine(img, point4, point1, color, thickness);
        }
        else
        {
            DrawDottedCircle(img, point1, point2, color, thickness);
            DrawDottedCircle(img, point2, point3, color, thickness);
            DrawDottedCircle(img, point3, point4, color, thickness);
            DrawDottedCircle(img, point4, point1, color, thickness);
        }
    }

}  // namespace cvkit
