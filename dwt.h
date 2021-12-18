/*
作者: lboscu
描述: 实现1维和2维数据的DWT及IDWT
修改日期:
*/

#ifndef DWT_H
#define DWT_H
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<json.hpp>

using json = nlohmann::json;

namespace dwt {

#define ODD 1
#define EVEN 0

class Dwt
{
public:
    Dwt();
    Dwt(std::string waveName);

protected:
    cv::Mat ld; // 低通滤波器
    cv::Mat hd; // 高通滤波器
    cv::Mat lr; //  重构时低通
    cv::Mat hr; // 重构时高通

protected:
    /**
     * @brief 一维的特殊卷积运算，包括卷积+边界对称延拓或卷积+去除边界延拓
     */
    void conv(cv::Mat& src,cv::Mat& dst,cv::Mat kernel,std::string shape="valid");

    /**
     * @brief 降采样
     */
    void downSampling(cv::Mat& Input,cv::Mat&Output);

    /**
     * @brief 上采样
     */
    void upSampling(cv::Mat& Input,cv::Mat& Output);

    /**
     * @brief 对称边界延拓
     */
    void symExtend(cv::Mat& src,cv::Mat& dst,int len);

    /**
     * @brief 去除延拓边界
     */
    void dropExtend(cv::Mat&src,cv::Mat& dst);

protected:
    bool evenDataLen = EVEN; // 数据长度是奇数还是偶数
};

/**
* @brief 一维小波分解
* @attention 数据的类型为double
* @example
*   dwt::Dwt1D d("db4");
*   std::vector<double> data = {1,2,3,4,5,6,7,8,9,10,11,12,13};
*   cv::Mat cA,cD;
*   d.dec(data,cA,cD);
*   std::vector<double> out;
*   d.rec(out,cA,cD);
*/
class Dwt1D: public Dwt
{
public:
    explicit Dwt1D(std::string waveName);

    /**
     * @brief dwt
     */
    void dec(std::vector<double>& data,cv::Mat& cA,cv::Mat& cD);

    /**
     * @brief idwt
     */
    void rec(std::vector<double>& dst,cv::Mat& cA,cv::Mat& cD);
};

/**
* @brief 二维小波分解
* @attention 数据的类型为double
* @example
*   dwt::Dwt2D d("db4");
*   cv::Mat img = {...};
*   cv::Mat cA,cH,cV,cD;
*   d.dec(img,cA,cH,cV,cD);
*   cv::Mat out;
*   d.rec(out,cA,cH,cV,cD);
*/
class Dwt2D: public Dwt
{
public:
    explicit Dwt2D(std::string waveName);

    /**
     * @brief dwt2
     */
    void dec(cv::Mat& img,cv::Mat& cA,cv::Mat& cH,cv::Mat& cV,cv::Mat& cD);

    /**
     * @brief idwt2
     */
    void rec(cv::Mat& dst,cv::Mat& cA,cv::Mat& cH,cv::Mat& cV,cv::Mat& cD);

    /**
     * @brief idwt2 unit
     */
    void recUnit(cv::Mat& src,cv::Mat& dst,cv::Mat k1,cv::Mat k2);

protected:
    int m_rows = 1;
    int m_cols = 1;
};

/**
 * HARD 硬阈值
 * SOFT 软阈值
 * SMOOTH 光滑函数
 * CUSTOM 自定义
*/
enum {HARD,SOFT,SMOOTH,CUSTOM};

/**
 * @brief 阈值降噪处理
 * @param cv::Mat& src, // 输入
 * @param cv::Mat& dst, // 输出
 * @param int type, // 输入阈值函数类型，HARD,SOFT,SMOOTH or CUSTOM
 * @param double (*f)(double), // 回调函数，当type=CUSTOM时，需要添加
 * @return
 */
void denoise(cv::Mat& src,cv::Mat& dst,int type,double (*f)(double)=nullptr);
}

#endif // DWT_H
