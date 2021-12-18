#include "dwt.h"


/**
* @brief 根据窗口名，选择常用已经定义好的低通和高通滤波器，及对应的重构滤波器
* @return
*/
dwt::Dwt::Dwt(std::string waveName)
{
    std::ifstream file("./waveFilter.json"); // 从文件中提取常用滤波器
    if(!file.is_open()) {
        std::cerr << "waveFilter.json not fond!";
    }

    json data;
    file >> data; // 解析json

    std::vector<double> tmp1 =  data.at(waveName).at("ld");
    this->ld = cv::Mat_<double>(tmp1).clone(); // 必须使用clone，不然会出现内存中数据被修改的错误
    std::vector<double> tmp2 =  data.at(waveName).at("hd");
    this->hd = cv::Mat_<double>(tmp2).clone();
    std::vector<double> tmp3 =  data.at(waveName).at("lr");
    this->lr = cv::Mat_<double>(tmp3).clone();
    std::vector<double> tmp4 =  data.at(waveName).at("hr");
    this->hr = cv::Mat_<double>(tmp4).clone();

    file.close();
}

/**
 * @brief 一维的特殊卷积运算，包括卷积+边界对称延拓或卷积+去除边界延拓
 * @param cv::Mat& src, // 输入被卷积的信号
 * @param cv::Mat& dst, // 输出卷积后结果
 * @param cv::Mat kernel, // 输入卷积核或滤波器
 * @param std::string shape   // 卷积方式，包括"valid"或者"full",default is valid
 * @return
 */
void dwt::Dwt::conv(cv::Mat& src,cv::Mat& dst,cv::Mat kernel,std::string shape)
{
    if(kernel.cols == 1) {
        kernel = kernel.t();
    }
    cv::flip(kernel,kernel,1); // 卷积时，卷积核左右翻转。因为卷积核数据不对称，所以这是必须的

    int m = src.total();
    int n = kernel.total();

    if(shape == "valid") { // 分解信号时，走这个代码块
        if(src.cols == 1) {
            dst = cv::Mat_<double>(m+n+1,1);
        }
        else if(src.rows == 1) {
            dst = cv::Mat_<double>(1,m+n+1);
        }

        cv::Mat A;
        symExtend(src,A,n); // 对称延拓后长度为m+2*n+0(1)

        cv::Mat tmp;
        for(int i=0;i<m+n+1;i++) { // 卷积后长度为m+2*n-(n-1)=m+n+1
            tmp = A(cv::Rect(i,0,n,1));
            dst.at<double>(i) =  tmp.dot(kernel);
        }
    }
    if(shape == "full") { // 重构信号时走这个代码块
        // full卷积后长度为2*len(cA)+n-1
        if(src.cols == 1) {
            dst = cv::Mat_<double>(m+n-1,1);
        }
        else if(src.rows == 1) {
            dst = cv::Mat_<double>(1,m+n-1);
        }

        cv::Mat tmp,A;
        A = cv::Mat_<double>(1,m+2*n-2); // 卷积前，边界补0
        for(int i=0;i<A.cols;i++) {
            if(i>=n-1 && i<m+n-1)
                A.at<double>(i) = src.at<double>(i-n+1);
            else
                A.at<double>(i) = 0;
        }
        for(int i=0;i<m+n-1;i++) {
            tmp = A(cv::Rect(i,0,n,1));
            dst.at<double>(i) =  tmp.dot(kernel);
        }
        dropExtend(dst,dst); // 去掉边界，或者从中提取有效数据，返回len(dst)<2*len(cA)+n-1
    }
}

/**
 * @brief 降采样
 * @param cv::Mat& Input, // 输入数据
 * @param cv::Mat& Output, // 输出降采样后的行向量
 * @return
 */
void dwt::Dwt::downSampling(cv::Mat& Input,cv::Mat& Output)
{
    // 去掉首尾边界，再采样偶数列
    unsigned len = Input.total();
    Output = cv::Mat_<double>(1,(len-2)/2);
    for(unsigned i=0;i<(len-2)/2;i++) {
        Output.at<double>(i) = Input.at<double>(2*i+2);
    }
}

/**
 * @brief 上采样,向后插值0
 * @param cv::Mat& Input, // 输入数据
 * @param cv::Mat& Output, // 输出上采样后的向量
 * @return
 */
void dwt::Dwt::upSampling(cv::Mat& Input,cv::Mat& Output)
{
    unsigned len = Input.total();

    if(Input.cols == 1) {   // 使得输出和输入一样都是行向量或列向量
        Output = cv::Mat_<double>(Input.rows*2,1);
    }
    else if(Input.rows == 1) {
        Output = cv::Mat_<double>(1,Input.cols*2);
    }

    for(unsigned i=0;i<len;i++) {
        Output.at<double>(i*2) = Input.at<double>(i);
        Output.at<double>(i*2+1) = 0;
    }   //  2468 => 20406080

}

/**
 * @brief 对称边界延拓
 * @param cv::Mat& src, // 输入数据
 * @param cv::Mat& dst, // 输出边界延拓后的行向量
 * @param int len, // 输入两边延拓的长度，等于卷积核长度
 * @return
 */
void dwt::Dwt::symExtend(cv::Mat& src,cv::Mat& dst,int len)
{
    // 对称延拓  12345 => 4321 12345 5432
    unsigned slen = src.total();
    unsigned dlen = slen + 2 * len;
    if(slen % 2 != 0) { // 数据长度是奇数，右延拓就增加1个数
        dlen += 1;
    }

    dst = cv::Mat_<double>(1,dlen); // 分配空间给输出向量
    for(unsigned i=len;i<len+slen;i++) {
        dst.at<double>(i) = src.at<double>(i-len);
    }

    for(int i=len-1;i>=0;i--) { // 0,1,2,...,len-1,len,len+1,len+2,...
        dst.at<double>(i) = src.at<double>(len-i-1);
    }

    for(unsigned i=len+slen;i<dlen;i++) { // slen-1 - (i-len-slen)
        dst.at<double>(i) = src.at<double>(-i+len +2*slen -1);
    }
}

/**
 * @brief 去除边界，提取有效数据
 * @param cv::Mat& src, // 输入数据
 * @param cv::Mat& dst, // 输出向量
 * @return
 */
void dwt::Dwt::dropExtend(cv::Mat&src,cv::Mat& dst)
{
    int m = src.total();
    int n = this->lr.total();

    // 增加的长度，如果原始数据长度是偶数，则加1，否则不加。
    // 目的是为了保持重构后长度与原始长度一致
    int bl = evenDataLen ? 0 : 1;

    // 保持输入输出同为行或列向量
    if(src.rows==1) {
        dst = src(cv::Range(0,1),cv::Range(n-2,m-n+bl)).clone();
    }
    else if(src.cols==1) {
        dst = src(cv::Range(n-2,m-n+bl),cv::Range(0,1)).clone();
    }
}

/**
 * @brief
 * @param std::string waveName, // 小波名字
 * @return
 */
dwt::Dwt1D::Dwt1D(std::string waveName):Dwt(waveName)
{

}

/**
 * @brief 1维dwt分解
 * @param std::vector<double>& data, // 输入数据
 * @param cv::Mat& cA, // 输出低频近似系数
 * @param cv::Mat& cD， // 输出高频细节系数
 * @return
 */
void dwt::Dwt1D::dec(std::vector<double>& data,cv::Mat& cA,cv::Mat& cD)
{
    cv::Mat A(data);
    if(A.total() % 2 == 0) { // 如果原始数据长度是偶数，则...
        this->evenDataLen = EVEN;
    }
    else {  // 如果原始数据长度是偶数，则...
        this->evenDataLen = ODD;
    }

    cv::Mat xld;
    conv(A,xld,this->ld); // 低通卷积
    downSampling(xld,cA); // 下采样

    cv::Mat xhd;
    conv(A,xhd,this->hd); // 高通卷积
    downSampling(xhd,cD); // 下采样
}

/**
 * @brief 1维dwt重构
 * @param std::vector<double>& dst, // 输出重构结果
 * @param cv::Mat& cA, // 输入低频近似系数
 * @param cv::Mat& cD， // 输入高频细节系数
 * @return
 */
void dwt::Dwt1D::rec(std::vector<double>& dst,cv::Mat& cA,cv::Mat& cD)
{
    cv::Mat xld,A,B;
    upSampling(cA,xld);
    conv(xld,A,this->lr,"full"); // 2*cA+1-n+0(1)

    upSampling(cD,xld);
    conv(xld,B,this->hr,"full");
  // 12345  1020304050 full conv   8*2+8-1=23
    cv::Mat last = A + B; // 这两个表达式不能合在一起
    dst = last;
}

/**
 * @brief 2维数据离散小波变换
 * @param std::string waveName, // 输入小波名字
 * @return
 */
dwt::Dwt2D::Dwt2D(std::string waveName):Dwt(waveName)
{

}

/**
 * @brief 2维dwt分解
 * @param cv::Mat& img, // 输入二维数据
 * @param cv::Mat& cA, // 输出低频近似系数
 * @param cv::Mat& cH, // 输出竖直细节系数
 * @param cv::Mat& cV, // 输出水平细节系数
 * @param cv::Mat& cD, // 输出对角细节系数
 * @return
 */
void dwt::Dwt2D::dec(cv::Mat& img,cv::Mat& cA,cv::Mat& cH,cv::Mat& cV,cv::Mat& cD)
{
    // img是double类型，单通道
    int rows = m_rows = img.rows;
    int cols = m_cols = img.cols;

    int drows = (rows + this->ld.total() - 1)/2; // 分解后系数矩阵的行数
    int dcols = (cols + this->ld.total() - 1)/2; // 分解后系数矩阵的列数
    cv::Mat dst,tmp,ds;
    cv::Mat last,last2;

    // 将每行数据分解，采用低通
    last = cv::Mat_<double>(rows,dcols);
    for(int i=0;i<rows;i++) {
        tmp = img(cv::Range(i,i+1),cv::Range(0,cols));
        conv(tmp,dst,this->ld);
        downSampling(dst,ds);
        ds.copyTo(last(cv::Range(i,i+1),cv::Range(0,dcols)));
    }

    // 将每列数据分解，采用低通
    last2 = cv::Mat_<double>(drows,dcols);
    for(int i=0;i<dcols;i++) {
        tmp = last(cv::Range(0,rows),cv::Range(i,i+1));
        conv(tmp,dst,this->ld);
        downSampling(dst,ds);
        cv::Mat(ds.t()).copyTo(last2(cv::Range(0,drows),cv::Range(i,i+1)));
    }
    cA = last2.clone(); // 得到cA

    // 将每列数据分解，采用高通
    for(int i=0;i<dcols;i++) {
        tmp = cv::Mat(last(cv::Range(0,rows),cv::Range(i,i+1)));
        conv(tmp,dst,this->hd);
        downSampling(dst,ds);
        cv::Mat(ds.t()).copyTo(last2(cv::Range(0,drows),cv::Range(i,i+1)));
    }
    cH = last2.clone(); // 得到cH

    // 将每行数据分解，采用高通
    for(int i=0;i<rows;i++) {
        tmp = img(cv::Range(i,i+1),cv::Range(0,cols));
        conv(tmp,dst,this->hd);
        downSampling(dst,ds);
        ds.copyTo(last(cv::Range(i,i+1),cv::Range(0,dcols)));
    }

    // 将每列数据分解，采用低通
    for(int i=0;i<dcols;i++) {
        tmp = cv::Mat(last(cv::Range(0,rows),cv::Range(i,i+1)));
        conv(tmp,dst,this->ld);
        downSampling(dst,ds);
        cv::Mat(ds.t()).copyTo(last2(cv::Range(0,drows),cv::Range(i,i+1)));
    }
    cV = last2.clone(); // 得到cV

    // 将每列数据分解，采用高通
    for(int i=0;i<dcols;i++) {
        tmp = cv::Mat(last(cv::Range(0,rows),cv::Range(i,i+1)));
        conv(tmp,dst,this->hd);
        downSampling(dst,ds);
        cv::Mat(ds.t()).copyTo(last2(cv::Range(0,drows),cv::Range(i,i+1)));
    }
    cD = last2.clone(); // 得到cD
}


/**
 * @brief 2维dwt重构
 * @param cv::Mat& dst, // 输出二维数据
 * @param cv::Mat& cA, // 输入低频近似系数
 * @param cv::Mat& cH, // 输入竖直细节系数
 * @param cv::Mat& cV, // 输入水平细节系数
 * @param cv::Mat& cD, // 输入对角细节系数
 * @return
 */
void dwt::Dwt2D::rec(cv::Mat& dst,cv::Mat& cA,cv::Mat& cH,cv::Mat& cV,cv::Mat& cD)
{
    cv::Mat A;
    recUnit(cA,A,this->lr,this->lr);    // 重构cA部分

    cv::Mat B;
    recUnit(cH,B,this->hr,this->lr);    // 重构cH部分

    cv::Mat C;
    recUnit(cV,C,this->lr,this->hr);    // 重构cV部分

    cv::Mat D;
    recUnit(cD,D,this->hr,this->hr);    // 重构cD部分

    dst = A + B + C + D;
}

/**
 * @brief 2维dwt重构的一个unit,重构一个系数的数据
 * @param cv::Mat& cA, // 输入要重构的系数
 * @param cv::Mat& dst, // 输出重构后的数据
 * @param cv::Mat k1, // 输入卷积核1
 * @param cv::Mat k2, // 输入卷积核2
 * @return
 */
void dwt::Dwt2D::recUnit(cv::Mat& cA,cv::Mat& dst,cv::Mat k1,cv::Mat k2)
{
    int nc = cA.cols;
    int nr = cA.rows;
    cv::Mat A,A0;
    cv::Mat tmp,tmp2,last2,last;

    // 先重构每一列的数据
    int nl = 2*nr - this->lr.total()+1; // 重构后的行数
    if(m_rows % 2 == 0) {
        this->evenDataLen = EVEN;
        nl += 1;    //  偶数时，行数+1
    }
    else {
        this->evenDataLen = ODD;
    }

    last2 = cv::Mat_<double>(nl,nc);
    A0 = cv::Mat_<double>(nl,1);    // 每一列数据重构后的结果
    for(int i=nc-1;i>=0;i--) {
        tmp = cv::Mat(cA(cv::Range(0,nr),cv::Range(i,i+1)));
        upSampling(tmp,tmp2);
        conv(tmp2,A0,k1,"full");
        A0.copyTo(last2(cv::Range(0,nl),cv::Range(i,i+1)));
    }

    // 再重构每一行的数据
    int nt = 2*nc - this->lr.total()+1; // 重构后的列数
    if(m_cols % 2 == 0) {
        this->evenDataLen = EVEN;
        nt += 1;
    }
    else {
        this->evenDataLen = ODD;
    }

    last = cv::Mat_<double>(nl,nt);
    A0 = cv::Mat_<double>(1,nt);    // 每一行数据重构后的结果
    for(int i=nl-1;i>=0;i--) {
        tmp = cv::Mat(last2(cv::Range(i,i+1),cv::Range(0,nc)));
        upSampling(tmp,tmp2);
        conv(tmp2,A0,k2,"full");
        A0.copyTo(last(cv::Range(i,i+1),cv::Range(0,nt)));
    }

    dst = last.clone();
}

/**
 * @brief 阈值降噪处理
 * @param cv::Mat& src, // 输入
 * @param cv::Mat& dst, // 输出
 * @param int type, // 输入阈值类型,HARD,SOFT,SMOOTH,CUSTOM
 * @param double (*f)(double), // 回调函数
 * @return
 */
void dwt::denoise(cv::Mat& src,cv::Mat& dst,int type,double (*f)(double))
{
    dst = src.clone();
    int len = src.total();

    // 固定阈值
    cv::resize(dst,dst,cv::Size(src.total(),1));
    cv::Mat tmpM = cv::abs(dst);
    cv::sort(tmpM,tmpM,CV_SORT_EVERY_ROW);
    cv::resize(dst,dst,cv::Size(src.cols,src.rows));
    // 噪声标准差估计
    double sigma = len % 2 != 0 ? tmpM.at<double>(len/2) :
                  (tmpM.at<double>(len/2)+tmpM.at<double>((len-1)/2))/2.0;
    sigma /= 0.6745;
    double lambda = sigma * std::sqrt(2*std::log(len));

    int tmp;
    if(dwt::HARD == type) {
        for(int i=0;i<len;i++) {
            tmp = std::abs(src.at<double>(i));
            if(tmp >= lambda) {
                dst.at<double>(i) = src.at<double>(i);
            }
            else {
                dst.at<double>(i) = 0;
            }
        }
    }

    if(dwt::SOFT == type) {
        for(int i=0;i<len;i++) {
            tmp = std::abs(src.at<double>(i));
            if(tmp >= lambda) {
                dst.at<double>(i) = (src.at<double>(i) > 0 ? 1 : -1)*(tmp - lambda);
            }
            else {
                dst.at<double>(i) = 0;
            }
        }
    }

    if(dwt::SMOOTH == type) {
        double alpha = 0.9; // 参数1
        double m = 2; // 参数2
        for(int i=0;i<len;i++) {
            tmp = std::abs(src.at<double>(i));
            if(tmp >= lambda) {
                dst.at<double>(i) = src.at<double>(i) * (1 - alpha / std::exp(m*(tmp-lambda)*(tmp-lambda)));
            }
            else {
                dst.at<double>(i) = src.at<double>(i) * (1 - alpha) / std::exp((tmp-lambda)*(tmp-lambda));
            }
        }
    }

    if(dwt::CUSTOM == type) {
        for(int i=0;i<len;i++) {
            dst.at<double>(i) = f(src.at<double>(i));
        }
    }
}
