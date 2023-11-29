#ifndef CNN_HPP
#define CNN_HPP

#include "NvInfer.h"
#include "postprocess.hpp"
#include <opencv2/opencv.hpp>

class CNN
{
public:
    CNN(const std::string &OnnxFilePath, const std::string &SaveTrtFilePath, int BatchSize, int InputChannel, int InputImageWidth, int InputImageHeight, int ModelOutHeadNum);
    ~CNN();
    void ModelInit();
    void Inference(cv::Mat &SrcImage);

    std::vector<float> DetectiontRects_;

private:
    void EngineInference(cv::Mat &SrcImage, void **Buffers, const std::vector<int64_t> &BufferSize, cudaStream_t Stream);
    std::vector<float> PrepareImage(cv::Mat &vec_img);

    std::string OnnxFilePath_;
    std::string SaveTrtFilePath_;

    int BatchSize_ = 0;
    int InputChannel_ = 0;
    int InputImageWidth_ = 0;
    int InputImageHeight_ = 0;
    int ModelOutHeadNum_ = 0;
    int ModelOutputSize_ = 0;

    nvinfer1::ICudaEngine *PtrEngine_ = nullptr;
    nvinfer1::IExecutionContext *PtrContext_ = nullptr;
};

#endif // YOLOV6_TRT_YOLOV6_H
