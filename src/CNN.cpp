#include "CNN.hpp"
#include "common/common.hpp"
#include <algorithm>
#include <chrono>

CNN::CNN(const std::string &OnnxFilePath, const std::string &SaveTrtFilePath, int BatchSize, int InputChannel, int InputImageWidth, int InputImageHeight, int ModelOutHeadNum)
{
    OnnxFilePath_ = OnnxFilePath;
    SaveTrtFilePath_ = SaveTrtFilePath;
    BatchSize_ = BatchSize;
    InputChannel_ = InputChannel;
    InputImageWidth_ = InputImageWidth;
    InputImageHeight_ = InputImageHeight;
    ModelOutHeadNum_ = ModelOutHeadNum;
}

CNN::~CNN()
{
}

void CNN::ModelInit()
{
    std::fstream existEngine;
    existEngine.open(SaveTrtFilePath_, std::ios::in);
    if (existEngine)
    {
        ReadTrtFile(SaveTrtFilePath_, PtrEngine_);
        assert(PtrEngine_ != nullptr);
    }
    else
    {
        OnnxToTRTModel(OnnxFilePath_, SaveTrtFilePath_, PtrEngine_, BatchSize_);
        assert(PtrEngine_ != nullptr);
    }
}

void CNN::Inference(cv::Mat &SrcImage)
{
    assert(PtrEngine_ != nullptr);
    PtrContext_ = PtrEngine_->createExecutionContext();
    assert(PtrContext_ != nullptr);

    const int ModelOutHeadNum = ModelOutHeadNum_;
    void *Buffers[ModelOutHeadNum];

    std::vector<int64_t> BufferSize;
    BufferSize.resize(ModelOutHeadNum);

    int nbBindings = PtrEngine_->getNbBindings();
    int64_t TotalSize = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = PtrEngine_->getBindingDimensions(i);
        nvinfer1::DataType dtype = PtrEngine_->getBindingDataType(i);
        TotalSize = Volume(dims) * 1 * GetElementSize(dtype);
        BufferSize[i] = TotalSize;
        cudaMalloc(&Buffers[i], TotalSize);

        // std::cout << "binding:" << i << ": " << TotalSize << std::endl;
    }

    // get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    EngineInference(SrcImage, Buffers, BufferSize, stream);

    // release the stream and the Buffers
    cudaStreamDestroy(stream);
    for (int i = 0; i < nbBindings; i++)
    {
        cudaFree(Buffers[i]);
    }

    // destroy the engine
    PtrContext_->destroy();
    PtrEngine_->destroy();
}

void CNN::EngineInference(cv::Mat &SrcImage, void **Buffers, const std::vector<int64_t> &BufferSize, cudaStream_t Stream)
{
    std::vector<float> curInput = PrepareImage(SrcImage);

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    std::cout << "host2device" << std::endl;
    cudaMemcpyAsync(Buffers[0], curInput.data(), BufferSize[0], cudaMemcpyHostToDevice, Stream);

    // do inference
    auto t_start = std::chrono::high_resolution_clock::now();
    PtrContext_->execute(BatchSize_, Buffers);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference take: " << total_inf << " ms." << std::endl;

    const int ModelOutHeadNum = BufferSize.size() - 1;
    float *ModelOutput[ModelOutHeadNum];
    for (int i = 0; i < BufferSize.size() - 1; i++)
    {
        ModelOutput[i] = new float[int(BufferSize[i + 1] / sizeof(float))];
        cudaMemcpyAsync(ModelOutput[i], Buffers[i + 1], BufferSize[i + 1], cudaMemcpyDeviceToHost, Stream);
    }

    cudaStreamSynchronize(Stream);

    // Postprocess
    auto t_start1 = std::chrono::high_resolution_clock::now();
    static GetResultRectYolov8 Postprocess;
    int ret = Postprocess.GetConvDetectionResult(ModelOutput, DetectiontRects_);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Postprocess take: " << total_inf1 << " ms." << std::endl;

    for (int i = 0; i < BufferSize.size() - 1; i++)
    {
        delete[] ModelOutput[i];
    }
}

std::vector<float> CNN::PrepareImage(cv::Mat &SrcImage)
{
    std::vector<float> Result(BatchSize_ * InputImageWidth_ * InputImageHeight_ * InputChannel_);
    float *Imagedata = Result.data();

    cv::Mat rsz_img;
    cv::resize(SrcImage, rsz_img, cv::Size(InputImageWidth_, InputImageHeight_));
    rsz_img.convertTo(rsz_img, CV_32FC3, 1.0 / 255);

    // HWC TO CHW
    int channelLength = InputImageWidth_ * InputImageHeight_;
    std::vector<cv::Mat> split_img = {cv::Mat(InputImageHeight_, InputImageWidth_, CV_32FC1, Imagedata + channelLength * 2),
                                      cv::Mat(InputImageHeight_, InputImageWidth_, CV_32FC1, Imagedata + channelLength * 1),
                                      cv::Mat(InputImageHeight_, InputImageWidth_, CV_32FC1, Imagedata + channelLength * 0)};

    cv::split(rsz_img, split_img);

    return Result;
}