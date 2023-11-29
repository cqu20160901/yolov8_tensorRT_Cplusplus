#ifndef COMMON_HPP
#define COMMON_HPP

#include "NvOnnxParser.h"
#include "logging.h"
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <fstream>
#include <numeric>

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

inline unsigned int GetElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t Volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}

bool ReadTrtFile(const std::string &engineFile, nvinfer1::ICudaEngine *&engine)
{
    std::string cached_engine;
    std::fstream file;
    std::cout << "loading filename from:" << engineFile << std::endl;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engineFile, std::ios::binary | std::ios::in);

    if (!file.is_open())
    {
        std::cout << "read file error: " << engineFile << std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    std::cout << "deserialize done" << std::endl;

    return true;
}

void OnnxToTRTModel(const std::string &modelFile, const std::string &filename, nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE)
{
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }

    // Build the engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "start building engine" << std::endl;
    engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine done" << std::endl;
    assert(engine);

    // we can destroy the parser
    parser->destroy();
    // save engine
    nvinfer1::IHostMemory *data = engine->serialize();
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char *)data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();

    // then close everything down
    network->destroy();
    builder->destroy();
}

#endif
