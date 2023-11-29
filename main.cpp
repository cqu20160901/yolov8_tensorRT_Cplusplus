#include "src/CNN.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    std::string OnnxFile = "/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/models/yolov8n_ZQ.onnx";
    std::string SaveTrtFilePath = "/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/models/yolov8n_ZQ.trt";
    cv::Mat SrcImage = cv::imread("/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;

    CNN YOLO(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640, 7);
    YOLO.ModelInit();
    YOLO.Inference(SrcImage);

    for (int i = 0; i < YOLO.DetectiontRects_.size(); i += 6)
    {
        int classId = int(YOLO.DetectiontRects_[i + 0]);
        float conf = YOLO.DetectiontRects_[i + 1];
        int xmin = int(YOLO.DetectiontRects_[i + 2] * float(img_width) + 0.5);
        int ymin = int(YOLO.DetectiontRects_[i + 3] * float(img_height) + 0.5);
        int xmax = int(YOLO.DetectiontRects_[i + 4] * float(img_width) + 0.5);
        int ymax = int(YOLO.DetectiontRects_[i + 5] * float(img_height) + 0.5);

        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(SrcImage, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(SrcImage, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    imwrite("/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/images/result.jpg", SrcImage);

    printf("== obj: %d \n", int(float(YOLO.DetectiontRects_.size()) / 6.0));

    return 0;
}
