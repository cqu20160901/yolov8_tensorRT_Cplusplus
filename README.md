# yolov8_tensorRT_Cplusplus
yolov8 tensorRT 的 C++部署。

TensorRT版本：TensorRT-7.1.3.4

编译前修改 CMakeLists.txt 对应的TensorRT版本

![17012433486778](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/62dbacce-3197-43b1-b44d-ed18f4619ba4)

## 编译
```powershell
cd yolov8_tensorRT_Cplusplus
mkdir build
cd build
cmake ..
make
```

## 运行
```powershell
cd build
./yolo_trt
```

## 测试效果

onnx 测试效果

![image](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/8574c0ce-fc56-4b3c-9c7e-ec31e29b01ed)

tensorRT 测试效果

![result](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/29a8115d-a5ce-4c58-9b1a-c48766cdfcd5)

tensorRT 时耗

![17012420664762](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/781af480-1f2c-473e-a254-366598866141)


修改相关的路径
```cpp
    std::string OnnxFile = "/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/models/yolov8n_ZQ.onnx";
    std::string SaveTrtFilePath = "/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/models/yolov8n_ZQ.trt";
    cv::Mat SrcImage = cv::imread("/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;

    CNN YOLO(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640, 7);  // 1, 3, 640, 640, 7 前四个为模型输入的NCWH, 7为模型输出叶子节点的个数+1，（本示例中的onnx模型输出有6个叶子节点，再+1=7）
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

```



