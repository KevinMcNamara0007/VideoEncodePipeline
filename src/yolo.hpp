// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#pragma once
/// @file yolo.hpp
/// A YOLOv8 / YOLO-World detection head exported to ONNX
/// (tools/export_onnx.py), run with OpenCV DNN. Post-processing is the
/// Ultralytics predict path: letterbox to the export size (pad 114), RGB/255,
/// best class per anchor above `conf`, per-class greedy NMS at IoU 0.7,
/// max_det 300, boxes mapped back and clipped to the image.
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

namespace eworld {

struct YoloBox {
    int    cls{0};
    float  conf{0.0f};
    double xyxy[4]{0, 0, 0, 0};
};

class YoloHead {
public:
    /// `onnx` is the model; class names come from <onnx stem>.names.json.
    YoloHead(const std::string& onnx, int imgsz);
    /// Boxes for one BGR image, best confidence first.
    std::vector<YoloBox> predict(const cv::Mat& bgr, float conf, float iou = 0.7f, int max_det = 300);
    const std::vector<std::string>& names() const { return names_; }

private:
    cv::dnn::Net             net_;
    std::vector<std::string> names_;
    int                      imgsz_;
};

}  // namespace eworld
