// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
/// yolo_dump <model.onnx> <conf> <image>... -- one JSON line per image:
/// {"image", "boxes": [{"name", "conf", "xyxy"}]}. Compare it with
/// Ultralytics predict (conf, imgsz 640) on the same files.
#include "yolo.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: yolo_dump <model.onnx> <conf> <image>...\n";
        return 2;
    }
    eworld::YoloHead head(argv[1], 640);
    const float conf = std::stof(argv[2]);
    for (int i = 3; i < argc; ++i) {
        nlohmann::json boxes = nlohmann::json::array();
        for (const auto& b : head.predict(cv::imread(argv[i], cv::IMREAD_COLOR), conf))
            boxes.push_back({{"name", head.names()[static_cast<size_t>(b.cls)]},
                             {"conf", b.conf},
                             {"xyxy", {b.xyxy[0], b.xyxy[1], b.xyxy[2], b.xyxy[3]}}});
        std::cout << nlohmann::json{{"image", argv[i]}, {"boxes", boxes}}.dump() << "\n";
    }
    return 0;
}
