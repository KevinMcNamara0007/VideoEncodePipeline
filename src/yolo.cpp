// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#include "yolo.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace eworld {
namespace fs = std::filesystem;

YoloHead::YoloHead(const std::string& onnx, int imgsz) : imgsz_(imgsz) {
    if (!fs::is_regular_file(onnx)) throw std::runtime_error("model not found: " + onnx);
    const fs::path names_path = fs::path(onnx).replace_extension(".names.json");
    std::ifstream in(names_path);
    if (!in) throw std::runtime_error("class names not found: " + names_path.string() +
                                      " (tools/export_onnx.py writes it beside the .onnx)");
    names_ = nlohmann::json::parse(in).get<std::vector<std::string>>();
    net_ = cv::dnn::readNetFromONNX(onnx);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<YoloBox> YoloHead::predict(const cv::Mat& bgr, float conf, float iou, int max_det) {
    std::vector<YoloBox> out;
    if (bgr.empty()) return out;
    // Letterbox: scale the long side to imgsz, centre, pad with 114.
    const double r  = std::min(static_cast<double>(imgsz_) / bgr.rows, static_cast<double>(imgsz_) / bgr.cols);
    const int    nw = static_cast<int>(std::lround(bgr.cols * r));
    const int    nh = static_cast<int>(std::lround(bgr.rows * r));
    const double dw = (imgsz_ - nw) / 2.0, dh = (imgsz_ - nh) / 2.0;
    cv::Mat img = bgr;
    if (nw != bgr.cols || nh != bgr.rows) cv::resize(bgr, img, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    const int top = static_cast<int>(std::lround(dh - 0.1)), bottom = static_cast<int>(std::lround(dh + 0.1));
    const int left = static_cast<int>(std::lround(dw - 0.1)), right = static_cast<int>(std::lround(dw + 0.1));
    if (top || bottom || left || right)
        cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(imgsz_, imgsz_), cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat pred = net_.forward();   // [1, 4 + nc, anchors]
    if (pred.dims != 3 || pred.size[1] != 4 + static_cast<int>(names_.size()))
        throw std::runtime_error("unexpected YOLO output shape for " + std::to_string(names_.size()) + " classes");
    const int nc = pred.size[1] - 4, na = pred.size[2];
    const float* p = pred.ptr<float>();

    struct Cand { int cls; float conf; double b[4]; };
    std::vector<Cand> cands;
    for (int a = 0; a < na; ++a) {
        int best = 0;
        float score = p[4 * na + a];
        for (int c = 1; c < nc; ++c) {
            const float s = p[(4 + c) * na + a];
            if (s > score) { score = s; best = c; }
        }
        if (score <= conf) continue;
        const double cx = p[a], cy = p[na + a], w = p[2 * na + a], h = p[3 * na + a];
        cands.push_back({best, score, {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2}});
    }
    std::vector<size_t> order(cands.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return cands[x].conf > cands[y].conf; });

    auto iou_of = [](const double* a, const double* b) {
        const double iw = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
        const double ih = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
        const double inter = iw * ih;
        const double u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        return u > 0 ? inter / u : 0.0;
    };
    std::vector<size_t> kept;
    for (size_t i : order) {
        bool drop = false;
        for (size_t k : kept)
            if (cands[k].cls == cands[i].cls && iou_of(cands[k].b, cands[i].b) > iou) { drop = true; break; }
        if (drop) continue;
        kept.push_back(i);
        if (static_cast<int>(kept.size()) >= max_det) break;
    }
    for (size_t i : kept) {
        YoloBox yb{cands[i].cls, cands[i].conf, {}};
        const double lim[4] = {static_cast<double>(bgr.cols), static_cast<double>(bgr.rows),
                               static_cast<double>(bgr.cols), static_cast<double>(bgr.rows)};
        for (int j = 0; j < 4; ++j) {
            const double v = (cands[i].b[j] - (j % 2 == 0 ? dw : dh)) / r;
            yb.xyxy[j] = std::clamp(v, 0.0, lim[j]);
        }
        out.push_back(yb);
    }
    return out;
}

}  // namespace eworld
