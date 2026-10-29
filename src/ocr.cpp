// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#include "ocr.hpp"

#include "rules.hpp"

#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace eworld {

Tesseract::Tesseract() : api_(std::make_unique<tesseract::TessBaseAPI>()) {
    // nullptr datapath: the compiled-in tessdata, as the CLI uses it. Leave
    // TESSDATA_PREFIX unset (recipe_aerial_overlay.md).
    if (api_->Init(nullptr, "eng") != 0)
        throw std::runtime_error("tesseract: cannot load the eng traineddata");
    // The CLI's chatter ("Image too small to scale!!") went to a captured
    // stderr. Here it would flood the log on every thin crop.
    api_->SetVariable("debug_file", "/dev/null");
}

Tesseract::~Tesseract() {
    if (api_) api_->End();
}

std::vector<OcrWord> Tesseract::words(const cv::Mat& bgr, int psm) {
    std::vector<OcrWord> rows;
    if (bgr.empty()) return rows;
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    api_->SetPageSegMode(static_cast<tesseract::PageSegMode>(psm));
    api_->SetImage(rgb.data, rgb.cols, rgb.rows, 3, static_cast<int>(rgb.step));
    api_->SetSourceResolution(70);
    if (api_->Recognize(nullptr) != 0) {
        api_->Clear();
        return rows;
    }
    std::unique_ptr<char[]> tsv(api_->GetTSVText(0));
    api_->Clear();
    if (!tsv) return rows;
    // level page_num block_num par_num line_num word_num left top width height conf text
    std::istringstream in(tsv.get());
    for (std::string line; std::getline(in, line);) {
        std::vector<std::string> parts;
        std::string cell;
        std::istringstream ls(line);
        while (std::getline(ls, cell, '\t')) parts.push_back(cell);
        if (line.size() && line.back() == '\t') parts.emplace_back();
        if (parts.size() < 12 || parts[0] != "5") continue;
        double conf;
        int left, top, width, height;
        try {
            conf   = std::stod(parts[10]);
            left   = static_cast<int>(std::stod(parts[6]));
            top    = static_cast<int>(std::stod(parts[7]));
            width  = static_cast<int>(std::stod(parts[8]));
            height = static_cast<int>(std::stod(parts[9]));
        } catch (const std::exception&) {
            continue;
        }
        std::string text = parts[11];
        const auto a = text.find_first_not_of(" \t\r\n\f\v");
        const auto b = text.find_last_not_of(" \t\r\n\f\v");
        text = a == std::string::npos ? "" : text.substr(a, b - a + 1);
        if (conf < kOcrMinConf || text.empty()) continue;
        rows.push_back({text, conf, {left, top, left + width, top + height}});
    }
    return rows;
}

}  // namespace eworld
