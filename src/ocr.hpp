// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#pragma once
/// @file ocr.hpp
/// Tesseract word reads, through libtesseract instead of the CLI. Same rows
/// as `tesseract stdin stdout --psm N -l eng tsv`: level-5 words with
/// conf >= kOcrMinConf and non-blank text. The image is read at 70 dpi, the
/// value the CLI falls back to for a PNG with no resolution.
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tesseract { class TessBaseAPI; }

namespace eworld {

struct OcrWord {
    std::string text;
    double      conf{0.0};
    int         xyxy[4]{0, 0, 0, 0};
};

class Tesseract {
public:
    Tesseract();
    ~Tesseract();
    /// Words in a BGR image at page segmentation mode `psm`.
    std::vector<OcrWord> words(const cv::Mat& bgr, int psm);

private:
    std::unique_ptr<tesseract::TessBaseAPI> api_;
};

}  // namespace eworld
