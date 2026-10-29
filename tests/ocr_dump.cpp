// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
/// ocr_dump <psm> <image>... -- the words eworld::Tesseract keeps, one line
/// per word: <image>\t<text>\t<conf>\t<x0,y0,x1,y1>. Compare with
/// `tesseract <image> stdout --psm N -l eng tsv` filtered the same way.
#include "ocr.hpp"

#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ocr_dump <psm> <image>...\n";
        return 2;
    }
    eworld::Tesseract tess;
    const int psm = std::stoi(argv[1]);
    for (int i = 2; i < argc; ++i)
        for (const auto& w : tess.words(cv::imread(argv[i], cv::IMREAD_COLOR), psm))
            std::cout << argv[i] << "\t" << w.text << "\t" << w.conf << "\t" << w.xyxy[0] << "," << w.xyxy[1] << ","
                      << w.xyxy[2] << "," << w.xyxy[3] << "\n";
    return 0;
}
