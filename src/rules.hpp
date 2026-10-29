// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#pragma once
/// @file rules.hpp
/// The label rules of the aerial encode: OCR tokens, the scene tag, and the
/// stencil words in a Florence read. No model runs here. Text handling is
/// ASCII: a non-ASCII letter is not alphanumeric for ocr_token / figure_words.
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace eworld {

constexpr int    kMaxOcrTags = 8;
constexpr double kOcrMinConf = 40.0;   // Tesseract word confidence, 0-100. Stencils often land in the 40s.

/// One object in one frame, in encoder-frame pixels. `object_id` is the
/// object it belongs to; `cls` and `ocr` are that object's class and text
/// from the index, `detected` what the head said in this frame.
struct Box {
    std::string              cls;
    std::string              detected;
    double                   confidence{0.0};
    double                   xyxy[4]{0, 0, 0, 0};
    std::vector<std::string> ocr;
    std::vector<std::string> scout;    // text boxes: what the Tesseract scout spelled
    std::string              source;   // coco | world | tesseract
    long                     object_id{-1};
};

/// "ocr:B127" from a raw read, or nullopt outside 2-16 alphanumerics.
std::optional<std::string> ocr_token(const std::string& text);

double area(const double xyxy[4]);
double iou(const double a[4], const double b[4]);

/// One rule tag from the class set. Empty when the frame has no classes.
std::string scene_tag(const std::set<std::string>& class_names);

/// A plate token: not the prompt echo, and not a one- or two-letter speck.
bool keep_stencil(const std::string& token);

/// Stencil tokens from the door's "Text in the figure:" line (at most 8).
std::vector<std::string> figure_words(const std::string& text);

/// One read of an object: the words one reader saw in one view.
struct Read {
    std::vector<std::string> words;
    bool                     florence{false};   // false: the Tesseract scout
};

/// The object's text from all its reads. Spellings within edit distance
/// half their length are one word (B127 / B27 / 8127); a word is kept when
/// at least two reads support it, unless only one read saw any text (then
/// nothing can corroborate it). Each kept word is its most frequent Florence
/// spelling. Most supported first, at most kMaxOcrTags.
std::vector<std::string> corroborate(const std::vector<Read>& reads);

}  // namespace eworld
