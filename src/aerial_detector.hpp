// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#pragma once
/// @file aerial_detector.hpp
/// Objects first, then text, as one ehc::media::FrameDetector.
///
///   1. Object layer: YOLOv8 (COCO, filtered to classes that occur in aerial
///      footage) and YOLO-World (prompts for what COCO has no name for),
///      merged class-agnostically: overlapping boxes are one object, the
///      more confident label wins. Scene text is a class too: Tesseract
///      locates text lines ("text" boxes); it does not read them.
///   2. Dedup: an object in the next kept frame is the same object when its
///      box overlaps (IoU), whatever the heads called it, across short gaps.
///      Each object gets an object_id and one class: the confidence-weighted
///      vote over its frames.
///   3. OCR: each object is read on a few crops spread over its frames
///      (Florence-2 through the HaloCore door; Tesseract when Florence is
///      off) and the words merged. Text written on another object (AMMO on
///      a truck) is that object's text; a text object read as empty is
///      dropped.
///   4. Index: objects() is the lookup table (object_id -> class, text,
///      frames). Each frame's boxes and ocr: tags come from it.
///
/// detect() runs once per video on all survivors, so the whole video is
/// deduplicated before any OCR runs. records()[i] lines up with survivor i.
#include "florence.hpp"
#include "ocr.hpp"
#include "rules.hpp"
#include "yolo.hpp"

#include "ehc/media/detector.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace eworld {

struct FrameRecord {
    std::vector<Box> boxes;
    std::string      scene;
};

/// One deduplicated object: the row of objects.jsonl.
struct ObjectEntry {
    long                     object_id{-1};
    std::string              cls;
    std::string              source;
    double                   confidence{0.0};   // best over its frames
    std::vector<std::string> ocr;
    std::vector<int>         survivors;         // survivor ordinals it was seen in
    std::vector<std::array<double, 4>> boxes;   // its box in each of those survivors
    long                     host_id{-1};       // text only: the object it is written on
    std::vector<Read>        reads;             // every read of it; ocr = corroborate(reads)
    std::map<std::string, double> votes;        // detected class -> summed confidence
    std::map<std::string, std::string> sources; // detected class -> head
    int                      best_survivor{-1}; // the frame its crop was read from
    double                   best_xyxy[4]{0, 0, 0, 0};
};

/// config/object_layer.json, with paths resolved against `root`.
struct ObjectLayerConfig {
    std::string coco_model;
    float       coco_conf{0.25f};
    std::set<std::string> coco_keep;
    std::string world_model;                       // "" = no world head
    float       world_conf{0.20f};
    std::map<std::string, std::string> world_tokens;   // prompt -> index token
    double merge_iou{0.5};
    double track_iou{0.3};
    int    track_max_gap{2};
    double track_min_appearance{0.5};
    std::map<std::string, int> track_group;   // label -> group; absent = its own group
    bool same_group(const std::string& a, const std::string& b) const;
    int    ocr_min_side{48};
    double ocr_pad{0.1};
    int    ocr_views{3};
    std::set<std::string> ocr_skip;
    bool   text_enabled{true};
    int    text_psm{11};
    double text_min_conf{40.0};
    int    text_min_height{10};
    double text_pad{0.3};
    double text_host_containment{0.6};

    static ObjectLayerConfig load(const std::string& path, const std::string& root);
    nlohmann::ordered_json to_json() const;
};

struct DetectorOptions {
    ObjectLayerConfig layer;
    int         imgsz{640};
    std::string ocr{"florence"};   // florence | tesseract | none
};

class AerialDetector final : public ehc::media::FrameDetector {
public:
    explicit AerialDetector(DetectorOptions opt);
    std::vector<ehc::media::FrameDetections> detect(const std::vector<std::string>& image_paths) override;
    std::string name() const override { return "aerial_objects_ocr"; }
    const std::vector<FrameRecord>& records() const { return records_; }
    const std::vector<ObjectEntry>& objects() const { return objects_; }

private:
    std::vector<Box> object_layer(const cv::Mat& bgr);
    std::vector<Box> text_boxes(const cv::Mat& bgr);
    void read_objects(const std::vector<cv::Mat>& images);
    void attach_text(std::vector<std::vector<Box>>& frames);

    DetectorOptions                opt_;
    std::unique_ptr<YoloHead>      coco_, world_;
    std::unique_ptr<HaloFlorence>  florence_;
    std::unique_ptr<Tesseract>     tess_;    // OCR when --ocr tesseract
    std::unique_ptr<Tesseract>     scout_;   // locates scene text for the object layer
    std::vector<FrameRecord>       records_;
    std::vector<ObjectEntry>       objects_;
    long                           next_object_id_{0};   // unique across videos
};

}  // namespace eworld
