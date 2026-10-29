// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#pragma once
/// @file florence.hpp
/// HaloCore Florence-2 through the document door (libhalocore_door). One
/// door call over a batch of stills; the model loads once per call. The
/// module is opened at run time, so this binary links neither onnxruntime
/// nor PDFium.
///
/// Module:  $HALOCORE_DOOR_LIB, default /opt/mpl/lib/libhalocore_door.dylib
/// Model:   $HALOCORE_FLORENCE_DIR, default /opt/mpl/models/florence-2-base-onnx
#include <map>
#include <string>
#include <vector>

namespace eworld {

class HaloFlorence {
public:
    explicit HaloFlorence(const std::string& model_dir = "");
    ~HaloFlorence();
    /// Map each file stem to the door's caption text. Empty on failure.
    std::map<std::string, std::string> read_many(const std::vector<std::string>& image_paths);

private:
    void* lib_{nullptr};
    void* fn_{nullptr};
};

}  // namespace eworld
