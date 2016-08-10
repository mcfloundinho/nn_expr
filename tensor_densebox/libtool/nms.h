#pragma once

#include <vector>

namespace nms {
    struct Rect {
        int x, y, w, h;
    };

    /*!
     * merge boxes with iou >= @THRESHOLD as the first box
     * @param THRESHOLD: iou threshold
     * @return indexes
     */
    std::vector<int> iou_nms(const std::vector<Rect> &, const double THRESHOLD=0.5);

    /*!
     * merge boxes with iou >= @THRESHOLD as the average box
     */
    std::vector<Rect> iou_nms_avgbox(const std::vector<Rect> &, const double THRESHOLD=0.5);
}
