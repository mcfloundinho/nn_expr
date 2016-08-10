#include "nms.h"
#include <iostream>

namespace nms {

double iou(const Rect &l, const Rect &r) {
    int x0 = std::max(l.x, r.x), y0 = std::max(l.y, r.y);
    int x1 = std::min(l.x + l.w, r.x + r.w), y1 = std::min(l.y + l.h, r.y + r.h);
    int l_area = l.w * l.h, r_area = r.w * r.h;
    int union_area = std::max(0, x1 > x0 ? (x1 - x0) * (y1 - y0) : 0);
    int tot_area = l_area + r_area - union_area;
    double iou = tot_area > 0 ? double(union_area) / tot_area : 0;
    return iou;
}

std::vector<int> iou_nms(const std::vector<Rect> &boxes, const double THRESHOLD) {
    std::vector<int> ret;
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto &now = boxes[i];
        bool legal = true;
        for (auto j: ret) {
            auto &t_box = boxes[j];
            auto r = iou(now, t_box);
            if (r >= THRESHOLD) {
                legal = false;
                break;
            }
        }
        if (legal) ret.emplace_back(i);
    }
    return ret;
}

std::vector<Rect> iou_nms_avgbox(const std::vector<Rect> &boxes, const double THRESHOLD) {
    std::vector<Rect> ret;
    std::vector<int> sizes;
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto &now = boxes[i];
        bool legal = true;
        for (size_t j = 0; j < sizes.size(); ++j) {
            auto t_box = ret[j];
            auto s = sizes[j];
            t_box.x /= s, t_box.y /= s, t_box.w /= s, t_box.h /= s;
            auto r = iou(now, t_box);
            if (r >= THRESHOLD) {
                legal = false;
                ret[j].x += now.x, ret[j].y += now.y, ret[j].w += now.w, ret[j].h += now.h;
                sizes[j] += 1;
                break;
            }
        }
        if (legal) {
            ret.emplace_back(now);
            sizes.emplace_back(1);
        }
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto &box = ret[i];
        auto s = sizes[i];
        box.x /= s, box.y /= s, box.w /= s, box.h /= s;
    }
    return ret;
}

}
