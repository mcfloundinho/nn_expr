from libcpp.vector cimport vector

cdef extern from "nms.h" namespace "nms":
    ctypedef struct Rect:
        int x, y, w, h
    vector[int] iou_nms(const vector[Rect] &, const double)
    vector[Rect] iou_nms_avgbox(const vector[Rect] &, const double)

cdef vector[int] iou_nms_wrapper(boxes, threshold):
    cdef vector[Rect] c_boxes
    cdef Rect rect
    for x, y, w, h in boxes:
        rect.x, rect.y, rect.w, rect.h = x, y, w, h
        c_boxes.push_back(rect)
    ret = iou_nms(c_boxes, <double> threshold)
    return ret

cdef vector[Rect] iou_nms_avgbox_wrapper(boxes, threshold):
    cdef vector[Rect] c_boxes
    cdef Rect rect
    for x, y, w, h in boxes:
        rect.x, rect.y, rect.w, rect.h = x, y, w, h
        c_boxes.push_back(rect)
    ret = iou_nms_avgbox(c_boxes, <double> threshold)
    return ret

def perform_iou_nms(boxes, threshold=0.5):
    c_ret = iou_nms_wrapper(boxes, threshold)
    ret = []
    for i from <unsigned int> 0 <= i < c_ret.size():
        ret.append(c_ret[i])
    return ret

def perform_iou_nms_avgbox(boxes, threshold=0.5):
    c_ret = iou_nms_avgbox_wrapper(boxes, threshold)
    ret = []
    for i from <unsigned int> 0 <= i < c_ret.size():
        box = c_ret[i]
        ret.append((box.x, box.y, box.w, box.h))
    return ret
