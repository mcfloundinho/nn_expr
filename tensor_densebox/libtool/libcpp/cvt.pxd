#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: 2015年10月21日 星期三 11时57分14秒
# Mail: hewr2010@gmail.com

from .string cimport string
from libc.string cimport memcpy

cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef cppclass CVMat "cv::Mat":
        CVMat() except+
        void create(int, int, int)
        void* data
        int type() const

cdef extern from "opencv2/opencv.hpp":        
    cdef int CV_8UC3
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_64FC1

cdef extern from "opencv2/highgui/highgui.hpp" namespace "cv":
    cdef CVMat imread(const string &filename, int flags)

