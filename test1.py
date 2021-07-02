#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @License  : (C) Copyright 2019-2020,****
"""
@Modify Time: 2021/6/29 20:02
@Author     : erqiang Xu@fiberhome.com
@version    : 1.0
@Desc       :
-------------------------------------------------------------------------------------
"""
import cv2
import skimage
import numpy as np
from obj_info import ObjInfo
from motion_object_detection import MotionObjectDetection

if __name__ == '__main__':
    rtsp = "../data_video/01_3.avi"
    stream = cv2.VideoCapture(rtsp)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_w=int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h=int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter("01_3.mp4", fourcc, 25, (frame_w, frame_h*2))
    fgbg=cv2.createBackgroundSubtractorMOG2(1, 40, False)
    # fgbg=cv2.createBackgroundSubtractorMOG2(varThreshold=16,detectShadows=False)
    # fgbg.set
    frame_count=0
    while 1:
        (ret, frame) = stream.read()
        if ret:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            h,w,c = frame.shape
            frame_count+=1
            print(frame_count)
            # print(frame_size)
            # mask = [[[0, 0], [w,0],[w,h],[0,h],[0,0]]]
            # motion_detect = MotionObjectDetection((h, w), mask)
            # motion_boxes, foreground_mask_term = motion_detect.motion_detect(frame)
            fgmask = fgbg.apply(frame,0.01)
            kernel_type= cv2.MORPH_ELLIPSE
            cv2.imshow('fgmask', fgmask)
            kernel_size=15
            u_image = fgmask.astype(np.uint8)
            kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
            # u_image = cv2.morphologyEx(u_image, cv2.MORPH_ERODE, kernel)
            u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
            u_image = cv2.morphologyEx(u_image, cv2.MORPH_ERODE, kernel)
            u_image = cv2.morphologyEx(u_image, cv2.MORPH_ERODE, kernel)
            # u_image = cv2.morphologyEx(u_image, cv2.MORPH_CLOSE, kernel)
            # u_image = cv2.morphologyEx(u_image, cv2.MORPH_ERODE, kernel)
            # u_image = cv2.morphologyEx(u_image, cv2.MORPH_OPEN, kernel)
            u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
            # u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
            # u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)

            found_conn_objs = []
            # img = u_image.astype(np.uint8)
            img_nccomps = cv2.connectedComponentsWithStats(u_image)
            found_objs = img_nccomps[2]
            for i in range(img_nccomps[0]):
                obj = found_objs[i]
                area = obj[4]
                if 1000 > area > 500:
                    found_conn_objs.append(ObjInfo(obj).get_box()[:4])

            for box in found_conn_objs:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            mask = cv2.cvtColor(u_image, cv2.COLOR_GRAY2BGR)
            ROI = cv2.bitwise_and(mask, frame)

            # chull = skimage.morphology.convex_hull_object(ROI)
            ret, thresh = cv2.threshold(u_image, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area=cv2.contourArea(cnt)
                print("area :",area)
                if area>10000:
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

            cv2.imshow('frame', frame)
            cv2.imshow('ROI', ROI)
            # cv2.imshow('imag', imag)
            cv2.imshow('long', u_image)
            cv2.waitKey(100)
            show_img=np.zeros(shape=(h*2,w,3),dtype=np.uint8)
            show_img[:h,:,:]=frame
            show_img[h:,:,0]=fgmask
            show_img[h:,:,1]=fgmask
            show_img[h:,:,2]=fgmask
            video_writer.write(show_img)
            if frame_count>=500:
                video_writer.release()
        else:
            video_writer.release()
            stream.release()
