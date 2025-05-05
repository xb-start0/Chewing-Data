from pstats import count_calls

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import argparse
import os
import cv2
from PyQt5.QtGui import QImage, QPixmap

class FrameGenerationTest():


    def on_cd_frame_cb(self, ts, cd_frame):
        img_cv = cv2.cvtColor(cd_frame, cv2.COLOR_BGR2RGB)
        h, w, c = img_cv.shape  # 获取图片的高度、宽度和通道数
        QImg = QImage(img_cv, w, h, QImage.Format_RGB888)  # 将numpy数组转换成QImage格式
        QImg.save(self.savePath + "/img{}.png".format(self.count), 'png')
        print('正在保存第{}张图片'.format(self.count))
        self.count += 1  # 计数加1

    def __init__(self):
        # self.event_file_path = r'C:\Users\YZ\Desktop\recording_202407090863.raw'
        self.event_file_path = r'D:\Desktop\make_dataset\data\mengqingguo0001\recording_2024-08-11_23-06-14.raw'
        self.acc_time = 100 * 1000 # int(1000 * 1000) # 微秒 # 这么多时间的事件积累产生一个帧
        # self.fps = 1 # 一秒时间内保存这么多帧
        self.count = 0
        self.savePath = ''
        filename =  self.event_file_path.split('\\')[-2]
        self.savePath = f'D:/Desktop/make_dataset/event2frame/{filename}_acc{self.acc_time}μm_fps{self.fps}'
        self.disTime = 0 # 10 * 60 * 1000  # 结束时间戳

        self.start_ts = 0 + self.disTime
        if not os.path.exists(self.savePath):
            # 如果路径不存在，创建它
            os.makedirs(self.savePath)

        self.mv_iterator = EventsIterator(input_path=self.event_file_path, delta_t=self.acc_time, start_ts=self.start_ts)
        self.height, self.width = self.mv_iterator.get_size()  # Camera Geometry

        # Helper iterator to emulate realtime
        if not is_live_camera(self.event_file_path):
            mv_iterator = LiveReplayEventsIterator(self.mv_iterator)

        # Event Frame Generator
        self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                                sensor_width=self.width, 
                                sensor_height=self.height, 
                                # fps=self.fps,
                                accumulation_time_us = self.acc_time,
                                palette=ColorPalette.Dark)
        self.event_frame_gen.set_output_callback(self.on_cd_frame_cb)


if __name__ == '__main__':
    frameGen = FrameGenerationTest()

    # Process events
    for evs in frameGen.mv_iterator:
        # evs包含了self.acc_time内积累的所有事件
        # Dispatch system events to the window
        EventLoop.poll_and_dispatch()
        frameGen.event_frame_gen.process_events(evs)

