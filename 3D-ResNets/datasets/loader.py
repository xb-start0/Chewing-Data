import io

import h5py
from PIL import Image


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB').resize((112,112))

class EventLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('L').resize((112,112))


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path)).resize(112,112)


class VideoLoader(object):

    def __init__(self, image_name_formatter, event_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        self.event_name_formatter = event_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
            self.event_loader = EventLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, event_path, frame_indices):
        video = []
        event = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            event_image_path = event_path / self.event_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))
                event.append(self.event_loader(event_image_path))

        return video,event


class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:

            flow_data = []
            for flow in self.flows:
                flow_data.append(f[f'video_{flow}'])

            video = []
            for i in frame_indices:
                if i < len(flow_data[0]):
                    frame = [
                        Image.open(io.BytesIO(video_data[i]))
                        for video_data in flow_data
                    ]
                    frame.append(frame[-1])  # add dummy data into third channel
                    video.append(Image.merge('RGB', frame))

        return video