import json
from pathlib import Path
from yeaudio.audio import AudioSegment
import torch
import torch.utils.data as data
import numpy as np
from .loader import VideoLoader
from macls.data_utils.featurizer import AudioFeaturizer
from datasets.transform import normalize
def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


# def get_database(data, subset, root_path, video_path_formatter):
#     video_ids = []
#     video_paths = []
#     annotations = []

#     for key, value in data['database'].items():
#         this_subset = value['subset']
#         if this_subset == subset:
#             video_ids.append(key)
#             annotations.append(value['annotations'])
#             if 'video_path' in value:
#                 video_paths.append(Path(value['video_path']))
#             else:
#                 label = value['annotations']['label']
#                 video_paths.append(video_path_formatter(root_path, label, key))

#     return video_ids, video_paths, annotations

def get_chew_database(data, root_path, event_root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    event_paths = []
    annotations = []

    for key, value in data['database'].items():
        video_ids.append(key)
        annotations.append(value['annotations'])
        if 'video_path' in value:
            video_paths.append(Path(value['video_path']))
        else:
            video_paths.append(video_path_formatter(root_path, key))
            event_paths.append(video_path_formatter(event_root_path, key))

    return video_ids, video_paths, event_paths, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 root_event_path,
                 audio_configs,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        # self.data, self.class_names = self.__make_dataset(
        #     root_path, annotation_path, subset, video_path_formatter)
        self.data = self.__make_chew_dataset(
            root_path, root_event_path, annotation_path, video_path_formatter)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        # self.audio_configs = dict_to_object(audio_configs)
        self.audio_featurizer = AudioFeaturizer(feature_method="Fbank",
                                        use_hf_model=False,
                                        method_args={"sample_frequency": 16000,"num_mel_bins": 80})
        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        
    def __make_chew_dataset(self, root_path, root_event_path, annotation_path,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, event_img_paths, annotations = get_chew_database(
            data, root_path, root_event_path, video_path_formatter)

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            video_path = video_paths[i]
            event_img_path = event_img_paths[i]
            audio_id = video_ids[i]
            audio_path = "/media/Storage2/xb/Chew/Chew_data_all/video/"+audio_id+"/"+audio_id+"_bread_front.mp4"
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                "event": event_img_path,
                "audio": audio_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': annotations[i]['label']
            }
            dataset.append(sample)
        return dataset
    # def __make_dataset(self, root_path, annotation_path, subset,
    #                    video_path_formatter):
    #     with annotation_path.open('r') as f:
    #         data = json.load(f)
    #     video_ids, video_paths, annotations = get_database(
    #         data, subset, root_path, video_path_formatter)
    #     class_to_idx = get_class_labels(data)
    #     idx_to_class = {}
    #     for name, label in class_to_idx.items():
    #         idx_to_class[label] = name

    #     n_videos = len(video_ids)
    #     dataset = []
    #     for i in range(n_videos):
    #         if i % (n_videos // 5) == 0:
    #             print('dataset loading [{}/{}]'.format(i, len(video_ids)))

    #         if 'label' in annotations[i]:
    #             label = annotations[i]['label']
    #             label_id = class_to_idx[label]
    #         else:
    #             label = 'test'
    #             label_id = -1

    #         video_path = video_paths[i]
    #         if not video_path.exists():
    #             continue

    #         segment = annotations[i]['segment']
    #         if segment[1] == 1:
    #             continue

    #         frame_indices = list(range(segment[0], segment[1]))
    #         sample = {
    #             'video': video_path,
    #             'segment': segment,
    #             'frame_indices': frame_indices,
    #             'video_id': video_ids[i],
    #             'label': label_id
    #         }
    #         dataset.append(sample)
    #     return dataset, idx_to_class

    def __loading(self, path, event_path, frame_indices):
        clip , event_clip= self.loader(path, event_path, frame_indices)
        frames = np.zeros((64, 112, 112, 4), np.dtype('uint8'))
        for i, img in enumerate(clip):
            frames[i,:,:,:3] = np.array(img)
            frames[i, :, :, 3] = np.array(event_clip[i])
        clip_fusion = self.spatial_transform(frames) 
        # if self.spatial_transform is not None:
        #     self.spatial_transform.randomize_parameters()
        #     clip = [self.spatial_transform(img) for img in clip]
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip_fusion

    def __getitem__(self, index):
        path = self.data[index]['video']
        event_path = self.data[index]['event']
        audio_path = self.data[index]['audio']
        audio_segment = AudioSegment.from_file(audio_path )
        samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
        feature = self.audio_featurizer(samples)
        resize = torch.nn.AdaptiveAvgPool2d((112, 112))
        audio_item = resize(feature).transpose(1, 2)
        # audio_item = normalize(audio_item, 0.5, 0.5, inplace=False)
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, event_path, frame_indices)
        audio_item = audio_item.repeat(clip.shape[0],1,1,1)
        if self.target_transform is not None:
            target = self.target_transform(target)
        clip = torch.cat((clip, audio_item), dim=1)
        return clip, target

    def __len__(self):
        return len(self.data)