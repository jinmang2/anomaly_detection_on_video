import os
import av
import ffmpeg
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from utils import process_feat
import torch
from torchvision import transforms

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import option
import gtransforms

args = option.parse_args()


class FeatureDataset(Dataset):
    def __init__(
        self,
        args,
        is_normal=True,
        transform=None,
        test_mode=False,
        is_preprocessed=False,
    ):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.is_preprocessed = args.preprocessed

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if args.datasetname == "UCF":
                if self.is_normal:
                    self.list = self.list[810:]  # ucf 810; sht63; xd 9525
                else:
                    self.list = self.list[:810]  # ucf 810; sht 63; 9525
            elif args.datasetname == "XD":
                if self.is_normal:
                    self.list = self.list[9525:]
                else:
                    self.list = self.list[:9525]

    def __getitem__(self, index):
        label = self.get_label(index)  # get video level label 0/1
        if args.datasetname == "UCF":
            features = np.load(self.list[index].strip("\n"), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split("/")[-1].strip("\n")[:-4]
        elif args.datasetname == "XD":
            features = np.load(self.list[index].strip("\n"), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split("/")[-1].strip("\n")[:-4]
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            if args.datasetname == "UCF":
                mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
                features = np.concatenate((features, mag), axis=2)
            elif args.datasetname == "XD":
                mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                features = np.concatenate((features, mag), axis=1)
            return features, name
        else:
            if args.datasetname == "UCF":
                if self.is_preprocessed:
                    return features, label
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, args.seg_length)  # ucf(32,2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate(
                    (divided_features, divided_mag), axis=2
                )
                return divided_features, label

            elif args.datasetname == "XD":
                feature = process_feat(features, 32)
                if args.add_mag_info == True:
                    feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                    feature = np.concatenate((feature, feature_mag), axis=1)
                return feature, label

    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):

        return len(self.list)

    def get_num_frames(self):
        return self.num_frame


class TenCropVideoFrameDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        output_folder_name: str = "outputs",  # for ffmpeg-python
        frames_per_clip: int = 16,
        resize: int = 256,
        cropsize: int = 224,
        to_frame: str = "pyav",
    ):
        to_frame = to_frame.lower()
        assert to_frame in ("pyav", "ffmpeg")

        self.images = []
        if to_frame == "pyav":
            self.container = av.open(video_path)
            self.container.seek(0)

            for frame in self.container.decode(video=0):
                self.images.append(frame.to_image())

            assert len(self.images) == self.container.streams.video[0].frames
        else:
            if not output_folder_name.endswith(os.sep):
                output_folder_name += os.sep

            output_path = os.path.join(os.getcwd(), output_folder_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # @TODO: Image path가 들어왔을 경우 처리
            ffmpeg.input(video_path).output(
                "{}%d.jpg".format(output_path),
                start_number=0,
            ).global_args("-loglevel", "quiet").run()

            from natsort import natsorted

            rgb_files = natsorted(list(os.listdir(output_path)))

            for file in rgb_files:
                self.images.append(Image.open(os.path.join(output_path, file)))

            assert len(self.images) == len(rgb_files)

        self.frames_per_clip = frames_per_clip
        self.indices = list(range(len(self.images) // frames_per_clip + 1))

        self.transform = transforms.Compose(
            [
                gtransforms.GroupResize(size=resize),
                gtransforms.GroupTenCrop(size=cropsize),
                gtransforms.ToTensorTenCrop(),
                gtransforms.GroupNormalizeTenCrop(),
                gtransforms.LoopPad(max_len=frames_per_clip),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx = idx * self.frames_per_clip
        end_idx = (idx + 1) * self.frames_per_clip
        image = self.images[start_idx:end_idx]
        tensor = self.transform(image)
        return tensor.permute(1, 0, 2, 3, 4)
