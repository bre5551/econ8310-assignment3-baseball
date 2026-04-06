from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BaseballVideoDataset(Dataset):
    def __init__(self, data_dir="data", img_size=128, frame_step=4):
        self.data_dir = Path(data_dir)
        self.video_dir = self.data_dir / "videos"
        self.ann_dir = self.data_dir / "annotations"

        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video folder not found: {self.video_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotation folder not found: {self.ann_dir}")

        self.img_size = img_size
        self.frame_step = frame_step

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.samples = []
        self.classes = ["no_baseball", "baseball"]
        self.class_to_idx = {"no_baseball": 0, "baseball": 1}

        xml_files = sorted(self.ann_dir.glob("*.xml"))
        if not xml_files:
            raise RuntimeError("No XML annotation files found.")

        for xml_path in xml_files:
            video_name, positive_frames = self._parse_xml(xml_path)
            video_path = self.video_dir / video_name

            if not video_path.exists():
                print(f"Warning: missing video for {xml_path.name}: {video_path.name}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                cap.release()
                print(f"Warning: could not read frames from {video_path.name}")
                continue

            for frame_idx in range(0, total_frames, self.frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = torch.tensor(frame.tolist(), dtype=torch.uint8).permute(2, 0, 1)
                x = self.transform(x)

                label = 1 if frame_idx in positive_frames else 0
                y = torch.tensor(label, dtype=torch.long)

                self.samples.append((x, y))

            cap.release()

        if not self.samples:
            raise RuntimeError("No valid video/frame samples found.")

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        source_tag = root.find("./meta/task/source")
        if source_tag is None or not source_tag.text:
            raise ValueError(f"No source video found in {xml_path}")

        video_name = source_tag.text.strip()
        positive_frames = set()

        for track in root.findall("track"):
            label = track.attrib.get("label", "")
            if label != "baseball":
                continue

            for box in track.findall("box"):
                outside = int(box.attrib.get("outside", "0"))
                frame = int(box.attrib["frame"])
                if outside == 0:
                    positive_frames.add(frame)

        return video_name, positive_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data"
    ds = BaseballVideoDataset(data_dir=root, img_size=128, frame_step=8)

    print("Total samples:", len(ds))
    print("Classes:", ds.classes)

    x, y = ds[0]
    print("Sample shape:", x.shape)
    print("Sample dtype:", x.dtype)
    print("Sample label:", y.item())