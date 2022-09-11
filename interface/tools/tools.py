import torch
import torchvision
from typing import Tuple

class MyPredictor():
    def __init__(
        self, 
        colors: dict = {
            0: (67, 214, 30),
            1: (191, 191, 11),
            2: (191, 134, 11),
            3: (179, 11, 191),
            4: (11, 107, 191)
        }) -> None:
        self.model = ""
        self.label = {
            0: "Coverall",
            1: "Face Shield",
            2: "Gloves",
            3: "Goggles",
            4: "Mask"
        }
        self.colors = colors
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.load_models()

    def load_models(self, path: str = "./models/CPPEModel.pt", device: str = "cpu") -> None:
        model = torch.load(f=path, map_location=torch.device(device))
        self.model = model
        self.model.eval()

    def predict(self, image, iou_threshold: float = 0.3) -> Tuple:
        image_transformed = self.transform(image)
        batch_image = torch.unsqueeze(image_transformed, 0)

        # Inference
        output = self.model(batch_image)[0]
        boxes = output['boxes']
        labels = output['labels'].detach().numpy()
        scores = output['scores']

        # NMS
        NMS = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
        boxes_np = boxes.detach().numpy()
        scores_np = scores.detach().numpy()
        boxes = [boxes_np[i] for i in range(len(boxes_np)) if i in NMS]
        scores = [scores_np[i] for i in range(len(scores_np)) if i in NMS]
        labels = [labels[i] for i in range(len(labels)) if i in NMS]
        return boxes, labels, scores