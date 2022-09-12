import torch
import torchvision
import cv2
import numpy as np

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

    def draw_bounding_box(self, image: np.ndarray, boxes: list, labels: list, scores: list) -> np.ndarray:
        image_copy = image.copy()
        for i, box in enumerate(boxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            color = self.colors[labels[i]]
            category = self.label[labels[i]]
            percentage = scores[i] * 100
            text = f"{category} ({percentage:.1f}%)"
            cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image_copy, text, (xmin, ymax - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image_copy

    def predict(self, image: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
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

        # Draw bounding box
        image_labelled = self.draw_bounding_box(image, boxes, labels, scores)
        return image_labelled