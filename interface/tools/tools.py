import os
import torch
import torchvision

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

    def predict(self, image) -> list:
        image_transformed = self.transform(image)
        batch_image = torch.unsqueeze(image_transformed, 0)

        # Inference
        output = self.model(batch_image)[0]
        boxes = output['boxes'].detach().numpy()
        labels = output['labels'].detach().numpy()
        scores = output['scores'].detach().numpy()
        return [self.label[label] for label in labels]