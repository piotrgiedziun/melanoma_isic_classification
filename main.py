import io
import warnings

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import transforms

warnings.simplefilter('ignore')


app = FastAPI()


class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(
                in_features=512, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(
                in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 250, 1)

    def forward(self, inputs):
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output


@app.post("/")
async def create_upload_file(file: UploadFile, age: int = 49, sex: int = 1, site: str = "site_nan"):
    device = torch.device('cpu')
    pred_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    request_object_content = await file.read()
    x = Image.open(io.BytesIO(request_object_content))
    x = Image.fromarray(np.array(x)[:, :, ::-1])
    x = pred_transform(x)
    x = x.unsqueeze(0)
    # 'male': 1, 'female': 0
    meta = np.array([[sex, float(age/90), *[1 if site.lower() == s else 0 for s in ['site_head/neck', 'site_lower extremity', 'site_oral/genital', 'site_palms/soles', 'site_torso', 'site_upper extremity', 'site_nan']]]], dtype=np.float32)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    meta = torch.tensor(meta, device=device, dtype=torch.float32)
    preds = []

    for fold in range(1, 6):
        model_path = f'model_{fold}.pth'
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            z_test = model((x, meta))
            z_test = torch.sigmoid(z_test)
            preds.append(z_test[0][0].numpy())

    return {"filename": file.filename, "pred": float(np.average(preds))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
