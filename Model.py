import torch
import torch.nn as nn
import torchvision.models as models

def gem(x, p=3, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
class Model(nn.Module):
    def __init__(self,num_classes,model_path=None):
        super(Model, self).__init__()

        self.model = models.resnet50(pretrained=False)  
        self.model.avgpool = GeM()  
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  
        
        # Si tienes un modelo preentrenado, cargarlo
        if model_path:
            self.load_model(model_path)

    def forward(self,x):
        return self.model(x)
    
    
    def load_model(self, model_path):
       # Aquí carga los pesos del modelo desde el archivo
       checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
       if 'state_dict' in checkpoint:
           self.model.load_state_dict(checkpoint['state_dict'])
       else:
           self.model.load_state_dict(checkpoint)
       print("Modelo cargado correctamente.")

    