import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch3d.renderer import (
    look_at_rotation,
    FoVPerspectiveCameras,
    RasterizationSettings,
    Materials,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    PointLights
)
from paramvectordef import ParamVectorDef


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def conv_layer_set(self, in_c, out_c, k, p, s):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_c),
        )
        return conv_layer

    def fc_layer_set(self, in_c, out_c):
        fc_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(),
        )
        return fc_layer

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CameraModule(BaseModel):
    def __init__(self, num_cameras):
        super(CameraModule, self).__init__()
        self.num_cameras = num_cameras
        self.camera_latent = nn.Parameter(torch.randn((num_cameras, 128)), requires_grad=True)
        self.camera_mlp = nn.Sequential(
            self.fc_layer_set(128, 128),
            self.fc_layer_set(128, 64),
            self.fc_layer_set(64, 16),
            nn.Linear(16, 3),
        )

    def forward(self):
        cameras = self.camera_mlp(self.camera_latent)
        cameras = F.normalize(cameras, p=2, dim=1)
        return cameras


class FeatureExtractor(BaseModel):
    def __init__(self, in_c):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        for param in vgg.parameters():
            param.requires_grad = False
        self.out_c = vgg.classifier[0].in_features
        self.features = nn.Sequential(
            self.conv_layer_set(in_c, 3, 3, 1, 1),
            vgg.features,
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = torch.flatten(x, 1)
        return x


class TaskModule(BaseModel):
    def __init__(self, task_type, in_c, out_c):
        super(TaskModule, self).__init__()
        self.task_type = task_type
        module = list()
        module.append(self.fc_layer_set(in_c, 256))
        module.append(self.fc_layer_set(256, 64))
        module.append(self.fc_layer_set(64, 16))
        module.append(nn.Linear(16, out_c))
        if task_type == 'scalar':
            module.append(nn.Sigmoid())
        module = nn.Sequential(*module)
        if task_type == 'scalar':
            self.scalar_module = module
        elif task_type == 'type':
            self.type_module = module
        elif task_type == 'binary':
            self.binary_module = module

    def forward(self, x):
        if self.task_type == 'scalar':
            x = self.scalar_module(x)
        elif self.task_type == 'type':
            x = self.type_module(x)
        elif self.task_type == 'binary':
            x = self.binary_module(x)
        return x


class MeshModel(BaseModel):
    def __init__(self, version):
        super(MeshModel, self).__init__()
        self.param_def = ParamVectorDef(version)
        self.num_cameras = 3
        self.cameras = CameraModule(self.num_cameras)
        self.paramtypes = [param.paramtype for param in self.param_def.params]
        self.feature_extractor = FeatureExtractor(self.num_cameras)
        in_c = self.feature_extractor.out_c
        self.task_modules = nn.ModuleList()
        for param in self.param_def.params:
            if param.paramtype == 'scalar':
                self.task_modules.append(TaskModule('scalar', in_c, len(param.data)))
            elif param.paramtype == 'type':
                self.task_modules.append(TaskModule('type', in_c, len(param.data)))
            elif param.paramtype == 'binary':
                self.task_modules.append(TaskModule('binary', in_c, 1))

    def render_batch(self, meshes, image_size=128):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        dist = 1.2
        batch_size = len(meshes)
        camera_location = (self.cameras() * dist).repeat(batch_size, 1)
        R = look_at_rotation(camera_location, device=device)
        T = -torch.bmm(R.transpose(1, 2), camera_location[:, :, None])[:, :, 0]
        camera = FoVPerspectiveCameras(device=device, znear=-10000, zfar=10000)
        lights = PointLights(device=device, location=camera_location)
        material = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=True)
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
                                shader=HardFlatShader(device=device, cameras=camera, lights=lights)
                                )
        images = renderer(meshes_world=meshes.extend(self.num_cameras), materials=material, R=R, T=T)
        images = images[:, :, :, 0:1]
        images = images.permute(0, 3, 1, 2)
        images = images.reshape([batch_size, self.num_cameras, image_size, image_size])
        return images

    def forward(self, meshes, predict=False):
        image_batch = self.render_batch(meshes, 256)
        predictions = []
        features = self.feature_extractor(image_batch)
        for module in self.task_modules:
            predictions.append(module(features))
        if predict:
            for i, param in enumerate(self.param_def.params):
                if param.paramtype == 'binary':
                    predictions[i] = torch.sigmoid(predictions[i]).detach().cpu()
                    predictions[i] = torch.where(predictions[i] > 0.5, 1.0, 0.0)
                elif param.paramtype == 'type':
                    predictions[i] = torch.softmax(predictions[i], dim=1).detach().cpu()
                    predictions[i] = torch.FloatTensor(predictions[i].shape).zero_().scatter_(1, torch.argmax(predictions[i], dim=1, keepdim=True), 1)
                else:
                    predictions[i] = predictions[i].detach().cpu()
        return predictions


class MultiTaskLoss(nn.Module):
    def __init__(self, version):
        super(MultiTaskLoss, self).__init__()
        self.param_def = ParamVectorDef(version)
        self.loss_fn = []
        for param in self.param_def.params:
            if param.paramtype == 'scalar':
                self.loss_fn.append(nn.L1Loss(reduction='sum'))
            elif param.paramtype == 'type':
                self.loss_fn.append(nn.CrossEntropyLoss(reduction='sum'))
            elif param.paramtype == 'binary':
                self.loss_fn.append(nn.BCEWithLogitsLoss(reduction='sum'))

    def forward(self, predictions, targets):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        losses = torch.zeros(len(self.loss_fn), device=device)
        for i, loss in enumerate(self.loss_fn):
            losses[i] = loss(predictions[i], targets[i])
        total_loss = torch.sum(losses)
        return losses, total_loss
