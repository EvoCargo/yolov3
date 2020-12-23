from pathlib import Path

import torch
from torch import Tensor, device as Device

File = Path


class DetectionFacade(torch.nn.Module):
    '''Exposes traced NN to final inference via TorchScript.

    Loads the traced trained model and perform `forward` method of it
    with post-processing.
    '''

    def __init__(
        self, yolo_path: File,
        device: Device,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        agnostic_nms: bool = True,
    ):
        '''
        Args:
            yolo_path: path to traced model
            device: cuda or cpu
        '''
        super().__init__()

        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms

        self.model = torch.jit.load(yolo_path.as_posix(), map_location=self.device)
        self.model.eval()

        self._grid = [torch.tensor(()), torch.tensor(()), torch.tensor(())]

    def forward(
        self,
        batch: Tensor,
    ) -> Tensor:
        '''
        Args:
            batch: 4D tensor (BxHxWxC) of stacked preprocessed images

        Returns:
            Bounding boxes and confidences of detections
        '''
        batch = batch.to(self.device)
        raw = self.model(batch)

        for i in range(len(raw)):
            ny, nx = raw[i].shape[2:4]
            if self._grid[i].shape[2:4] != raw[i].shape[2:4]:
                self._grid[i] = self._make_grid(nx, ny)

            # y = raw[i].sigmoid()
            # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self._grid[i].to(raw[i].device)) * self.stride[i]  # xy
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            # z.append(y.view(bs, -1, self.no))

        return torch.tensor(()).cpu()

    def _make_grid(self, nx: int = 20, ny: int = 20):
        yv, xv = torch.meshgrid((torch.arange(ny, device=self.device), torch.arange(nx, device=self.device)))
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @classmethod
    def script(cls, traced_path: File, scripted_path: File, device: Device):
        '''Saves scripted version of Facade to scripted_path

        Args:
            traced_path: tha path to traced trained model
            scripted_path: the path for scripted model
            device: the device
        '''
        facade = cls(traced_path, device)
        scripted = torch.jit.script(facade)
        scripted.save(scripted_path.as_posix())

    @classmethod
    def load(cls, scripted_path: File, device: Device):
        '''Loads previously scripted Facade.
        '''
        return torch.jit.load(scripted_path.as_posix(), device)
