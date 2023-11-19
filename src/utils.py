
import torch
from torch import nn
import numpy as np
import random
from typing import Tuple
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms_if_available()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class L2ConstraintedNet(nn.Module):
    def __init__(self, org_model, alpha=16, num_classes=2):
        super().__init__()
        self.org_model = org_model
        self.alpha = alpha

    def forward(self, x):
        x = self.org_model(x)
        # モデルの出力をL2ノルムで割り、定数alpha倍する
        l2 = torch.sqrt((x**2).sum()) # 基本的にこの行を追加しただけ
        x = self.alpha * (x / l2)     # 基本的にこの行を追加しただけ
        return x

def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam

def criterion(logits, targets, activated=False):
    device = torch.device('cuda')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    if activated:
        losses = nn.BCELoss(reduction='none')(logits.view(-1), targets.view(-1))
    else:
        losses = bce(logits.view(-1), targets.view(-1))
    losses[targets.view(-1) > 0] *= 2.
    norm = torch.ones(logits.view(-1).shape[0]).to(device)
    norm[targets.view(-1) > 0] *= 2
    return losses.sum() / norm.sum()

def tta(tensor):
    # 入力テンソルの形状: torch.Size([8, 32, 5, 384, 384])
    
    # 結果を格納するためのテンソルを確保
    result_shape = (tensor.size(0), 4, *tensor.size()[1:])
    augmented = torch.empty(result_shape, device=tensor.device, dtype=tensor.dtype)

    # 元のデータ
    augmented[:, 0] = tensor

    # Horizontal Flip
    augmented[:, 1] = torch.flip(tensor, [-1])

    # Vertical Flip
    augmented[:, 2] = torch.flip(tensor, [-2])

    # Horizontal + Vertical Flip
    augmented[:, 3] = torch.flip(augmented[:, 1], [-2])

    return augmented




class OcclusionSensitivityMap:
    def __init__(
        self,
        model: nn.Module,
        mask_size: Tuple[int, int, int] = (16, 16, 16),
        stride: Tuple[int, int, int] = (1, 1, 1),
        num_classes: int = 9
    ):
        self.model = model
        self.mask_size = mask_size  # (depth, height, width)
        self.stride = stride  # (stride_depth, stride_height, stride_width)
        self.num_classes = num_classes
    
    def generate_sensitivity_map(
        self,
        input_tensor: torch.Tensor,
        batch_size: int,
        occlusion_value: float = 0.0
    ) -> torch.Tensor:
        # Assuming input_tensor is of shape (1, d, c, w, h)
        _, D, C, W, H = input_tensor.shape
        # Initialize sensitivity map with zeros
        sensitivity_map = torch.zeros((D, W, H))

        # Get the model's output without occlusion for comparison
        with torch.no_grad():
            baseline_output = self.model(input_tensor).detach()
            baseline_output = baseline_output.view(32,self.num_classes).mean(0)
            baseline_output = baseline_output[-1].cpu()

        # Create a large tensor to hold all occluded images for a batch
        batch_tensor = torch.zeros((batch_size, D, C, W, H))

        # Index to keep track of how many occlusions we have in our batch tensor
        batch_index = 0

        # Iterate over the 3D image
        for z in tqdm(range(0, D, self.stride[0])):
            for y in range(0, W, self.stride[1]):
                for x in range(0, H, self.stride[2]):
                    # Occlude the input_tensor
                    occluded = input_tensor.clone()
                    occluded[0,
                             z: min(z + self.mask_size[0], D),
                             :,
                             y: min(y + self.mask_size[1], W),
                             x: min(x + self.mask_size[2], H)] = occlusion_value

                    # Store the occluded image in the batch tensor
                    batch_tensor[batch_index] = occluded
                    batch_index += 1

                    # If the batch tensor is full, or we are at the end, process the batch
                    if batch_index == batch_size or (z == D - self.stride[0] and y == W - self.stride[1] and x == H - self.stride[2]):
                        with torch.no_grad():
                            outputs = self.model(batch_tensor[:batch_index]).detach()
                            outputs = outputs.view(-1,32,self.num_classes).mean(1)
                            outputs = outputs[:,-1].cpu()
                            #print(outputs.size(),'dfjiow')

                        for i in range(batch_index):
                            # The prediction for the current occluded image in the batch
                            prediction = outputs[i]
                            # Calculate the difference from the baseline output
                            diff = baseline_output - prediction

                            # Get the current occlusion coordinates
                            current_z = (i * self.stride[0]) % D
                            current_y = ((i * self.stride[1]) % W) % W
                            current_x = ((i * self.stride[2]) % H) % H

                            # Update the sensitivity map with the difference
                            sensitivity_map[
                                current_z: min(current_z + self.mask_size[0], D),
                                current_y: min(current_y + self.mask_size[1], W),
                                current_x: min(current_x + self.mask_size[2], H)
                            ] += diff

                        # Reset the batch tensor and index for the next batch
                        batch_tensor.zero_()
                        batch_index = 0

        return sensitivity_map

# モデルと入力テンソルを定義したら、以下のようにクラスを使用できます：
# model = ...  # あなたのモデルを定義
# input_tensor = ...  # 入力テンソルを定義 (1, d, c, w, h)
# occlusion_sensitivity = OcclusionSensitivityMap(model)
# batch_size = 8  # 任意のバッチサイズを設定
# sensitivity_map = occlusion_sensitivity.generate_sensitivity_map(input_tensor, batch_size)

##matploglibなど

def create_combined_colormap(n_bins=100):
    # 下部のカラーマップ（赤から黄色）
    top = LinearSegmentedColormap.from_list('top', [(1, 1, 0), (1, 0, 0)], N=n_bins//2)

    # 上部のカラーマップ（水色から濃い青）
    bottom = LinearSegmentedColormap.from_list('bottom', [(0, 0, 0.5), (0, 1, 1)], N=n_bins//2)

    # 2つのカラーマップを結合（結合部分が黄色と水色になるように）
    colors = np.vstack((bottom(np.linspace(0, 1, n_bins//2)),
                        top(np.linspace(0, 1, n_bins//2))))

    # 結合されたカラーマップを作成
    combined_cmap = LinearSegmentedColormap.from_list('combined_v3', colors, N=n_bins)

    return combined_cmap


