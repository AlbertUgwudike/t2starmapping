import torch

sobel3D = torch.tensor([
    [ [-1, -2, -1], [-2, -4, -2], [-1, -2, -1] ],
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],
    [ [ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1] ],
], dtype=torch.float32) / 32

prewitt3D = torch.tensor([
    [ [-1, -1, -1], [-1, -1, -1], [-1, -1, -1] ],
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],
    [ [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1] ],
], dtype=torch.float32) / 32

upSobel3D = torch.tensor([
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],
    [ [ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1] ],
], dtype=torch.float32) / 64

dnSobel3D = torch.tensor([
    [ [ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1] ],
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],
    [ [ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0] ],
], dtype=torch.float32) / 64