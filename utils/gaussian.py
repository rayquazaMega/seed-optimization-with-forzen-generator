import torch
import torch.nn.functional as F

def gaussian_blur(image, kernel_size, sigma):
    x = torch.arange(kernel_size).float() - kernel_size // 2
    y = torch.arange(kernel_size).float() - kernel_size // 2
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.to('cuda')

    blurred_image = F.conv2d(image, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size // 2)

    return blurred_image

if __name__ == '__main__':
    image = torch.randn(1, 1, 384, 576)

    blurred_image = gaussian_blur(image, kernel_size=15, sigma=4.0)

    print(blurred_image.size())