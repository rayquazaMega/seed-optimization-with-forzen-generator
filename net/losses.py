import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHistogram(nn.Module):
    def __init__(self, bins=255, min=0.0, max=1.0, sigma=30 * 255):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x


class HistEntropyLoss(nn.Module):
    def __init__(self, bins=256, min=0.0, max=1.0, sigma=30 * 256):
        super(HistEntropyLoss, self).__init__()
        self.softhist = SoftHistogram(bins=bins, min=min, max=max, sigma=sigma)

    def forward(self, x):
        x = torch.exp(torch.mean(torch.log(x.clamp_min(1e-6)), dim=1))
        x = x.view(-1)
        p = self.softhist(x)
        p = p / x.shape[0]
        return 8 + p.mul(p.clamp_min(1e-6).log2()).sum()


class FiedelityLoss(nn.Module):
    def __init__(self):
        super(FiedelityLoss, self).__init__()

    def forward(self, y, y_pred):
        loss = (F.mse_loss(y_pred, y, reduction='none') / (y_pred.abs() + y.abs()).clamp(min=1e-2))
        return loss.mean()

class BrightLoss(nn.Module):
    def __init__(self,gamma):
        super(BrightLoss, self).__init__()
        self.gamma = gamma
    def downsample(self,image):
        return F.avg_pool2d(image, kernel_size=32, stride=16, padding=0)
    def forward(self, illum_en):
        loss = torch.mean(torch.abs(illum_en-self.gamma))# + F.mse_loss(self.downsample(illum_en), illum_gamma)
        return loss

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x,weight_map=None):
        self.h_x = x.size()[2]
        self.w_x = x.size()[3]
        self.batch_size = x.size()[0]
        if weight_map is None:
            self.TVLoss_weight=(1, 1)
        else:
            # self.h_x = x.size()[2]
            # self.w_x = x.size()[3]
            # self.batch_size = x.size()[0]
            self.TVLoss_weight = self.compute_weight(weight_map)

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        h_tv = (self.TVLoss_weight[0]*torch.abs((x[:,:,1:,:]-x[:,:,:self.h_x-1,:]))).sum()
        w_tv = (self.TVLoss_weight[1]*torch.abs((x[:,:,:,1:]-x[:,:,:,:self.w_x-1]))).sum()
        # print(self.TVLoss_weight[0],self.TVLoss_weight[1])
        return (h_tv/count_h+w_tv/count_w)/self.batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def compute_weight(self, img):
        gradx = torch.abs(img[:, :, 1:, :] - img[:, :, :self.h_x-1, :])
        grady = torch.abs(img[:, :, :, 1:] - img[:, :, :, :self.w_x-1])
        TVLoss_weight_x = torch.div(1,torch.exp(gradx))
        TVLoss_weight_y = torch.div(1, torch.exp(grady))

        # TVLoss_weight_x = torch.div(1, torch.abs(gradx)+0.0001)
        # TVLoss_weight_y = torch.div(1, torch.abs(grady)+0.0001)

        # TVLoss_weight_x = torch.log2(1+gradx*gradx)
        # TVLoss_weight_y = torch.log2(1+grady*grady)
        return TVLoss_weight_x, TVLoss_weight_y

class TVLoss_jit(nn.Module):
    def __init__(self,h_x,w_x):
        super(TVLoss_jit,self).__init__()
        self.h_x = h_x
        self.w_x = w_x
        self.batch_size = 1
    def forward(self,x,weight_map=None):
        if weight_map is None:
            TVLoss_weight = (1,1)
        else:
            TVLoss_weight = self.compute_weight(weight_map)

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        h_tv = (TVLoss_weight[0]*torch.abs((x[:,:,1:,:]-x[:,:,:self.h_x-1,:]))).sum()
        w_tv = (TVLoss_weight[1]*torch.abs((x[:,:,:,1:]-x[:,:,:,:self.w_x-1]))).sum()
        # print(self.TVLoss_weight[0],self.TVLoss_weight[1])
        return (h_tv/count_h+w_tv/count_w)/self.batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def compute_weight(self, img):
        gradx = torch.abs(img[:, :, 1:, :] - img[:, :, :self.h_x-1, :])
        grady = torch.abs(img[:, :, :, 1:] - img[:, :, :, :self.w_x-1])
        TVLoss_weight_x = torch.div(1,torch.exp(gradx))
        TVLoss_weight_y = torch.div(1, torch.exp(grady))

        # TVLoss_weight_x = torch.div(1, torch.abs(gradx)+0.0001)
        # TVLoss_weight_y = torch.div(1, torch.abs(grady)+0.0001)

        # TVLoss_weight_x = torch.log2(1+gradx*gradx)
        # TVLoss_weight_y = torch.log2(1+grady*grady)
        return TVLoss_weight_x, TVLoss_weight_y

class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class HEP_Smooth_loss(nn.Module):
    # smooth loss for hep URetinex and RetinexNet
    def __init__(self):
        super(HEP_Smooth_loss, self).__init__()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def forward(self, input_I, input_R):
        rgb_weights = torch.Tensor([0.2990, 0.5870, 0.1140]).cuda()
        input_gray = torch.tensordot(input_R, rgb_weights, dims=([1], [-1]))
        input_gray = torch.unsqueeze(input_gray, 1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.gradient(input_gray, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.gradient(input_gray, "y")))

class ColorMapLoss(nn.Module):
    def __init__(self):
        super(ColorMapLoss, self).__init__()
    
    def calculate_color_map(self, image):
        # input image should be T2P prosessed
        color_gt = nn.functional.avg_pool2d(image, 11, 1, 5)
        color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)

        return color_gt

    def forward(self, image1, image2):
        # Calculate the Mean Squared Error (MSE) loss between the two color maps
        color_map1 = self.calculate_color_map(image1)
        color_map2 = self.calculate_color_map(image2)
        loss = nn.MSELoss()(color_map1, color_map2)
        return loss