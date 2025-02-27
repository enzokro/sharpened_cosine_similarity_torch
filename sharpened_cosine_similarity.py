"""
Based on  and copy/pasted heavily from code
https://github.com/ZeWang95/scs_pytorch/blob/main/scs.py
from Ze Wang
https://twitter.com/ZeWang46564905/status/1488371679936057348?s=20&t=lB_T74PcwZmlJ1rrdu8tfQ

and code
https://github.com/oliver-batchelor/scs_cifar/blob/main/src/scs.py
from Oliver Batchelor
https://twitter.com/oliver_batch/status/1488695910875820037?s=20&t=QOnrCRpXpOuC0XHApi6Z7A

and the TensorFlow implementation
https://colab.research.google.com/drive/1Lo-P_lMbw3t2RTwpzy1p8h0uKjkCx-RB
and blog post
https://www.rpisoni.dev/posts/cossim-convolution/
from Raphael Pisoni
https://twitter.com/ml_4rtemi5
"""
import torch
from torch import nn
import torch.nn.functional as F

class SharpenedCosineSimilarity(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0,
        eps=1e-12,
    ):
        super(SharpenedCosineSimilarity, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_uniform_(w)
        self.w = nn.Parameter(
            w.view(out_channels, in_channels, -1), requires_grad=True)

        self.p_scale = 10
        p_init = 2**.5 * self.p_scale
        self.register_parameter("p", nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, p_init)

        self.q_scale = 100
        self.register_parameter("q", nn.Parameter(torch.empty(1)))
        nn.init.constant_(self.q, 10)

    def forward(self, x):
        x = unfold2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding)
        n, c, h, w, _, _ = x.shape
        x = x.reshape(n,c,h,w,-1)

        # After unfolded and reshaped, dimensions of the images x are
        # dim 0, n: batch size
        # dim 1, c: number of input channels
        # dim 2, h: number of rows in the image
        # dim 3, w: number of columns in the image
        # dim 4, l: kernel size, squared
        #
        # The dimensions of the weights w are
        # dim 0, v: number of output channels
        # dim 1, c: number of input channels
        # dim 2, l: kernel size, squared

        square_sum = torch.sum(torch.square(x), [1, 4], keepdim=True)
        x_norm = torch.add(
            torch.sqrt(square_sum + self.eps),
            torch.square(self.q / self.q_scale))

        square_sum = torch.sum(torch.square(self.w), [1, 2], keepdim=True)
        w_norm = torch.add(
            torch.sqrt(square_sum + self.eps),
            torch.square(self.q / self.q_scale))

        x = torch.einsum('nchwl,vcl->nvhw', x / x_norm, self.w / w_norm)
        sign = torch.sign(x)

        x = torch.abs(x) + self.eps
        x = x.pow(torch.square(self.p / self.p_scale).view(1, -1, 1, 1))
        return sign * x


def unfold2d(x, kernel_size:int, stride:int, padding:int):
    x = F.pad(x, [padding]*4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided(
        (bs, in_c, (h - ks) // stride + 1, (w - ks) // stride + 1, ks, ks),
        (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x


class SharpenedCosineSimilarity_ConvImpl(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0,
        eps=1e-12,
    ):
        super(SharpenedCosineSimilarity_ConvImpl, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_uniform_(w)
        # TODO: this could be initialized as
        # (out_channels, in_channel, kernel_size, kernel_size)
        # right off the bat, but we leave it in this format to retain compat
        # with the einsum implementation
        self.w = nn.Parameter(
            w.view(out_channels, in_channels, -1), requires_grad=True)

        self.p_scale = 10
        p_init = 2**.5 * self.p_scale
        self.register_parameter("p", nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, p_init)

        self.q_scale = 100
        self.register_parameter("q", nn.Parameter(torch.empty(1)))
        nn.init.constant_(self.q, 10)

    def forward(self, x):
        # reshaping for compatibility with the einsum-based implementation
        w = self.w.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        w_norm = torch.linalg.vector_norm(
            w,
            dim=(1, 2, 3),
            keepdim=True,
        )

        q_sqr = (self.q / self.q_scale) ** 2

        # a small difference: we add eps outside of the norm
        # instead of inside in order to reuse the performant
        # code of torch.linalg.vector_norm
        w_normed = w / ((w_norm + self.eps) + q_sqr)

        x_norm_squared = F.avg_pool2d(
            x ** 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=1, # we actually want sum_pool
        ).sum(dim=1, keepdim=True)

        y_denorm = F.conv2d(
            x,
            w_normed,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

        y = y_denorm / ((x_norm_squared + self.eps).sqrt() + q_sqr)

        sign = torch.sign(y)

        y = torch.abs(y) + self.eps
        p_sqr = (self.p / self.p_scale) ** 2
        y = y.pow(p_sqr.reshape(1, -1, 1, 1))
        return sign * y


class SharpenedCosineSimilarityAnnotated(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0,
        eps=1e-12,
    ):
        super(SharpenedCosineSimilarityAnnotated, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)

        # initialize weight kernel for sharpened cosine similarity
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.register_parameter("w", nn.Parameter(w))
        nn.init.xavier_uniform_(self.w)

        # initialize the learned channel-wise exponents
        self.p_scale = 10
        p_init = 2**.5 * self.p_scale
        self.register_parameter("p", nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, p_init)

        # initialize the learned noise threshold 
        # TODO: channel-wise threshold?
        self.q_scale = 100
        self.register_parameter("q", nn.Parameter(torch.empty(1)))
        nn.init.constant_(self.q, 10)

    def forward(self, x):

        # get the kernel norm
        w_norm = torch.linalg.vector_norm(
            self.w,
            dim=(1, 2, 3),
            keepdim=True,
        )

        # find the scaled noise threshold
        q_sqr = (self.q / self.q_scale) ** 2

        # normalize the convolutional kernel
        w_normed = self.w / ((w_norm + self.eps) + q_sqr)

        # find squared norm of x
        x_norm_squared = F.avg_pool2d(
            (x + self.eps) ** 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=1, # we actually want sum_pool
        ).sum(dim=1, keepdim=True)


        # convolve input with the normalized kernel
        y_denorm = F.conv2d(
            x,
            w_normed,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

        # normalize convolution output to get cosine similarities
        # y = y_denorm / ((x_norm_squared + self.eps).sqrt() + q_sqr)
        y = y_denorm / ((x_norm_squared).sqrt() + q_sqr)

        # find sign to amplify positive/negative matches appropriately
        sign = torch.sign(y)

        # get the absolute value 
        y = torch.abs(y) + self.eps # eps for stability

        # sharpen the cosine similarities per-channel to highlight matches and bury noise
        p_sqr = (self.p / self.p_scale) ** 2
        y = y.pow(p_sqr.reshape(1, -1, 1, 1))

        # return sharpened cosine similarity
        return sign * y