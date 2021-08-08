import torch
import torch.nn as nn
import sparseconvnet as scn

DIMENSION = 3

def UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2], leakiness=0, n_input_planes=-1):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
    def U(nPlanes,n_input_planes=-1): #Recursive function
        m = scn.Sequential()
        for i in range(reps):
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
            n_input_planes=-1
        if len(nPlanes) > 1:
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        U(nPlanes[1:])).add(
                        scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                            downsample[0], downsample[1], False))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
        return m
    m = U(nPlanes,n_input_planes)
    return m


class UNetSCN(nn.Module):
    def __init__(self,
                 in_channels,
                 m=16,  # number of unet features (multiplied in each layer)
                 block_reps=1,  # depth
                 residual_blocks=False,  # ResNet style basic blocks
                 full_scale=4096,
                 num_planes=7
                 ):
        super(UNetSCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]

        # self.sparseModel = scn.Sequential().add(
        #     scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
        #     scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
        #     scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
        #     scn.BatchNormReLU(m)).add(
        #     scn.OutputLayer(DIMENSION))

        self.layer1 = scn.InputLayer(DIMENSION, full_scale, mode=4)
        self.layer2 = scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)
        self.layer3 = UNet(DIMENSION, block_reps, n_planes, residual_blocks)
        self.layer4 = scn.BatchNormReLU(m)
        self.layer5 = scn.OutputLayer(DIMENSION)

    def forward(self, x):
        # x = self.sparseModel(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def test():
    b, n = 2, 100
    coords = torch.randint(4096, [b, n, DIMENSION])
    batch_idxs = torch.arange(b).reshape(b, 1, 1).repeat(1, n, 1)
    coords = torch.cat([coords, batch_idxs], 2).reshape(-1, DIMENSION + 1)

    in_channels = 3
    feats = torch.rand(b * n, in_channels)

    x = [coords, feats.cuda()]

    net = UNetSCN(in_channels).cuda()
    out_feats = net(x)

    print('out_feats', out_feats.shape)


if __name__ == '__main__':
    test()
