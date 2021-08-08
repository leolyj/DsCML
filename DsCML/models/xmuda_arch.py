import torch
import torch.nn as nn
import torch.nn.functional as F
from DsCML.models.resnet34_unet import UNetResNet34
from DsCML.models.scn_unet import UNetSCN


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.con1_1_avg = nn.Conv2d(feat_channels,num_classes,kernel_size=1,stride=1)
        # self.con1_1_max = nn.Conv2d(feat_channels,num_classes,kernel_size=1,stride=1)
        # self.con1_1_min = nn.Conv2d(feat_channels,num_classes,kernel_size=1,stride=1)
        # self.linear = nn.Linear(feat_channels, num_classes)
        self.dow_avg = nn.AvgPool2d((5,5),stride=(1,1),padding=(2,2))
        # self.dow_max = nn.MaxPool2d((3,3),stride=(1,1),padding=(1,1))
        # self.dow_min = nn.MaxPool2d((3,3),stride=(1,1),padding=(1,1))
        # 2nd segmentation head
        self.dual_head = dual_head


    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        # 2D network
        out_2D_feature = self.net_2d(img)
        out_2D_feature_avg = self.dow_avg(out_2D_feature)
        out_2D_feature_avg = self.con1_1_avg(out_2D_feature_avg)
        # 2D-3D feature lifting
        img_feats = []
        for i in range(out_2D_feature_avg.shape[0]):
            img_feats.append(out_2D_feature_avg.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        x_avg = torch.cat(img_feats, 0)

        x = x_avg

        # linear
        # x = self.linear(img_feats)

        preds = {
            'seg_logit': x,
        }

        return preds, out_2D_feature, img_indices



class L2G_classifier_2D(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 ):
        super(L2G_classifier_2D, self).__init__()



        # segmentation head
        self.con1_1_avg = nn.Conv2d(input_channels, num_classes, kernel_size=1, stride=1)
        self.con1_1_max = nn.Conv2d(input_channels, num_classes, kernel_size=1, stride=1)
        self.con1_1_min = nn.Conv2d(input_channels, num_classes, kernel_size=1, stride=1)
        self.linear = nn.Linear(input_channels, num_classes)
        self.dow_avg = nn.AvgPool2d((5, 5), stride=(1, 1), padding=(2, 2))
        self.dow_max = nn.MaxPool2d((5, 5), stride=(1, 1), padding=(2, 2))
        self.dow_min = nn.MaxPool2d((5, 5), stride=(1, 1), padding=(2, 2))
        self.dow_ = nn.AdaptiveAvgPool2d((1,1))



    def forward(self, input_2D_feature,img_indices):
        # (batch_size, 3, H, W)

        avg_feature = self.dow_avg(input_2D_feature)
        avg_feature = self.con1_1_avg(avg_feature)
        global_wise_line = self.dow_(input_2D_feature).squeeze()

        avg_line = []
        for i in range(avg_feature.shape[0]):
            avg_line.append(avg_feature.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        avg_line = torch.cat(avg_line, 0)

        max_feature = self.dow_max(input_2D_feature)
        max_feature = self.con1_1_max(max_feature)

        max_line = []
        for i in range(max_feature.shape[0]):
            max_line.append(max_feature.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        max_line = torch.cat(max_line, 0)

        min_feature = -self.dow_min(-input_2D_feature)
        min_feature = self.con1_1_min(min_feature)

        min_line = []
        for i in range(min_feature.shape[0]):
            min_line.append(min_feature.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        min_line = torch.cat(min_line, 0)

        # linear
        # point_wise_pre = self.linear(point_wise_line)
        global_wise_pre = self.linear(global_wise_line)

        preds = {
            'feats': (avg_line+max_line+min_line)/3,
            'seg_logit_avg': avg_line,
            'seg_logit_max': max_line,
            'seg_logit_min': min_line,
            'seg_logit_global': global_wise_pre
        }



        return preds





class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head


    def forward(self, data_batch):
        out_3D_feature = self.net_3d(data_batch['x'])
        x = self.linear(out_3D_feature)

        preds = {
            'seg_logit': x,
        }



        return preds,out_3D_feature



class L2G_classifier_3D(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 ):
        super(L2G_classifier_3D, self).__init__()


        # segmentation head
        self.linear_point = nn.Linear(input_channels, num_classes)
        self.linear_global = nn.Linear(input_channels, num_classes)
        self.dow = nn.AvgPool1d(kernel_size=3,stride=1,padding=1)
        self.dow_ = nn.AdaptiveAvgPool1d(8)

    def forward(self, input_3D_feature):

        x = torch.transpose(input_3D_feature,0,1)
        x = x.unsqueeze(0)
        # local_wise_line = self.dow(x).squeeze(0)
        # local_wise_line = torch.transpose(local_wise_line,0,1)

        global_wise_line = self.dow_(x).squeeze(0)
        global_wise_line = torch.transpose(global_wise_line, 0, 1)

        # linear
        point_wise_pre = self.linear_point(input_3D_feature)
        # local_wise_pre = self.linear(local_wise_line)
        global_wise_pre = self.linear_global(global_wise_line)

        preds = {
            'feats': input_3D_feature,
            'seg_logit_point': point_wise_pre,
            'seg_logit_global': global_wise_pre
        }



        return preds






def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
