from torch import nn
import timm


class TimmModel(nn.Module):
    def __init__(self, backbone,input_size=384, num_classes=1, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=5,
            num_classes=1,
            features_only=False,
            drop_rate=0.,
            drop_path_rate=0.,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        self.input_size = input_size

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0., bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * 32, 5,self.input_size, self.input_size)
        feat = self.encoder(x)
        feat = feat.view(bs, 32, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * 32, -1)
        feat = self.head(feat)
        
        return feat