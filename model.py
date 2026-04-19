import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        """
        Parametreler:
            in_channels: Giriş kanal sayisi (spektrogram için 1)
            out_channels: Çikiş kanal sayisi (maske için 1)
            features: 4 katmanli encoder yapisi için kanal listesi
        """
        super(UNet, self).__init__()

        
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 simetrik pooling

        current_in = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(current_in, feature))
            current_in = feature

        # --- Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for feature in reversed(features):
            
            self.up_convs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        # Son katman
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Maske değerlerini [0, 1] araliğina çeker

    def forward(self, x):
        
        skip_connections = []
        for encode_block in self.encoder_blocks:
            x = encode_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        
        skip_connections = skip_connections[::-1]

        for idx, (up_conv, decode_block) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            x = up_conv(x)
            skip = skip_connections[idx]

            # Boyut uyuşmazliklarini pad ile düzelt (tek sayi boyutlari için)
            if x.shape != skip.shape:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                          diff_h // 2, diff_h - diff_h // 2])

            # Skip bağlantisini birleştir
            x = torch.cat((skip, x), dim=1)
            x = decode_block(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)  # Maske çiktisi
        return x


if __name__ == "__main__":
    print(">>> 4 Katmanli U-Net Maske Modeli Test Ediliyor...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    model = UNet(in_channels=1, out_channels=1).to(device)
    print("Model başariyla oluşturuldu.")

    # transforms.py'den gelen boyutlar: (1, 1, 256, 63)
    dummy_input = torch.randn(1, 1, 256, 63).to(device)
    print(f"Giriş Tensor Boyutu: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Maske Çikiş Boyutu : {output.shape}")
    print(f"Maske Değer Araliği : [{output.min().item():.3f}, {output.max().item():.3f}]")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Eğitilebilir Parametre Sayisi: {total_params:,}")

    print("Test geçti.")