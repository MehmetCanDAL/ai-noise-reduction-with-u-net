import torch
import torchaudio

class AudioToSpectrogram:
    def __init__(self, n_fft=510, hop_length=256):
        """
        n_fft=510     → freq_bins=256, U-Net pooling için ideal
        hop_length=256 → %50 örtüşme
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

    def __call__(self, waveform):
        """
        waveform : (1, samples)
        return   : magnitude (1, freq_bins, time_frames) — log ölçekli
                   phase     (1, freq_bins, time_frames)
        """
        # Window'u waveform ile aynı device'a taşı
        window = self.window.to(waveform.device)

        stft = torch.stft(
            waveform.squeeze(0),        # (samples,)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )                               # → (freq_bins, time_frames) complex

        magnitude = stft.abs().unsqueeze(0)   # (1, freq_bins, time_frames)
        phase     = stft.angle().unsqueeze(0) # (1, freq_bins, time_frames)

        magnitude = torch.log1p(magnitude)    # Log scaling

        return magnitude, phase

    def inverse(self, magnitude, phase, length=None):
        """
        magnitude : (1, freq_bins, time_frames) — log ölçekli
        phase     : (1, freq_bins, time_frames)
        return    : (1, samples)
        """
        window = self.window.to(magnitude.device)

        magnitude = torch.expm1(magnitude)    # log1p'nin tersi

        # Polar formdan complex'e: magnitude * e^(j*phase)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        stft_complex = torch.complex(real, imag).squeeze(0)  # (freq_bins, time_frames)

        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=length
        )                                     # → (samples,)

        return waveform.unsqueeze(0)          # (1, samples)
    

if __name__ == "__main__":
    print("Test başlıyor...")
    dummy_waveform = torch.randn(1, 16000)

    transform = AudioToSpectrogram()
    magnitude, phase = transform(dummy_waveform)

    print(f"Giriş dalga boyutu   : {dummy_waveform.shape}")
    print(f"Magnitude boyutu     : {magnitude.shape}")   # (1, 256, 63) beklenir
    print(f"Phase boyutu         : {phase.shape}")       # (1, 256, 63) beklenir

    recovered = transform.inverse(magnitude, phase, length=16000)
    print(f"Geri dönüştürülen ses: {recovered.shape}")

    # Geri dönüşüm kalitesini kontrol et
    error = (dummy_waveform - recovered).abs().mean()
    print(f"Ortalama hata        : {error:.6f}")         # ~0 olmalı
    print("Test geçti.")