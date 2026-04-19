import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class VoiceBankDemandDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000, segment_length=16384):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        # Sadece her iki dizinde de var olan dosyalari al
        self.file_list = [
            f for f in os.listdir(clean_dir)
            if f.endswith('.wav') and os.path.exists(os.path.join(noisy_dir, f))
        ]

        if len(self.file_list) == 0:
            raise ValueError(f"Eşleşen .wav dosyasi bulunamadi: {clean_dir} / {noisy_dir}")

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, path):
        """Yükle, mono'ya çevir, gerekirse resample yap."""
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _fix_length(self, waveform, start=None):
        """Segment uzunluğuna pad veya crop uygula."""
        length = waveform.shape[1]
        if length < self.segment_length:
            pad_len = self.segment_length - length
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, start:start + self.segment_length]
        return waveform

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        clean_waveform = self._load_audio(clean_path)
        noisy_waveform = self._load_audio(noisy_path)

        length = min(clean_waveform.shape[1], noisy_waveform.shape[1])

        if length >= self.segment_length:
            # Ayni start ile hizali kirp
            start = torch.randint(0, length - self.segment_length + 1, (1,)).item()
            clean_waveform = self._fix_length(clean_waveform, start)
            noisy_waveform = self._fix_length(noisy_waveform, start)
        else:
            # Zero-pad
            clean_waveform = self._fix_length(clean_waveform)
            noisy_waveform = self._fix_length(noisy_waveform)

        return noisy_waveform, clean_waveform


if __name__ == "__main__":
    train_clean = "data/clean_trainset_28spk_wav"
    train_noisy = "data/noisy_trainset_28spk_wav"

    dataset = VoiceBankDemandDataset(train_clean, train_noisy)
    print(f"Toplam eğitim örneği: {len(dataset)}")

    noisy, clean = dataset[0]
    print(f"Gürültülü tensor boyutu: {noisy.shape}")
    print(f"Temiz tensor boyutu: {clean.shape}")
    assert noisy.shape == clean.shape, "Boyutlar eşleşmiyor!"
    print("Test geçti.")