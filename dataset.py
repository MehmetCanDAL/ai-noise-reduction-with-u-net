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

        # hem clean hem de noisy dizinlerindeki aynı isimdeki .wav dosyalarını listele
        self.file_list = [
            f for f in os.listdir(clean_dir)
            if f.endswith('.wav') and os.path.exists(os.path.join(noisy_dir, f))
        ]
        # Eğer hiç dosya yoksa hata verir
        if len(self.file_list) == 0:
            raise ValueError(f"Eşleşen .wav dosyasi bulunamadi: {clean_dir} / {noisy_dir}")

    def __len__(self):
        # Datasettki örnek sayısı verir
        return len(self.file_list)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path) #.waw dosyasını tensor olarak yükler (sr: sample rate)
        if sr != self.sample_rate: # sr farklıysa sr'yi sample_rate'e dönüştürür
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate) 
        if waveform.shape[0] > 1: # stereo ise mono yapar
            waveform = waveform.mean(dim=0, keepdim=True) # (1, samples) boyutuna getirir
        return waveform

    def _fix_length(self, waveform, start=None): # segment_lenght uzunluğa göre kırpar veya pad eder
        length = waveform.shape[1] # (1, samples) boyutunda olduğu varsayılır
        if length < self.segment_length:
            pad_len = self.segment_length - length # eksik uzunluk kadar pad ekler
            waveform = F.pad(waveform, (0, pad_len)) # sonunda 0 ekleyerek uzatır
        else:
            waveform = waveform[:, start:start + self.segment_length] # segment_lenght uzunluğunda kırp
        return waveform

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        clean_waveform = self._load_audio(clean_path)
        noisy_waveform = self._load_audio(noisy_path)

        length = min(clean_waveform.shape[1], noisy_waveform.shape[1]) # her iki dalga boyutunun minimum uzunluğunu alır

        if length >= self.segment_length: # segment_lenght uzunluğunda kırpma yapacak kadar uzun ise
            # Ayni start ile hizali kirp , rastgele bir başlangıç noktası seçer 
            start = torch.randint(0, length - self.segment_length + 1, (1,)).item()
            clean_waveform = self._fix_length(clean_waveform, start) 
            noisy_waveform = self._fix_length(noisy_waveform, start)
        else:
            # segment_lenght uzunluğunda kırpacak kadar uzun değilse, her iki dalga boyunu da segment_lenght uzunluğuna pad eder
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