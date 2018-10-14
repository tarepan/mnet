from audioLoader import VCC2016Dataset, ToTensor
from torch.utils.data import DataLoader

train_A_dir = "./data/vcc2016/vcc2016_training/smallSF1"
sampling_rate = 16000

vcc2016 = VCC2016Dataset(
    train_A_dir,
    sampling_rate,
    transform=ToTensor())

dataloader = DataLoader(vcc2016, batch_size=1, shuffle=False)

for wav in dataloader:
    # print(f"wav through loader. tyep is {wav.dtype}")
    print(wav)
    print(wav[0])
    print(wav[0,0])
