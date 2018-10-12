import time
from audioProcess.getAudioStats import getAudioStats

def test():
    start_time = time.time()
    train_A_dir = "./data/vcc2016/vcc2016_training/smallSF1"
    # train_A_dir = "./data/vcc2016/vcc2016_training/SF1"
    sampling_rate = 16000
    logF0_mean, logF0_std, MCEP_means, MCEP_stds = getAudioStats(train_A_dir, sampling_rate)
    print(f"LogF0\nmean: {logF0_mean}, std: {logF0_std}");
    print(f"MCEP means:")
    print(MCEP_means)
    print(f"MCEP stds:")
    print(MCEP_stds)
    time_elapsed = time.time() - start_time
    print(time_elapsed)
    print(f'Time Elapsed for Data Preprocessing: {time_elapsed // 3600}:{time_elapsed % 3600 // 60}:{time_elapsed % 60 // 1}')

if __name__ == "__main__":
    test()
