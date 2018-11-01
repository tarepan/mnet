import time
import numpy as np
import torch
import librosa
from mnet.audioProcess.AudioDataset import MonoAudioDataset
from .conversion_CycleGAN_VC import convertVoice

def convert1DInto2D(input1d):
    N_batch, channels, width = input1d.size()
    image2d = input1d.view(N_batch, 1, channels, width)
    return image2d

def test(args, net_G_A2B, net_G_B2A, net_D_A, net_D_B, device, sr, featStats_A, featStats_B, evalDirA, evalDirB, epoch, writer):
    """
    Test and evaluate models with test data.
    """
    print(f"epoch {epoch} Evaluation start")
    start = time.time()
    # initialization
    ## Nets
    net_G_A2B.eval(), net_G_B2A.eval(), net_D_A.eval(), net_D_B.eval()
    ## Lossess
    # criterion_GAN, criterion_cycle, criterion_identity = torch.nn.BCELoss(), torch.nn.L1Loss(), torch.nn.L1Loss()
    ## configs
    # lambda_cyc, lambda_id = 10.0, 5.0
    ## dataset
    vcc2016_eval_wavs_a = MonoAudioDataset(evalDirA/"wavs", sr)
    vcc2016_eval_wavs_b = MonoAudioDataset(evalDirB/"wavs", sr)

    # evaluation
    for idx, (wav_A, wav_B) in enumerate(zip(vcc2016_eval_wavs_a, vcc2016_eval_wavs_b)):
        if type(args.batch_num_limit_test) is not int or idx <= args.batch_num_limit_test:
            wav_b_gened, normedMCEPseq_a, normedMCEPseq_gened_b = convertVoice(wav_A, sr, featStats_A, featStats_B, net_G_A2B, device)
            wav_a_gened, normedMCEPseq_b, normedMCEPseq_gened_a = convertVoice(wav_B, sr, featStats_B, featStats_A, net_G_B2A, device)
            librosa.output.write_wav(evalDirA/"generated"/f"{idx}.wav", wav_a_gened, sr)
            librosa.output.write_wav(evalDirB/"generated"/f"{idx}.wav", wav_b_gened, sr)
            # batch_size = real_A.size()[0]
            # zeros, ones = torch.zeros(batch_size, 1, device=device), torch.ones(batch_size, 1, device=device)
            # ## waveform conersion
            # with torch.no_grad():
            #     # MCEPs generation
            #     fake_A = net_G_B2A(real_B)
            #     fake_B = net_G_A2B(real_A)
                ## loss G
                ### Adversarial loss
                # Lgen_A2B = criterion_GAN(net_D_B(convert1DInto2D(fake_B)), ones)
                # Lgen_B2A = criterion_GAN(net_D_A(convert1DInto2D(fake_A)), ones)
                # ### Cycle consistency loss
                # Lcyc_A2B2A = criterion_cycle(net_G_B2A(fake_B), real_A)*lambda_cyc
                # Lcyc_B2A2B = criterion_cycle(net_G_A2B(fake_A), real_B)*lambda_cyc
                # ### Identity mapping loss
                # Lid_B = criterion_identity(net_G_A2B(real_B), real_B)*lambda_id
                # Lid_A = criterion_identity(net_G_B2A(real_A), real_A)*lambda_id
                # ### Total loss
                # Ltotal_G = Lgen_A2B + Lgen_B2A + Lcyc_A2B2A + Lcyc_B2A2B + Lid_B + Lid_A
                ######################### record Ltotal_G
                # MCEPs discrimination A
                # Lreal_A = criterion_GAN(net_D_A(convert1DInto2D(real_A)), ones)
                # Lfake_A = criterion_GAN(net_D_A(convert1DInto2D(fake_A)), zeros)
                # Ltotal_D_A = Lreal_A*0.5 + Lfake_A*0.5
                # # MCEPs discrimination B
                # Lreal_B = criterion_GAN(net_D_B(convert1DInto2D(real_B)), ones)
                # Lfake_B = criterion_GAN(net_D_B(convert1DInto2D(fake_B)), zeros)
                # Ltotal_D_B = Lreal_B*0.5 + Lfake_B*0.5
    print(f"Epoch {epoch} evaluation finished! {time.time() - start} sec")
