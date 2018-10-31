import numpy as np
import torch
from modules.audioProcess.basics.getFeatures import convertWavIntoFeatures, convertFeaturesIntoWav
def convertVoice(waveform, sr, fStats_src, fStats_tgt, Generator, device):
    """
    Convert a source domain wave into a target domain wave
    Args:
        waveform (numpy.ndarray(T,)): a source waveform which is converted into target domain
        FeatureStats_src:
        FeatureStats_tgt:
        Generator (torch.nn.Modules): Generator network which convert waveform. should be .eval()-nized.
    Returns:
        (numpy.ndarray(T,)): converted source waveform, which is converted into target domain (tiny end sound can be cut out because of Conv)
        (Tensor(1, MCEPdim, t)): normedMCEPseq_tensor is source waveform MCEPseq Tensor
        (Tensor(1, MCEPdim, t)): normedMCEPseq_tensor_verted is converted waveform MCEPseq Tensor
    """
    ############ waveform to argumented features ############
    f0seq, MCEPseq, APseq = convertWavIntoFeatures(waveform, sr, frame_period = 5.0, MCEPdim = 24)
    normedMCEPseq = (MCEPseq - fStats_src["MCEP_mean"])/fStats_src["MCEP_std"]
    ## clipping for fully convolution
    clipping = f0seq.shape[0] - f0seq.shape[0]%4
    f0seq, normedMCEPseq, APseq = f0seq[0:clipping], normedMCEPseq[:, 0:clipping], APseq[0:clipping, :]
    ## from numpy To PyTorch Tensor
    normedMCEPseq_tensor = torch.from_numpy(normedMCEPseq).view(1, 24, -1).to(device)

    ############ feature conversion ############
    ## f0 (Logarithm Gaussian normalization. Liu, et al.. "High Quality Voice Conversion through Phoneme-based Linear Mapping Functions with STRAIGHT for Mandarin")
    f0seq_verted = np.exp((np.log(f0seq) - fStats_src["logF0_mean"]) / fStats_src["logF0_std"] * fStats_tgt["logF0_std"] + fStats_tgt["logF0_mean"])
    ### no conversion test: f0seq_verted = f0seq
    ## NormedMCEPseq
    with torch.no_grad():
        normedMCEPseq_tensor_verted = Generator(normedMCEPseq_tensor)
        ### no conversion test:  normedMCEPseq_tensor_verted = normedMCEPseq_tensor
    ## AP - No conversion

    ############ argumented features to waveform ############
    ## from PyTorch Tensor To numpy
    trimed = normedMCEPseq_verted = normedMCEPseq_tensor_verted.view(24, -1)
    normedMCEPseq_verted = trimed.cpu().numpy()
    ## argumented features to waveform
    MCEPseq_verted = normedMCEPseq_verted * fStats_tgt["MCEP_std"] + fStats_tgt["MCEP_mean"]
    # no conversion test: MCEPseq_verted = normedMCEPseq_verted * fStats_src["MCEP_std"] + fStats_src["MCEP_mean"]

    waveform_verted = convertFeaturesIntoWav(f0seq_verted, MCEPseq_verted, APseq, sr, frame_period = 5.0)
    return waveform_verted, normedMCEPseq_tensor, normedMCEPseq_tensor_verted
