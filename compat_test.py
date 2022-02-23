import math
import torch

from sharpened_cosine_similarity import SharpenedCosineSimilarity, SharpenedCosineSimilarity_ConvImpl, SharpenedCosineSimilarity_ConvImplAnnot

def test():
    original = SharpenedCosineSimilarity(5, 5, 3)
    faster = SharpenedCosineSimilarity_ConvImpl(5, 5, 3)
    faster.load_state_dict(original.state_dict())

    test_values = torch.randn(1, 5, 32, 32)

    orig_output = original(test_values)
    faster_output = faster(test_values)

    print((orig_output - faster_output).abs().max().item())

def reshape_w(original):
    state_dict = original.state_dict()
    nin, nout, ksqr = state_dict['w'].shape
    k = int(math.sqrt(ksqr))
    state_dict['w'] = state_dict['w'].reshape(nin, nout, k, k)
    return state_dict

def test_annot():
    original = SharpenedCosineSimilarity(5, 5, 3)
    faster = SharpenedCosineSimilarity_ConvImplAnnot(5, 5, 3)
    state_dict = reshape_w(original)
    faster.load_state_dict(state_dict)

    test_values = torch.randn(1, 5, 32, 32)

    orig_output = original(test_values)
    faster_output = faster(test_values)

    print((orig_output - faster_output).abs().max().item())

test(),test_annot()
