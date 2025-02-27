import math
import torch

from sharpened_cosine_similarity import SharpenedCosineSimilarity, SharpenedCosineSimilarity_ConvImpl, SharpenedCosineSimilarityAnnotated

def test():
    original = SharpenedCosineSimilarity(5, 5, 3)
    faster = SharpenedCosineSimilarity_ConvImpl(5, 5, 3)
    faster.load_state_dict(original.state_dict())

    test_values = torch.randn(1, 5, 32, 32)

    orig_output = original(test_values)
    faster_output = faster(test_values)

    print((orig_output - faster_output).abs().max().item())


def reshape_w(state_dict):
    '''Small helper for compatability between einsum and conv2d `w` shapes.'''
    nin, nout, ksqr = state_dict['w'].shape
    k = int(math.sqrt(ksqr))
    state_dict['w'] = state_dict['w'].reshape(nin, nout, k, k)
    return state_dict

def test_annot():
    original = SharpenedCosineSimilarity(5, 5, 3)
    faster = SharpenedCosineSimilarityAnnotated(5, 5, 3)
    faster.load_state_dict(reshape_w(original.state_dict()))

    test_values = torch.randn(1, 5, 32, 32)

    orig_output = original(test_values)
    faster_output = faster(test_values)

    print((orig_output - faster_output).abs().max().item())

def test_conv_and_annot():
    original = SharpenedCosineSimilarity(5, 5, 3)
    # load original conv implementation
    faster = SharpenedCosineSimilarity_ConvImpl(5, 5, 3)
    faster.load_state_dict(original.state_dict())
    # load annoated conv implementation
    annotated = SharpenedCosineSimilarityAnnotated(5, 5, 3)
    annotated.load_state_dict(reshape_w(original.state_dict()))

    test_values = torch.randn(1, 5, 32, 32)

    orig_output = original(test_values)
    faster_output = faster(test_values)
    annotated_output = annotated(test_values)

    conv_delta = (orig_output - faster_output).abs().max()
    annotated_delta = (orig_output - annotated_output).abs().max()

    print(conv_delta.item())
    print(annotated_delta.item())

    # final sanity check
    assert torch.allclose(conv_delta, annotated_delta)

# run tests
test()
test_annot()
test_conv_and_annot()