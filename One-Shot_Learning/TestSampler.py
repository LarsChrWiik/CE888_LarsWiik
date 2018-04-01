
from Sampler import Sampler

"""
Test for Sample generation. 
"""
def testSampler(path):

    # Generate examples.
    count = 10
    example = Sampler.get_samples(
        path=path,
        count=count
    )

    # Test Examples.
    assert len(example[0]) == count
    assert len(example[1]) == count
    assert len([x for x in example[1] if x == 0]) == count / 2
    assert len([x for x in example[1] if x == 1]) == count / 2
    assert len(example[0][0]) == 2*105*105
