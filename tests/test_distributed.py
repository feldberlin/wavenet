import pytest

from wavenet import datasets, distributed, model, train


@pytest.mark.integration
def test_distributed_data_parallel():
    p = model.HParams(n_audio_chans=2, n_layers=8).with_all_chans(2)
    tp = train.HParams(max_epochs=1, batch_size=8)
    ds, ds_test = datasets.tracks("fixtures/goldberg/short.wav", 0.8, p)
    m = model.Wavenet(p)
    t = distributed.DDP(m, ds, None, tp)
    t.train()


@pytest.mark.integration
def test_data_parallel():
    p = model.HParams(n_audio_chans=2, n_layers=8).with_all_chans(2)
    tp = train.HParams(max_epochs=1, batch_size=8)
    ds, ds_test = datasets.tracks("fixtures/goldberg/short.wav", 0.8, p)
    m = model.Wavenet(p)
    t = distributed.DP(m, ds, None, tp)
    t.train()
