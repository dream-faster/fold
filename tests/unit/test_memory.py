from fold.loop.memory import postprocess_X_y_into_memory_, preprocess_X_y_with_memory
from fold.models.baseline import Naive
from fold.utils.tests import generate_sine_wave_data


def test_memory_in_sample():
    X, y = generate_sine_wave_data()
    sample_weights = y.copy()
    naive = Naive()
    postprocess_X_y_into_memory_(naive, X, y, sample_weights, in_sample=True)
    assert naive._state.memory_X.equals(X)
    assert naive._state.memory_y.equals(y)
    assert naive._state.memory_sample_weights.equals(sample_weights)
    X_with_mem, y_with_mem, sample_w_with_mem = preprocess_X_y_with_memory(
        naive, X, y, sample_weights, in_sample=True
    )
    assert X_with_mem.equals(X)
    assert y_with_mem.equals(y)
    assert sample_w_with_mem.equals(sample_weights)


def test_memory_out_of_sample():
    X, y = generate_sine_wave_data()
    sample_weights = y.copy()
    X_train, y_train, sample_weights_train = (
        X.iloc[:-10],
        y.iloc[:-10],
        sample_weights.iloc[:-10],
    )
    X_test, y_test, sample_weights_test = (
        X.iloc[-10:],
        y.iloc[-10:],
        sample_weights.iloc[-10:],
    )
    naive = Naive()
    # First postprocess_X_y_into_memory is called with in_sample=True, during training
    postprocess_X_y_into_memory_(
        naive, X_train, y_train, sample_weights_train, in_sample=True
    )
    # Then postprocess_X_y_into_memory is called with in_sample=False, during inference
    postprocess_X_y_into_memory_(
        naive, X_train, y_train, sample_weights_train, in_sample=False
    )
    assert naive._state.memory_X.equals(X_train.iloc[-1:None])
    assert naive._state.memory_y.equals(y_train.iloc[-1:None])
    assert naive._state.memory_sample_weights.equals(sample_weights_train.iloc[-1:None])
    X_with_mem, y_with_mem, sample_weights_with_mem = preprocess_X_y_with_memory(
        naive, X_test, y_test, sample_weights_test, in_sample=False
    )
    assert len(X_with_mem) == len(y_with_mem)
    assert X_with_mem.index.equals(y_with_mem.index)
    assert X_with_mem.index.equals(sample_weights_with_mem.index)
    assert X_with_mem.equals(X.iloc[-11:])
    assert y_with_mem.equals(y.iloc[-11:])
    assert sample_weights_with_mem.equals(sample_weights.iloc[-11:])
