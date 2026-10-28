from activation_steering import device_name

def test_explicit_device_is_preserved():
    assert device_name("cpu") == "cpu"
