import numpy as np
from input_encoding import FlyInputEncoder

def test_encoder():
    num_pns = 297   # PNs that connect to KCs (ff["n_pn"])
    encoder = FlyInputEncoder(num_pns=num_pns)

    obs = np.array([0.8, 0.2, 0.5], dtype=np.float32)
    activity = encoder.encode(obs, e3_mode=False)
    assert len(activity) == num_pns
    assert np.all(activity >= 0.0), "PN activity cannot be negative!"

    activity_e3 = encoder.encode(obs, e3_mode=True)
    assert np.all(activity_e3[encoder.smell_pn_indices] > 0.0), "Smell PNs must keep baseline input in E3 mode!"
    print("[PASS] Input Encoder tests passed successfully!")

if __name__ == "__main__":
    test_encoder()