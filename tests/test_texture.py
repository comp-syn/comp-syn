from __future__ import annotations

from pathlib import Path

import pytest

from compsyn.config import CompsynConfig
from compsyn.vectors import WordToColorVector
from compsyn.trial import Trial


def arrays_within_n_percent(array_a, array_b, n_percent) -> bool:
    for a, b in zip(array_a, array_b):
        percent_diff = (abs(a - b) / ((a + b) / 2)) * 100
        if percent_diff > n_percent:
            return False
    return True


@pytest.mark.integration
def test_get_wavelet_embedding() -> None:

    CompsynConfig(
        work_dir=Path(__file__).parent.joinpath("test-assets"),
    )
    trial = Trial(
        experiment_name="test-downloads",
        trial_id="known-dist",
        hostname="pytester",
        trial_timestamp="testoclock",
    )

    w2cv = WordToColorVector(label="earth", trial=trial)
    w2cv.load_data()
    w2cv.run_analysis()

    # fmt: off
    known_jzazbz_wavelet_embedding = [ 7.64314235e-04, 5.94415315e-06, 8.54728479e-06, 1.28341026e-05, 2.22079477e-05, 4.32342356e-05, 7.41187073e-07, 5.65949599e-07, 4.50868593e-07, 4.97627483e-07, 8.77906159e-07, 1.00500840e-06, 9.45680958e-07, 1.66989418e-06, 1.68426592e-06, 3.30888528e-06, -5.08988449e-05, 1.84844485e-06, 2.20531119e-06, 2.87183729e-06, 3.65866043e-06, 4.53035901e-06, 2.28084719e-07, 1.61017391e-07, 1.11515502e-07, 8.99797212e-08, 2.22999186e-07, 2.36072406e-07, 2.37612258e-07, 3.58962428e-07, 3.56402998e-07, 5.63441499e-07, -2.04611221e-04, 3.68319496e-06, 5.01756031e-06, 6.52662104e-06, 8.68146600e-06, 1.27296403e-05, 4.66510263e-07, 3.45701103e-07, 2.58091563e-07, 2.62670435e-07, 5.28109289e-07, 5.83997817e-07, 5.80993810e-07, 8.49996602e-07, 8.61557638e-07, 1.28373680e-06]
    known_rgb_wavelet_embedding = [1.0149459838867188, 0.009396257810294628, 0.013252515345811844, 0.019200582057237625, 0.03169100731611252, 0.0597832128405571, 0.001177848200313747, 0.0008956858655437827, 0.0006961669423617423, 0.0007526640547439456, 0.001356705790385604, 0.0015081641031429172, 0.0014198219869285822, 0.002444450045004487, 0.0024045188911259174, 0.0046163457445800304, 1.1359686851501465, 0.009039577096700668, 0.01293253805488348, 0.019352681934833527, 0.03348885476589203, 0.06518232822418213, 0.0011238295119255781, 0.0008525265147909522, 0.000675839779432863, 0.0007351068779826164, 0.0013168096775189042, 0.0015156366862356663, 0.0014302206691354513, 0.002517778892070055, 0.0025533251464366913, 0.004978210665285587, 1.3184771537780762, 0.008817654103040695, 0.012831225991249084, 0.019689790904521942, 0.034744977951049805, 0.06812523305416107, 0.0010847593657672405, 0.0008324516820721328, 0.0006748223095200956, 0.0007393427076749504, 0.0013635988580062985, 0.001524904859252274, 0.0014678940642625093, 0.0025919084437191486, 0.0026492660399526358, 0.005247402470558882]
    known_grey_wavelet_embedding = [1.1207001209259033, 0.008954653516411781, 0.012728508561849594, 0.01901235617697239, 0.03277358412742615, 0.0636843740940094, 0.0011132140643894672, 0.0008454968919977546, 0.0006713253096677363, 0.0007336997659876943, 0.0013040185440331697, 0.0014946818118914962, 0.001397418323904276, 0.0024632513523101807, 0.002477526431903243, 0.004860419314354658]
    # fmt: on

    print(
        len(w2cv.jzazbz_wavelet_embedding),
        len(w2cv.rgb_wavelet_embedding),
        len(w2cv.grey_wavelet_embedding),
    )

    assert len(w2cv.jzazbz_wavelet_embedding) == len(
        known_jzazbz_wavelet_embedding
    )  # 48 (3 channels)
    assert len(w2cv.rgb_wavelet_embedding) == len(
        known_rgb_wavelet_embedding
    )  # 48 (3 channels)
    assert len(w2cv.grey_wavelet_embedding) == len(
        known_grey_wavelet_embedding
    )  # 16 (1 channel)

    assert arrays_within_n_percent(
        w2cv.jzazbz_wavelet_embedding, known_jzazbz_wavelet_embedding, 1
    )
    assert arrays_within_n_percent(
        w2cv.rgb_wavelet_embedding, known_rgb_wavelet_embedding, 1
    )
    assert arrays_within_n_percent(
        w2cv.grey_wavelet_embedding, known_grey_wavelet_embedding, 1
    )
