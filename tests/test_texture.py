from __future__ import annotations

from pathlib import Path

import pytest

from compsyn.config import CompsynConfig
from compsyn.vectors import WordToColorVector
from compsyn.trial import Trial


@pytest.mark.integration
def test_get_wavelet_embedding() -> None:

    CompsynConfig(work_dir=Path(__file__).parent.joinpath("test-assets"),)
    trial = Trial(
        experiment_name="test-downloads",
        trial_id="known-dist",
        hostname="pytester",
        trial_timestamp="testoclock",
    )

    w2cv = WordToColorVector(label="earth", trial=trial)
    w2cv.load_data()
    w2cv.run_analysis()

    assert [round(col.item(), 9) for col in w2cv.jzazbz_wavelet_embedding] == [
        0.000764322,
        5.939e-06,
        8.541e-06,
        1.2832e-05,
        2.2209e-05,
        4.3232e-05,
        7.16e-07,
        5.23e-07,
        4.1e-07,
        5.02e-07,
        6.85e-07,
        5.3e-07,
        4.15e-07,
        4.76e-07,
        7.47e-07,
        5.65e-07,
        4.43e-07,
        5.06e-07,
        8.12e-07,
        6.42e-07,
        5.38e-07,
        5.09e-07,
        9.76e-07,
        7.97e-07,
        8.9e-07,
        1.003e-06,
        8.2e-07,
        8.72e-07,
        1.073e-06,
        9.01e-07,
        9.2e-07,
        1.188e-06,
        1.02e-06,
        8.6e-07,
        1.534e-06,
        1.636e-06,
        1.483e-06,
        1.718e-06,
        1.747e-06,
        1.76e-06,
        1.988e-06,
        1.546e-06,
        3.321e-06,
        3.392e-06,
        3.489e-06,
        3.034e-06,
        -5.0918e-05,
        1.934e-06,
        2.287e-06,
        2.912e-06,
        3.672e-06,
        4.536e-06,
        2.27e-07,
        1.6e-07,
        1.13e-07,
        9.4e-08,
        2.29e-07,
        1.69e-07,
        1.16e-07,
        9e-08,
        2.33e-07,
        1.61e-07,
        1.15e-07,
        9.5e-08,
        2.7e-07,
        2.11e-07,
        1.45e-07,
        1.12e-07,
        2.93e-07,
        2.29e-07,
        1.88e-07,
        2.81e-07,
        2.17e-07,
        1.65e-07,
        2.82e-07,
        2.23e-07,
        1.87e-07,
        3.45e-07,
        2.63e-07,
        2.12e-07,
        3.77e-07,
        3.3e-07,
        3.36e-07,
        2.65e-07,
        3.95e-07,
        3.49e-07,
        4.18e-07,
        4.14e-07,
        5.49e-07,
        4.8e-07,
        5.45e-07,
        6.81e-07,
        -0.000204587,
        3.907e-06,
        5.216e-06,
        6.603e-06,
        8.695e-06,
        1.2734e-05,
        4.75e-07,
        3.34e-07,
        2.51e-07,
        2.73e-07,
        4.73e-07,
        3.62e-07,
        2.77e-07,
        2.84e-07,
        4.99e-07,
        3.74e-07,
        2.78e-07,
        2.85e-07,
        5.3e-07,
        3.98e-07,
        3.17e-07,
        3.16e-07,
        6.26e-07,
        4.87e-07,
        5.04e-07,
        6.48e-07,
        5.28e-07,
        5.12e-07,
        6.9e-07,
        5.33e-07,
        5.43e-07,
        7.21e-07,
        6.1e-07,
        5.96e-07,
        8.47e-07,
        8.03e-07,
        8.09e-07,
        7.03e-07,
        9.31e-07,
        8.49e-07,
        1.017e-06,
        9.47e-07,
        1.251e-06,
        9.76e-07,
        1.46e-06,
        1.466e-06,
    ]
    assert [round(col.item(), 9) for col in w2cv.rgb_wavelet_embedding] == [
        1.014945984,
        0.009396258,
        0.013252515,
        0.019200582,
        0.031691007,
        0.059783213,
        0.001140333,
        0.000836609,
        0.000639198,
        0.000745256,
        0.00110587,
        0.000837386,
        0.000641303,
        0.000737013,
        0.001194611,
        0.000895079,
        0.000690989,
        0.000761543,
        0.001270579,
        0.001013669,
        0.000813177,
        0.000766844,
        0.001549269,
        0.001226852,
        0.001260856,
        0.001572235,
        0.001285431,
        0.00129194,
        0.001672111,
        0.001386931,
        0.001365271,
        0.001807749,
        0.001498375,
        0.001221747,
        0.002351799,
        0.002238571,
        0.002247381,
        0.002341452,
        0.002617436,
        0.002569995,
        0.002880126,
        0.002149119,
        0.004468621,
        0.004596817,
        0.005034585,
        0.00436536,
        1.135968685,
        0.009039577,
        0.012932538,
        0.019352682,
        0.033488855,
        0.065182328,
        0.001082545,
        0.000781028,
        0.000607226,
        0.000741217,
        0.001050415,
        0.000815197,
        0.000626508,
        0.000704953,
        0.001127835,
        0.000847677,
        0.000662576,
        0.000747948,
        0.001234524,
        0.000966204,
        0.00080705,
        0.00074631,
        0.001459279,
        0.001198518,
        0.001340469,
        0.001520763,
        0.001230105,
        0.001305779,
        0.001618752,
        0.001359622,
        0.001379338,
        0.001797546,
        0.001552639,
        0.001287858,
        0.002309467,
        0.002458176,
        0.002221298,
        0.002577471,
        0.002666234,
        0.002637238,
        0.003057637,
        0.002356894,
        0.004989001,
        0.005077529,
        0.005261533,
        0.00458478,
        1.318477154,
        0.008817654,
        0.012831226,
        0.019689791,
        0.034744978,
        0.068125233,
        0.001062313,
        0.000790347,
        0.000611719,
        0.000745692,
        0.000988547,
        0.000753743,
        0.000588729,
        0.000679672,
        0.001089166,
        0.000832044,
        0.000667845,
        0.000757916,
        0.001199011,
        0.000953672,
        0.000830996,
        0.000774091,
        0.001556255,
        0.001252134,
        0.001388045,
        0.001505357,
        0.001235502,
        0.001300348,
        0.001608279,
        0.001369383,
        0.001410504,
        0.001802948,
        0.001628201,
        0.001368636,
        0.002376638,
        0.002660308,
        0.002261992,
        0.002738453,
        0.002596188,
        0.0027345,
        0.003029612,
        0.002567008,
        0.005395136,
        0.00533828,
        0.005356061,
        0.004900134,
    ]
    assert [round(col.item(), 9) for col in w2cv.grey_wavelet_embedding] == [
        1.120700121,
        0.008954654,
        0.012728509,
        0.019012356,
        0.032773584,
        0.063684374,
        0.001075433,
        0.000782494,
        0.00060924,
        0.0007363,
        0.001035572,
        0.000795929,
        0.000618756,
        0.000700899,
        0.001125535,
        0.000845621,
        0.00066069,
        0.000745989,
        0.001216316,
        0.000957944,
        0.000796614,
        0.00075161,
        0.001452177,
        0.001177006,
        0.001313262,
        0.001492847,
        0.001223507,
        0.00130005,
        0.001597925,
        0.001337112,
        0.001363384,
        0.001767491,
        0.001501109,
        0.001258604,
        0.002267609,
        0.002399141,
        0.002197139,
        0.002518061,
        0.002588583,
        0.002597672,
        0.00292962,
        0.002265286,
        0.004858273,
        0.004969786,
        0.005145674,
        0.004467945,
    ]
