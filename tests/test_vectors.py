from compsyn import datahelper, vectors
from pytest import approx
import os

def test_vector():
	"""
	creates vector object for the saved love image set and tests distributions and ratios.
	TODO: load generic image and test instead for downloaded images
		   make path more elegant
	"""

	path = os.getcwd() + "/downloads/paper_categories"
	vec = vectors.Vector("love", path)

	expected_rgb_dist = [1.40895633e-07, 7.41199147e-09, 8.63679137e-11, 5.33250120e-09, 1.23785039e-07, 1.61645538e-08, 2.79873735e-08, 1.60805576e-07]
	expected_jzazbz_dist = [ 43.70086125,  82.88902835,  20.36144642, 207.55729541, 11.45945804, 131.10485973,  10.49319587, 334.18735373]
	expected_rgb_ratio = [0.43899919, 0.27914204, 0.28185877]

	assert expected_rgb_dist[0] == approx(vec.rgb_dist[0], rel=1e-6, abs=1e-12)
	assert expected_jzazbz_dist[0] == approx(vec.jzazbz_dist[0], rel=1e-6, abs=1e-12)
	assert expected_rgb_ratio[0] == approx(vec.rgb_ratio[0], rel=1e-6, abs=1e-12)
