from compsyn import datahelper, analysis, visualisation
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

class Vector():
	def __init__(self, word, path):

		self.word = word 

		img_object = datahelper.ImageData()
		img_object.load_image_dict_from_folder(os.path.join(path, word))
		img_analysis = analysis.ImageAnalysis(img_object)
		img_analysis.compute_color_distributions(word, ["jzazbz", "rgb"])
		img_analysis.get_composite_image()

		self.jzazbz_vector = np.mean(img_analysis.jzazbz_dict[word], axis=0)
		self.jzazbz_dist = np.mean(img_analysis.jzazbz_dist_dict[word], axis=0)

		self.rgb_vector = np.mean(img_analysis.rgb_dict[word], axis=0)
		self.rgb_dist = np.mean(img_analysis.rgb_dist_dict[word], axis=0)
		self.rgb_ratio = np.mean(img_analysis.rgb_ratio_dict[word], axis=0)

		self.colorgram = PIL.Image.fromarray(img_analysis.compressed_img_dict[word].astype(np.uint8))

	def print_word_color(self, size=30, color_magnitude=1.65):

		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])

		plt.text(0.35, 0.5, self.word, color=color_magnitude*pself.rgb_ratio, fontsize=size)
		ax.set_axis_off()
		plt.show()