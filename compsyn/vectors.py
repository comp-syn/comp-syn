from compsyn import datahelper, analysis, visualisation
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import json


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
		self.colorgram_vector = img_analysis.compressed_img_dict[word]

		self.colorgram = PIL.Image.fromarray(self.colorgram_vector.astype(np.uint8))

	def print_word_color(self, size=30, color_magnitude=1.65):

		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])

		plt.text(0.35, 0.5, self.word, color=color_magnitude*pself.rgb_ratio, fontsize=size)
		ax.set_axis_off()
		plt.show()

	def save_json_to_disk(self):

		vector_properties = {}
		vector_properties['jzazbz_vector'] = self.jzazbz_vector
		vector_properties['jzazbz_dist'] = self.jzazbz_dist
		vector_properties['rgb_vector'] = self.rgb_vector
		vector_properties['rgb_dist'] = self.rgb_dist
		vector_properties['rgb_ratio'] = self.rgb_ratio
		vector_properties['colorgram_vector'] = self.colorgram_vector

		with open(self.word + '_vector_properties', 'w') as fp:
		    json.dump(vector_properties, fp)