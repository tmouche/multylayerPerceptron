
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def	normalize(data, min_max) -> list:
	normalized_data = []
	np.array(normalized_data)
	x = 0
	for n in range(len(data)):
		normalized_line = []
		for x in range(len(min_max)):
			normalized_line.append((data[n][x] - min_max[x][0]) / (min_max[x][1] - min_max[x][0]))
		normalized_data.append(normalized_line)
	return normalized_data

def	visualize(label, data_raw):

	figure, axes = plt.subplots(2,3)

	# 1->histogramme: on fait la moyenne pour une feature de tous les B et tous les M et on affiche tout ca
	# 	1-radius
	# 	2-texture
	# 	3-perimeter
	# 	4-area
	# 	5-smoothness
	# 	6-compactness
	# 	7-concavity
	# 	8-concave points
	# 	9-symmetry
	# 	10-fractal dimension
	bar = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

	M_means = []
	B_means = []
	M_se = []
	B_se = []
	M_worth = []
	B_worth = []

	x = 0
	offset = len(bar)
	while x < len(bar):
		data_means = (np.transpose([row[x:x+1] for row in data_raw])[0])
		data_se = (np.transpose([row[x+offset:x+offset+1] for row in data_raw])[0])
		data_worth = (np.transpose([row[x+offset*2:x+offset*2+1] for row in data_raw])[0])
		temp_M_means = list([])
		temp_B_means = list([])
		temp_M_se = list([])
		temp_B_se = list([])
		temp_M_worth = list([])
		temp_B_worth = list([])
		for i in range(len(label)):
			if label[i] == 'M':
				temp_M_means.append(data_means[i])
				temp_M_se.append(data_se[i])
				temp_M_worth.append(data_worth[i])
			else:
				temp_B_means.append(data_means[i])
				temp_B_se.append(data_se[i])
				temp_B_worth.append(data_worth[i])
		M_means.append(np.mean(temp_M_means))
		B_means.append(np.mean(temp_B_means))
		M_se.append(np.mean(temp_M_se))
		B_se.append(np.mean(temp_B_se))
		M_worth.append(np.mean(temp_M_worth))
		B_worth.append(np.mean(temp_B_worth))
		x += 1
	
	M_peri = []
	B_peri = []
	M_area = []
	B_area = []
	M_concav = []
	B_concav = []
	M_concav_point = []
	B_concav_point = []
	M_sym = []
	B_sym = []
	M_frac_dim = []
	B_frac_dim = []

	for i in range(len(label)):
		if label[i] == 'M':
			M_peri.append(data_raw[i][3])
			M_area.append(data_raw[i][4])
			M_concav.append(data_raw[i][7])
			M_concav_point.append(data_raw[i][8])
			M_sym.append(data_raw[i][9])
			M_frac_dim.append(data_raw[i][10])
		else:
			B_peri.append(data_raw[i][3])
			B_area.append(data_raw[i][4])
			B_concav.append(data_raw[i][7])
			B_concav_point.append(data_raw[i][8])
			B_sym.append(data_raw[i][9])
			B_frac_dim.append(data_raw[i][10])

	bar_width = 0.25
	axes[0][0].bar(np.arange(len(bar)), M_means, width=bar_width, label="M")
	axes[0][0].bar(np.arange(len(bar)) + bar_width, B_means, width=bar_width, label="B")
	axes[0][0].set_xticks(np.arange(len(bar)) + bar_width)
	axes[0][0].set_xticklabels(bar)
	axes[0][0].set_xlabel("features")
	axes[0][0].set_ylabel("value")
	axes[0][0].set_title("Comparaison of normalized means of means features between M and B")
	axes[0][0].legend(["Malignant", "Benign"], loc="lower right")

	axes[0][1].bar(np.arange(len(bar)), M_se, width=bar_width, label="M")
	axes[0][1].bar(np.arange(len(bar)) + bar_width, B_se, width=bar_width, label="B")
	axes[0][1].set_xticks(np.arange(len(bar)) + bar_width)
	axes[0][1].set_xticklabels(bar)
	axes[0][1].set_xlabel("features")
	axes[0][1].set_ylabel("value")
	axes[0][1].set_title("Comparaison of normalized means of se features between M and B")
	axes[0][1].legend(["Malignant", "Benign"], loc="lower right")

	axes[0][2].bar(np.arange(len(bar)), M_worth, width=bar_width, label="M")
	axes[0][2].bar(np.arange(len(bar)) + bar_width, B_worth, width=bar_width, label="B")
	axes[0][2].set_xticks(np.arange(len(bar)) + bar_width)
	axes[0][2].set_xticklabels(bar)
	axes[0][2].set_xlabel("features")
	axes[0][2].set_ylabel("value")
	axes[0][2].set_title("Comparaison of normalized means of worth features between M and B")
	axes[0][2].legend(["Malignant", "Benign"], loc="lower right")

	axes[1][0].plot(M_peri, M_area, "ro", B_peri, B_area, "bo")
	axes[1][0].legend(["Malignant", "Benign"], loc="lower right")
	axes[1][0].set_title("Perimeter on area")

	axes[1][1].plot(M_concav, M_concav_point, "ro", B_concav, B_concav_point, "bo")
	axes[1][1].legend(["Malignant", "Benign"], loc="lower right")
	axes[1][1].set_title("Concativy on concave points")

	axes[1][2].plot(M_frac_dim, M_sym, "ro", B_frac_dim, B_sym, "bo")
	axes[1][2].legend(["Malignant", "Benign"], loc="lower right")
	axes[1][2].set_title("Fractal dimension on symmetry")

	plt.show()

def	main():
	if len(sys.argv) > 3 or len(sys.argv) < 2 or (len(sys.argv) == 3 and sys.argv[2] != "--visualize"):
		print("Error: python process_dataset.py *dataset* %--visualize%")
		return 1
	visu = True if len(sys.argv) == 3 else False
	ds_path = sys.argv[1]
	df = pd.read_csv(ds_path)
	data_raw = np.array(df.iloc[:,2:])
	label = np.array(df.iloc[:,1:2])
	min_max = []
	x = 0
	while x < len(data_raw[0]):
		base = np.transpose(data_raw[:, x:x+1])
		min = np.nanmin(base)
		max = np.nanmax(base)
		min_max.append([min, max])
		x += 1
	normalized = normalize(data_raw, min_max)
	try:
		f_test = open("test_dataset.csv", "w")
		f_train = open("train_dataset.csv", "w")
	except:
		return print("Error log: can not create dataset files")
	for x in range(len(label)):
		rand = np.random.randint(10)
		f = f_train if rand < 8 else f_test
		f.write(label[x][0])
		for n in normalized[x]:
			f.write(f",{n}")
		f.write("\n")
	f_test.close()
	f_train.close()
	if visu == True:
		visualize(label, normalized)

if __name__ == "__main__":
	main()