
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

def	visualize(df : pd.DataFrame):
	print("to do")

def	main():
	if len(sys.argv) > 3 or len(sys.argv) < 2 or (len(sys.argv) == 3 and sys.argv[2] != "--visualize"):
		print("Error: python3 process_dataset.py *dataset* %--visualize%")
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
	f_test = open("test_dataset.csv", "w")
	f_train = open("train_dataset.csv", "w")
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
		visualize(df)

if __name__ == "__main__":
	main()