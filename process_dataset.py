
import sys
import pandas

SPLIT = 90

def	main():
	if len(sys.argv) != 2:
		print("Error: python process_dataset.py *dataset*")
		return 1
	ds_path = sys.argv[1]
	try:
		df:pandas.DataFrame = pandas.read_csv(ds_path)
	except Exception as e:
		print(e)
		return 
	try:
		f_test = open("datasets/test_dataset.csv", "w")
		f_train = open("datasets/train_dataset.csv", "w")
	except:
		return print("Error log: can not create dataset files")
	slice:int = int(len(df) / 100) * SPLIT
	df_train = df.iloc[:slice, 1:]
	df_test = df.iloc[slice+1:, 1:]

	df_train.to_csv(f_train)
	df_test.to_csv(f_test)

if __name__ == "__main__":
	main()