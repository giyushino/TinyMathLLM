from datasets import load_dataset

dataset = load_dataset("json", data_files = r"C:\Users\allan\nvim\TinyMath\TinyMathLLM\datasets\dataset_extract.json")

for i in range(5):
    print(dataset["eval"][i])

