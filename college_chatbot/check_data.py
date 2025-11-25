import os

DATA_PATH = "data"

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        path = os.path.join(DATA_PATH, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"{file}: {len(text)} characters")
        print(text[:150], "\n" + "-"*50)