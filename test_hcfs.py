from src.config import DATASET_PATH
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.hcfs_builder import HCFSBuilder

loader = DataLoader(DATASET_PATH)
datasets = loader.load_datasets()

preprocessor = Preprocessor()

cleaned = {}

for name, df in datasets.items():

    if df.shape[1] != 24:
        continue

    cleaned[name] = preprocessor.clean_dataset(df)

hcfs = HCFSBuilder()

ranking = hcfs.build({
    "ant": cleaned["ant-1.7"]
})

print(ranking)

print("\nTop 10 Features")
print(hcfs.get_top_features(
    {"ant": cleaned["ant-1.7"]},
    k=10
))