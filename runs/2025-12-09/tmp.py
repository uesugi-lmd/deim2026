import numpy as np

f = np.load("./cirr_features/cirr_train_image_features.npz")

print(f.files)              # ['names', 'embs'] のはず
print(f["names"][:10])      # 最初の10個を確認
print(len(f["names"]))      # 数も確認