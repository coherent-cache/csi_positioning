### This is a brief demonstration for importing 5G CSI dataset from .mat file format
import h5py

data = h5py.File("dataset/dataset_SNR50_outdoor.mat")

features = data["features"]
print("The size of features")
print(features.shape)

position = data["labels"]["position"]
print("The size of labels")
print(position.shape)
