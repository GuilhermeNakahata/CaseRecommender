from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.most_popular import MostPopular
import numpy as np

train_file = "train1.dat"
test_file = "test1.dat"

train_file2 = "train2.dat"
test_file2 = "test2.dat"

train_file3 = "train3.dat"
test_file3 = "test3.dat"

train_file4 = "train4.dat"
test_file4 = "test4.dat"

train_file5 = "train5.dat"
test_file5 = "test5.dat"

metrics = ('MAE', 'RMSE')

ItemKNN(train_file, test_file).compute(metrics = metrics)
ItemKNN(train_file2, test_file2).compute(metrics = metrics)
ItemKNN(train_file3, test_file3).compute(metrics = metrics)
ItemKNN(train_file4, test_file4).compute(metrics = metrics)
ItemKNN(train_file5, test_file5).compute(metrics = metrics)

MostPopular(train_file, test_file).compute(metrics = metrics)
MostPopular(train_file2, test_file2).compute(metrics = metrics)
MostPopular(train_file3, test_file3).compute(metrics = metrics)
MostPopular(train_file4, test_file4).compute(metrics = metrics)
MostPopular(train_file5, test_file5).compute(metrics = metrics)

MAE = [0,0,0,0,0]
RMSE = [1,1,1,1,1]

dPadrao = np.std(MAE)
media = np.mean(MAE)

print(dPadrao)
print(media)