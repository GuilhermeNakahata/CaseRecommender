from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.most_popular import MostPopular
from caserec.utils.cross_validation import CrossValidation
import numpy as np

db = 'u.data'
folds_path = ''



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

ItemKNN(train_file, test_file).compute()
ItemKNN(train_file2, test_file2).compute()
ItemKNN(train_file3, test_file3).compute()
ItemKNN(train_file4, test_file4).compute()
ItemKNN(train_file5, test_file5).compute()

recommender = ItemKNN()
CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

Precision5 = [0.14322,0.142149,0.146563,0.144531,0.145574]
Precision10 = [0.121058,0.120803,0.125278,0.121569,0.123057]
Recall5 = [0.056669,0.059846,0.059505,0.060537,0.056998]
Recall10 = [0.093983,0.097995,0.1025,0.099444,0.094651]
Map5 = [0.28052,0.278123,0.288626,0.283141,0.279541]
Map10 = [0.272674,0.272406,0.279898,0.277427,0.273422]
NDCG5 = [0.350294,0.348877,0.364232,0.35496,0.354235]
NDCG10 = [0.366323,0.371003,0.380844,0.375039,0.371405]

MAE = [0.800444,0.79628,0.808652,0.802582,0.805707]
RMSE = [1.043307,1.034178,1.052225,1.042158,1.050752]

desviopadrao = np.std(MAE)
media = np.mean(MAE)

print(desviopadrao)
print(media)
