from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.most_popular import MostPopular
from caserec.utils.cross_validation import CrossValidation

db = 'u.data'
folds_path = ''

recommender = ItemKNN()
CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

recommender = MostPopular()
CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()
