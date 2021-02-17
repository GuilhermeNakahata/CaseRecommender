from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.utils.cross_validation import CrossValidation

db = 'u.data'
folds_path = ''

metrics = ('PREC', 'RECALL', 'NDCG', 'MAP')

recommender = ItemKNN()
CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute(metrics = metrics)

recommender = MostPopular()
CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute(metrics = metrics)
