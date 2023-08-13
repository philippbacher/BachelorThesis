from collections import defaultdict
from itertools import product
from surprise import Dataset, Reader, SVD, KNNBasic, KNNWithMeans, NMF
from surprise.model_selection import GridSearchCV, cross_validate, train_test_split, KFold
from surprise import accuracy
from dbconnect import get_connection
import pandas
import numpy
import gc
from CustomGridSearchCV import CustomGridSearchCV, GridSearchResult

def get_data():
    cursor = get_connection()
    print('Fetching data...')
    cursor.execute('''select users.user_id, track_id,round((( count(*) - playcountmintrack )::numeric / ( case when playcountmaxtrack - playcountmintrack = 0 then 1 else playcountmaxtrack - playcountmintrack end ) )* ( 1000 - 1) + 1 , 2)::float as rating, users.usergroup, users.isbyms
                    from users join events on users.user_id = events.user_id
                    group by users.user_id, track_id''')
    print('Data fetched...')
    return cursor.fetchall()

def filter_data(df, beyms=None, group=None):
    if beyms != None:
        df = df[df['isbyms'] == beyms]
    if group != None and beyms == 1:
        df = df[df['usergroup'] == group]
    
    return df[['user_id', 'item_id', 'rating']]

def print_current_eval(model, byms, group):
    str_byms = model + ' for beyond mainstrem listeners' + ' in usergroup ' + str(group) if byms == 1 else model + ' for mainstream listeners' 
    print(str_byms)

def evaluate_svd(dataset):
    trainset, testset = train_test_split(dataset, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    accuracy.rmse(predictions)

def evaluate_KNNBasic(useritem_dataset):

    param_grid = {
         'k': [10,20,40,70,100,160,200],           # Number of neighbors to consider
         'sim_options': {
             'name': ['cosine', 'pearson', 'msd'],  # Similarity measure to use
             'user_based': [True]     # User-based or item-based filtering
         }
    }

    # param_grid = {
    #     'k': [40],
    #     'sim_options': {
    #         'name': ['cosine']
    #     }
    # }
    print("Starting KNNBasic hyperparametertuning...")
    gs = CustomGridSearchCV(KNNBasic, param_grid, measures= ["rmse", "mae"], cv=5, n_jobs=-1)
    gs.fit(useritem_dataset)
    
    print("Best results for each group:")
    for gsr in gs.gridsearch_results:
        print(gsr.group)
        print(gsr.best_score["mae"])
        print(gsr.best_params["mae"])
        print(gsr.best_score["rmse"])
        print(gsr.best_params["rmse"])

    # print("\nAll Results:")
    # for params, rmse, mae in zip(gs.cv_results['params'], gs.cv_results['mean_test_rmse'], gs.cv_results['mean_test_mae']):
    #     print(params)
    #     print("RMSE:", rmse)
    #     print("MAE:", mae)
    #     print("-" * 40)

    #GridSearchCV doesn't work here because it uses way to much memory, so I decided to just do it manually:
    #combinations = product([10,20,30,40], ['cosine','pearson'], [True, False])
    #paramResults = list()
    #cross product of all parameters, because they could influence each other so it isn't enough to just test each once
    # for k, name, user_based in combinations:
    #     print(k, name, user_based)
    #     gc.collect()
    #     model = KNNBasic(k, sim_options={'name': name, 'user_based': user_based})
    #     result = cross_validate(model, useritem_dataset, measures=["rmse", "mae"], cv=5, verbose=False)
    #     mean_rmse = result['test_rmse'].mean()
    #     mean_mae = result['test_mae'].mean()

    #     result = {
    #             'Algorithm': 'KNNBasic',
    #             'mae': mean_mae,
    #             'rmse': mean_rmse,
    #             'k': k,
    #             'sim_options':{
    #                 'name': name,
    #                 'user_based': user_based
    #             }
    #         }

    #     print(result)
    #     paramResults.append(result)

    # trainset, testset = train_test_split(dataset, test_size=0.2)
    # model = KNNBasic(k=40, sim_options={'user-based': True})
    # model.fit(trainset)
    # predictions = model.test(testset)
    # accuracy.rmse(predictions)

def evaluate_KNNWithMeans(dataset):
    trainset, testset = train_test_split(dataset, test_size=0.2)
    model = KNNWithMeans(k=40, sim_options={'user-based': True, 'name': 'cosine'})
    model.fit(trainset)
    predictions = model.test(testset)
    accuracy.rmse(predictions)

def evaluate_nmf(dataset):
    param_grid = {'n_factors': [10, 20, 30],
            'n_epochs': [10, 50,90,150,200,250],
            'biased': [True, False]
            }
    
    print("Starting NMF hyperparametertuning...")
    gs = GridSearchCV(NMF, param_grid, measures= ["rmse", "mae"], cv=5, n_jobs=-1)
    gs.fit(dataset)

    print(gs.best_score["mae"])
    print(gs.best_params["mae"])
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])

    print("\nAll Results:")
    for params, rmse, mae in zip(gs.cv_results['params'], gs.cv_results['mean_test_rmse'], gs.cv_results['mean_test_mae']):
        print(params)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("-" * 40)

def eval_results(model, predictions):
    absolute_errors = defaultdict(list)
    for user_id, item_id, r, r_, details in predictions:
        ae = numpy.abs(r - r_)
        absolute_errors["all"].append(ae)
    print(numpy.mean(absolute_errors["all"]))
    

data = get_data()
raw_data = pandas.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'usergroup', 'isbyms' ])
# = Reader(line_format='user item rating', sep=",", rating_scale=(1,1000))

reader = Reader(rating_scale=(1,1000))

# for i in (True, False): # this is for beyms/ms
#     for j in range(4): # this is for the groups
#         #only one iteration for ms because there are no groups
#         if i == 0 and j == 1:
#             break
#         filtered_data = filter_data(df, i,j)
#         dataset = Dataset.load_from_df(filtered_data, reader=reader)
#         print_current_eval('NMF', i, j)
#         evaluate_nmf(dataset)
#         print_current_eval('SVD', i, j)
#         evaluate_svd(dataset)
#         print_current_eval('KNNBasic', i, j)
#         evaluate_KNNBasic(dataset)
#         print_current_eval('KNNWithMeans', i, j)
#         evaluate_KNNWithMeans(dataset)


filtered_data = filter_data(raw_data)

#just for testing to make it faster
#filtered_data = filtered_data.head(10000)

useritem_dataset = Dataset.load_from_df(filtered_data, reader=reader)
# folds_it = KFold(n_splits=5).split(dataset)

evaluate_KNNBasic(raw_data)
#evaluate_nmf(useritem_dataset)

# for f, data in enumerate(folds_it):
#     print("nmf:" + str(f))
#     evaluate_nmf(data, None)

