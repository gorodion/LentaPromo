import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA_DIR = 'data'


def load_checks(data_dir):
    checks = pd.read_csv(os.path.join(data_dir, '20210518_checks.csv'), parse_dates=['day', 'time']) 
    checks = checks.rename({'selling_price': 'sales', 'supplier_price': 'supplier_sales'}, axis=1)
    checks['retail_price'] = checks['sales'] / checks['num_sales']
    checks['supplier_price'] = checks['supplier_sales'] / checks['num_sales']
    return checks


def generate_sku_features(data_dir):
    checks = load_checks(data_dir=data_dir)
    
    sales = checks.groupby(['sku', 'day']).agg({'sales': 'sum'})
    sku_importance = sales.groupby('sku').agg({'sales': 'sum'}).rank(ascending=False)
    sku_importance = sku_importance.rename({'sales': 'sku_rank'}, axis=1)
    return sku_importance

    
def load_target(data_dir):
    uplift = pd.read_csv(os.path.join(data_dir, '20210518_uplift.csv'))
    return uplift


def load_test_objects(data_dir):
    submission = pd.read_csv(os.path.join(data_dir, '20210521_sample_submission.csv'))
    submission['order'] = range(submission.shape[0])
    return submission


def load_hierarchy(data_dir):
    hierarchy = pd.read_csv(os.path.join(data_dir, '20210518_hierarchy.csv'))
    return hierarchy


def load_objects(data_dir):
    offers = pd.read_csv(os.path.join(data_dir, '20210521_offers.csv'), parse_dates=['start_date', 'end_date'])
    return offers
                 
    
def objects_features_aggregated(objects, sku_features):
    objects = objects.merge(sku_features, on='sku', how='left')

    objects = objects.groupby('Offer_ID').agg({
        'Promo_type': 'first',
        'sku': list,
        'start_date': 'first',
        'end_date': 'first',
        'train_test_group': 'first',
        'sku_rank': list,
    })
    objects['weekday'] = objects.start_date.apply(lambda x: x.weekday())
    objects['length'] = (objects.end_date - objects.start_date).apply(lambda x: 1 + x.days)
    objects['sku_count'] = objects['sku'].apply(len)
    
    rank_features = []
    for offer_index, offer in objects.iterrows():
        sku_list = list(zip(offer['sku_rank'], offer['sku']))
        best_sku_rank, best_sku = min(sku_list)
        worst_sku_rank, _ = max(sku_list)
        sku_rank_sum = sum([1.0 / rank for rank in offer['sku_rank']])
        rank_features.append({
            'best_sku_rank': best_sku_rank,
            'worst_sku_rank': worst_sku_rank,
            'best_sku': best_sku,
            'sku_rank_sum': sku_rank_sum
        })

    rank_features = pd.DataFrame(rank_features, index=objects.index)
    objects = pd.concat([objects, rank_features], axis=1)
    
    return objects

                         
def build_dataset(data_dir=DATA_DIR):
    objects = load_objects(data_dir)
    hierarchy = load_hierarchy(data_dir)
    sku_features = generate_sku_features(data_dir)
    target = load_target(data_dir)
    test_objects = load_test_objects(data_dir)
    
    dataset = objects_features_aggregated(objects, sku_features)
    
    return dataset, test_objects, target
    
    
def fit_predict(dataset, target, test_objects, target_column, object_column, features, categorical_features):
    dataset[categorical_features] = dataset[categorical_features].astype('category')

    train_Xy = dataset.merge(target, on=object_column)
    test_Xy = dataset.merge(test_objects, on=object_column)

    model = CatBoostRegressor(loss_function='MAE', random_seed=0)
    model.fit(train_Xy[features], train_Xy[target_column], cat_features=categorical_features, metric_period=100)
    test_Xy[target_column] = model.predict(test_Xy[features])

    submission_predict = test_Xy.sort_values('order')[[object_column, target_column]]
    return submission_predict, model


def predict_post_processing(submission, target_column='UpLift'):
    submission[target_column] = np.clip(submission[target_column], 0.5, None)
    return submission