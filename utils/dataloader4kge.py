from tqdm import tqdm
from datasets import Dataset
from utils import osUtils as ou, osUtils
# from data import filepaths as fp

import random

def readKGData( path = '/data/kg_final.txt' ):
    print('读取知识图谱数据...')
    entity_set = set( )
    relation_set = set( )
    triples = [ ]
    for h, r, t in ou.readTriple( path ):
        entity_set.add( int( h ) )
        entity_set.add( int( t ) )
        relation_set.add( int ( r ) )
        triples.append( [ int( h ), int(r), int( t ) ] )
        triples.append( [ int( t ), int(r), int( h ) ] )
    return list( entity_set ), list( relation_set ), triples

# 定义 readRecData 函数
def readRecData(path='/data/rate.txt', train_ratio=0.8):
    print('读取用户评分三元组...')
    print(len(path))
    user_set, item_set = set(), set()
    triples = []

    # 读取评分三元组
    for u, i, r in tqdm(osUtils.readTriple(path)):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u), int(i), int(r)))

    # 将 triples 转换为 Dataset 对象
    dataset = Dataset.from_dict({
        "user": [t[0] for t in triples],
        "item": [t[1] for t in triples],
        "rating": [t[2] for t in triples]
    })

    # 使用 train_test_split 进行数据集划分，test_size 为 0.2，seed 为 2025
    test_size = 1 - train_ratio
    split_datasets = dataset.train_test_split(test_size=test_size, seed=2025)

    # 获取训练集和测试集
    train_set = list(
        zip(split_datasets['train']['user'], split_datasets['train']['item'], split_datasets['train']['rating']))
    test_set = list(
        zip(split_datasets['test']['user'], split_datasets['test']['item'], split_datasets['test']['rating']))

    # 返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set), list(item_set), train_set, test_set

