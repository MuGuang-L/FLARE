import collections
import numpy as np

def construct_kg(kgTriples):
    print('生成知识图谱索引图')
    kg = dict()
    for triple in kgTriples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg

# 根据用户物品交互标注三元组得到用户的历史正例
def getUserHistoryPosDict(train_set):
    user_history_pos_dict = dict()

    for u, i, r in train_set:
        if r == 1:
            if str(u) not in user_history_pos_dict:
                user_history_pos_dict[str(u)] = []
            user_history_pos_dict[str(u)].append(i)
    return user_history_pos_dict

def getKgIndexsFromKgTriples( kg_triples ):
    kg_indexs = collections.defaultdict( list )
    for h, r, t in kg_triples:
        kg_indexs[ int( h ) ].append([ int( t ), int( r ) ])
        kg_indexs[int(t)].append([int(h), int(r)])
    return kg_indexs

# 过滤掉无正例的用户
def filetDateSet( dataSet, user_pos ):
    return [i for i in dataSet if str(i[0]) in user_pos]

# 根据kg邻接列表，得到实体邻接列表和关系邻接列表
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)

def construct_adj(neighbor_sample_size, kg_indexes, entity_num):
    logging.info('生成实体邻接列表和关系邻接列表')

    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)

    for entity in range(entity_num):
        neighbors = kg_indexes.get(int(entity), [])
        n_neighbors = len(neighbors)

        # 添加调试信息，打印出 n_neighbors 为 0 的实体
        if n_neighbors == 0:
            logging.warning(f"实体 {entity} 的邻居数量为 0，neighbors: {neighbors}")
            # 可选：抛出异常以中断执行，方便定位
            # raise ValueError(f"实体 {entity} 的邻居数量为 0")

        if n_neighbors >= neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)),
                                               size=neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)),
                                               size=neighbor_sample_size, replace=True)

        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation


