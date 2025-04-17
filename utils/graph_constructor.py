# coding=utf-8
import random
from datasets import load_from_disk
import pandas as pd
import networkx as nx

data = load_from_disk('/root/FLARE/processed_dataset_rec')
print(data.column_names)

temp_data = pd.DataFrame(data)
with open("/data/rate.txt", "w") as file:
    for _,row in temp_data.iterrows():
        user = int(row['user'])
        job_id = int(row['job_id'])
        label = int(row['label'])

        file.write(f"{user} {job_id} {label}\n")

split_data = data.train_test_split(test_size=0.2, seed=2025)
train_data = split_data['train'].to_pandas()
test_data = split_data['test'].to_pandas()
G = nx.Graph()

def add_edges(G,data, is_train=True):
    """
    构建图的边，包括用户和岗位的属性边，训练集中的用户-岗位交互边，以及辅助边（岗位共现边和用户共现边）。
    Args:
        data: 数据集（训练集或测试集）
        is_train: 是否为训练集（训练集需要处理 label=1 的 apply 边）
    """

    # 创建岗位和用户的共现字典
    job_co_occurrence = {}
    user_co_occurrence = {}

    for _, row in data.iterrows():
        job_id = row['job_id']
        user = row['user']

        # 为岗位添加属性边
        for col in [
            'first_level',
            'second_level',
            'third_level',
            'salary_floor',
            'salary_ceiling',
            'place',
            'job_experience',
            'job_education_requirement',
            '企业行业一级类别',
            '企业行业二级类别',
            '企业行业三级类别'
        ]:

            attribute_entity = row[col]
            G.add_edge(job_id, attribute_entity, relation=f'job_id_{col}')

        # 为用户添加属性边
        for col in ['gender','school', 'major', 'edu_background']:
            attribute_entity = row[col]
            G.add_edge(user, attribute_entity, relation=f'user_{col}')

        # 仅在训练集中为 label=1 的用户和岗位添加 apply 边
        if is_train and row['label'] == '1':
            G.add_edge(user, job_id, relation='apply')

            # 用户到岗位的共现
    #         if user not in user_co_occurrence:
    #             user_co_occurrence[user] = set()
    #         user_co_occurrence[user].add(job_id)
    #
    #         # 岗位到用户的共现
    #         if job_id not in job_co_occurrence:
    #             job_co_occurrence[job_id] = set()
    #         job_co_occurrence[job_id].add(user)
    #
    # # 添加岗位共现边
    # for job_id, users in job_co_occurrence.items():
    #     user_list = list(users)
    #     for i in range(len(user_list)):
    #         for j in range(i + 1, len(user_list)):
    #             G.add_edge(user_list[i], user_list[j], relation='job_co_occurrence')
    #
    # # 添加用户共现边
    # for user, jobs in user_co_occurrence.items():
    #     job_list = list(jobs)
    #     for i in range(len(job_list)):
    #         for j in range(i + 1, len(job_list)):
    #             G.add_edge(job_list[i], job_list[j], relation='user_co_occurrence')


# 构建训练集和测试集的边
add_edges(G,train_data, is_train=True)
add_edges(G,test_data, is_train=False)

 # 获取边的列表并映射关系为数字
relations_mapping = {}
edges_list = list(G.edges(data=True))
for index, edge in enumerate(edges_list):
    start_node = edge[0]
    end_node = edge[1]
    relation = edge[2]['relation']

    # 将关系映射为数字
    if relation not in relations_mapping:
        relations_mapping[relation] = len(relations_mapping)

    edges_list[index] = (start_node, end_node, relations_mapping[relation])
random.shuffle(edges_list)

with open("/data/kg_final.txt", "w") as file:
    for edge in edges_list:
        start_node = edge[0]
        end_node = edge[1]
        relation = edge[2]
        file.write(f"{start_node} {relation} {end_node}\n")


