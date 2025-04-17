import torch
def getEmbedding():

    user = torch.load('/data/emb/userid_to_embedding.pt')
    job = torch.load('/data/emb/jobid_to_embedding.pt')

    user_entitys = max(user.keys()) + 1
    job_entitys = max(job.keys()) - max(user.keys())
    dim = 3584
    print(user_entitys, job_entitys)
    print(max(user.keys()))
    user_tensor = torch.zeros((user_entitys, dim))  # Ĭ��ֵΪ������
    job_tensor = torch.zeros((job_entitys, dim))

    for idx, vec in user.items():
        user_tensor[idx] = vec.squeeze()  # ȥ�������ά��

    for idx, vec in job.items():
        mapped_idx = idx - user_entitys  # ��ְλ�ڵ� ID ӳ�䵽�� 0 ��ʼ������
        job_tensor[mapped_idx] = vec.squeeze()  # ȥ�������ά��

    return user_tensor,job_tensor

