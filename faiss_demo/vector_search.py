"""
Faiss 向量检索核心模块
提供数据生成、索引创建和查询功能
@author thomasliu319
"""
import faiss
import numpy as np


def generate_data(num_points, dimension, seed=None):
    """
    生成随机向量数据

    :param num_points: 向量数量
    :param dimension: 向量维度
    :param seed: 随机种子（可选，用于结果复现）
    :return: 形状为 (num_points, dimension) 的 float32 数组
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.random((num_points, dimension)).astype('float32')


def build_index(data, dimension):
    """
    创建基于 L2 距离的 Faiss 平面索引

    :param data: 待索引的数据集，形状为 (n, dimension)
    :param dimension: 向量维度
    :return: 构建好的 Faiss 索引对象
    """
    # 基于 L2 距离（欧几里得距离）的平面索引
    index = faiss.IndexFlatL2(dimension)
    # 将数据添加到索引中
    index.add(data)
    return index


def search(index, query_vectors, k):
    """
    在索引中查询最近邻向量

    :param index: Faiss 索引对象
    :param query_vectors: 查询向量，形状为 (num_queries, dimension)
    :param k: 返回前 k 个最近邻
    :return: (distances, indices) 距离数组和索引数组
    """
    distances, indices = index.search(query_vectors, k)
    return distances, indices
