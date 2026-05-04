import numpy as np
from typing import List, Dict
import torch
import torch.nn as nn
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import kendalltau
# from sklearn.decomposition import PCA

class MLPSimilarityAnalyzer:
    def __init__(self, mlp_list: List[nn.Module]):
        """
        初始化分析器
        Args:
            mlp_list: MLP网络列表，每个MLP应有相同的结构
        """
        self.mlp_list = mlp_list
        self.n_models = len(mlp_list)
        
    def extract_parameters(self) -> Dict[str, np.ndarray]:
        """提取所有模型的参数到字典"""
        params_dict = {
            'W1': [],
            'W2': [],
            'b1': [],
            'b2': []
        }
        
        for mlp in self.mlp_list:
            # 假设MLP结构：Linear1 -> ReLU -> Linear2
            # params_dict['W1'].append(mlp[0].weight.detach().numpy().flatten())
            # params_dict['b1'].append(mlp[0].bias.detach().numpy().flatten())
            # params_dict['W2'].append(mlp[2].weight.detach().numpy().flatten())
            # params_dict['b2'].append(mlp[2].bias.detach().numpy().flatten())

            num_heads = 8
            head_dim = 64
            params_dict['W1'].append(torch.normal(0, 0.02, size=(num_heads, head_dim, head_dim)).detach().numpy().flatten())
            params_dict['b1'].append(torch.zeros(num_heads, 1, 4 * head_dim).detach().numpy().flatten())
            params_dict['W2'].append(torch.normal(0, 0.02, size=(num_heads, 4 * head_dim, head_dim)).detach().numpy().flatten())
            params_dict['b2'].append(torch.zeros(num_heads, 1, head_dim).detach().numpy().flatten())
            
        # 转换为numpy数组
        for key in params_dict:
            params_dict[key] = np.array(params_dict[key])
            
        return params_dict
    
    def cosine_similarity_matrix(self, params: np.ndarray) -> np.ndarray:
        """计算余弦相似度矩阵"""
        # params shape: (n_models, n_parameters)
        norm = np.linalg.norm(params, axis=1, keepdims=True)
        normalized = params / norm
        similarity = np.dot(normalized, normalized.T)
        return similarity

    
    def compute_collective_similarity(self, normalize_by_layer=True) -> Dict[str, float]:
        """方法2：计算集合整体相似性指标"""
        params_dict = self.extract_parameters()
        similarity_scores = {}
        
        for param_name, params in params_dict.items():
            # 1. 平均成对余弦相似度
            cos_sim = self.cosine_similarity_matrix(params)
            # 取上三角（不包括对角线）
            upper_tri = cos_sim[np.triu_indices(self.n_models, k=1)]
            similarity_scores[f'{param_name}_mean_cosine'] = np.mean(upper_tri)
            similarity_scores[f'{param_name}_std_cosine'] = np.std(upper_tri)
            
            # 2. 中心性度量：到质心的平均距离
            centroid = np.mean(params, axis=0)
            distances_to_centroid = np.linalg.norm(params - centroid, axis=1)
            similarity_scores[f'{param_name}_mean_dist_to_centroid'] = np.mean(distances_to_centroid)
            similarity_scores[f'{param_name}_centroid_variance'] = np.var(distances_to_centroid)

        all_models_vectors = []
        for i in range(self.n_models):
            model_vector = []
            
            for param_name in ['W1', 'b1', 'W2', 'b2']:
                param_vector = params_dict[param_name][i]
                
                if normalize_by_layer:
                    # 对每层参数进行归一化，避免某些层主导相似度
                    norm = np.linalg.norm(param_vector)
                    if norm > 0:
                        param_vector = param_vector / norm
                
                model_vector.append(param_vector)
            
            combined_vector = np.concatenate(model_vector)
            all_models_vectors.append(combined_vector)
        
        all_models_vectors = np.array(all_models_vectors)

        cos_sim = self.cosine_similarity_matrix(all_models_vectors)
        upper_tri = cos_sim[np.triu_indices(self.n_models, k=1)]
        similarity_scores[f'MLP_mean_cosine'] = np.mean(upper_tri)
        similarity_scores[f'MLP_std_cosine'] = np.std(upper_tri)
        centroid = np.mean(all_models_vectors, axis=0)
        distances_to_centroid = np.linalg.norm(all_models_vectors - centroid, axis=1)
        similarity_scores[f'MLP_mean_dist_to_centroid'] = np.mean(distances_to_centroid)
        similarity_scores[f'MLP_centroid_variance'] = np.var(distances_to_centroid)       
            
        return similarity_scores
    
    
    def parameter_correlation_analysis(self) -> Dict[str, np.ndarray]:
        """方法4：参数相关性分析"""
        params_dict = self.extract_parameters()
        results = {}
        
        # 合并所有参数
        all_params = []
        for param_name in ['W1', 'W2', 'b1', 'b2']:
            all_params.append(params_dict[param_name])
            
        # 形状: (n_models, total_parameters)
        combined_params = np.hstack(all_params)
        
        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(combined_params)
        results['full_correlation_matrix'] = correlation_matrix
        
        # 计算平均相关性（不包括对角线）
        mask = ~np.eye(self.n_models, dtype=bool)
        results['mean_correlation'] = np.mean(correlation_matrix[mask])
        results['correlation_std'] = np.std(correlation_matrix[mask])
        
        return results


# 示例使用
def create_sample_mlps(n_models=10, input_dim=10, hidden_dim=20, output_dim=5):
    """创建示例MLP集合"""
    mlp_list = []
    for i in range(n_models):
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # 添加一些随机变化
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.1
        mlp_list.append(model)
    return mlp_list


def compare_two_collections(collection_a, collection_b):
    """比较两个MLP集合"""
    analyzer_a = MLPSimilarityAnalyzer(collection_a)
    analyzer_b = MLPSimilarityAnalyzer(collection_b)
    
    # 计算集合内相似性
    similarity_a = analyzer_a.compute_collective_similarity()
    similarity_b = analyzer_b.compute_collective_similarity()
    
    # # 计算相关性分析
    # corr_a = analyzer_a.parameter_correlation_analysis()
    # corr_b = analyzer_b.parameter_correlation_analysis()
    
    return {
        'collection_a': {
            'similarity': similarity_a,
            # 'correlation': corr_a
        },
        'collection_b': {
            'similarity': similarity_b,
            # 'correlation': corr_b
        }
    }


# 主程序示例
if __name__ == "__main__":
    # 创建两个MLP集合
    collection_a = create_sample_mlps(n_models=10)
    collection_b = create_sample_mlps(n_models=10)
    
    # 比较两个集合
    results = compare_two_collections(collection_a, collection_b)
    
    # 输出结果
    print("集合A相似性指标:")
    for key, value in results['collection_a']['similarity'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n集合B相似性指标:")
    for key, value in results['collection_b']['similarity'].items():
        print(f"  {key}: {value:.4f}")
    
