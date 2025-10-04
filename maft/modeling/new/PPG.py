import torch
import torch.nn as nn

class PseudoMaskGenerator(nn.Module):
    def __init__(self, num_clusters=5):
        """
        初始化伪提示生成器。

        参数:
            num_clusters (int): 每个类别要聚类的簇数（物体实例数）。对应论文中的 No。
        """
        super().__init__()
        self.num_clusters = num_clusters

    def forward(self, binary_mask):
        """
        对输入的二值化掩码进行聚类，并为每个簇生成一个单独的二值掩码。

        参数:
            binary_mask (torch.Tensor): 形状为 [B, 1, C, H, W] 的二值化掩码张量。

        返回:
            torch.Tensor: 形状为 [B, C, num_clusters, H, W] 的聚类后的掩码张量。
                          每个 num_clusters 通道都是一个二值掩码，代表一个独立的簇。
        """
        # 检查输入形状，移除单维度
        if binary_mask.dim() == 5 and binary_mask.shape[1] == 1:
            binary_mask = binary_mask.squeeze(1)
        
        B, C, H, W = binary_mask.shape

        # 初始化一个用于存储最终聚类结果的张量
        # 形状为 [B, C, num_clusters, H, W]
        clustered_masks = torch.zeros((B, C, self.num_clusters, H, W), 
                                      dtype=binary_mask.dtype, 
                                      device=binary_mask.device)

        # 遍历批次中的每个样本和每个类别
        for b in range(B):
            for c in range(C):
                # 获取当前样本和类别的二值化掩码
                mask = binary_mask[b, c]

                # 找到掩码中所有值为1的像素的坐标
                foreground_pixels = torch.nonzero(mask, as_tuple=False)

                # 如果前景像素太少，无法聚类，则将所有像素归为第一个簇
                if foreground_pixels.shape[0] <= self.num_clusters:
                    if foreground_pixels.shape[0] > 0:
                        clustered_masks[b, c, 0] = mask
                    continue

                # --- K-Means 聚类 ---
                # 1. 随机初始化聚类中心
                initial_indices = torch.randperm(foreground_pixels.shape[0])[:self.num_clusters]
                centroids = foreground_pixels[initial_indices].float()
                
                # 为了稳定性和效率，限制迭代次数
                for _ in range(10):
                    # 2. 分配每个点到最近的聚类中心
                    distances = torch.cdist(foreground_pixels.float(), centroids)
                    assignments = torch.argmin(distances, dim=1)

                    # 3. 更新聚类中心
                    new_centroids = torch.zeros_like(centroids)
                    for i in range(self.num_clusters):
                        points_in_cluster = foreground_pixels[assignments == i]
                        if points_in_cluster.shape[0] > 0:
                            new_centroids[i] = points_in_cluster.float().mean(dim=0)
                        else:
                            # 如果簇变空，重新从最远的点初始化，增加鲁棒性
                            all_distances_to_centroids = torch.cdist(foreground_pixels.float(), centroids).min(dim=1).values
                            farthest_point_idx = all_distances_to_centroids.argmax()
                            new_centroids[i] = foreground_pixels[farthest_point_idx].float()

                    if torch.allclose(centroids, new_centroids):
                        break
                    centroids = new_centroids
                
                # --- 将聚类结果转换为 [num_clusters, H, W] 的掩码 ---
                for i in range(self.num_clusters):
                    # 找到属于当前簇 i 的所有像素
                    current_cluster_pixels = foreground_pixels[assignments == i]
                    
                    if current_cluster_pixels.shape[0] > 0:
                        # 将这些像素在对应的通道 i 中设置为 1
                        clustered_masks[b, c, i, current_cluster_pixels[:, 0], current_cluster_pixels[:, 1]] = 1

        return clustered_masks


class PseudoPointGenerator(nn.Module):
    def __init__(self):
        """
        初始化伪点生成器。
        这个模块是无状态的，不需要可训练的参数。
        """
        super().__init__()

    def forward(self, clustered_masks, corr_prob):
        """
        根据聚类后的掩码和原始概率图生成伪点。

        参数:
            clustered_masks (torch.Tensor): 形状为 [B, C, num_clusters, H, W] 的聚类掩码。
                                            每个 num_clusters 通道都是一个二值掩码，代表一个独立的簇。
            corr_prob (torch.Tensor): 形状为 [B, 1, C, H, W] 或 [B, C, H, W] 的原始概率图谱。

        返回:
            torch.Tensor: 形状为 [B, C, num_clusters, 2] 的伪点坐标张量。
                          最后一个维度存储 (row, col) 坐标。
                          对于空的簇，坐标将被设置为 (-1, -1)。
        """
        # --- 1. 预处理和形状对齐 ---
        
        # 确保 corr_prob 的形状是 [B, C, H, W] 以便后续处理
        if corr_prob.dim() == 5 and corr_prob.shape[1] == 1:
            corr_prob = corr_prob.squeeze(1)
            
        B, C, num_clusters, H, W = clustered_masks.shape
        
        # [B, C, H, W] -> [B, C, num_clusters, H, W]
        corr_prob_expanded = corr_prob.unsqueeze(2)

        masked_corr_prob = corr_prob_expanded * clustered_masks

        # 将 H 和 W 维度展平, argmax 找到每个簇的最大值索引
        # [B, C, num_clusters, H, W] -> [B, C, num_clusters, H * W]
        flat_masked_corr = masked_corr_prob.view(B, C, num_clusters, H * W)

        flat_indices = torch.argmax(flat_masked_corr, dim=-1) # [B, C, num_clusters]

        # --- 3. 将一维索引转换回二维坐标 ---

        # 计算行坐标 (row) 和列坐标 (col)
        rows = flat_indices // W # [B, C, num_clusters]
        cols = flat_indices % W # [B, C, num_clusters]
        
        pseudo_points = torch.stack([rows, cols], dim=-1) # [B, C, num_clusters, 2]

        '''
        # --- 4. 处理掩码全0的情况 --- 
        # 这里取消了处理该部分的逻辑避免后续点坐标不合法，因为只需要最终掩码组合的时候把clustered_masks.sum(dim=(-1, -2)的项为0的部分忽略掉即可

        is_empty_cluster = (clustered_masks.sum(dim=(-1, -2)) == 0) # [B, C, num_clusters]
        
        # 使用这个布尔掩码将无效点的坐标设置为 -1,-1
        pseudo_points[is_empty_cluster] = -1
        '''
        
        return pseudo_points.long() # 返回整数类型的坐标
