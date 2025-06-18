# import os
# import cv2
# import h5py
# import scipy
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from utils_imgproc_NN import smallize_density_map, fix_singular_shape


# class CrowdCountingDataset(Dataset):
#     def __init__(self, img_paths, dm_paths, transform=None, stride=1, unit_len=16):
#         self.img_paths = img_paths
#         self.dm_paths = dm_paths
#         self.transform = transform
#         self.stride = stride
#         self.unit_len = unit_len
    
#     def __len__(self):
#         return len(self.img_paths)
    
#     def __getitem__(self, idx):
#         # Load image
#         img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB).astype(np.float32)
#         if self.unit_len:
#             img = fix_singular_shape(img, unit_len=self.unit_len)
        
#         # Load density map
#         dm = h5py.File(self.dm_paths[idx], 'r')['density'][()].astype(np.float32)
#         if self.unit_len:
#             dm = fix_singular_shape(dm, unit_len=self.unit_len)
#         dm = smallize_density_map(dm, stride=self.stride)
        
#         # Convert to tensors
#         img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
#         dm = torch.from_numpy(dm).unsqueeze(0)  # Add channel dimension
        
#         if self.transform:
#             img = self.transform(img)
            
#         return img, dm


# # def gen_paths_img_dm(path_file_root='/Users/soumyadip_iitk/Desktop/Dense Crowd Model/Sanghai_Data', dataset='A'):
# #     """Generate image and density map paths"""
# #     path_file_root_curr = os.path.join(path_file_root, 'part_'+dataset)
# #     img_paths = []
# #     dm_paths = []
# #     paths = os.listdir(path_file_root_curr)[:2]
    
# #     for i in sorted([os.path.join(path_file_root_curr, p) for p in paths]):
# #         with open(i, 'r') as fin:
# #             img_paths.append(
# #                 sorted(
# #                     [l.rstrip() for l in fin.readlines()],
# #                     key=lambda x: int(x.split('_')[-1].split('.')[0]))
# #             )
# #         with open(i, 'r') as fin:
# #             dm_paths.append(
# #                 sorted(
# #                     [l.rstrip().replace('images', 'ground').replace('.jpg', '.h5') for l in fin.readlines()],
# #                     key=lambda x: int(x.split('_')[-1].split('.')[0]))
# #             )
# #     return img_paths, dm_paths

# def gen_paths_img_dm_direct(data_root, dataset='A'):
#     """Generate paths directly from data structure"""
#     base_path = os.path.join(data_root, f'paths_{dataset}')
    
#     # Get train paths
#     train_img_dir = os.path.join(base_path, 'train_data', 'images')
#     train_img_paths = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.jpg')],
#                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
#     # Get test paths
#     test_img_dir = os.path.join(base_path, 'test_data', 'images') 
#     test_img_paths = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.jpg')],
#                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
#     # Generate corresponding density map paths
#     train_dm_paths = [p.replace('images', 'ground').replace('.jpg', '.h5') for p in train_img_paths]
#     test_dm_paths = [p.replace('images', 'ground').replace('.jpg', '.h5') for p in test_img_paths]
    
#     return (test_img_paths, train_img_paths), (test_dm_paths, train_dm_paths)


# def create_data_loader(img_paths, dm_paths, batch_size=4, shuffle=True, stride=1, unit_len=16):
#     """Create PyTorch DataLoader"""
#     # Define transforms for normalization (ImageNet stats)
#     transform = transforms.Compose([
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     # Flatten the paths if they are nested lists
#     if isinstance(img_paths[0], list):
#         img_paths_flat = [path for sublist in img_paths for path in sublist]
#         dm_paths_flat = [path for sublist in dm_paths for path in sublist]
#     else:
#         img_paths_flat = img_paths
#         dm_paths_flat = dm_paths
    
#     dataset = CrowdCountingDataset(
#         img_paths_flat, dm_paths_flat, 
#         transform=transform, stride=stride, unit_len=unit_len
#     )
    
#     dataloader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=shuffle, 
#         num_workers=4, pin_memory=True
#     )
    
#     return dataloader


# def gen_var_from_paths_torch(paths='/Users/soumyadip_iitk/Desktop/Dense Crowd Model/Sanghai_Data', stride=1, unit_len=16, device='cpu'):
#     """Generate variables from paths using PyTorch tensors"""
#     vars_list = []
#     format_suffix = paths[0].split('.')[-1]
    
#     if format_suffix == 'h5':
#         for ph in paths:
#             dm = h5py.File(ph, 'r')['density'][()].astype(np.float32)
#             if unit_len:
#                 dm = fix_singular_shape(dm, unit_len=unit_len)
#             dm = smallize_density_map(dm, stride=stride)
#             dm_tensor = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
#             vars_list.append(dm_tensor)
#     elif format_suffix == 'jpg':
#         for ph in paths:
#             raw = cv2.cvtColor(cv2.imread(ph), cv2.COLOR_BGR2RGB).astype(np.float32)
#             if unit_len:
#                 raw = fix_singular_shape(raw, unit_len=unit_len)
#             # Normalize to [0, 1] and convert to tensor
#             raw = raw / 255.0
#             raw_tensor = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
#             vars_list.append(raw_tensor)
#     else:
#         print('Format suffix is wrong.')
#         return None
    
#     # Stack all tensors
#     result = torch.cat(vars_list, dim=0).to(device)
#     return result


# def gen_density_map_gaussian_torch(im_shape, points, sigma=4, device='cpu'):
#     """
#     Generate density map using PyTorch tensors
#     Args:
#         im_shape: tuple (H, W) of image shape
#         points: numpy array or torch tensor of shape (N, 2) with (x, y) coordinates
#         sigma: gaussian kernel sigma
#         device: torch device
#     """
#     h, w = im_shape[:2]
#     density_map = torch.zeros((h, w), dtype=torch.float32, device=device)
    
#     if isinstance(points, torch.Tensor):
#         points = points.cpu().numpy()
    
#     points = np.squeeze(points)
#     if points.ndim == 1:
#         points = points.reshape(1, -1)
    
#     num_gt = points.shape[0]
#     if num_gt == 0:
#         return density_map
    
#     if sigma == 4:
#         # Adaptive sigma in CSRNet
#         leafsize = 2048
#         tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
#         distances, _ = tree.query(points, k=4)
    
#     for idx_p, p in enumerate(points):
#         p = np.round(p).astype(int)
#         p[0], p[1] = min(w-1, p[0]), min(h-1, p[1])  # Note: x,y vs row,col
#         gaussian_radius = sigma * 2 - 1
        
#         if sigma == 4:
#             # Adaptive sigma in CSRNet
#             sigma_adaptive = max(int(np.sum(distances[idx_p][1:4]) * 0.1), 1)
#             gaussian_radius = sigma_adaptive * 3
#         else:
#             sigma_adaptive = sigma
        
#         # Create Gaussian kernel using PyTorch
#         kernel_size = int(gaussian_radius * 2 + 1)
#         x = torch.arange(kernel_size, dtype=torch.float32, device=device) - gaussian_radius
#         y = torch.arange(kernel_size, dtype=torch.float32, device=device) - gaussian_radius
#         xx, yy = torch.meshgrid(x, y, indexing='ij')
#         gaussian_map = torch.exp(-(xx**2 + yy**2) / (2 * sigma_adaptive**2))
        
#         # Calculate bounds
#         x_left, x_right = 0, gaussian_map.shape[1]
#         y_up, y_down = 0, gaussian_map.shape[0]
        
#         # Cut the gaussian kernel if necessary
#         if p[1] < gaussian_radius:  # p[1] is y coordinate
#             y_up = gaussian_radius - p[1]
#         if p[0] < gaussian_radius:  # p[0] is x coordinate
#             x_left = gaussian_radius - p[0]
#         if p[1] + gaussian_radius >= h:
#             y_down = gaussian_map.shape[0] - (gaussian_radius + p[1] - h) - 1
#         if p[0] + gaussian_radius >= w:
#             x_right = gaussian_map.shape[1] - (gaussian_radius + p[0] - w) - 1
        
#         gaussian_map = gaussian_map[y_up:y_down, x_left:x_right]
        
#         if torch.sum(gaussian_map) > 0:
#             gaussian_map = gaussian_map / torch.sum(gaussian_map)
        
#         # Add to density map
#         y_start = max(0, p[1] - gaussian_radius)
#         y_end = min(h, p[1] + gaussian_radius + 1)
#         x_start = max(0, p[0] - gaussian_radius)
#         x_end = min(w, p[0] + gaussian_radius + 1)
        
#         density_map[y_start:y_end, x_start:x_end] += gaussian_map
    
#     # Normalize
#     if torch.sum(density_map) > 0:
#         density_map = density_map * num_gt / torch.sum(density_map)
    
#     return density_map


# # Example usage:
# def main():
#     # Get paths
#     img_paths, dm_paths = gen_paths_img_dm_direct('/Users/soumyadip_iitk/Desktop/Dense Crowd Model/Sanghai_Data')
    
#     # Create DataLoader
#     dataloader = create_data_loader(img_paths, dm_paths, batch_size=4)
    
#     # Example of iterating through data
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     for batch_idx, (images, density_maps) in enumerate(dataloader):
#         images = images.to(device)
#         density_maps = density_maps.to(device)
        
#         print(f"Batch {batch_idx}: Images shape: {images.shape}, Density maps shape: {density_maps.shape}")
        
#         if batch_idx == 0:  # Just show first batch
#             break


# if __name__ == "__main__":
#     main()
#updated model
import os
import cv2
import h5py
import scipy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
from utils_imgproc_NN import smallize_density_map, fix_singular_shape


class CrowdCountingDataset(Dataset):
    def __init__(self, img_paths, dm_paths, transform=None, stride=1, unit_len=16):
        self.img_paths = img_paths
        self.dm_paths = dm_paths
        self.transform = transform
        self.stride = stride
        self.unit_len = unit_len
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.unit_len:
            img = fix_singular_shape(img, unit_len=self.unit_len)
        
        # Load density map
        dm = h5py.File(self.dm_paths[idx], 'r')['density'][()].astype(np.float32)
        if self.unit_len:
            dm = fix_singular_shape(dm, unit_len=self.unit_len)
        dm = smallize_density_map(dm, stride=self.stride)
        
        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # HWC to CHW and normalize to [0,1]
        dm = torch.from_numpy(dm).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            img = self.transform(img)
            
        return img, dm


def gen_paths_img_dm_complete(data_root, dataset='A'):
    """Generate paths for images, density maps, and ground truth annotations"""
    base_path = os.path.join(data_root, f'paths_{dataset}')
    
    # Get train paths
    train_img_dir = os.path.join(base_path, 'train_data', 'images')
    train_img_paths = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.jpg')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Get test paths
    test_img_dir = os.path.join(base_path, 'test_data', 'images') 
    test_img_paths = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.jpg')],
                           key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Generate corresponding density map paths (.h5 files in 'ground' directory)
    train_dm_paths = [p.replace('images', 'ground').replace('.jpg', '.h5') for p in train_img_paths]
    test_dm_paths = [p.replace('images', 'ground').replace('.jpg', '.h5') for p in test_img_paths]
    
    # Generate corresponding ground truth annotation paths (.mat files in 'ground-truth' directory)
    train_gt_paths = [p.replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat') for p in train_img_paths]
    test_gt_paths = [p.replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat') for p in test_img_paths]
    
    return {
        'train': {
            'images': train_img_paths,
            'density_maps': train_dm_paths,
            'ground_truth': train_gt_paths
        },
        'test': {
            'images': test_img_paths,
            'density_maps': test_dm_paths,
            'ground_truth': test_gt_paths
        }
    }


def generate_density_maps_from_annotations(data_root, dataset='A'):
    """Generate .h5 density map files from .mat annotation files"""
    paths = gen_paths_img_dm_complete(data_root, dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for split in ['train', 'test']:
        img_paths = paths[split]['images']
        gt_paths = paths[split]['ground_truth']
        dm_paths = paths[split]['density_maps']
        
        print(f"Generating density maps for {split} set...")
        
        for img_path, gt_path, dm_path in zip(img_paths, gt_paths, dm_paths):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dm_path), exist_ok=True)
            
            # Skip if density map already exists
            if os.path.exists(dm_path):
                continue
                
            # Load image and ground truth
            img = cv2.imread(img_path)
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth file not found: {gt_path}")
                continue
                
            pts = loadmat(gt_path)
            
            # Set sigma based on dataset part
            sigma = 4 if 'part_A' in img_path else 15
            
            # Extract ground truth points
            gt = pts["image_info"][0, 0][0, 0][0]
            
            # Validate and filter points within image bounds
            valid_points = []
            for i in range(len(gt)):
                x, y = int(gt[i][0]), int(gt[i][1])
                if y < img.shape[0] and x < img.shape[1]:
                    valid_points.append([x, y])
            
            # Convert to numpy array
            if len(valid_points) > 0:
                points = np.array(valid_points)
            else:
                points = np.array([]).reshape(0, 2)
            
            # Generate density map using PyTorch function
            DM = gen_density_map_gaussian_torch(
                im_shape=(img.shape[0], img.shape[1]), 
                points=points, 
                sigma=sigma, 
                device=device
            )
            
            # Save density map to h5 file
            with h5py.File(dm_path, 'w') as hf:
                hf['density'] = DM.cpu().numpy()
        
        print(f"Completed {split} set density map generation")


def create_data_loader(img_paths, dm_paths, batch_size=4, shuffle=True, stride=1, unit_len=16):
    """Create PyTorch DataLoader"""
    # Define transforms for normalization (ImageNet stats)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CrowdCountingDataset(
        img_paths, dm_paths, 
        transform=transform, stride=stride, unit_len=unit_len
    )
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=4, pin_memory=True
    )
    
    return dataloader


def gen_density_map_gaussian_torch(im_shape, points, sigma=4, device='cpu'):
    """Generate density map using PyTorch tensors"""
    h, w = im_shape[:2]
    density_map = torch.zeros((h, w), dtype=torch.float32, device=device)
    
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    points = np.squeeze(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    num_gt = points.shape[0]
    if num_gt == 0:
        return density_map
    
    if sigma == 4:
        # Adaptive sigma in CSRNet
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, _ = tree.query(points, k=4)
    
    for idx_p, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(w-1, p[0]), min(h-1, p[1])
        gaussian_radius = sigma * 2 - 1
        
        if sigma == 4:
            sigma_adaptive = max(int(np.sum(distances[idx_p][1:4]) * 0.1), 1)
            gaussian_radius = sigma_adaptive * 3
        else:
            sigma_adaptive = sigma
        
        # Create Gaussian kernel using PyTorch
        kernel_size = int(gaussian_radius * 2 + 1)
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - gaussian_radius
        y = torch.arange(kernel_size, dtype=torch.float32, device=device) - gaussian_radius
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        gaussian_map = torch.exp(-(xx**2 + yy**2) / (2 * sigma_adaptive**2))
        
        # Calculate bounds and add to density map
        x_left, x_right = 0, gaussian_map.shape[1]
        y_up, y_down = 0, gaussian_map.shape[0]
        
        if p[1] < gaussian_radius:
            y_up = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            x_left = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[1] - h) - 1
        if p[0] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[0] - w) - 1
        
        gaussian_map = gaussian_map[y_up:y_down, x_left:x_right]
        
        if torch.sum(gaussian_map) > 0:
            gaussian_map = gaussian_map / torch.sum(gaussian_map)
        
        y_start = max(0, p[1] - gaussian_radius)
        y_end = min(h, p[1] + gaussian_radius + 1)
        x_start = max(0, p[0] - gaussian_radius)
        x_end = min(w, p[0] + gaussian_radius + 1)
        
        density_map[y_start:y_end, x_start:x_end] += gaussian_map
    
    if torch.sum(density_map) > 0:
        density_map = density_map * num_gt / torch.sum(density_map)
    
    return density_map


def main():
    data_root = '/Users/soumyadip_iitk/Desktop/Dense Crowd Model/Sanghai_Data'
    dataset = 'A'
    
    # First, generate density maps from .mat annotations if they don't exist
    print("Checking and generating density maps from annotations...")
    generate_density_maps_from_annotations(data_root, dataset)
    
    # Get all paths
    paths = gen_paths_img_dm_complete(data_root, dataset)
    
    print(f"Train images: {len(paths['train']['images'])}")
    print(f"Test images: {len(paths['test']['images'])}")
    
    # Create DataLoaders
    train_dataloader = create_data_loader(
        paths['train']['images'], 
        paths['train']['density_maps'], 
        batch_size=4, shuffle=True, stride=8, unit_len=None
    )
    
    test_dataloader = create_data_loader(
        paths['test']['images'], 
        paths['test']['density_maps'], 
        batch_size=1, shuffle=False, stride=8, unit_len=None
    )
    
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for batch_idx, (images, density_maps) in enumerate(train_dataloader):
        images = images.to(device)
        density_maps = density_maps.to(device)
        
        print(f"Batch {batch_idx}: Images {images.shape}, Density maps {density_maps.shape}")
        print(f"Density sum: {density_maps.sum().item():.2f}")
        
        if batch_idx == 0:
            break


if __name__ == "__main__":
    main()
