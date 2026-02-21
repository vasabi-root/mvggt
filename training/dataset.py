import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import json
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
from mvggt.utils.geometry import homogenize_points

import pandas as pd

import cv2
import numpy as np
from PIL import Image
import torch
import open3d as o3d
import os

def save_tensor_img(batch: torch.Tensor, path: Path):
    assert len(batch) != 3
    
    path = Path(path)
    
    for i, tensor in enumerate(batch):
        # if len(tensor.shape) == 2:
        #     img = tensor.cpu().detach().numpy()
        # if len(tensor.shape) == 3:
        #     img = tensor.permute(1, 2, 0).cpu().detach().numpy()
        
        img = F.to_pil_image(tensor)
        img.save(path.parent / f"{path.stem}_{i}{path.suffix}")
         
def draw_mask(img_tensor: torch.Tensor, mask_tensor: torch.Tensor, path: Path) -> None:
    mask = mask_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.cpu().detach().numpy()
    img = (np.stack(img[::-1], -1)*255).astype(np.uint8)
    contour = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
    cv2.imwrite(path, img)
    
def visualize_object(scannet_root: str, scene_id: str, object_id: int):
    scene_path = f"{scannet_root}/scans/{scene_id}"

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(f"{scene_path}/{scene_id}_vh_clean_2.ply")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.65, 0.65, 0.65])        # gray

    # Load aggregation + segmentation
    with open(f"{scene_path}/{scene_id}.aggregation.json") as f:
        agg = json.load(f)
    seg_groups = {str(g['objectId']): g['segments'] for g in agg['segGroups']}

    with open(f"{scene_path}/{scene_id}_vh_clean_2.0.010000.segs.json") as f:
        segs = json.load(f)
    seg_indices = np.asarray(segs['segIndices'])

    # Extract obj points
    segs_for_obj = seg_groups[str(object_id)]
    obj_vertex_mask = np.isin(seg_indices, segs_for_obj)
    vertices = np.asarray(mesh.vertices)
    obj_points = vertices[obj_vertex_mask]

    # Create PointCloud for the object (red)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # Visualize
    print(f"Visualizing scene {scene_id} | object {object_id} | {len(obj_points)} points")
    o3d.visualization.draw_geometries(
        [mesh, obj_pcd],
        window_name=f"ScanNet {scene_id} | object id {object_id} | {len(obj_points)} points",
        width=1280,
        height=720,
        mesh_show_back_face=True
    )
    
def convex_hull(u, v, H, W):
    pts2d = torch.stack([u, v], dim=1).cpu().numpy()   # (N, 2)

    hull = cv2.convexHull(pts2d)                                     # convex hull vertices

    mask_np = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_np, [hull], 1)                                 # fill the polygon

    mask = torch.from_numpy(mask_np)
    return mask

def dilate_erode(mask, H, W, k=0.1):
    kernel_sz = int(min(H,W) * k)
    kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
    dilated = cv2.dilate(mask.cpu().detach().numpy(), kernel)
    eroded = cv2.erode(dilated, kernel)
    return torch.from_numpy(eroded)
    

def project_points_to_mask(obj_points, pose_cam2world, K, H, W, fill_gaps=False):
    # world 3D -> camera 2D
    pose_world2cam = pose_cam2world.inverse()
    ones = torch.ones((len(obj_points),1))
    points_cam = (pose_world2cam @ torch.hstack([obj_points, ones]).T.float()).T[:, :3]
    # points_cam = obj_points
    
    # project
    points_proj = (K @ points_cam.T.float()).T
    
    u = (points_proj[:, 0] / points_proj[:, 2]).int()
    v = (points_proj[:, 1] / points_proj[:, 2]).int()
    z = points_proj[:, 2]
    
    valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = torch.zeros((H, W), dtype=torch.uint8)
    mask[v[valid], u[valid]] = 1
    
    # fill gaps with convex hull
    if fill_gaps and valid.sum() >= 3:
        mask = convex_hull(u[valid], v[valid], H, W)
        # mask = dilate_erode(mask, H, W)

    return mask.float()

def backproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Back-project a depth map to 3D points in camera coordinates.
    
    Args:
        depth: (H, W) depth map in meters (torch.float32)
        K:     (3, 3) intrinsics matrix
        
    Returns:
        (H, W, 3) tensor of points in camera coordinates (x, y, z)
    """
    H, W = depth.shape
    device = depth.device

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = torch.arange(W, dtype=torch.float32, device=device)
    v = torch.arange(H, dtype=torch.float32, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth.clone()

    return torch.stack([x, y, z], dim=-1)  # (H, W, 3)

class ScanReferMvggtDataset(Dataset):
    def __init__(self, scanrefer_path, scannet_root, tokenizer, num_views=8, img_size=(238, 322)): # TODO: add patch_size and fit img_size with patch_size
        '''
        ScanRefer_filtered_train.json not above 0706
        ScanRefer_filtered_val.json not above 0706
        '''
        with open(scanrefer_path) as f:
            self.data = [ann for ann in json.load(f)]
        self.scannet_root = scannet_root
        self.label_mapper = pd.read_csv(f"{self.scannet_root}/scannetv2-labels.combined.tsv", sep='\t')
        self.draw_masks = False # for debugging
        
        self.tokenizer = tokenizer
        self.num_views = num_views
        
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        scene_id = ann['scene_id']
        obj_id = ann['object_id']
        obj_name = ann['object_name']
        text = ann['description']

        frame_ids = self._sample_frames(scene_id)

        view_list = []
        for fid in frame_ids:
            rgb = self._load_rgb(scene_id, fid)                  # (3 H W)
            H, W = rgb.shape[-2:]
            
            depth = self._load_depth(scene_id, fid)              # (H W)
            pose_cam2world = self._load_pose(scene_id, fid)
            
            K_color, K_depth = self._load_intrinsics(scene_id)

            # backproject depth -> camera coordinates
            pts_local = backproject(depth, K_depth)              # (H W 3)

            # to world
            pts_world = (pose_cam2world @ homogenize_points(pts_local.view(-1, 3)).T)[:-1].T.view(H, W, 3) # [:-1] stands for dehomogenization

            # TODO: enhance efficiency of calculation. Do not load 3d masks. Try to load only 2d instance masks, instead of semantic ones
            mask_from_3d = self._load_mask_from_3d_label(scene_id, obj_id, pose_cam2world, K_color, H, W)
            mask_from_2d = torch.zeros_like(mask_from_3d)
            if mask_from_3d.sum() > 0:
                mask_from_2d = self._load_mask_from_2d_label(scene_id, fid, obj_name)
            
            # visualize_object(self.scannet_root, scene_id, obj_id)
            
            if self.draw_masks:
                masked_dir = Path('masked')/f'{scene_id}/{obj_name}'
                os.makedirs(masked_dir, exist_ok=True)
                draw_mask(rgb, mask_from_2d, masked_dir / f'{fid}.png')

            view_list.append({
                'img': rgb,
                'pts3d': pts_world,
                'valid_mask': depth > 0,
                'camera_pose': pose_cam2world,
                'referring_mask': mask_from_2d.bool()
            })

        # text
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
        return {
            'imgs': torch.stack([v['img'] for v in view_list]),           # (N 3 H W)
            'input_ids': tokens.input_ids[0],
            'attention_mask': tokens.attention_mask[0],
            'view_list': view_list
        }

    # collate_fn for batch > 1
    @staticmethod
    def collate_fn(batch):
        # batch = list of dicts
        imgs = torch.stack([b['imgs'] for b in batch])                    # (B N 3 H W)
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_masks = torch.stack([b['attention_mask'] for b in batch])
        # view_lists = [b['view_list'] for b in batch]                      # list[B] of list[N]
        
        view_lists = [] # N
        for _ in range(len(batch[0]['view_list'])):
            view_list_b3hw = {}
            for key in batch[0]['view_list'][0]:
                view_list_b3hw[key] = []
            view_lists.append(view_list_b3hw)
        
        for b in batch:
            for i, vl in enumerate(b['view_list']):
                for key in vl:
                    view_lists[i][key].append(vl[key])           # B 3 H W
                    
        for vl in view_lists:
            for key in vl:
                vl[key] = torch.from_numpy(np.array(vl[key]))
            
        return {'imgs': imgs, 'input_ids': input_ids, 'attention_masks': attention_masks, 'gt_raw': view_lists}
    
    def _map_obj_name_to_nyu40id(self, obj_name):
        row = self.label_mapper[self.label_mapper['raw_category'].str.lower() == obj_name.replace('_', ' ').lower()]
        return row['nyu40id'].values[0]
    
    def _load_rgb(self, scene_id, frame_idx):
        path = f"{self.scannet_root}/images/{scene_id}/color/{frame_idx}.jpg"      # :06d may be for new scannet
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def _load_depth(self, scene_id, frame_idx):
        path = f"{self.scannet_root}/images/{scene_id}/depth-erode/{frame_idx}.png"      # :06d may be for new scannet
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        return F.resize(depth, size=self.img_size, antialias=True).squeeze(0) # (H W)
    
    def _load_mask_from_2d_label(self, scene_id, frame_idx, obj_name):
        path = f"{self.scannet_root}/images/{scene_id}/labelv2-eroded/{frame_idx}.png"
        labels = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        labels = torch.from_numpy(labels).to(torch.int).unsqueeze(0)
        obj_id = self._map_obj_name_to_nyu40id(obj_name) # TODO: map obj_id instead of obj_name
        mask = labels == obj_id
        return F.resize(mask, size=self.img_size, antialias=True).squeeze(0) # (H W)
    
    def _load_mask_from_3d_label(self, scene_id, obj_id, pose_cam2world, K, H, W):
        '''
        Maps object 3D points -> 2D mask
        - `obj_id`: object id from aggregation.json
        - `pose_cam2world`: camera pose
        - `K`: camera intrinsics
        '''
        obj_pts = self._load_object_points(scene_id, obj_id)
        mask = project_points_to_mask(obj_pts, pose_cam2world, K, H, W, False)
        return mask

    def _load_pose(self, scene_id, frame_idx):
        path = f"{self.scannet_root}/images/{scene_id}/pose/{frame_idx}.txt"       # :06d may be for new scannet
        pose = np.loadtxt(path).astype(np.float32)  # 4x4 camera-to-world
        return torch.from_numpy(pose).float()

    def _load_intrinsics(self, scene_id):
        data = {}
        with open(f"{self.scannet_root}/scans/{scene_id}/{scene_id}.txt") as f:
            for line in f:
                pieces = line.strip().split(" = ")
                data[pieces[0]] = pieces[1]
        
        color, depth = "color", "depth"
        H, W = self.img_size
        K = {}
        for key in [color, depth]:
            K[key] = np.zeros((3, 3))
            K[key][0,0] = float(data[f"fx_{key}"])
            K[key][1,1] = float(data[f"fy_{key}"])
            K[key][0,-1] = float(data[f"mx_{key}"])
            K[key][1,-1] = float(data[f"my_{key}"])
            
            # correct intrinsics with real height/width of an image
            K[key][0] *= W/int(data[f"{key}Width"])
            K[key][1] *= H/int(data[f"{key}Height"])
            
            K[key][-1,-1] = 1
    
        return torch.from_numpy(K[color]).float(), torch.from_numpy(K[depth]).float()
    
    def _load_object_points(self, scene_id, object_id):
        scene_path = f"{self.scannet_root}/scans/{scene_id}"  # scene dir with .json и .ply
        
        # Aggregation
        with open(f"{scene_path}/{scene_id}.aggregation.json") as f:
            agg = json.load(f)
        seg_groups = {str(g['objectId']): g['segments'] for g in agg['segGroups']}
        
        # Segmentation (vertex -> segment)
        with open(f"{scene_path}/{scene_id}_vh_clean_2.0.010000.segs.json") as f:
            segs = json.load(f)
        seg_indices = segs['segIndices']  # list of segment id per vertex
        
        # Mesh vertices
        mesh = o3d.io.read_triangle_mesh(f"{scene_path}/{scene_id}_vh_clean_2.ply")
        vertices = np.asarray(mesh.vertices)  # (N 3)
        
        # Vertices extraction
        segs_for_obj = seg_groups[object_id]
        obj_vertex_mask = np.isin(seg_indices, segs_for_obj)
        obj_points = vertices[obj_vertex_mask]  # (M 3) world coordinates
        
        # visualize_object(self.scannet_root, scene_id, object_id)
        
        return torch.from_numpy(obj_points).float()
    
    def _sample_frames(self, scene_id):
        # Chooses `self.num_views` frames from the scene with id `scene_id`
        frame_idxs = [int(name[:-4]) for name in os.listdir(f"{self.scannet_root}/images/{scene_id}/color") if name[-4:] == ".jpg"]
        frame_idxs.sort()
            
        while len(frame_idxs) < self.num_views:
            frame_idxs.append(frame_idxs[0])
        if len(frame_idxs) == self.num_views:
            return frame_idxs
        
        step = len(frame_idxs)//self.num_views
        halt = step * self.num_views
        return frame_idxs[:halt:step]
        
if __name__ == '__main__':
    from scanrefer.lib.config import CONF
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_pooling_layer=False)
    
    dataset = ScanReferMvggtDataset(
        scanrefer_path=CONF.TRAIN_JSON,
        scannet_root=CONF.PATH.SCANNET,
        tokenizer=tokenizer,
        num_views=8
    )
    dataset[0]
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=ScanReferDataset.collate_fn)
    pass