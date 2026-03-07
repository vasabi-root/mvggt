import sys
from pathlib import Path

import os
import shutil
import json
from bisect import bisect_left

from tqdm import tqdm

def get_closest_frame_idx(valid_frame_idxs, frame_idx):
    frame_idx = int(frame_idx)
    closest_i = bisect_left(valid_frame_idxs, frame_idx)
    val_right = valid_frame_idxs[closest_i] if closest_i < len(valid_frame_idxs) else valid_frame_idxs[-1]
    val_left = valid_frame_idxs[closest_i-1] if closest_i-1 > 0 else valid_frame_idxs[0]
    closest_frame_idx = val_left if frame_idx - val_left < val_right - frame_idx else val_right
    return closest_frame_idx

def get_closest_frames_scores(valid_frame_idxs, frame_idxs_dict):
    closest = {}
    for frame_idx, score in frame_idxs_dict.items():
        closest_frame_idx = get_closest_frame_idx(valid_frame_idxs, frame_idx)
        if closest_frame_idx not in closest.keys():
            closest[closest_frame_idx] = score
        elif score > closest[closest_frame_idx]:
            old_score = closest[closest_frame_idx]
            closest[closest_frame_idx] = score
            closest_frame_idx_old = get_closest_frame_idx(valid_frame_idxs, int(frame_idx)-20)
            closest[closest_frame_idx_old] = old_score
        else:
            closest_frame_idx = get_closest_frame_idx(valid_frame_idxs, int(frame_idx)+20)
            closest[closest_frame_idx] = score
    assert len(frame_idxs_dict) == len(closest)
    return closest
        

def main():
    scannet_dir = Path('data/scannet_data')
    
    mvrefer_val_json = Path('data/mvrefer_val.json')
    with open(mvrefer_val_json, 'r') as f:
        mvrefer_val = json.load(f)
    
    for scene_id in tqdm(mvrefer_val):
        valid_frame_idxs = [int(name[:-4]) for name in os.listdir(scannet_dir / f'{scene_id}' / 'color')]
        valid_frame_idxs.sort()
        for obj_id in mvrefer_val[scene_id]:
            mvrefer_val[scene_id][obj_id] = get_closest_frames_scores(valid_frame_idxs, mvrefer_val[scene_id][obj_id])
            
    mvrefer_val_sparse_json = Path('data/mvrefer_val_sparse.json')
    with open(mvrefer_val_sparse_json, 'w') as f:
        json.dump(mvrefer_val, f)
    
if __name__ == '__main__':
    main()