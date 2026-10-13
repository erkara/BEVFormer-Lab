_base_ = [
    './bevformer_tiny.py',
]

# Experimental binary BEV occupancy/objectness model. This keeps the tiny
# BEVFormer dataset and encoder config, but uses a separate detector wrapper
# that calls the BEV encoder with only_bev=True and trains a small CNN mask head.
model = dict(
    type='BEVFormerOccupancy',
    class_names=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    occupancy_classes=(
        'car',
        'truck',
        'bus',
        'trailer',
        'construction_vehicle',
    ),
    occupancy_head=dict(
        type='BEVOccupancyHead',
        in_channels=256,
        hidden_channels=128,
        pos_weight=10.0,
    ),
)
