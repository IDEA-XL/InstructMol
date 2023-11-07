from .clip_encoder import CLIPVisionTower
from .gnn_graphmvp import GraphMVP
from .moleculeSTM_gnn_model import GNN_graphpred, GNN


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_graph_tower(graph_tower_cfg, **kwargs):
    graph_tower = getattr(graph_tower_cfg, 'mm_graph_tower', getattr(graph_tower_cfg, 'graph_tower', None))
    if graph_tower.startswith("graphmvp"):
        return GraphMVP(config=graph_tower_cfg)
    elif graph_tower.startswith("moleculestm"):
        # actually, 'graph_tower_cfg' is identical to 'model_args'
        molecule_node_model = GNN(
            num_layer=graph_tower_cfg.gin_num_layers,
            emb_dim=graph_tower_cfg.gin_hidden_dim,
            JK='last', # default to 'last'
            drop_ratio=graph_tower_cfg.drop_ratio,
            gnn_type='gin', # default to 'gin'
        )
        return GNN_graphpred(
            emb_dim=graph_tower_cfg.gin_hidden_dim,
            graph_pooling=graph_tower_cfg.graph_pooling,
            molecule_node_model=molecule_node_model,
            init_checkpoint=graph_tower_cfg.init_checkpoint,
        )
    
    raise ValueError(f'Unknown graph tower: {graph_tower}')