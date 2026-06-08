import time
import torch
from data_prep import DatasetConfig, build_datasets, num_classes_and_offset
from model import ModelConfig, EdgePredictor

cfg = DatasetConfig(
    predict_bond_change=True,
    selected_reactions=None,
    filter_unique_reactants=False,
    perturb_training=False,
    include_base_sample=True,
    copy_counts=[1],
    random_group_training=False,
    random_group_eval=False,
    show_progress=False,
)
bundle = build_datasets(cfg)
sample = bundle.train_data[0]
num_classes, _ = num_classes_and_offset(True)

def bench(use_latent: bool, iters: int = 40):
    mc = ModelConfig(
        node_in=sample.x.size(1),
        num_classes=num_classes,
        hidden=128,
        layers=3,
        edge_hidden=128,
        molecule_balanced_pool=True,
        use_branch_feature=not use_latent,
        branch_feature_mode='contextual',
        use_third_body_feature=True,
        use_latent_branching=use_latent,
        latent_dim=8,
        latent_hidden=64,
    )
    m = EdgePredictor(mc)
    m.train()
    s = sample

    for _ in range(5):
        out = m(s, return_aux=True)
        if isinstance(out, tuple):
            logits, aux = out
            loss = logits.square().mean() + aux.get('kl_loss', torch.tensor(0.0))
        else:
            loss = out.square().mean()
        loss.backward()
        m.zero_grad(set_to_none=True)

    t0 = time.time()
    for _ in range(iters):
        out = m(s, return_aux=True)
        if isinstance(out, tuple):
            logits, aux = out
            loss = logits.square().mean() + aux.get('kl_loss', torch.tensor(0.0))
        else:
            loss = out.square().mean()
        loss.backward()
        m.zero_grad(set_to_none=True)
    t1 = time.time()
    return (t1 - t0) / iters

base = bench(False)
lat = bench(True)
print(f'avg_sec_no_latent={base:.6f}')
print(f'avg_sec_latent={lat:.6f}')
print(f'ratio={lat/max(base,1e-9):.2f}x')
