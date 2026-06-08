# GNN_label

This directory contains the current graph-based reaction prediction pipeline for the 21 hydrogen reactions.

The code is organized into:

- data construction and augmentation
- GNN + edge decoder model
- training loop
- validation/reporting
- inference utilities (fixed-copy inference and stochastic event simulation)

## Overview

The model predicts a product bond graph from a reactant bond graph.

Current training can be multi-task:

- edge prediction (always)
- optional rate-coefficient regression per reactant-group (`predict_rate_coeffs=True`)

Two target modes are supported:

- `predict_bond_change=True`: classify edge delta in `[-3, -2, -1, 0, +1, +2, +3]` (7 classes)
- `predict_bond_change=False`: classify absolute bond order in `[0, 1, 2, 3]` (4 classes)

The current default in [reaction_dataset_prediction.py](/users/3/du000298/GNN_label/reaction_dataset_prediction.py) is bond-change prediction.

## File Map

- [hydrogen_adjacency.py](/users/3/du000298/GNN_label/hydrogen_adjacency.py): species graphs and 21 reaction definitions
- [data_prep.py](/users/3/du000298/GNN_label/data_prep.py): dataset generation / augmentation / train-eval split logic
- [model.py](/users/3/du000298/GNN_label/model.py): `EdgePredictor` (GINE encoder + edge decoder)
- [train.py](/users/3/du000298/GNN_label/train.py): optimization loop
- [validate.py](/users/3/du000298/GNN_label/validate.py): inference on eval set + output artifacts
- [plotting.py](/users/3/du000298/GNN_label/plotting.py): loss and graph plotting helpers
- [reaction_dataset_prediction.py](/users/3/du000298/GNN_label/reaction_dataset_prediction.py): main training entrypoint
- [reaction_inference_copies.py](/users/3/du000298/GNN_label/reaction_inference_copies.py): checkpoint-based inference for alternate copy counts
- [reaction_stochastic_inference.py](/users/3/du000298/GNN_label/reaction_stochastic_inference.py): stochastic event-by-event inference from a molecule pool

## Data Representation

### Node features

Each atom is represented by a 4D feature:

- `is_H`
- `is_O`
- `molecule_index_within_species`
- `atom_index_within_molecule`

Defined in [data_prep.py](/users/3/du000298/GNN_label/data_prep.py) via `_build_node_features`.

Optional training augmentation adds jitter to the last two features (`PERTURB_*` settings).

### Graph structure

For each sample:

- `edge_index`, `edge_attr`: only reactant bonds are message-passing edges (`edge_attr` = bond order)
- candidate prediction edges are all upper-triangle atom pairs (`pair_i`, `pair_j`)
- `react_edge`: reactant bond order for each candidate pair
- target classes in `y_edge`

### Atom mapping algorithm (reactant-product alignment)

Before building edge labels, product atoms are reordered to the reactant atom order with
`reorder_to_target(...)` in [hydrogen_adjacency.py](/users/3/du000298/GNN_label/hydrogen_adjacency.py).

Goal:

- align product adjacency to reactant atom indexing while respecting atom types
- minimize graph edits between reactant and aligned product

Objective (under a permutation `perm` from target index -> source index):

- `J(perm) = sum_{i<j} |A_target[i,j] - A_source[perm(i), perm(j)]|`
- here `A_target` is usually reactant adjacency and `A_source` is product adjacency

Algorithm:

1. Build a stable type-FIFO baseline mapping (`H->H`, `O->O`, first-to-first).
2. Build a signature-seeded mapping per atom type using local node signatures:
   - `(degree, bond_order_sum, neighbor_type_counts)`
3. Refine with same-type swap hill-climbing:
   - only swap target atoms of the same type
   - accept swap if it decreases `J`
   - use bounded passes/evaluation budget for robustness
4. Safety fallback:
   - compute `J` for refined and FIFO mappings
   - never degrade: keep refined only if `J_refined <= J_fifo`
5. Reorder product adjacency with the selected permutation.

This gives a deterministic mapping that is usually closer to minimal bond edits than pure FIFO, while preserving a guaranteed non-worse fallback.

### Group-aware metadata (key update)

To support multi-branch and multi-reaction packed samples, each sample can carry:

- `node_group_id`: group/block assignment per node
- `pair_group_id`: group/block assignment per candidate edge
- `branch_id_by_group`
- `branch_count_by_group`
- `branch_value_by_group`
- `third_body_by_group`
- `rate_target_by_group` (shape `[num_groups, 6]`)
- `rate_mask_by_group` (valid fields mask, same shape)

Legacy scalar fields are still present for compatibility:

- `branch_id`, `branch_count`, `branch_value`, `third_body_value`

### Rate target representation and standardization (current)

Rate targets are encoded per group as:

- `[main_logA, main_n, main_Ea, low_logA, low_n, low_Ea]`

where:

- Arrhenius / three-body reactions use first 3 fields, low fields masked out
- falloff reactions use all 6 fields

The encoding is built in [data_prep.py](/users/3/du000298/GNN_label/data_prep.py), then standardized component-wise:

- compute masked `mean/std` from training set only
- z-score train and eval targets with those stats
- attach `rate_mean` / `rate_std` to each sample

Validation de-standardizes predictions before writing `rate_coeffs.txt`, so logged physical values are in original units.

## Dataset Construction Modes

Implemented in [data_prep.py](/users/3/du000298/GNN_label/data_prep.py).

### Base reaction samples

- One sample per reaction (or per branch when identical reactants have multiple products)
- Optional copy factors (`copy_counts`)
- Optional feature jitter augmentation

### Random grouped training/eval

Enable `random_group_training=True` and/or `random_group_eval=True`:

- Randomly sample `k` reactions per sample (`k` in configurable range, e.g. 1 to 5)
- Pack selected reactions into disconnected blocks
- Group-wise branch/third-body metadata retained per block

Options:

- `*_replace_base=False`: append random packed samples to base set
- `*_replace_base=True`: replace base set with random packed samples

### Train/eval disjointness

`ensure_disjoint_train_eval=True` removes eval samples that overlap train by semantic key:

- key = `(sorted reaction ids in sample, reaction_repeat)`

This is done after dataset assembly.

## Model Architecture

Implemented in [model.py](/users/3/du000298/GNN_label/model.py).

### Encoder (reactant graph -> node embeddings)

- The encoder is a stack of `layers` GINEConv blocks.
- Input node feature size is `node_in` (default 4 from data prep).
- Hidden size is `hidden` (default 64 or 128 in your runs).
- Edge attributes in message passing are reactant bond orders (`edge_attr` with `edge_dim=1`).

Conceptually, for layer `l`:

- `h_i^(l+1) = MLP_l((1 + eps) * h_i^(l) + sum_{j in N(i)} phi(h_j^(l), e_ji))`
- `eps` is controlled by `self_eps` (`train_eps=False`, fixed scalar).

Because `edge_index` includes only existing reactant bonds, message passing stays on the reactant connectivity, while bond prediction is later done on all candidate atom pairs.

### Candidate edge set and decoder target

- Prediction candidates are all upper-triangle pairs `(i,j)` from `pair_i`, `pair_j`.
- For each pair, the model predicts class logits:
  - bond-change mode: 7 classes for `[-3..+3]`
  - direct-bond mode: 4 classes for `[0..3]`

### Group-aware pooling (critical for packed samples)

When multiple disconnected reaction blocks are packed into one `Data` object:

- `node_group_id` marks each node’s block.
- `pair_group_id` marks each candidate pair’s block.

The model computes a pooled context **per group** (not one global vector):

- If `molecule_balanced_pool=True`:
  - first mean-pool node embeddings per molecule inside each group
  - then sum those molecule means
- Otherwise:
  - mean-pool all nodes in that group directly

Then for each candidate edge `k`, gather pooled context from its group:

- `g_k = pair_group_id[k]`
- `z_k = z_group[g_k]`

This is what prevents branch/group leakage across packed blocks.

### Branching: metadata and conditioning

Branch metadata is built in [data_prep.py](/users/3/du000298/GNN_label/data_prep.py):

- reactions are grouped by reactant signature + third-body flag
- inside each group:
  - `branch_id` = index of this product branch
  - `branch_count` = total branches for that reactant group
  - `branch_value = branch_id / (branch_count - 1)` (or `0` if no branching)

For packed/mixed samples, this is carried per group:

- `branch_id_by_group`
- `branch_count_by_group`
- `branch_value_by_group`

The model supports two branch-conditioning modes inside the decoder:

1. `scalar`
- For each edge, gather `branch_value_by_group[g_k]` as a 1D scalar feature.

2. `contextual` (default in your current config)
- Learn embedding table `Embedding(max_branch_slots, branch_emb_dim)`.
- For each group `g`, lookup `e_g = Emb(branch_id_g)`.
- Concatenate with group pooled reactant context `z_g`.
- Pass through `branch_fuse` MLP:
  - `b_g = branch_fuse([z_g ; e_g])` with output size `branch_context_dim`.
- For each edge `k`, use `b_{g_k}`.

Why `contextual` helps:

- Same branch id can behave differently depending on reactant context.
- The control signal is stronger than a single scalar.

### Branching modes (current implementation)

#### Mode A: lookup-table / explicit branch index

Active when:

- `use_latent_branching=False`
- `use_branch_feature=True`

Behavior:

- branch identity is provided explicitly via `branch_id_by_group` / `branch_count_by_group`
- decoder receives explicit branch condition (`scalar` or `contextual`)
- third-body is also explicit (`third_body_by_group`)

In stochastic inference:

- branch can be selected by:
  - `random`, or
  - `best_confidence` over known branch IDs for the sampled reactant signature

This mode is stable and fast, but branch space is tied to known branch templates from training chemistry.

#### Mode B: latent branching (CVAE-style)

Active when:

- `use_latent_branching=True`

Behavior:

- explicit branch feature is disabled in the main script to avoid duplicate branch signals
- decoder branch variation is controlled by sampled latent `z` per group
- training uses posterior (`q`) with target statistics; inference uses prior (`p`)
- objective includes KL regularization: `L = CE + beta * KL (+ rate loss if enabled)`
- `beta` is warmed up via `latent_kl_warmup_epochs`

In stochastic inference:

- no explicit branch-id lookup is used for decoder conditioning
- multiple latent samples per event can be decoded (`LATENT_SAMPLES_PER_EVENT`)
- one sample is picked by `LATENT_SAMPLE_SELECTION` (`random` or confidence)

### Third-body conditioning

- Per group scalar `third_body_by_group` (`0/1`) is gathered per edge by `pair_group_id`.
- This gives the decoder an explicit channel for third-body vs non-third-body pathways even with similar reactant topology.

### Final edge decoder input

For candidate edge `k=(i,j)` the concatenated feature is:

- node pair embeddings: `h_i || h_j`
- reactant bond on that pair: `react_edge[k]`
- optional group context: `g_k` (if `molecule_balanced_pool=True`)
- optional branch feature:
  - scalar `branch_value_k`, or
  - contextual `b_k`
- optional third-body scalar `t_k`
- optional latent branch vector `z_k` (if `use_latent_branching=True`)

So decoder input dimension is:

- `2*hidden + 1`
- `+ hidden` if group pooled context enabled
- `+ 1` (scalar branch) or `+ branch_context_dim` (contextual branch)
- `+ 1` if third-body enabled
- `+ latent_dim` if latent branching enabled

`edge_mlp` then maps this to class logits.

### Practical summary

- Deterministic branch mode:
  - branch id/count/value are explicit metadata features.
  - branch selection in stochastic inference can be `random` or `best_confidence` over known template variants.
- Latent branch mode:
  - branch is represented by sampled `z` (no explicit branch id lookup for decoder conditioning).
  - at stochastic inference, multiple latent samples can be decoded per event and selected by confidence.
- Group-aware conditioning still applies in both modes, which is critical for packed multi-reaction samples.

## Training Pipeline

Entry script: [reaction_dataset_prediction.py](/users/3/du000298/GNN_label/reaction_dataset_prediction.py)

### Loop

- Build dataset bundle from config
- Instantiate model from `ModelConfig`
- Train with [train.py](/users/3/du000298/GNN_label/train.py):
  - class-weighted cross entropy
  - optional latent KL regularization (`CE + beta * KL`) with warmup
  - optional rate regression loss on standardized targets
    - `rate_loss = masked MSE(rate_pred_by_group, rate_target_by_group)`
    - total = `CE + beta*KL + rate_loss_weight*rate_loss`
  - AdamW
  - optional ReduceLROnPlateau scheduler
  - optional gradient clipping
  - snapshot checkpoints at configured epochs
- Save loss plot
- Save final model checkpoint (`model.pt`)
- Run evaluation + artifact export

### Batching detail

For `batch_size>1`, train uses list-collation mode due variable-sized graph-level tensors (e.g., `react_adj` has different `N x N` across samples).

## Validation / Output Artifacts

Implemented in [validate.py](/users/3/du000298/GNN_label/validate.py).

For each eval sample, outputs include:

- `pred_mat.txt`
- `true_mat.txt`
- `reactant_adj.txt`
- `node_embeddings.txt`
- `bond_class_probs.txt` (top classes + full probabilities + error flags)
- `rate_coeffs.txt` (pred vs target for each valid rate field, plus physical values)
- `error.txt`
- `graph_reactant.png`
- `graph_predicted.png`
- `graph_target.png`

And global:

- `summary.txt`
- optional snapshot predicted graphs by epoch

## Inference Scripts

### 1) Copy-count inference

[reaction_inference_copies.py](/users/3/du000298/GNN_label/reaction_inference_copies.py)

Purpose:

- load saved `model.pt`
- evaluate on a new set of duplication factors (`INFERENCE_COPY_COUNTS`)
- reuse standard evaluation outputs

### 2) Stochastic pool inference

[reaction_stochastic_inference.py](/users/3/du000298/GNN_label/reaction_stochastic_inference.py)

Purpose:

- simulate repeated reaction events from an initial molecule pool
- evaluate many candidate reaction channels per step (not a single pre-picked pair)
- include both branching modes:
  - deterministic mode: all branch-id variants for matching templates become channels
  - latent mode: sampled latent decodes become channels (`LATENT_SAMPLES_PER_EVENT`)
- use temperature-dependent rate coefficients to weight channels by probability

Per step (current implementation):

1. Enumerate feasible reactant groups from the current pool:
   - unimolecular: `[A]`
   - bimolecular: `[A,B]` (including `[A,A]` when count >= 2)
2. Expand each reactant group into channels using:
   - third-body state (`False/True`)
   - branch variants (or latent samples)
3. For each channel:
   - run model inference to decode product graph
   - get rate coefficients from model auxiliary rate head (`rate_pred_by_group`)
   - de-standardize rates using checkpoint `rate_mean` / `rate_std`
   - fallback to template reaction rate if needed
4. Convert rates to effective kinetics:
   - Arrhenius: `ln k = ln A + n ln T - Ea / (R T)`
   - three-body and falloff use pressure-based effective concentration scaling
5. Compute log-propensity:
   - `log_prop = log(k_eff) + log(count_factor) + log(collision_prior) + log(third_body_prior)`
6. Softmax-normalize over all channels and sample one channel.
7. Apply selected channel product update to the molecule pool.

Outputs:

- `events.txt` (event log)
- per-event folder with reactant/predicted graphs and matrices
- per-event `selected_channel.txt`:
  - number of candidate channels
  - selected probability
  - `log_k`, rate type, template reaction id
- `overall_steps/`:
  - overall pool graph per step
  - pool composition per step
  - overall adjacency per step
- `overall_steps.gif` (animated global evolution)
- `species_counts.png` (species count vs step)

## Typical Workflow

1. Train:

```bash
cd /users/3/du000298/GNN_label
python reaction_dataset_prediction.py
```

2. Evaluate same model on alternate copy factors:

```bash
python reaction_inference_copies.py
```

3. Run stochastic simulation from molecule pool:

```bash
python reaction_stochastic_inference.py
```

## Notes / Limitations

- Species mapping in stochastic inference is driven by `hydrogen_adjacency.py`.
- Unknown predicted components can be:
  - rejected (`ENFORCE_KNOWN_SPECIES_MAPPING=True`), or
  - auto-registered as `UNK_####` species (`ENFORCE_KNOWN_SPECIES_MAPPING=False`).
- Pressure is currently used in a simplified way:
  - third-body prior probability
  - coarse effective concentration scaling for three-body/falloff rates
  It is not a full physically calibrated reactor model.
- The simulator currently always executes one sampled reaction event per step when channels exist.
  It does not yet include explicit no-event probability or Gillespie waiting-time updates.
- Enabling unseen reactant channels can increase channel count significantly and make inference slower.
- Disjoint train/eval filtering is key-based, not full graph-isomorphism-based.
- Latent branching can increase training cost and may require KL warmup/weight tuning to avoid posterior collapse or unstable branch usage.
