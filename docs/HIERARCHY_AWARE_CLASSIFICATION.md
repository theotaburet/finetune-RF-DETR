# Hierarchy-Aware Classification for RF-DETR Fine-Tuning

> **Status**: Proposal -- not yet implemented
> **Date**: 2026-02-24
> **Author**: Auto-generated from pipeline analysis

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Current Architecture](#2-current-architecture)
- [3. Hierarchy Structure](#3-hierarchy-structure)
- [4. Proposed Approaches](#4-proposed-approaches)
  - [4.1. Approach A -- Label Coarsening (No Model Change)](#41-approach-a--label-coarsening-no-model-change)
  - [4.2. Approach B -- Hierarchy-Weighted Soft Labels](#42-approach-b--hierarchy-weighted-soft-labels)
  - [4.3. Approach C -- Multi-Head Hierarchical Classification](#43-approach-c--multi-head-hierarchical-classification)
  - [4.4. Approach D -- Post-Hoc Confusion Reweighting via Evaluation](#44-approach-d--post-hoc-confusion-reweighting-via-evaluation)
  - [4.5. Approach E -- Hierarchical Label Smoothing in COCO Targets](#45-approach-e--hierarchical-label-smoothing-in-coco-targets)
- [5. Comparative Analysis](#5-comparative-analysis)
  - [5.1. Summary Table](#51-summary-table)
  - [5.2. Performance Analysis](#52-performance-analysis)
  - [5.3. Flexibility Analysis](#53-flexibility-analysis)
  - [5.4. Why Runtime Patching is Not a Fork](#54-why-runtime-patching-is-not-a-fork)
  - [5.5. Ruling Out Approach E](#55-ruling-out-approach-e)
- [6. Recommended Strategy](#6-recommended-strategy)
- [7. Implementation Roadmap](#7-implementation-roadmap)
  - [7.1. Hierarchy Fetching and Caching](#71-hierarchy-fetching-and-caching)
  - [7.2. Distance Matrix](#72-distance-matrix)
  - [7.3. Soft Target Matrix](#73-soft-target-matrix)
  - [7.4. Runtime Patch for SetCriterion.loss_labels()](#74-runtime-patch-for-setcriterionloss_labels)
  - [7.5. Configurable Label Depth](#75-configurable-label-depth-for-curriculum-phase-2)
  - [7.6. Hierarchical Evaluator](#76-hierarchical-evaluator)
  - [7.7. Configuration Schema](#77-configuration-schema)
  - [7.8. Integration Points in the Pipeline](#78-integration-points-in-the-pipeline)
- [8. Appendix: EKB Hierarchy API](#8-appendix-ekb-hierarchy-api)

## 1. Problem Statement

### 1.1. The Fine-Grained Classification Challenge

The EKB database exposes a **tree-structured class taxonomy** with 5 top-level branches
(anthropogenic, biological, geoacoustic, unknown, soundscape) and up to 4 levels of depth.
The pipeline currently extracts only the **leaf label** from each event's `label_hierarchy`
string (e.g., `"anthropogenic + ship + cargo + container ship"` becomes `"container ship"`),
discarding all hierarchical structure before training.

This creates two concrete problems:

1. **Flat cross-entropy treats all confusions equally.** Predicting "container ship" when
   the truth is "bulk carrier" (same parent: `ship/cargo`) incurs the same loss penalty as
   predicting "container ship" when the truth is "odontoceti" (different top-level branch).
   From a domain perspective, the first confusion is far less severe.

2. **Data sparsity at the leaves.** Some leaf classes have very few examples (e.g., "USV",
   "AUV", "sirenian"). Training a flat classifier across 29 potential leaf classes with
   limited and imbalanced data leads to poor generalization at the leaves while wasting
   model capacity on distinctions the model cannot yet learn.

### 1.2. Desired Behavior

- **Within-branch confusion should be less penalized** than cross-branch confusion.
- The model should first learn to distinguish the coarse categories (anthropogenic vs.
  biological vs. geoacoustic) and then progressively refine within branches.
- The evaluation metrics should reflect hierarchical correctness: a prediction within the
  right branch should score higher than one in the wrong branch.
- The solution must work **without forking the `rfdetr` library**, or if forking is
  required, the changes must be minimal and maintainable.

### 1.3. Hardware and Training Constraints

| Constraint            | Value                                          |
| --------------------- | ---------------------------------------------- |
| GPU                   | RTX 4070 Laptop (8 GB VRAM)                    |
| Batch size            | 2 (OOM above this)                             |
| Model                 | RF-DETR Base (resolution 560, DINOv2 encoder)  |
| Current class count   | 10 leaf classes from downloaded data           |
| Potential class count | 29 leaf classes across the full EKB taxonomy   |
| Training framework    | `rfdetr` library (pip-installed, not editable) |

## 2. Current Architecture

### 2.1. Label Flow (Status Quo)

```
EKB API  ──label_hierarchy──►  Sidecar JSON  ──_extract_label()──►  Leaf name
                                                                        │
                                                                        ▼
COCO JSON  ◄──category_id (1-based)──  PreprocessStep  ◄──  bbox.category
     │
     ▼
RF-DETR  ──_load_classes()──►  num_classes = len(categories) + 1
     │
     ▼
nn.Linear(hidden_dim, num_classes + 1)  ──sigmoid──►  Per-class BCE loss
```

Key details:

- `_extract_label()` in `src/rf_detr_finetuning/dataprocessor/chunker.py:41-59` splits on
  `" + "` or `" > "` and returns the last segment.
- Category IDs are assigned dynamically during preprocessing in encounter order (1-based).
- All categories get `"supercategory": "object"` -- RF-DETR's `_load_classes()` filters
  out categories with `supercategory == "none"`.

### 2.2. RF-DETR Classification Loss

The `rfdetr` library uses **sigmoid-based per-class binary cross-entropy**, not softmax
cross-entropy. The default mode (when fine-tuning via `TrainConfig`) is **IoU-Aware BCE**
(`ia_bce_loss=True`):

```python
# In rfdetr/models/lwdetr.py SetCriterion.loss_labels()
# Soft target: blend of predicted probability and IoU quality
t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)  # alpha=0.25

# BCE with soft targets
loss_ce = neg_weights * logits - F.logsigmoid(logits) * (pos_weights + neg_weights)
```

This is significant because:

- **Each class dimension is an independent binary classifier** (sigmoid, not softmax).
  There is no explicit mutual exclusion between classes.
- The loss for positive samples is modulated by IoU quality -- high-quality localizations
  get stronger gradient signals.
- The matching is done via the Hungarian algorithm with focal-loss-based classification
  cost.

### 2.3. What the `supercategory` Field Does (and Does Not Do)

Despite the COCO format supporting `supercategory`, RF-DETR uses it **only** as a filter
(`!= "none"`). It does **not** use `supercategory` for hierarchical loss computation,
grouped evaluation, or label smoothing. Setting `supercategory` to actual parent names
(e.g., `"ship"`) has zero effect on training behavior.

## 3. Hierarchy Structure

### 3.1. The EKB Taxonomy Tree

The hierarchy is available via `GET /ekb/api/hierarchy` on the EKB instance at
`http://192.168.50.103:8080/`. Each node has:

```json
{
  "id": "anthropogenic/ship/cargo/container ship",
  "name": "container ship",
  "parentId": "anthropogenic/ship/cargo",
  "color": "#FF5733",
  "description": "...",
  "customFields": {},
  "children": []
}
```

The full tree (simplified):

```
ROOT
├── anthropogenic
│   ├── explosion
│   │   └── airgun
│   ├── other anthropogenic
│   │   └── inherent sound
│   ├── ship
│   │   ├── cargo
│   │   │   ├── container ship
│   │   │   └── bulk carrier
│   │   ├── passenger ship
│   │   ├── sailboat
│   │   ├── tanker
│   │   ├── trawler
│   │   ├── tugboat
│   │   ├── yacht
│   │   ├── warship
│   │   ├── USV
│   │   ├── outboard
│   │   └── speedboat
│   ├── sonar
│   ├── aircraft
│   └── sub
│       └── AUV
├── biological
│   ├── cetacean
│   │   ├── mysticeti
│   │   └── odontoceti
│   ├── fish
│   ├── invertebrate
│   ├── other biological
│   ├── pinniped
│   └── sirenian
├── geoacoustic
├── unknown
└── soundscape
```

### 3.2. Hierarchy Depth and Distance Metric

Given two leaf nodes, the **taxonomic distance** can be defined as the number of edges
to their Lowest Common Ancestor (LCA):

| Pair                          | LCA           | Distance |
| ----------------------------- | ------------- | -------- |
| container ship ↔ bulk carrier | cargo         | 2 (1+1)  |
| container ship ↔ tanker       | ship          | 3 (2+1)  |
| container ship ↔ airgun       | anthropogenic | 4 (3+1)  |
| container ship ↔ odontoceti   | ROOT          | 6 (3+3)  |
| mysticeti ↔ odontoceti        | cetacean      | 2 (1+1)  |
| mysticeti ↔ fish              | biological    | 3 (1+2)  |
| mysticeti ↔ container ship    | ROOT          | 6 (3+3)  |

This distance can be normalized to `[0, 1]` by dividing by the maximum possible distance
in the tree (currently 6 for the deepest pair across branches).

### 3.3. Hierarchy Data Source

The `/ekb/api/hierarchy` endpoint is **not currently called** by the pipeline. The
`EKBAPIClient` in `run_download_data.py` only queries `/sdk/labels` and `/sdk/sources`.

**Action required**: Implement an `EKBAPIClient.fetch_hierarchy()` method that queries
`GET /ekb/api/hierarchy` and caches the result locally as `data/hierarchy.json`.

## 4. Proposed Approaches

### 4.1. Approach A -- Label Coarsening (No Model Change)

**Principle**: Instead of training on 29 leaf classes, collapse the taxonomy to a
configurable depth and train on the resulting coarser labels.

**Mechanism**:

```yaml
# config/pipeline.yaml
hierarchy:
  training_depth: 2   # 0=root, 1=top-level, 2=second level, null=leaf
```

At depth 2, the label `"anthropogenic/ship/cargo/container ship"` becomes `"ship"`,
and `"biological/cetacean/odontoceti"` becomes `"cetacean"`. The resulting class set
is smaller and more balanced:

| Depth | Classes | Example labels                                              |
| ----- | ------- | ----------------------------------------------------------- |
| 1     | 5       | anthropogenic, biological, geoacoustic, unknown, soundscape |
| 2     | ~14     | explosion, ship, sonar, aircraft, sub, cetacean, fish, ...  |
| 3     | ~22     | airgun, cargo, passenger ship, mysticeti, odontoceti, ...   |
| leaf  | ~29     | container ship, bulk carrier, AUV, ...                      |

**Implementation**:

1. Fetch the hierarchy tree from `/ekb/api/hierarchy`.
2. In `_extract_label()`, instead of always taking the leaf, truncate the path at
   `training_depth` levels.
3. The rest of the pipeline (COCO categories, training) is unchanged.

**Advantages**:

- Zero changes to `rfdetr` library.
- Fewer classes means more samples per class, better convergence.
- The model implicitly cannot make cross-branch errors at the coarsened level.
- Can be used as a **curriculum**: train at depth 1, fine-tune at depth 2, then depth 3.

**Disadvantages**:

- Loses fine-grained discrimination entirely at the chosen depth.
- Requires reprocessing spectrograms when changing depth (or storing hierarchy in sidecar).
- Not a smooth trade-off -- discrete depth levels.

**Complexity**: Low. ~50 lines of code in `_extract_label()` + config changes.

### 4.2. Approach B -- Hierarchy-Weighted Soft Labels

**Principle**: Instead of a hard one-hot target, assign soft probability mass to
classes that are hierarchically close to the ground truth. This is analogous to label
smoothing, but the smoothing distribution follows the tree structure.

**Mechanism**:

For a ground-truth class $c$, construct a soft target vector $\\mathbf{y}$ where:

$$y_i = \\begin{cases}
1 - \\epsilon & \\text{if } i = c \\
\\epsilon \\cdot w(i, c) / Z & \\text{if } i \\neq c
\\end{cases}$$

where $w(i, c) = \\exp(-\\lambda \\cdot d(i, c))$ is a weight that decays exponentially
with the taxonomic distance $d(i, c)$, and $Z = \\sum\_{j \\neq c} w(j, c)$ normalizes
the smoothed mass.

Example with $\\epsilon = 0.1$, $\\lambda = 1.0$:

| Ground truth: container ship | Target |
| ---------------------------- | ------ |
| container ship               | 0.90   |
| bulk carrier (dist=2)        | 0.033  |
| tanker (dist=3)              | 0.012  |
| airgun (dist=4)              | 0.004  |
| odontoceti (dist=6)          | 0.001  |
| ...                          | ...    |

**Implementation challenge**: RF-DETR's IoU-Aware BCE constructs its own soft targets
internally (blending predicted probability with IoU quality). Injecting external soft
targets requires modifying `SetCriterion.loss_labels()` in `rfdetr/models/lwdetr.py`.

**Two sub-options**:

**(B1) Fork `rfdetr` and patch `loss_labels()`**:

Replace the hard `target_classes_onehot` construction with a precomputed soft target
matrix. This is ~30 lines of change in `lwdetr.py` but requires maintaining a fork.

```python
# In SetCriterion.loss_labels(), replace:
target_classes_onehot = torch.zeros([...], device=...)
target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

# With:
target_classes_onehot = self.hierarchy_soft_targets[target_classes]
# Where hierarchy_soft_targets is a [num_classes, num_classes] matrix
# precomputed from the taxonomy tree
```

**(B2) Custom loss wrapper without forking**:

Register a training callback that modifies the loss after computation. This is fragile
and not supported by the `rfdetr` training API.

**Advantages**:

- Directly encodes the hierarchy into the training signal.
- Smooth trade-off via $\\epsilon$ and $\\lambda$ hyperparameters.
- Compatible with the sigmoid per-class architecture -- each class target can be
  independently softened.

**Disadvantages**:

- Requires forking `rfdetr` (option B1) or fragile monkey-patching (option B2).
- Interaction with IoU-Aware BCE is non-trivial -- the soft targets would need to be
  blended with the IoU-based target modulation.
- Hyperparameter tuning required ($\\epsilon$, $\\lambda$).

**Complexity**: Medium-High. Fork maintenance + careful loss integration.

### 4.3. Approach C -- Multi-Head Hierarchical Classification

**Principle**: Instead of a single flat classification head, add separate classification
heads for each level of the hierarchy. The final prediction is the product of
conditional probabilities along the path.

**Mechanism**:

```
                        ┌─ Head L1: 5 classes (anthropogenic, biological, ...)
Decoder output ────────┤
                        ├─ Head L2: 14 classes (ship, explosion, cetacean, ...)
                        │
                        └─ Head L3: 29 classes (container ship, bulk carrier, ...)
```

During training, each head receives supervision at its level. The final inference
prediction is:

$$P(\\text{container ship}) = P(\\text{anthropogenic}) \\times P(\\text{ship} | \\text{anthropogenic}) \\times P(\\text{container ship} | \\text{ship})$$

**Implementation**:

This requires significant changes to `rfdetr`:

- Add multiple `nn.Linear` heads to `LWDETRTransformer`.
- Modify `SetCriterion` to compute loss at each level.
- Modify `PostProcess` to combine predictions across levels.

**Advantages**:

- The model explicitly learns the hierarchy.
- Coarse-level heads converge fast (few classes, more data per class).
- Natural curriculum effect: early in training, the coarse heads dominate the gradient.
- Evaluation can report per-level accuracy.

**Disadvantages**:

- Major fork of `rfdetr` -- deep changes to model architecture and loss.
- Increased VRAM usage (multiple heads, though each is small).
- Complex inference logic to combine head outputs.
- Requires careful handling of leaf-only nodes (geoacoustic, unknown, soundscape appear
  only at level 1 with no children).

**Complexity**: Very High. Full fork with architectural changes.

### 4.4. Approach D -- Post-Hoc Confusion Reweighting via Evaluation

**Principle**: Train the model with flat labels as-is, but modify the **evaluation
metrics** to be hierarchy-aware. This doesn't change training at all -- it changes how
we interpret and report results.

**Mechanism**:

1. **Hierarchical mAP**: Instead of binary correct/incorrect for class matching,
   use a graded score based on taxonomic distance:

   $$\\text{match_score}(pred, gt) = \\begin{cases}
   1.0 & \\text{if } pred = gt \\
   \\gamma^{d(pred, gt)} & \\text{if } d(pred, gt) \\leq \\tau \\
   0.0 & \\text{otherwise}
   \\end{cases}$$

   where $\\gamma \\in (0, 1)$ is a decay factor and $\\tau$ is a maximum distance
   threshold.

2. **Hierarchical confusion matrix**: Group classes by their parent branch and
   report within-branch vs. cross-branch confusion rates separately.

3. **Branch-level mAP**: Compute mAP at each hierarchy level by collapsing predictions
   to that level before evaluation.

**Implementation**:

```python
class HierarchicalEvaluator:
    """COCO-style evaluation with hierarchy-aware matching."""

    def __init__(self, hierarchy_tree: dict, gamma: float = 0.5):
        self.distance_matrix = self._compute_distances(hierarchy_tree)
        self.gamma = gamma

    def compute_hierarchical_map(self, predictions, ground_truths):
        # Standard COCO eval but with graded class matching
        ...
```

This requires writing a custom evaluator but **zero changes** to `rfdetr`.

**Advantages**:

- No training changes -- works with any trained model.
- Can be applied retroactively to existing training runs.
- Gives richer insight into model behavior.
- Can inform decisions about which approach to adopt for training.

**Disadvantages**:

- Does not actually improve the model's training signal.
- The model still wastes capacity penalizing within-branch confusion.
- Non-standard metrics may be harder to compare with published results.

**Complexity**: Low-Medium. Custom evaluator code, no `rfdetr` changes.

### 4.5. Approach E -- Hierarchical Label Smoothing in COCO Targets

**Principle**: Manipulate the COCO annotation file before training to encode
hierarchy information through **auxiliary annotations**. Instead of modifying the
loss, modify the data.

**Mechanism**:

For each annotated bounding box with leaf class $c$, add **additional annotations**
at the same bbox coordinates for each ancestor class. Use a separate category namespace
for ancestor labels.

Example: A bbox labeled "container ship" generates:

| Annotation | Category               | Source       |
| ---------- | ---------------------- | ------------ |
| Original   | container ship (id=15) | ground truth |
| Auxiliary  | cargo (id=30)          | parent       |
| Auxiliary  | ship (id=31)           | grandparent  |
| Auxiliary  | anthropogenic (id=32)  | top-level    |

The model now sees `num_leaf_classes + num_ancestor_classes` total categories. At
inference, only the leaf class predictions are used; ancestor predictions are discarded.

**Variant (E2) -- Merged parent labels**:

Instead of separate ancestor categories, add **duplicate annotations** where the leaf
label is replaced by the parent label. This means some bboxes will have both
"container ship" and "ship" as labels. Since RF-DETR uses sigmoid (not softmax),
multiple positive labels per box are valid.

**Implementation**:

Modify `PreprocessStep.run()` to:

1. Load the hierarchy tree.
2. For each bbox annotation, look up the full ancestor path.
3. Generate additional annotations for ancestor levels.
4. Add ancestor categories to the COCO categories list.

**Advantages**:

- No changes to `rfdetr` library.
- The sigmoid-based loss naturally supports multi-label targets.
- The model learns to activate both "container ship" and "ship" and "anthropogenic"
  for the same detection, encoding hierarchy implicitly.
- At inference, the ancestor activations provide a confidence measure for branch
  membership.

**Disadvantages**:

- **Hungarian matcher incompatibility (deal-breaker).** The Hungarian matcher performs
  1:1 bipartite assignment between predicted queries and ground-truth annotations. If
  we create duplicate annotations for the same physical bounding box (one for
  "container ship", another for "ship"), the matcher will assign **separate queries**
  to each. This forces the model to predict N detections per object (one per hierarchy
  level), doubling or tripling detection count. NMS cannot reliably deduplicate because
  the class IDs differ. This fundamentally breaks the detection paradigm.
- Inflates annotation count (3-4x depending on tree depth).
- The model may learn to rely on easy ancestor predictions and underfit leaf predictions.
- Category count increases, increasing the classification head size.

**Complexity**: Medium for implementation, but the matcher incompatibility makes this
approach **not viable** for RF-DETR or any DETR-family detector that uses bipartite
matching.

## 5. Comparative Analysis

### 5.1. Summary Table

| Criterion               | A: Coarsening         | B: Soft Labels         | C: Multi-Head     | D: Eval Only | E: Aux Annotations      |
| ----------------------- | --------------------- | ---------------------- | ----------------- | ------------ | ----------------------- |
| `rfdetr` fork required  | No                    | **No** (runtime patch) | Yes               | No           | No                      |
| Training signal change  | Indirect              | **Direct**             | Direct            | None         | Indirect                |
| Implementation effort   | Low                   | **Medium**             | Very High         | Low-Medium   | Medium                  |
| VRAM overhead           | None (fewer classes)  | None                   | Small             | None         | Small (more categories) |
| Hyperparameters added   | 1 (depth)             | 2 (epsilon, lambda)    | 1 (level weights) | 1 (gamma)    | 1 (ancestor depth)      |
| Reversibility           | Full                  | **Full** (toggle flag) | Difficult         | Full         | Full                    |
| Can combine with others | Yes (A+B, A+D)        | **Yes (B+A, B+D)**     | Standalone        | Yes (any)    | N/A (broken)            |
| Curriculum potential    | Natural (depth 1-2-3) | Via epsilon schedule   | Natural           | N/A          | N/A                     |
| **Viable**              | Yes                   | **Yes**                | Yes (impractical) | Yes          | **No** (matcher)        |

### 5.2. Performance Analysis

The approaches can be ranked on their expected impact on the training signal:

**Tier 1 -- Direct hierarchy encoding in the loss (B, C)**

These approaches directly modify how the gradient flows during backpropagation. When the
model confuses "container ship" with "bulk carrier" (distance=2), the loss is reduced
compared to confusing it with "odontoceti" (distance=6). The gradient signal is
**proportional to the severity of the mistake**.

Literature strongly supports this:

- Bertinetto et al. (*Making Better Mistakes*, NeurIPS 2020) showed that hierarchy-aware
  cross-entropy consistently improves "mistake severity" metrics while maintaining or
  improving flat accuracy.
- Wu et al. (*Learning with Hierarchical Complementary Labels*) demonstrated that
  soft-target approaches converge faster on fine-grained tasks with limited data.

Between B and C: Multi-head (C) is theoretically optimal but requires a deep rfdetr fork
with architectural changes (new `nn.Linear` heads, modified forward pass, custom
`PostProcess`). Soft labels (B) achieves 80-90% of the benefit with 10% of the
implementation cost by modifying only the target construction, not the architecture.

**Tier 2 -- Indirect hierarchy encoding via data (A, E)**

These approaches modify the training data rather than the loss. Coarsening (A) reduces
the problem to fewer classes, which helps convergence but permanently loses fine-grained
distinctions. Auxiliary annotations (E) is architecturally incompatible with DETR's
Hungarian matcher (see section 4.5), so it is ruled out.

Coarsening alone limits the model ceiling: once trained at depth 2, the model *cannot*
distinguish leaf classes. However, as a **warm-up phase** before applying Approach B at
the leaf level, it provides a strong initialization.

**Tier 3 -- No training modification (D)**

Hierarchical evaluation (D) does not improve the model. However, it is **essential as a
diagnostic tool** for any of the training approaches. Without hierarchy-aware metrics,
you cannot distinguish "the model is confused within the ship branch" from "the model is
confused between biological and anthropogenic". This distinction drives all subsequent
decisions.

### 5.3. Flexibility Analysis

Flexibility means: how easily can the approach adapt when the hierarchy changes, classes
are added/removed, or hyperparameters need tuning?

**Approach B (Soft Labels via Runtime Patch) is the most flexible** because:

1. **Hierarchy changes require zero retraining infrastructure changes.** The distance
   matrix is computed from the API hierarchy tree at pipeline start. If the EKB taxonomy
   is updated (e.g., a new ship subtype is added), the next training run automatically
   picks it up. No data reprocessing, no model architecture change.

2. **Continuous control over hierarchy influence.** The two hyperparameters (epsilon and
   lambda) provide smooth, independent control:

   - **epsilon** (0 to 1): how much total mass is shifted from the true class to related
     classes. At epsilon=0, the loss is identical to standard flat BCE.
   - **lambda** (0 to infinity): how sharply the mass decays with distance. At lambda=0,
     all non-true classes get equal smoothing (uniform label smoothing). At lambda=inf,
     only the closest sibling gets any mass (nearest-neighbor smoothing).

3. **Runtime toggle.** The patch is applied conditionally based on a config flag. Setting
   `hierarchy.enabled: false` restores vanilla RF-DETR training with zero code changes.
   This makes A/B testing trivial.

4. **No fork maintenance.** Unlike a git fork of rfdetr, the monkey-patch lives in our
   codebase. When rfdetr releases a new version, we verify that `loss_labels()` still has
   the same signature and internal structure. If it does, the patch works as-is. If not,
   we adapt the patch (a single function). Compare this to rebasing a fork with merge
   conflicts across multiple files.

**Approach A (Coarsening)** is moderately flexible: changing depth is easy, but it
requires reprocessing spectrograms (the label is baked into the COCO JSON during
preprocessing). It also cannot be smoothly interpolated between depths.

**Approach C (Multi-Head)** is the least flexible: adding a hierarchy level requires
adding a new classification head, retraining from scratch, and modifying inference logic.

### 5.4. Why Runtime Patching is Not a Fork

The original document (section 4.2) described B2 as "fragile monkey-patching." This
deserves clarification. The implementation is a **class-level method replacement**:

```python
# Before calling rfdetr's train_from_config():
from rfdetr.models.lwdetr import SetCriterion

SetCriterion._original_loss_labels = SetCriterion.loss_labels
SetCriterion.loss_labels = hierarchy_aware_loss_labels  # Our function
```

This works because Python resolves instance methods via the class's MRO. Any
`SetCriterion` instance created after the patch uses our function. The original is
preserved as `_original_loss_labels` for fallback.

**Why this is robust for rfdetr specifically:**

- `loss_labels()` has a stable, well-defined signature: `(self, outputs, targets, indices, num_boxes, log=True)`. This is the standard DETR loss interface shared across
  DETR, Deformable-DETR, DINO, and RT-DETR. It is unlikely to change.
- The `ia_bce_loss` branch (our target) constructs `pos_weights` and `neg_weights` as
  dense `[B, Q, C]` tensors. Our patch extends these tensors with additional entries for
  related classes. The rest of the loss computation is unchanged.
- The patch does not modify the model architecture, the matcher, the optimizer, the
  data loader, or any other rfdetr component.

### 5.5. Ruling Out Approach E

Approach E (auxiliary annotations) was initially attractive because it requires no rfdetr
changes. However, the Hungarian matcher incompatibility is a **fundamental blocker**, not
a tuning concern:

1. The matcher minimizes a cost matrix over `[num_queries, num_gt_annotations]`. Adding K
   duplicate annotations per box multiplies the GT count by K.
2. The matcher will assign K different queries to the K duplicates of the same physical
   box (same coordinates, different class labels).
3. The model learns to predict K overlapping detections per object.
4. At inference, NMS cannot merge detections with different class IDs, resulting in K
   output boxes per object with different class predictions.
5. This corrupts both precision and the COCO mAP metric.

A potential workaround -- injecting multi-label targets into a single annotation rather
than duplicating annotations -- leads back to Approach B (modifying the loss targets),
confirming that B is the correct framing.

## 6. Recommended Strategy

### Overview

**Primary approach: Hierarchy-Aware Sigmoid BCE via Runtime Patch (Approach B2)**,
combined with **Curriculum Warm-Up (Approach A)** and **Hierarchical Evaluation
(Approach D)**.

This combination delivers:

- **Best expected performance**: direct hierarchy encoding in the loss gradient
- **Best flexibility**: continuous hyperparameters, auto-updating from API, runtime toggle
- **No rfdetr fork**: the patch is a single function in our codebase
- **Graceful fallback**: set `hierarchy.enabled: false` for vanilla training

### Phase 1: Foundation (Approach D + Infrastructure)

**Goal**: Build the hierarchy infrastructure and evaluation tools. No training changes yet.

1. **Fetch and cache hierarchy** from `GET /ekb/api/hierarchy`.
2. **Build the distance matrix** from the hierarchy tree.
3. **Implement `HierarchicalEvaluator`** that reports:
   - Standard COCO mAP (unchanged baseline).
   - Branch-level confusion rates (within-branch vs. cross-branch).
   - Per-depth accuracy (collapsing predictions to each hierarchy level).
4. **Run a baseline training** with flat labels. Use the evaluator to quantify the
   confusion patterns. This establishes the performance floor and identifies which
   branches have the most within-branch confusion.

**Deliverable**: Baseline mAP + confusion analysis showing where hierarchy-awareness
would help most.

### Phase 2: Curriculum Warm-Up (Approach A)

**Goal**: Give the model strong coarse-level features before fine-grained training.

1. Train at **depth 2** (~14 classes) for N epochs using the standard flat loss.
   At this depth, all ship subtypes collapse to "ship", all cetacean subtypes collapse
   to "cetacean", etc. The model learns to separate the major acoustic categories with
   high data density per class.
2. Save the checkpoint as the **warm-up initialization** for Phase 3.

**Why depth 2, not depth 1**: At depth 1 (5 classes), the distinctions are too coarse
to be useful as a pre-training signal -- "anthropogenic" covers everything from airguns
to ship noise. At depth 2, the model learns meaningful acoustic boundaries (ship vs.
sonar vs. explosion; cetacean vs. fish vs. pinniped) that transfer directly to the
fine-grained task.

**Deliverable**: Warm-up checkpoint with strong branch-level features.

### Phase 3: Hierarchy-Aware Fine-Tuning (Approach B2)

**Goal**: Fine-tune at the leaf level with hierarchy-weighted soft targets.

1. **Apply the runtime patch** to `SetCriterion.loss_labels()` before training.
2. **Load the Phase 2 checkpoint** and switch labels to full leaf depth.
3. Train with the hierarchy-aware loss. The soft targets ensure that:
   - Confusing "container ship" with "bulk carrier" (distance=2) produces a weak gradient.
   - Confusing "container ship" with "odontoceti" (distance=6) produces a strong gradient.
4. **Tune epsilon and lambda** using the hierarchical evaluator:
   - Start with epsilon=0.1, lambda=1.0 (moderate smoothing).
   - If cross-branch confusion is still high, increase lambda (sharper decay).
   - If the model underfits leaf distinctions, decrease epsilon (less smoothing).

**Deliverable**: Final model with hierarchy-aware training + evaluation report.

### Phase 4: Iteration and Ablation

Compare three models using the hierarchical evaluator:

- **Flat baseline** (Phase 1): no hierarchy awareness.
- **Coarsened only** (Phase 2): trained at depth 2, evaluated at depth 2.
- **Hierarchy-aware** (Phase 3): trained at leaf level with soft targets.

Report:

- Standard COCO mAP at the leaf level.
- "Mistake severity" metric: average taxonomic distance of incorrect predictions.
- Per-branch mAP breakdown.
- Within-branch vs. cross-branch confusion ratio.

## 7. Implementation Roadmap

### 7.1. Hierarchy Fetching and Caching

```python
# In run_download_data.py, add to EKBAPIClient:


def fetch_hierarchy(self) -> dict:
    """Fetch the full class hierarchy tree from the EKB API.

    Returns:
        Nested dict representing the taxonomy tree. Each node has keys:
        id, name, parentId, children, color, description, customFields.
    """
    url = f"{self.base_url}/ekb/api/hierarchy"
    response = self.session.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def cache_hierarchy(self, output_path: Path) -> dict:
    """Fetch hierarchy and cache to local JSON file."""
    tree = self.fetch_hierarchy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    return tree
```

### 7.2. Distance Matrix

```python
import numpy as np


def build_distance_matrix(hierarchy_tree: dict, class_names: list[str]) -> np.ndarray:
    """Build a pairwise taxonomic distance matrix for the given classes.

    The distance between two classes is the number of edges to their Lowest
    Common Ancestor (LCA), normalized to [0, 1].

    Args:
        hierarchy_tree: Nested dict from /ekb/api/hierarchy.
        class_names: Ordered list of class names (matching COCO categories).

    Returns:
        Square numpy array of shape [n_classes, n_classes].
    """
    # Build node_name -> ancestor path mapping
    ancestors: dict[str, list[str]] = {}

    def _walk(node: dict, path: list[str]) -> None:
        current_path = path + [node["name"]]
        ancestors[node["name"]] = current_path
        for child in node.get("children", []):
            _walk(child, current_path)

    # Handle both root-with-children and list-of-roots formats
    roots = hierarchy_tree.get("children", [hierarchy_tree])
    for root_node in roots:
        _walk(root_node, [])

    n = len(class_names)
    dist = np.zeros((n, n), dtype=np.float32)

    for i, name_i in enumerate(class_names):
        for j, name_j in enumerate(class_names):
            if i == j:
                continue
            path_i = ancestors.get(name_i, [name_i])
            path_j = ancestors.get(name_j, [name_j])

            # Find LCA depth (length of shared prefix)
            lca_depth = 0
            for k in range(min(len(path_i), len(path_j))):
                if path_i[k] == path_j[k]:
                    lca_depth = k + 1
                else:
                    break

            dist[i, j] = (len(path_i) - lca_depth) + (len(path_j) - lca_depth)

    max_dist = dist.max()
    if max_dist > 0:
        dist /= max_dist

    return dist
```

### 7.3. Soft Target Matrix

```python
def build_soft_target_matrix(
    distance_matrix: np.ndarray,
    epsilon: float = 0.1,
    lam: float = 1.0,
) -> np.ndarray:
    """Build a hierarchy-weighted soft target matrix from pairwise distances.

    For ground-truth class c, the soft target vector y is:
        y[c] = 1 - epsilon
        y[i] = epsilon * exp(-lambda * d(i,c)) / Z   for i != c

    where Z normalizes the off-diagonal weights to sum to epsilon.

    Args:
        distance_matrix: [C, C] normalized distance matrix from build_distance_matrix().
        epsilon: Total probability mass redistributed to related classes.
        lam: Exponential decay rate. Higher = sharper (less mass to distant classes).

    Returns:
        [C, C] matrix where row c is the soft target vector for ground-truth class c.
    """
    C = distance_matrix.shape[0]
    soft_targets = np.zeros((C, C), dtype=np.float32)

    for c in range(C):
        # Compute raw weights for all other classes
        weights = np.exp(-lam * distance_matrix[c])
        weights[c] = 0.0  # Exclude self

        # Normalize off-diagonal to sum to epsilon
        total = weights.sum()
        if total > 0:
            soft_targets[c] = epsilon * weights / total
        soft_targets[c, c] = 1.0 - epsilon

    return soft_targets
```

**Example output** for the EKB taxonomy with epsilon=0.1, lambda=1.0:

```
Ground truth: container ship
  container ship:   0.900   (self)
  bulk carrier:     0.033   (dist=0.33, same parent "cargo")
  tanker:           0.012   (dist=0.50, same grandparent "ship")
  passenger ship:   0.012   (dist=0.50)
  trawler:          0.012   (dist=0.50)
  airgun:           0.004   (dist=0.67, same top-level "anthropogenic")
  odontoceti:       0.001   (dist=1.00, cross-branch)
  fish:             0.001   (dist=1.00, cross-branch)
  ...
```

### 7.4. Runtime Patch for SetCriterion.loss_labels()

This is the core of the approach. The patch replaces the `ia_bce_loss` branch of
`loss_labels()` to inject hierarchy-weighted soft targets into the `pos_weights` and
`neg_weights` tensors.

```python
import torch
import torch.nn.functional as F
from rfdetr.models import box_ops
from rfdetr.models.lwdetr import SetCriterion


def _make_hierarchy_aware_loss_labels(soft_target_matrix_np: "np.ndarray"):
    """Create a patched loss_labels method with hierarchy-aware soft targets.

    Args:
        soft_target_matrix_np: [C, C] soft target matrix from
            build_soft_target_matrix(). Row c is the target vector for GT class c.

    Returns:
        A replacement method for SetCriterion.loss_labels.
    """
    # Convert to tensor once; will be moved to device on first call
    _soft_targets_cpu = torch.from_numpy(soft_target_matrix_np)
    _device_cache = {}

    def _get_soft_targets(device):
        if device not in _device_cache:
            _device_cache[device] = _soft_targets_cpu.to(device)
        return _device_cache[device]

    def hierarchy_aware_loss_labels(
        self, outputs, targets, indices, num_boxes, log=True
    ):
        """Hierarchy-aware classification loss (patched).

        Identical to the original ia_bce_loss branch, except that positive
        weights are spread across related classes according to the hierarchy
        soft target matrix. Falls back to the original for non-ia_bce_loss
        modes.
        """
        # Fall back to original for non-ia_bce_loss modes
        if not self.ia_bce_loss:
            return SetCriterion._original_loss_labels(
                self,
                outputs,
                targets,
                indices,
                num_boxes,
                log=log,
            )

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # [B, Q, C]
        device = src_logits.device
        B, Q, C = src_logits.shape

        soft_targets = _get_soft_targets(device)  # [C, C]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        alpha = self.focal_alpha
        gamma = 2

        # Compute IoU between matched predictions and targets
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
            dim=0,
        )
        iou_targets = torch.diag(
            box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )[0]
        )
        pos_ious = iou_targets.clone().detach()

        prob = src_logits.sigmoid()

        # Initialize weights (default: all positions are negative)
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = prob**gamma

        # --- Original behavior: only GT class gets positive weight ---
        # pos_ind = [id for id in idx]
        # pos_ind.append(target_classes_o)
        # t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
        # t = torch.clamp(t, 0.01).detach()
        # pos_weights[pos_ind] = t
        # neg_weights[pos_ind] = 1 - t

        # --- Hierarchy-aware: spread positive weight to related classes ---
        # For each matched detection, look up the soft target row for its GT class.
        # Shape: [num_matched, C]
        gt_soft = soft_targets[target_classes_o]  # [M, C]

        # IoU-aware base strength for each matched detection
        # t_base[m] = prob_at_gt[m]^alpha * iou[m]^(1-alpha)
        pos_ind_gt = [id for id in idx]
        pos_ind_gt.append(target_classes_o)
        t_base = prob[pos_ind_gt].pow(alpha) * pos_ious.pow(1 - alpha)
        t_base = torch.clamp(t_base, 0.01).detach()  # [M]

        # Scale soft targets by the IoU-aware base strength:
        # pos_weight[m, c] = t_base[m] * soft_target[gt_class[m], c]
        # This gives the GT class ~(1-eps)*t_base, siblings get eps*w*t_base
        weighted_soft = t_base.unsqueeze(1) * gt_soft  # [M, C]

        # Scatter into the full [B, Q, C] pos_weights tensor
        batch_idx, query_idx = idx  # each [M]
        pos_weights[
            batch_idx.unsqueeze(1).expand_as(weighted_soft),
            query_idx.unsqueeze(1).expand_as(weighted_soft),
            torch.arange(C, device=device).unsqueeze(0).expand_as(weighted_soft),
        ] = weighted_soft.to(pos_weights.dtype)

        # Update neg_weights for all positions that received positive weight
        # neg_weight = 1 - pos_weight (for positions with pos_weight > 0)
        mask = pos_weights > 0
        neg_weights[mask] = 1.0 - pos_weights[mask]

        # Standard IA-BCE loss computation (unchanged)
        loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (
            pos_weights + neg_weights
        )
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        if log:
            from rfdetr.util.misc import accuracy

            losses["class_error"] = (
                100
                - accuracy(
                    src_logits[idx],
                    target_classes_o,
                )[0]
            )

        return losses

    return hierarchy_aware_loss_labels


def apply_hierarchy_patch(soft_target_matrix: "np.ndarray") -> None:
    """Apply the hierarchy-aware loss patch to SetCriterion.

    Must be called BEFORE rfdetr's train_from_config() builds the criterion.

    Args:
        soft_target_matrix: [C, C] matrix from build_soft_target_matrix().
    """
    # Preserve original for fallback
    if not hasattr(SetCriterion, "_original_loss_labels"):
        SetCriterion._original_loss_labels = SetCriterion.loss_labels

    SetCriterion.loss_labels = _make_hierarchy_aware_loss_labels(soft_target_matrix)


def remove_hierarchy_patch() -> None:
    """Remove the hierarchy-aware loss patch, restoring vanilla behavior."""
    if hasattr(SetCriterion, "_original_loss_labels"):
        SetCriterion.loss_labels = SetCriterion._original_loss_labels
        del SetCriterion._original_loss_labels
```

**How it integrates with the existing training flow:**

```python
# In run_pipeline.py TrainStep.run(), before calling trainer.train():

if self.config.hierarchy.enabled:
    from rf_detr_finetuning.hierarchy import (
        apply_hierarchy_patch,
        build_distance_matrix,
        build_soft_target_matrix,
    )

    hierarchy_tree = load_cached_hierarchy(self.config.hierarchy.cache_path)
    class_names = [cat["name"] for cat in coco_data["categories"]]

    dist_matrix = build_distance_matrix(hierarchy_tree, class_names)
    soft_targets = build_soft_target_matrix(
        dist_matrix,
        epsilon=self.config.hierarchy.epsilon,
        lam=self.config.hierarchy.lam,
    )
    apply_hierarchy_patch(soft_targets)
    console.print(
        f"  [green]✓[/green] Hierarchy-aware loss enabled "
        f"(epsilon={self.config.hierarchy.epsilon}, "
        f"lambda={self.config.hierarchy.lam})"
    )

# ... trainer.train() runs with patched loss ...
```

### 7.5. Configurable Label Depth (for Curriculum, Phase 2)

```python
def _extract_label(
    hierarchy: str,
    training_depth: int | None = None,
) -> str:
    """Extract label from hierarchy string at the configured depth.

    Args:
        hierarchy: Full hierarchy string, e.g. "anthropogenic + ship + cargo + container ship".
        training_depth: If set, truncate to this depth (1=first level, 2=second, ...).
            None means use the leaf (deepest level).

    Returns:
        Label at the requested depth, or leaf if depth exceeds available levels.
    """
    if not isinstance(hierarchy, str) or not hierarchy:
        return ""

    if " + " in hierarchy:
        parts = [p.strip() for p in hierarchy.split(" + ")]
    elif " > " in hierarchy:
        parts = [p.strip() for p in hierarchy.split(" > ")]
    else:
        return hierarchy.strip()

    if training_depth is None or training_depth >= len(parts):
        return parts[-1]

    return parts[min(training_depth, len(parts) - 1)]
```

### 7.6. Hierarchical Evaluator

```python
class HierarchicalEvaluator:
    """Hierarchy-aware detection evaluation.

    Extends standard COCO mAP with:
    - Branch-level confusion analysis (within vs. cross-branch)
    - Per-depth accuracy reporting
    - Mistake severity metric (average taxonomic distance of errors)
    """

    def __init__(
        self,
        hierarchy_tree: dict,
        class_names: list[str],
        distance_matrix: "np.ndarray | None" = None,
    ):
        self.hierarchy_tree = hierarchy_tree
        self.class_names = class_names

        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = build_distance_matrix(hierarchy_tree, class_names)

        # Precompute branch membership (top-level ancestor for each class)
        self.class_to_branch: dict[str, str] = {}
        for node in hierarchy_tree.get("children", []):
            self._assign_branch(node, node["name"])

    def _assign_branch(self, node: dict, branch: str) -> None:
        self.class_to_branch[node["name"]] = branch
        for child in node.get("children", []):
            self._assign_branch(child, branch)

    def mistake_severity(
        self,
        pred_class_ids: list[int],
        gt_class_ids: list[int],
    ) -> float:
        """Average normalized taxonomic distance for incorrect predictions.

        Returns 0.0 if all predictions are correct. Returns values in [0, 1]
        where higher means worse mistakes (more cross-branch confusion).
        """
        errors = []
        for pred, gt in zip(pred_class_ids, gt_class_ids):
            if pred != gt:
                errors.append(self.distance_matrix[pred, gt])
        return float(np.mean(errors)) if errors else 0.0

    def confusion_analysis(
        self,
        pred_class_ids: list[int],
        gt_class_ids: list[int],
    ) -> dict:
        """Analyze confusion patterns relative to the hierarchy.

        Returns:
            Dict with:
            - total_errors: total number of misclassifications
            - within_branch: errors where pred and gt share the same top-level branch
            - cross_branch: errors where pred and gt are in different branches
            - within_ratio: within_branch / total_errors
            - mistake_severity: average taxonomic distance of errors
            - branch_matrix: {gt_branch: {pred_branch: count}}
        """
        within = 0
        cross = 0
        branch_matrix: dict[str, dict[str, int]] = {}

        for pred, gt in zip(pred_class_ids, gt_class_ids):
            if pred == gt:
                continue
            pred_name = self.class_names[pred]
            gt_name = self.class_names[gt]
            pred_branch = self.class_to_branch.get(pred_name, "unknown")
            gt_branch = self.class_to_branch.get(gt_name, "unknown")

            if gt_branch not in branch_matrix:
                branch_matrix[gt_branch] = {}
            branch_matrix[gt_branch][pred_branch] = (
                branch_matrix[gt_branch].get(pred_branch, 0) + 1
            )

            if pred_branch == gt_branch:
                within += 1
            else:
                cross += 1

        total = within + cross
        return {
            "total_errors": total,
            "within_branch": within,
            "cross_branch": cross,
            "within_ratio": within / total if total > 0 else 0.0,
            "mistake_severity": self.mistake_severity(pred_class_ids, gt_class_ids),
            "branch_matrix": branch_matrix,
        }
```

### 7.7. Configuration Schema

```yaml
# config/pipeline.yaml (additions)
hierarchy:
  enabled: true                            # Master toggle for hierarchy-aware training

  # Hierarchy source
  source: "api"                            # "api" or "file"
  cache_path: "data/hierarchy.json"        # Local cache
  api_endpoint: "/ekb/api/hierarchy"       # API route

  # Label depth for curriculum training (null = leaf)
  training_depth: null

  # Soft target parameters (only used when enabled: true)
  epsilon: 0.1                             # Mass redistributed to related classes [0, 1]
  lam: 1.0                                # Distance decay rate (higher = sharper)

  # Evaluation
  eval:
    report_per_depth: true
    report_branch_confusion: true
    report_mistake_severity: true
```

### 7.8. Integration Points in the Pipeline

```
run_pipeline.py
│
├── DownloadStep
│   └── (optional) cache hierarchy via fetch_hierarchy()
│
├── PreprocessStep
│   └── _extract_label(hierarchy, training_depth=config.hierarchy.training_depth)
│       Uses training_depth for curriculum; null = leaf level
│
├── SplitStep
│   └── (unchanged)
│
├── TrainStep
│   ├── IF hierarchy.enabled:
│   │   ├── Load hierarchy tree from cache_path
│   │   ├── Build distance matrix for current COCO categories
│   │   ├── Build soft target matrix (epsilon, lambda)
│   │   ├── apply_hierarchy_patch(soft_targets)
│   │   └── Log configuration
│   │
│   ├── Train model (rfdetr internally uses patched SetCriterion)
│   │
│   └── IF hierarchy.enabled:
│       └── remove_hierarchy_patch()  # Clean up after training
│
└── EvalStep
    ├── Standard COCO mAP
    └── HierarchicalEvaluator
        ├── Mistake severity
        ├── Within/cross branch confusion
        └── Per-depth accuracy
```

## 8. Appendix: EKB Hierarchy API

### 8.1. Endpoint

```
GET /ekb/api/hierarchy
Host: 192.168.50.103:8080
```

### 8.2. Response Schema

```json
{
  "id": "root",
  "name": "root",
  "children": [
    {
      "id": "anthropogenic",
      "name": "anthropogenic",
      "parentId": "root",
      "color": "#E74C3C",
      "description": "Human-generated sounds",
      "customFields": {},
      "children": [
        {
          "id": "anthropogenic/ship",
          "name": "ship",
          "parentId": "anthropogenic",
          "children": [
            {
              "id": "anthropogenic/ship/cargo",
              "name": "cargo",
              "parentId": "anthropogenic/ship",
              "children": [
                {
                  "id": "anthropogenic/ship/cargo/container ship",
                  "name": "container ship",
                  "parentId": "anthropogenic/ship/cargo",
                  "children": []
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### 8.3. `label_hierarchy` String Mapping

The `label_hierarchy` field on events uses `" + "` or `" > "` separators that correspond
to the path from root to leaf in the hierarchy tree:

| `label_hierarchy` string                          | Tree path                                 |
| ------------------------------------------------- | ----------------------------------------- |
| `"anthropogenic + ship + cargo + container ship"` | `anthropogenic/ship/cargo/container ship` |
| `"biological > cetacean > odontoceti"`            | `biological/cetacean/odontoceti`          |
| `"geoacoustic"`                                   | `geoacoustic` (leaf at depth 1)           |
| `"unknown"`                                       | `unknown` (leaf at depth 1)               |

### 8.4. Current vs. Potential Class Counts

| Depth         | Classes from current download (10) | Classes from full EKB (29) |
| ------------- | ---------------------------------- | -------------------------- |
| 1 (top-level) | 4                                  | 5                          |
| 2             | 7                                  | ~14                        |
| 3             | 9                                  | ~22                        |
| Leaf          | 10                                 | 29                         |

The 10 classes currently in the downloaded data:
`airgun`, `anthropogenic`, `biological`, `explosion`, `fish`, `mysticeti`,
`odontoceti`, `other anthropogenic`, `ship`, `unknown`.

Note: some of these (e.g., `anthropogenic`, `biological`) are intermediate nodes that
appear as leaves because no finer-grained annotation was provided for those events. This
is expected behavior -- the hierarchy allows annotation at any depth.
