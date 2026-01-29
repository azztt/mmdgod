# Category Mapping Fix - Critical Bug Resolution

## The Problem

The previous training run (20k iterations) produced 0% AP on both:
- SUNRGBD (target domain)
- Hypersim val (source domain)

This indicated the model wasn't learning at all, not just domain gap.

## Root Cause

**Category ID mismatch between model and data loader:**

1. **Config defines categories in specific order:**
   ```
   0: person
   1: books  
   2: chair
   3: towel
   ...
   ```

2. **Data loader was creating alphabetical mapping:**
   ```
   0: bathtub (alphabetically first)
   1: bed
   2: blinds
   7: chair (7th alphabetically)
   ...
   ```

3. **Model classifier was initialized with 31 classes matching config order**

4. **But training data had annotations with alphabetical indices!**
   - A "chair" annotation was labeled as class 7 (alphabetical)
   - But model expected "chair" at class 2 (config order)
   - Loss was computed on wrong class indices throughout training

## The Fix

### 1. Modified `register_and_store_rgbd_model_metadata` in `datasets_rgbd.py`:
- Now accepts `category_names` parameter from config
- Uses config order instead of sorting alphabetically

### 2. Modified `load_rgbd_json` in `datasets_rgbd.py`:
- Checks for global `omni3d_model` metadata first
- Uses global category order for mapping if available
- Falls back to local mapping only if global not set

### 3. Modified `train_rgbd.py`:
- Sets up global metadata BEFORE registering datasets
- Passes `category_names` from config to ensure consistent ordering

## Verification

After the fix, category mapping is correct:
```
manifest_id 18 (chair) -> contiguous 2 (chair) ✅
manifest_id 25 (lamp)  -> contiguous 6 (lamp)  ✅
```

## Action Required

**The old model must be discarded and training restarted from scratch.**

The old model's weights are meaningless because:
- Classifier weight for class 2 learned "blinds" patterns (alphabetical position)
- But we need class 2 to detect "chair" (config position)

## Training Command

```bash
conda activate cube
python tools/train_rgbd.py --config-file configs/rgbd/Base_RGBD.yaml \
    --num-gpus 4 \
    SOLVER.MAX_ITER 50000
```

## Files Changed

1. `cubercnn/data/datasets_rgbd.py`:
   - `register_and_store_rgbd_model_metadata()` - accepts category_names param
   - `load_rgbd_json()` - uses global metadata for category mapping

2. `tools/train_rgbd.py`:
   - Sets global metadata before dataset registration
   - Passes category_names to metadata function

## Old Model Location

The old (incorrectly trained) model was moved to:
`output/rgbd_base_old/`
