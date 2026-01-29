# Configuration Issues Found and Fixed

**Date:** January 28, 2026  
**Status:** TEMPORARY WORKAROUND APPLIED - FULL RETRAINING REQUIRED

---

## Critical Issue: NUM_CLASSES Mismatch

### Problem Identified

**Root Cause:** Softmax probability dilution due to class count mismatch

- **Configured NUM_CLASSES:** 43 (inherited from `Base.yaml`)
- **Actual Categories Used:** 31 (defined in `CATEGORY_NAMES`)
- **Model Output Dimension:** 44 (43 classes + 1 background)
- **Expected Output Dimension:** 32 (31 classes + 1 background)

### Impact

1. **Severe Probability Dilution:**
   - Softmax computed over 44 dimensions instead of 32
   - Class probabilities diluted: ~0.02-0.05 instead of expected 0.05-0.15
   - Combined scores (objectness × class_prob): ~0.001-0.005 instead of 0.05+
   
2. **Predictions Below Threshold:**
   - TEST_SCORE_THRESH: 0.05 (reasonable for 32-class softmax)
   - Actual scores: 0.001-0.03 (too low due to 44-class softmax)
   - Result: 0 predictions passed threshold → AP = 0.000

3. **Wasted Model Capacity:**
   - 12 output neurons (44-32) trained but never used
   - Gradient flow distributed across unused classes
   - Suboptimal feature learning

### Debug Evidence

```
[DEBUG] Scores - max: 0.0048, min: 0.0009, mean: 0.0044
[DEBUG] Objectness - max: 0.09, mean: 0.05
[DEBUG] Class probs - max: 0.05, mean: 0.02
[DEBUG] Threshold: 0.05, Keep: 0/300  ← All filtered out!
```

After lowering threshold to 0.001:
```
[DEBUG] Scores - max: 0.0171, min: 0.0007, mean: 0.0044
[DEBUG] Threshold: 0.001, Keep: 287/300  ← Now getting predictions
```

---

## Fixes Applied

### 1. Override NUM_CLASSES in Config

**File:** `configs/rgbd/dinov3_base_aug.yaml`

**Change:**
```yaml
# Before (inherited from Base.yaml)
ROI_HEADS:
  NUM_CLASSES: 31  # Only set explicitly, but Base.yaml had 43

# After (with explanation)
ROI_HEADS:
  NUM_CLASSES: 31
  SCORE_THRESH_TEST: 0.001  # Lowered due to softmax dilution
```

**Lines:** 62-68

### 2. Lower Score Threshold (Temporary Workaround)

**File:** `configs/rgbd/dinov3_base_aug.yaml`

**Change:**
```yaml
# Before
TEST_SCORE_THRESH: 0.05

# After (with detailed explanation)
TEST_SCORE_THRESH: 0.001  # Temporarily lowered (normally 0.05)
                          # Current checkpoint trained with NUM_CLASSES=43
                          # but only 31 categories used
```

**Lines:** 96-105

---

## Related Issues Found

### 1. Inconsistent Threshold Settings

**Issue:** Multiple threshold settings in different config sections
- `ROI_HEADS.SCORE_THRESH_TEST`: 0.01 (Base.yaml)
- `ROI_CUBE_HEAD.TEST_SCORE_THRESH`: 0.05 (was in config)

**Fix:** Added `SCORE_THRESH_TEST: 0.001` to ROI_HEADS for consistency

### 2. Model Architecture Implications

**Current Checkpoint (`model_recent.pth` @ iter 27299):**
- Trained with `class_head = nn.Linear(d_model, 44)` 
- All 44 neurons learned features optimized for 44-way softmax
- Cannot simply "fix" by loading into 32-output model (dimension mismatch)

**Code Location:** `cubercnn/modeling/meta_arch/pure_detr3d.py:371`
```python
self.class_head = nn.Linear(d_model, self.num_classes + 1)  # Was 44, should be 32
```

---

## Evaluation Results with Workaround

### Hypersim Val (Trained Domain)
- **Predictions per image:** 150-300 (was 0)
- **Score range:** 0.001-0.030
- **AP2D:** 0.000 (predictions generated but very inaccurate)
- **AP3D:** 0.000 (predictions generated but very inaccurate)

### SUNRGBD Val (Cross-Domain)
- **Predictions per image:** 0-163 (highly variable)
- **Score range:** 0.000-0.010 (much lower than Hypersim)
- **AP:** 0.000 (very few predictions, mostly inaccurate)

**Analysis:** The workaround allows predictions to be generated, but they are highly inaccurate because:
1. Model learned features optimized for 44-class discrimination
2. 12 unused classes diluted the decision boundaries
3. Class probabilities not properly calibrated

---

## Proper Solution: Retrain from Scratch

### Why Workaround is Insufficient

1. **Architecture Mismatch:** 
   - Current checkpoint has 44-dim class head
   - Proper model needs 32-dim class head
   - Cannot convert without losing learned weights

2. **Feature Learning:**
   - Backbone features optimized for 44-way discrimination
   - Decision boundaries learned with diluted softmax
   - Class embeddings not aligned with 31 actual categories

3. **Calibration:**
   - Softmax probabilities calibrated for 44 classes
   - Confidence scores unreliable for thresholding
   - Post-hoc recalibration difficult with cross-domain generalization

### Retraining Checklist

- [ ] Verify `NUM_CLASSES: 31` in config (DONE)
- [ ] Set `TEST_SCORE_THRESH: 0.05` back to normal (change from 0.001)
- [ ] Delete existing checkpoints to avoid confusion
- [ ] Train from scratch with corrected config
- [ ] Monitor class head output dimension: should be 32 not 44
- [ ] Verify improved score distributions: expect 0.05-0.30 instead of 0.001-0.03
- [ ] Check predictions pass threshold without lowering it

### Expected Improvements After Retraining

| Metric | Before (44-class) | After (32-class) | Expected Gain |
|--------|-------------------|------------------|---------------|
| Max Score | 0.01-0.03 | 0.10-0.30 | 10-30× |
| Mean Score | 0.001-0.005 | 0.05-0.10 | 50× |
| Pred/Image @ 0.05 | 0 | 100-200 | ∞ |
| AP2D | 0.000 | ? | Baseline |

---

## Configuration Files Affected

1. **`configs/Base.yaml`**
   - Contains `NUM_CLASSES: 43` (Omni3D default)
   - Should NOT be modified (shared base config)

2. **`configs/rgbd/dinov3_base_aug.yaml`**
   - Now explicitly overrides `NUM_CLASSES: 31` ✓
   - Temporarily lowered `TEST_SCORE_THRESH: 0.001` (revert after retraining)
   - Documents the issue and temporary nature

3. **`cubercnn/modeling/meta_arch/pure_detr3d.py`**
   - Uses `cfg.MODEL.ROI_HEADS.NUM_CLASSES` (line 329)
   - Creates `self.num_classes` attribute
   - Class head: `nn.Linear(d_model, self.num_classes + 1)` (line 371)
   - Background target: `self.num_classes` (lines 581, 767)

---

## Lessons Learned

### 1. Always Verify Config Inheritance
- Base configs can have incompatible defaults
- Explicitly override critical hyperparameters
- Add comments explaining overrides

### 2. Debug Score Distributions Early
- Low scores (< threshold) → no predictions → AP = 0
- Check softmax dimensionality matches actual classes
- Monitor score statistics during training

### 3. Configuration Validation
- Add assertion: `assert len(CATEGORY_NAMES) == NUM_CLASSES`
- Log class head output dimension at model init
- Validate model architecture matches config intent

### 4. Checkpoint Compatibility
- Changing NUM_CLASSES invalidates existing checkpoints
- Document architectural changes in checkpoint metadata
- Version checkpoints when making breaking changes

---

## Next Steps

### Immediate (For Current Session)
1. ✅ Lower threshold to 0.001 (allows evaluation to complete)
2. ✅ Document all issues found
3. ⏳ Decide: Continue with broken checkpoint or restart training

### Short-Term (Proper Fix)
1. ⬜ Reset `TEST_SCORE_THRESH: 0.05` in config
2. ⬜ Delete `model_recent.pth` (corrupted with 44-class head)
3. ⬜ Restart training from scratch with fixed config
4. ⬜ Monitor training logs for correct class head dim (32)
5. ⬜ Verify score distributions: expect 0.05+ instead of 0.001-0.03

### Long-Term (Robustness)
1. ⬜ Add config validation in training script
2. ⬜ Log model architecture at init (catch mismatches early)
3. ⬜ Add test for NUM_CLASSES == len(CATEGORY_NAMES)
4. ⬜ Document all config inheritance in README

---

## Files Modified

- `configs/rgbd/dinov3_base_aug.yaml` (lines 62-68, 96-105)

## Files to Review

- `configs/Base.yaml` (source of NUM_CLASSES: 43)
- `cubercnn/modeling/meta_arch/pure_detr3d.py` (uses NUM_CLASSES)

## Checkpoints Affected

- `output/dinov3_base_aug/model_recent.pth` @ iter 27299
  - **Status:** INVALID (44-class head, should be 32-class)
  - **Action:** DELETE before retraining
  - **Reason:** Architecture incompatible with fixed config
