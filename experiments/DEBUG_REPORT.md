# DATA_VALIDATION.IPYNB - COMPLETE DEBUG REPORT
## Senior Engineer Review - Cell-by-Cell Analysis

---

## CELL 1: Change Directory
**Status**: ✅ OK
**Code**: `os.chdir("..")` and `os.getcwd()`
**Issue**: None
**Solution**: N/A

---

## CELL 2: Import Libraries and Load Data
**Status**: ✅ OK
**Code**: Imports and data loading
**Issue**: None
**Solution**: N/A

---

## CELL 3-8: Basic Data Exploration
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 9: Check Unique Values
**Status**: ✅ OK
**Code**: `df_ratings["user_id"].nunique(), df_ratings["movie_id"].nunique()`
**Result**: (6040, 3706)
**Issue**: None
**Solution**: N/A

---

## CELL 10: Check Unique Movies
**Status**: ⚠️ INCONSISTENCY DETECTED
**Code**: `df_movies["movie_id"].nunique()`
**Result**: 3883
**Issue**: 
- Ratings has 3706 unique movies
- Movies dataframe has 3883 unique movies
- **177 movies in movies.dat have NO ratings**
**Solution**:
```python
# Add after cell 10 to verify
rated_movies = set(df_ratings["movie_id"].unique())
all_movies = set(df_movies["movie_id"].unique())
unrated_movies = all_movies - rated_movies
print(f"Movies with no ratings: {len(unrated_movies)}")
# Decision: Keep all movies for content-based cold-start recommendations
```

---

## CELL 14: Sparsity Calculation
**Status**: ✅ OK
**Result**: 4.47% density
**Issue**: None
**Solution**: N/A

---

## CELL 15-17: User Activity Analysis
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 18: Power Law Analysis
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 21: Item Popularity
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 23: Temporal Analysis
**Status**: ⚠️ POTENTIAL ISSUE
**Code**: Creates `datetime` column
**Issue**: This modifies `df_ratings` in place, which affects downstream cells
**Solution**: This is fine, but be aware the dataframe is being modified

---

## CELL 24-28: Content Validation
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 29: Create Implicit Feedback
**Status**: ⚠️ CRITICAL - VALUES NOT USED LATER
**Code**: 
```python
df_ratings["implicit"] = (df_ratings["rating"] >= 3).astype(int)
df_ratings["confidence"] = 1 + 40 * np.log1p(df_ratings["rating"])
```
**Issue**: Confidence scores are calculated but NOT used in matrix construction (Cell 72)
**Solution**: 
```python
# In Cell 72, replace:
# (np.ones(len(train), dtype=np.float32),
# WITH:
(train["confidence"].values.astype(np.float32),
```

---

## CELL 30: Train/Val Split
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 31: Text Embedding Preparation
**Status**: ⚠️ MINOR ISSUE
**Code**: Creates `text_for_embedding` and `clean_text`
**Issue**: `clean_text` is created but never used
**Solution**: Remove unused column or use it instead of `text_for_embedding`

---

## CELL 33: Save Processed Data
**Status**: ❌ CRITICAL ERROR - CELL NOT EXECUTED
**Code**: `train.to_csv(...)`, `val.to_csv(...)`, `df_movies.to_csv(...)`
**Issue**: This cell has no execution count - files may not exist or be outdated
**Solution**: **MUST EXECUTE THIS CELL** before proceeding to evaluation

---

## CELL 34: Generate Embeddings
**Status**: ✅ OK
**Result**: Shape (3883, 384)
**Issue**: None
**Solution**: N/A

---

## CELL 35: Embedding Validation
**Status**: ❌ BUG DETECTED
**Code**: Line 1380
```python
idx2 = indices[+1]  # WRONG!
```
**Issue**: Should be `indices[i+1]`
**Solution**:
```python
for i in range(0, len(indices), 2):
    idx1, idx2 = indices[i], indices[i+1]  # FIX THIS LINE
    if idx1 != idx2:
        sim = np.dot(E_items[idx1], E_items[idx2])
        cosine_similarities.append(sim)
```

---

## CELL 36: Content Similarity Function
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 44: FAISS Index Creation
**Status**: ⚠️ POTENTIAL ALIGNMENT ISSUE
**Code**: 
```python
index_to_movie_id = df_movies["movie_id"].values
```
**Issue**: Assumes df_movies is in the same order as E_items embeddings
**Verification Needed**: Confirm embeddings were created in df_movies row order
**Solution**: Add assertion:
```python
assert len(index_to_movie_id) == len(E_items), "Mismatch in embedding count"
assert index_to_movie_id[0] == df_movies.iloc[0]["movie_id"], "Order mismatch"
```

---

## CELL 61: Recommendation Functions
**Status**: ✅ OK
**Issue**: None
**Solution**: N/A

---

## CELL 72: Build Interaction Matrix
**Status**: ❌ CRITICAL ERROR - DIMENSION SWAP
**Code**:
```python
interaction_matrix = coo_matrix(
    (np.ones(len(train), dtype=np.float32),
     (train["user_idx"].values, train["item_idx"].values))
)
interaction_csr = interaction_matrix.tocsr().T  # WRONG TRANSPOSE!
interaction_csr = interaction_matrix.tocsr()  # Overwritten!
```

**Issues**:
1. Using `np.ones()` instead of confidence scores
2. Matrix dimensions are **(users, items)** but implicit library needs **(items, users)**
3. Transpose is applied then immediately overwritten
4. Cell has KeyError - not executed properly

**Solution**:
```python
import implicit
from scipy.sparse import csr_matrix, coo_matrix

# Create mappings for users and items
train["user_idx"] = train["user_id"].astype("category").cat.codes
train["item_idx"] = train["movie_id"].astype("category").cat.codes

user_id_to_idx = dict(zip(train["user_id"], train["user_idx"]))
item_id_to_idx = dict(zip(train["movie_id"], train["item_idx"]))
user_idx_to_id = dict(zip(train["user_idx"], train["user_id"]))
item_idx_to_id = dict(zip(train["item_idx"], train["movie_id"]))

# Build sparse interaction matrix with confidence weights
# implicit library expects (items, users) shape
interaction_matrix = coo_matrix(
    (train["confidence"].values.astype(np.float32),  # USE CONFIDENCE!
     (train["item_idx"].values, train["user_idx"].values))  # (items, users)
)

# Convert to CSR format for implicit library
interaction_csr = interaction_matrix.tocsr()  # NO TRANSPOSE!

print(f"Interaction matrix shape: {interaction_csr.shape}")
print(f"Expected: (n_items={train['item_idx'].nunique()}, n_users={train['user_idx'].nunique()})")
print(f"Number of non-zero entries: {interaction_csr.nnz}")
print(f"Sparsity: {100 * (1 - interaction_csr.nnz / (interaction_csr.shape[0] * interaction_csr.shape[1])):.2f}%")
```

---

## CELL 63: ALS Training
**Status**: ❌ CRITICAL - TRAINED ON WRONG MATRIX
**Code**: `model.fit(interaction_csr)`
**Issue**: 
- Model was trained on incorrectly shaped matrix
- Result: user_factors (3662, 64), item_factors (5400, 64)
- **These dimensions are SWAPPED!**
- Should be: user_factors (5400, 64), item_factors (3662, 64)

**Solution**: Re-train after fixing Cell 72
```python
# After fixing Cell 72, verify before training:
n_users = train['user_idx'].nunique()
n_items = train['item_idx'].nunique()
print(f"Expected matrix shape: ({n_items}, {n_users})")
print(f"Actual matrix shape: {interaction_csr.shape}")
assert interaction_csr.shape == (n_items, n_users), "Matrix shape mismatch!"

# Then train
model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.01,
    alpha=40,
    iterations=20,
    random_state=42,
    calculate_training_loss=True,
    use_gpu=False,
    use_native=True,
    dtype=np.float32
)

model.fit(interaction_csr)

# Verify output dimensions
print(f"User factors shape: {model.user_factors.shape}")  # Should be (n_users, 64)
print(f"Item factors shape: {model.item_factors.shape}")  # Should be (n_items, 64)
assert model.user_factors.shape[0] == n_users, "User factors dimension wrong!"
assert model.item_factors.shape[0] == n_items, "Item factors dimension wrong!"
```

---

## CELL 73: Sanity Checks
**Status**: ❌ FAILS DUE TO DIMENSION SWAP
**Error**: `IndexError: index 5399 is out of bounds for axis 0 with size 3662`
**Issue**: 
- Trying to access `train_predictions[u, i]`
- `u` can be up to 5399 (user_idx)
- But `train_predictions` has only 3662 rows (because user_factors has 3662 rows)
- **This confirms the dimension swap**

**Solution**: Fix Cell 72 and 63 first, then:
```python
user_norms = np.linalg.norm(user_factors, axis=1)
item_norms = np.linalg.norm(item_factors, axis=1)

print(f"User factors norms - Mean: {user_norms.mean():.4f}, Std: {user_norms.std():.4f}")
print(f"Item factors norms - Mean: {item_norms.mean():.4f}, Std: {item_norms.std():.4f}")

# Check reconstruction error on training data
train_predictions = user_factors @ item_factors.T  # (n_users, n_items)
train_errors = []
for u, i in zip(train["user_idx"], train["item_idx"]):
    pred = train_predictions[u, i]
    actual = 1  # All training interactions are positive (implicit=1)
    train_errors.append((actual - pred) ** 2)

print(f"Mean squared training error: {np.mean(train_errors):.6f}")

# Verify dimensions
print(f"Number of users in training: {user_factors.shape[0]}")
print(f"Number of items in training: {item_factors.shape[0]}")
assert user_factors.shape[0] == train['user_idx'].nunique()
assert item_factors.shape[0] == train['item_idx'].nunique()
```

---

## CELL 68: Evaluation Setup
**Status**: ❌ FAILS DUE TO UPSTREAM ERRORS
**Error**: `AssertionError: ALS has 3662 users but train has 5400 users`
**Issue**: Confirms the dimension swap problem
**Solution**: Fix Cells 72 and 63 first

---

## CELL 69: Verification
**Status**: ⚠️ REVEALS ROOT CAUSE
**Output**:
```
Users seen by ALS: 5400
Unique user_idx used for training: 5400
```
**Issue**: This shows the matrix had 5400 in the user dimension, but ALS interpreted it as items
**Solution**: Fix the matrix construction in Cell 72

---

## SUMMARY OF CRITICAL ISSUES

### 🔴 CRITICAL (Must Fix):
1. **Cell 72**: Matrix dimensions swapped - **(users, items)** instead of **(items, users)**
2. **Cell 72**: Not using confidence scores (using `np.ones()` instead)
3. **Cell 63**: ALS trained on wrong matrix dimensions
4. **Cell 33**: Not executed - processed data files may not exist
5. **Cell 35**: Bug in similarity calculation (`indices[+1]` should be `indices[i+1]`)

### ⚠️ WARNING (Should Fix):
1. **Cell 10**: 177 movies have no ratings (acceptable for cold-start)
2. **Cell 31**: Unused `clean_text` column
3. **Cell 44**: No verification of embedding-to-movie_id alignment

### ✅ WORKING CORRECTLY:
- Data loading and basic EDA
- Embedding generation
- Content similarity functions
- Hybrid recommendation functions (will work once ALS is fixed)

---

## RECOMMENDED FIX ORDER

1. **Fix Cell 35** (embedding validation bug)
2. **Execute Cell 33** (save processed data)
3. **Fix Cell 72** (matrix construction)
4. **Re-run Cell 63** (ALS training)
5. **Run Cell 73** (sanity checks - should pass now)
6. **Run Cell 68** (evaluation setup - should pass now)

---

## VERIFICATION CHECKLIST

After fixes, verify:
- [ ] `interaction_csr.shape[0]` == number of unique items
- [ ] `interaction_csr.shape[1]` == number of unique users
- [ ] `user_factors.shape[0]` == number of unique users
- [ ] `item_factors.shape[0]` == number of unique items
- [ ] `E_items.shape[0]` == total movies in df_movies (3883)
- [ ] No IndexError in sanity checks
- [ ] Evaluation runs without assertion errors
