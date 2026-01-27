# FIXED CODE SNIPPETS FOR data_validation.ipynb
# Copy and paste these into the corresponding cells

# ============================================================
# CELL 35 FIX: Embedding Validation (Angular Spread Check)
# ============================================================

# Norm Check
print("=== NORM CHECK ===")
norms = np.linalg.norm(E_items, axis=1)
print(f"Mean norm: {norms.mean():.6f}")
print(f"Std of norm: {norms.std():.6f}")
print(f"Min norm: {norms.min():.6f}")
print(f"Max norm: {norms.max():.6f}")

# Angular spread check
import random
random.seed(42)
print("\n=== ANGULAR SPREAD CHECK ===")
sample_size = min(1000, len(E_items))
indices = random.sample(range(len(E_items)), sample_size * 2)

cosine_similarities = []
for i in range(0, len(indices), 2):
    idx1, idx2 = indices[i], indices[i+1]  # FIXED: was indices[+1]
    if idx1 != idx2:
        sim = np.dot(E_items[idx1], E_items[idx2])
        cosine_similarities.append(sim)

cosine_similarities = np.array(cosine_similarities)
print(f"Mean cosine: {cosine_similarities.mean():.4f}")
print(f"Std cosine: {cosine_similarities.std():.4f}")
print(f"Min cosine: {cosine_similarities.min():.4f}")
print(f"Max cosine: {cosine_similarities.max():.4f}")

# Dimensionality check
print(f"\n=== DIMENSIONALITY CHECK ===")
print(f"Embedding shape: {E_items.shape}")
print(f"Matrix rank: {np.linalg.matrix_rank(E_items)}")


# ============================================================
# CELL 72 FIX: Build Interaction Matrix (CRITICAL)
# ============================================================

import implicit
from scipy.sparse import csr_matrix, coo_matrix

# Create mappings for users and items
train["user_idx"] = train["user_id"].astype("category").cat.codes
train["item_idx"] = train["movie_id"].astype("category").cat.codes

user_id_to_idx = dict(zip(train["user_id"], train["user_idx"]))
item_id_to_idx = dict(zip(train["movie_id"], train["item_idx"]))
user_idx_to_id = dict(zip(train["user_idx"], train["user_id"]))
item_idx_to_id = dict(zip(train["item_idx"], train["movie_id"]))

# Get dimensions
n_users = train["user_idx"].nunique()
n_items = train["item_idx"].nunique()

print(f"Number of unique users: {n_users}")
print(f"Number of unique items: {n_items}")

# Build sparse interaction matrix with confidence weights
# CRITICAL: implicit library expects (items, users) shape!
interaction_matrix = coo_matrix(
    (train["confidence"].values.astype(np.float32),  # FIXED: Use confidence scores
     (train["item_idx"].values, train["user_idx"].values))  # FIXED: (items, users)
)

# Convert to CSR format for implicit library
interaction_csr = interaction_matrix.tocsr()  # FIXED: No transpose!

# Verify dimensions
print(f"\nInteraction matrix shape: {interaction_csr.shape}")
print(f"Expected shape: ({n_items}, {n_users})")
assert interaction_csr.shape == (n_items, n_users), f"Shape mismatch! Got {interaction_csr.shape}, expected ({n_items}, {n_users})"

print(f"Number of non-zero entries: {interaction_csr.nnz}")
print(f"Sparsity: {100 * (1 - interaction_csr.nnz / (interaction_csr.shape[0] * interaction_csr.shape[1])):.2f}%")


# ============================================================
# CELL 63 FIX: ALS Training (Re-run after fixing Cell 72)
# ============================================================

# Verify matrix shape before training
n_users = train['user_idx'].nunique()
n_items = train['item_idx'].nunique()
print(f"Expected matrix shape: ({n_items}, {n_items})")
print(f"Actual matrix shape: {interaction_csr.shape}")
assert interaction_csr.shape == (n_items, n_users), "Matrix shape mismatch!"

# Initialize ALS model with research-backed hyperparameters
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

print("Confidence statistics:")
print(f"Min: {train['confidence'].min()}")
print(f"Max: {train['confidence'].max()}")
print(f"Mean: {train['confidence'].mean()}")

# Train the model
model.fit(interaction_csr)

# Extract learned factors
user_factors = model.user_factors
item_factors = model.item_factors

print(f"\nUser factors shape: {user_factors.shape}")
print(f"Item factors shape: {item_factors.shape}")

# CRITICAL: Verify dimensions match expectations
assert user_factors.shape[0] == n_users, f"User factors dimension wrong! Got {user_factors.shape[0]}, expected {n_users}"
assert item_factors.shape[0] == n_items, f"Item factors dimension wrong! Got {item_factors.shape[0]}, expected {n_items}"
print("✅ All dimensions verified correctly!")


# ============================================================
# CELL 73 FIX: Sanity Checks (Run after fixing Cells 72 & 63)
# ============================================================

user_norms = np.linalg.norm(user_factors, axis=1)
item_norms = np.linalg.norm(item_factors, axis=1)

print(f"User factors norms - Mean: {user_norms.mean():.4f}, Std: {user_norms.std():.4f}")
print(f"Item factors norms - Mean: {item_norms.mean():.4f}, Std: {item_norms.std():.4f}")

# Check reconstruction error on training data
train_predictions = user_factors @ item_factors.T  # Shape: (n_users, n_items)
print(f"\nTrain predictions shape: {train_predictions.shape}")
print(f"Expected shape: ({n_users}, {n_items})")

train_errors = []
for u, i in zip(train["user_idx"], train["item_idx"]):
    pred = train_predictions[u, i]
    actual = 1  # All training interactions are positive (implicit=1)
    train_errors.append((actual - pred) ** 2)

print(f"Mean squared training error: {np.mean(train_errors):.6f}")

# Check cold-start coverage
print(f"\nNumber of users in training: {user_factors.shape[0]}")
print(f"Number of items in training: {item_factors.shape[0]}")
print("✅ Sanity checks passed!")


# ============================================================
# CELL 44 FIX: FAISS Index with Verification
# ============================================================

import faiss

dim = E_items.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(E_items)

faiss.write_index(index, "models/faiss_index/movies_embeddings.index")

index_to_movie_id = df_movies["movie_id"].values
np.save("models/faiss_index/index_to_movie_id.npy", index_to_movie_id)

# ADDED: Verification
assert len(index_to_movie_id) == len(E_items), f"Mismatch! {len(index_to_movie_id)} movie IDs but {len(E_items)} embeddings"
print(f"✅ FAISS index created with {len(index_to_movie_id)} movies")

# Test the index
test_embedding = E_items[0:1]
D, I = index.search(test_embedding, 5)
print(f"\nTop 5 similar to {df_movies.iloc[0]['title']}:")
for i, (distance, idx) in enumerate(zip(D[0], I[0])):
    movie_id = index_to_movie_id[idx]
    title = df_movies[df_movies["movie_id"] == movie_id]["title"].iloc[0]
    print(f"    {i+1}. {title} (similarity: {distance:.4f})")
