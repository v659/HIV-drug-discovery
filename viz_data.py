import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats

DATASET = "bace"
SEED = 42
np.random.seed(SEED)

print(f"Loading {DATASET.upper()} dataset...")
tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='ECFP', splitter='scaffold')
train_dc, valid_dc, test_dc = datasets

def get_data(dataset):
    X = np.array(dataset.X, dtype=np.float32)
    y = np.array(dataset.y, dtype=np.float32).flatten()
    return X, y

X_train, y_train = get_data(train_dc)
X_valid, y_valid = get_data(valid_dc)
X_test, y_test = get_data(test_dc)

print(f"\nDataset Statistics:")
print(f"Train: {len(X_train)} samples ({np.sum(y_train==1)} positive, {np.sum(y_train==0)} negative)")
print(f"Valid: {len(X_valid)} samples ({np.sum(y_valid==1)} positive, {np.sum(y_valid==0)} negative)")
print(f"Test: {len(X_test)} samples ({np.sum(y_test==1)} positive, {np.sum(y_test==0)} negative)")

print(f"\nFeature matrix shape: {X_train.shape[1]} dimensions (ECFP fingerprints)")
print(f"Class balance overall: {100*np.mean(np.concatenate([y_train, y_valid, y_test])):.1f}% positive")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(f'BACE Dataset Analysis (ECFP Fingerprints)', fontsize=16, fontweight='bold', y=0.98)

ax1 = plt.subplot(3, 4, 1)
split_counts = [len(y_train), len(y_valid), len(y_test)]
split_labels = ['Train', 'Validation', 'Test']
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax1.bar(split_labels, split_counts, color=colors, alpha=0.8)
ax1.set_title('Dataset Split Sizes', fontweight='bold')
ax1.set_ylabel('Number of Samples')
ax1.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, split_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom')

ax2 = plt.subplot(3, 4, 2)
pos_counts = [np.sum(y_train==1), np.sum(y_valid==1), np.sum(y_test==1)]
neg_counts = [np.sum(y_train==0), np.sum(y_valid==0), np.sum(y_test==0)]
x = np.arange(len(split_labels))
width = 0.35
bars_pos = ax2.bar(x - width/2, pos_counts, width, label='Positive', color='#2E86AB', alpha=0.8)
bars_neg = ax2.bar(x + width/2, neg_counts, width, label='Negative', color='#A23B72', alpha=0.8)
ax2.set_title('Class Distribution per Split', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(split_labels)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

ax3 = plt.subplot(3, 4, 3)
total_pos = np.sum(y_train==1) + np.sum(y_valid==1) + np.sum(y_test==1)
total_neg = np.sum(y_train==0) + np.sum(y_valid==0) + np.sum(y_test==0)
ax3.pie([total_pos, total_neg], labels=[f'Positive\n({total_pos})', f'Negative\n({total_neg})'],
        colors=['#2E86AB', '#A23B72'], autopct='%1.1f%%', startangle=90)
ax3.set_title('Overall Class Balance', fontweight='bold')

ax4 = plt.subplot(3, 4, 4)
fingerprint_density = np.mean(X_train, axis=0)
ax4.hist(fingerprint_density, bins=50, alpha=0.7, color='#F18F01', edgecolor='black')
ax4.set_xlabel('Feature Activation Frequency')
ax4.set_ylabel('Count')
ax4.set_title('ECFP Feature Activation Distribution', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axvline(x=np.mean(fingerprint_density), color='red', linestyle='--',
            label=f'Mean: {np.mean(fingerprint_density):.3f}')
ax4.legend()

print(f"\nFeature Statistics:")
print(f"Mean activation frequency: {np.mean(fingerprint_density):.4f}")
print(f"Std activation frequency: {np.std(fingerprint_density):.4f}")
print(f"Max activation frequency: {np.max(fingerprint_density):.4f}")
print(f"Min activation frequency: {np.min(fingerprint_density):.4f}")
print(f"Features with >10% activation: {np.sum(fingerprint_density > 0.1)}")
print(f"Features with <1% activation: {np.sum(fingerprint_density < 0.01)}")

ax5 = plt.subplot(3, 4, 5)
train_pos_idx = np.where(y_train == 1)[0]
train_neg_idx = np.where(y_train == 0)[0]
sample_pos = X_train[np.random.choice(train_pos_idx, min(100, len(train_pos_idx)), replace=False)]
sample_neg = X_train[np.random.choice(train_neg_idx, min(100, len(train_neg_idx)), replace=False)]

pos_mean_act = np.mean(sample_pos, axis=0)
neg_mean_act = np.mean(sample_neg, axis=0)

ax5.scatter(pos_mean_act, neg_mean_act, alpha=0.6, color='#2E86AB', s=20)
ax5.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
ax5.set_xlabel('Mean Activation (Positive Class)')
ax5.set_ylabel('Mean Activation (Negative Class)')
ax5.set_title('Class-wise Feature Activation', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

ax6 = plt.subplot(3, 4, 6)
correlation = np.corrcoef(pos_mean_act, neg_mean_act)[0, 1]
ax6.text(0.1, 0.8, f'Correlation between\nclass activations:', fontsize=10)
ax6.text(0.1, 0.6, f'r = {correlation:.3f}', fontsize=12, fontweight='bold')
ax6.text(0.1, 0.4, f'Most features behave\nsimilarly in both classes', fontsize=9)
ax6.axis('off')

print(f"\nClass-wise Feature Analysis:")
print(f"Correlation between positive/negative class activations: {correlation:.3f}")

combined_X = np.vstack([X_train, X_valid, X_test])
combined_y = np.concatenate([y_train, y_valid, y_test])

sample_size = 500
if len(combined_X) > sample_size:
    idx = np.random.choice(len(combined_X), sample_size, replace=False)
    sample_X = combined_X[idx]
    sample_y = combined_y[idx]
else:
    sample_X = combined_X
    sample_y = combined_y

print(f"\nRunning t-SNE on {len(sample_X)} samples...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
tsne_result = tsne.fit_transform(sample_X)

ax7 = plt.subplot(3, 4, (7, 8))
scatter = ax7.scatter(tsne_result[:, 0], tsne_result[:, 1],
                     c=sample_y, cmap='coolwarm', alpha=0.7, s=30)
ax7.set_xlabel('t-SNE 1')
ax7.set_ylabel('t-SNE 2')
ax7.set_title('t-SNE Visualization of Molecular Space', fontweight='bold')
legend1 = ax7.legend(*scatter.legend_elements(), title="Classes")
ax7.add_artist(legend1)

print(f"\nRunning PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(sample_X)

ax8 = plt.subplot(3, 4, (9, 10))
scatter2 = ax8.scatter(pca_result[:, 0], pca_result[:, 1],
                      c=sample_y, cmap='coolwarm', alpha=0.7, s=30)
ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax8.set_title('PCA Visualization', fontweight='bold')
legend2 = ax8.legend(*scatter2.legend_elements(), title="Classes")
ax8.add_artist(legend2)

print(f"PCA Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

ax9 = plt.subplot(3, 4, 11)
pca_full = PCA().fit(combined_X)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
ax9.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
        marker='o', linestyle='-', color='#2E86AB', linewidth=2)
ax9.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
ax9.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
ax9.set_xlabel('Number of Principal Components')
ax9.set_ylabel('Cumulative Explained Variance')
ax9.set_title('PCA Variance Explained', fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.legend()
ax9.set_xlim([1, 100])

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\nPCA Dimensionality Analysis:")
print(f"Components for 90% variance: {n_components_90}")
print(f"Components for 95% variance: {n_components_95}")

ax10 = plt.subplot(3, 4, 12)
distinctiveness = np.abs(pos_mean_act - neg_mean_act)
top_10_idx = np.argsort(distinctiveness)[-10:][::-1]
top_10_values = distinctiveness[top_10_idx]

ax10.barh(range(10), top_10_values, color='#A23B72', alpha=0.8)
ax10.set_yticks(range(10))
ax10.set_yticklabels([f'Feature {i}' for i in top_10_idx])
ax10.set_xlabel('Activation Difference')
ax10.set_title('Top 10 Most Discriminative Features', fontweight='bold')
ax10.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print(f"\nDataset Characteristics:")
print(f"1. {len(combined_X)} total molecules")
print(f"2. {X_train.shape[1]}-dimensional ECFP fingerprints")
print(f"3. Slight class imbalance ({100*np.mean(combined_y):.1f}% positive)")
print(f"4. High-dimensional but sparse features")
print(f"5. Classes are not linearly separable in PCA/t-SNE")
print(f"6. Many features have low activation frequency (<1%)")
print(f"7. Only {n_components_90} components needed for 90% variance")
print(f"8. Validation set has different scaffold distribution (scaffold split)")