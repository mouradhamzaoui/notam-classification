"""
Script de régénération des splits et du pipeline de features.
À exécuter si train_test_splits.pkl est manquant.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
from src.data.data_loader import NOTAMDataLoader
from src.features.feature_engineering import NOTAMFeaturePipeline

# ── Chargement ────────────────────────────────────────────────────────────────
loader = NOTAMDataLoader(path="data/processed/notams_clean.csv")
df     = loader.load()
X_train, X_test, y_train, y_test = loader.split(df)

# ── Feature pipeline ──────────────────────────────────────────────────────────
pipeline      = NOTAMFeaturePipeline(max_tfidf_features=5000)
X_train_final = pipeline.fit_transform(X_train, y_train)
X_test_final  = pipeline.transform(X_test)

y_train_enc = pipeline.encode_labels(y_train)
y_test_enc  = pipeline.encode_labels(y_test)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
Path("data/processed").mkdir(exist_ok=True)

pipeline.save("data/processed/feature_pipeline.pkl")

joblib.dump({
    "X_train":     X_train_final,
    "X_test":      X_test_final,
    "y_train":     y_train_enc,
    "y_test":      y_test_enc,
    "y_train_str": y_train,
    "y_test_str":  y_test,
    "classes":     pipeline.label_encoder.classes_,
}, "data/processed/train_test_splits.pkl")

print("\n✅ Fichiers régénérés avec succès :")
print(f"   → data/processed/feature_pipeline.pkl")
print(f"   → data/processed/train_test_splits.pkl")
print(f"\n   X_train shape : {X_train_final.shape}")
print(f"   X_test  shape : {X_test_final.shape}")
print(f"   Classes       : {list(pipeline.label_encoder.classes_)}")