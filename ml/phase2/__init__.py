"""
Phase 2 — unimodal text and voice research baselines on the verified EATD
participant-level splits.

Modules:
    dataset         — manifest / split access (participant key preserved)
    evaluation      — metrics, participant bootstrap CIs, artifact provenance
    text_tfidf      — Experiment T1 (TF-IDF + logistic regression)
    text_transformer— Experiment T2 (frozen multilingual encoder + head)
    voice_egemaps   — Experiment V1 (openSMILE eGeMAPS + logistic regression)
    voice_wav2vec2  — Experiment V2 (frozen Wav2Vec2 embeddings + head)

Nothing here writes to models/final or changes the serving gate.
"""
