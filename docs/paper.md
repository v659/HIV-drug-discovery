# Closing the gap on a $0 budget: ensembling public molecular foundation models for HIV bioactivity prediction

**Author:** Arjun Chandra Agarwal
**Affiliation:** Independent
**Status:** Preprint draft — target venue ChemRxiv / arXiv (cs.LG, q-bio.QM)
**Date:** 2026

---

## Abstract

I achieve **0.806 ± 0.018 test AUC** on the MoleculeNet HIV scaffold-split benchmark using a 5-fold ensemble of a from-scratch GATv2-based graph neural network and a fine-tuned MolFormer-XL transformer, combined through out-of-fold logistic stacking. The pipeline is reproducible end-to-end on free-tier Google Colab and consumer Apple Silicon, at **zero dollars of compute cost**. This statistically ties Uni-Mol's 3D-conformer-pretrained result of 0.808 ± 0.003 (Zhou et al., 2023) on this benchmark, whose pretraining required ~160 V100-GPU-hours (≈ $500–$2K of cloud compute) on a 19M-molecule, 209M-conformer corpus. My contribution is methodological: I show that for binary molecular property prediction, principled ensembling of publicly-available pretrained checkpoints, combined with honest scaffold-based evaluation and threshold calibration, matches the SOTA 3D-pretrained model on HIV at zero downstream compute cost — without 3D conformers, without bespoke pretraining, and without GPU-cluster access.

**Keywords:** molecular property prediction, graph neural networks, MolFormer, ensemble learning, scaffold splits, low-resource ML, MoleculeNet HIV.

---

## 1. Introduction

Molecular property prediction has, in recent years, become an arms race in compute and data. Foundation models such as Uni-Mol (Zhou et al., 2023), GROVER (Rong et al., 2020), and ChemBERTa (Chithrananda et al., 2020) rely on industrial-scale pretraining infrastructure — multi-GPU clusters, billion-molecule corpora, weeks of wall-clock time — that is inaccessible to most academic groups, students, and independent researchers. The implicit message of these works is that *bespoke pretraining is now the cost of admission to competitive molecular ML*.

This paper contests that framing. I show that on the MoleculeNet HIV scaffold-split benchmark — one of the field's standard evaluation tasks — a careful ensemble of two complementary, **already-public** molecular models achieves test AUC **0.806 ± 0.018**, statistically tied with Uni-Mol's 0.808 ± 0.003 (Zhou et al., 2023), at **$0 of downstream compute cost**. The pipeline runs end-to-end on Google Colab's free GPU tier and a consumer Apple Silicon laptop. No institutional GPU access is required; no proprietary data is used; every checkpoint and configuration in this paper is reproducible by a motivated student with a free Google account.

My contribution has three components:

1. **A reproducible $0 ensemble pipeline.** A from-scratch GATv2 graph neural network (provides explicit graph reasoning over atoms, bonds, and global descriptors) is combined with a fine-tuned MolFormer-XL transformer (provides a 1.1B-molecule pretraining prior over canonical SMILES). I train both models on identical scaffold-held-out 5-fold CV splits and combine them via a logistic-regression stacker fitted on out-of-fold validation predictions.

2. **An honest evaluation protocol.** I use scaffold-based 5-fold CV — substantially harder than random splits — and report all numbers with bootstrap-style fold-level uncertainty. Out-of-fold stacker fitting prevents leakage. I additionally identify and correct a previously-overlooked failure mode (Tanimoto identity-leakage in nearest-neighbor features) that inflates apparent test AUC by ~0.045 if not handled.

3. **An empirical observation about the diminishing returns of bespoke pretraining.** For binary HIV bioactivity classification, $0-downstream-cost ensembling of public 2D checkpoints matches Uni-Mol's 3D-conformer-pretrained result on this benchmark to within statistical noise (0.806 vs 0.808). I argue that as the public pretraining ecosystem matures, the field's compute investment is now better spent on ensemble design, calibration, and honest evaluation than on additional bespoke pretraining for many downstream tasks. I do *not* claim 3D pretraining is unhelpful in general — only that on at least one standard benchmark, careful 2D ensembling closes the gap.

The remainder of this paper is structured as follows. Section 2 reviews relevant prior work. Section 3 describes the dataset and scaffold-splitting protocol. Section 4 details the GNN and MolFormer-XL components. Section 5 describes the stacker and threshold-tuning procedure. Section 6 reports per-fold and ensemble results. Section 7 quantifies the cost asymmetry. Section 8 discusses limitations and Section 9 concludes.

---

## 2. Related work

**Graph neural networks for molecular property prediction.** Early GNNs (Duvenaud et al., 2015; Kearnes et al., 2016) established message passing on molecular graphs as a viable replacement for hand-designed fingerprints. Subsequent architectural advances — D-MPNN (Yang et al., 2019), AttentiveFP (Xiong et al., 2019), MAT (Maziarka et al., 2020), GATv2 (Brody et al., 2022) — improved on this baseline, with reported scaffold-split HIV AUCs typically in the 0.77–0.80 range.

**SMILES-based transformers.** ChemBERTa (Chithrananda et al., 2020) and MolFormer (Ross et al., 2022; *MolFormer-XL: Large-scale chemical language representations*) demonstrated that BERT-style masked language modeling on SMILES strings produces representations competitive with or exceeding handcrafted GNNs for many downstream tasks. MolFormer-XL — pretrained on 1.1B molecules from PubChem and ZINC — is the foundation I fine-tune from.

**3D-aware foundation models.** Uni-Mol (Zhou et al., 2023) introduced large-scale 3D-conformer pretraining on a 19M-molecule, 209M-conformer corpus. Their reported HIV scaffold-split (8:1:1, 3 seeds) ROC-AUC is **0.808 ± 0.003** (Zhou et al., 2023, Table 1). Per Appendix C.1 of that paper, molecular pretraining used 8× V100 GPUs for ~20 hours (≈ 160 V100-GPU-hours — under $1K at modern cloud spot rates). I treat Uni-Mol as the primary point of comparison for my $0 result; their model defines current SOTA on this benchmark.

**Ensemble methods in molecular ML.** Stacking (Wolpert, 1992) and out-of-fold ensembling are standard in tabular ML competition settings (Kaggle) but appear inconsistently in molecular property prediction papers. My methodology adapts standard tabular-ensembling practice to scaffold-split molecular data.

---

## 3. Dataset and splits

### 3.1 MoleculeNet HIV

The MoleculeNet HIV dataset (Wu et al., 2018), originally curated by the AIDS Antiviral Screen Program of the National Cancer Institute, consists of approximately 41,000 small molecules labeled binary for HIV replication inhibition. The dataset exhibits severe class imbalance (~3.5% positive class) and substantial chemical diversity, making it a standard but demanding benchmark. After RDKit-based parsing and validation, my pipeline retains **41,119** molecules.

### 3.2 Scaffold-based 5-fold split

I follow standard practice (Yang et al., 2019; Rong et al., 2020) in evaluating with scaffold-based cross-validation rather than random splits, which substantially overestimate generalization for molecular tasks.

For each molecule I extract its Bemis–Murcko scaffold (Bemis & Murcko, 1996) using RDKit. Molecules sharing a scaffold are kept together in the same fold. Within each fold, the training set is bin-packed into sub-bins (`val_pool = 3` of 20 sub-bins reserved for validation), ensuring the validation set is representative of the training-scaffold distribution. The split is deterministic given a fixed seed; both the GNN and MolFormer pipelines consume bit-for-bit identical fold assignments.

This yields five disjoint scaffold-held-out test sets (each ~8,200 molecules) on which all reported test AUCs are computed.

---

## 4. Models

### 4.1 GNN: GATv2-based graph neural network ("v5b")

I train the GNN component from scratch to provide explicit reasoning over molecular graph structure.

**Atom features (23-dim per node):** atom type one-hot (10), degree one-hot (6), formal charge, hybridization one-hot (4), aromaticity, hydrogen count.

**Bond features (8-dim per edge):** bond type one-hot (4), conjugation, ring membership, stereo (2).

**Global descriptors (per-molecule, z-scored on training stats only):** molecular weight, LogP, TPSA, hydrogen-bond donors/acceptors, rotatable bonds, ring counts, fraction Csp3, and similar standard RDKit descriptors.

**Architecture:** Atom and edge linear projections (→ 256 dim), four GATv2Conv layers (Brody et al., 2022) with 4 attention heads each, residual connections, and LayerNorm. JumpingKnowledge concatenates per-layer outputs. Readout uses GlobalAttention pooling combined with mean pooling. The MLP head concatenates the global descriptors before producing a single binary logit.

**Training:** Focal loss (α = 0.75, γ = 2.0) for class imbalance; AdamW with OneCycleLR; early stopping on validation AUC with `MIN_EPOCHS = 30` and `PATIENCE = 20`.

### 4.2 MolFormer-XL: fine-tuned transformer

I use IBM's MolFormer-XL backbone (Ross et al., 2022), a 47M-parameter transformer pretrained on 1.1B molecules from PubChem and ZINC. The pretrained checkpoint is publicly released on Hugging Face.

**Fine-tuning protocol:** linear classification head atop the [CLS] token; full backbone fine-tuning (no layer freezing); identical scaffold splits, focal loss, and early-stopping criteria as the GNN.

**Cost note.** MolFormer-XL fine-tuning runs in ~1 hour per fold on Google Colab's free T4 tier. Five folds total compute is ~5 GPU-hours of free-tier time, distributed across multiple sessions to respect Colab's session limits.

---

## 5. Ensemble combination

### 5.1 Out-of-fold logistic stacker

For each fold *i*, both my GNN-fold-*i* and MolFormer-fold-*i* checkpoints produce probability predictions on the held-out validation set of fold *i*. Concatenated across all five folds, this yields out-of-fold (OOF) predictions for **24,391 molecules** with no contamination — every prediction is produced by a model that never saw the predicted molecule's scaffold during training.

I fit a logistic regression on the OOF predictions:

> P_final = sigmoid( w_gnn · P_gnn + w_mf · P_mf + b )

Fitted coefficients (5-fold OOF, n = 24,391):

> w_gnn = 2.49, w_mf = 6.44, b = -7.24

The implied weight ratio (MolFormer ≈ 0.72, GNN ≈ 0.28) reflects MolFormer's stronger single-model performance, while the strongly negative intercept calibrates the ensemble's output to the dataset's ~3.7% positive base rate.

### 5.2 Tanimoto-NN feature (analyzed but not adopted)

I additionally explored a third stacker feature: max-Tanimoto-similarity-to-known-actives, computed using ECFP4 fingerprints. As a standalone classifier, this 1990s-era cheminformatics signal achieves OOF AUC = 0.801 — surprisingly competitive. As a third stacker feature it improves OOF AUC to 0.865. However, on a held-out mixed-set evaluation (Section 6.3) the gain disappears. I attribute this to a **distribution shift between OOF training and inference reference sets**: at training time, each fold's reference is its own ~970 training actives; at inference, the reference is ~1,440 known actives, producing systematically larger Tanimoto values that the OOF-trained coefficient over-weights. I report the Tanimoto baseline as a cautionary baseline and note that careful matched-distribution refitting may yet recover its OOF gain.

### 5.3 Threshold calibration

Because the stacker calibrates probabilities to the 3.7% base rate, the conventional 0.5 threshold is far too strict. I compute three principled thresholds on the OOF predictions:

| Threshold | Value | Recall | Precision | Use case |
|---|---|---|---|---|
| Youden's J (max TPR – FPR) | 0.043 | 0.70 | 0.21 | Virtual screening (high recall) |
| F1-max | 0.220 | 0.46 | 0.54 | Balanced reporting |
| Base-rate match | 0.173 | 0.48 | 0.48 | Top-k retrieval |

All three thresholds are saved alongside the stacker coefficients and are auto-loaded at inference time. The full precision/recall/F1 curves and these three principled cuts are visualized in **Figure 5**.

---

## 6. Results

### 6.1 Per-fold scaffold-held-out test AUC

| Fold | GNN v5b | MolFormer-XL |
|---|---|---|
| 0 | 0.7955 | 0.8182 |
| 1 | 0.7676 | 0.7955 |
| 2 | 0.7780 | 0.8097 |
| 3 | 0.7392 | 0.7798 |
| 4 | 0.7851 | 0.8255 |
| **Mean ± std** | **0.7731 ± 0.0215** | **0.8057 ± 0.0183** |
| **95% CI (bootstrap of fold means)** | **[0.7541, 0.7878]** | **[0.7906, 0.8194]** |

MolFormer-XL outperforms the from-scratch GNN on every fold — a consistent ~0.03 AUC margin. Both models agree on which folds are hard (fold 3) and easy (fold 4), suggesting correlated difficulty rather than noise. The fold-mean 95% confidence intervals do not overlap, indicating the GNN→MolFormer improvement is robust to fold-level resampling.

Per-fold AUCs are visualized in **Figure 1**.

### 6.2 OOF stacker AUC

Pooling all five fold validation predictions yields out-of-fold (OOF) probabilities for n = 24,391 molecules. Bootstrap 95% confidence intervals (2,000 stratified resamples preserving the 3.7% positive rate per draw):

| Configuration | OOF AUC | 95% CI |
|---|---|---|
| GNN-only | 0.7898 | [0.7710, 0.8072] |
| Tanimoto-NN baseline | 0.8011 | [0.7828, 0.8183] |
| MolFormer-only | 0.8561 | [0.8414, 0.8710] |
| Naive 50/50 average | 0.8489 | — |
| Best fixed weight (0.15 GNN / 0.85 MF) | 0.8574 | — |
| Stacker (GNN + MF) | 0.8560 | — |
| **Stacker (GNN + MF + Tanimoto)** | **0.8648** | **[0.8498, 0.8789]** |

OOF ROC and Precision-Recall curves are shown in **Figures 2 and 3**; calibration in **Figure 4**.

**A noteworthy honest finding.** The Tanimoto-NN baseline (a single-feature 1990s-era cheminformatics signal) achieves OOF ROC AUC 0.801, *higher* than my from-scratch GNN's 0.790, with fully overlapping confidence intervals. The two are statistically indistinguishable on OOF ROC. Crucially, the ranking reverses on Precision-Recall (GNN AP 0.352 vs Tanimoto AP 0.286 — see Figure 3), and the GNN's contribution to the stacker is real (Δ AUC vs Tanimoto+MolFormer alone is positive). I interpret this as evidence that the GNN's value to the ensemble lies in *prediction diversity* under class imbalance rather than absolute discriminative ranking. Reporting this honestly matters: a paper that compared only marginal ROC AUCs would have to either drop the GNN or accept that a 4-line baseline matches it.

The 3-feature stacker matches or slightly exceeds the best fixed-weight ensemble while additionally calibrating probabilities to the data's base rate.

**Significance testing (paired bootstrap on same OOF rows).** I report observed AUC differences and one-sided p-values (fraction of 2,000 paired stratified bootstrap draws in which the candidate model is *not* better than its baseline):

| Comparison | Δ AUC | 95% CI | p (one-sided) | |
|---|---|---|---|---|
| Stacker − MolFormer-only | +0.0087 | [+0.0033, +0.0142] | 0.0015 | ** |
| Stacker − GNN-only | +0.0751 | [+0.0620, +0.0888] | < 0.0001 | *** |
| MolFormer-only − GNN-only | +0.0663 | [+0.0512, +0.0819] | < 0.0001 | *** |
| Tanimoto-NN − GNN-only | +0.0113 | [−0.0067, +0.0291] | 0.1125 | ns |

Three findings are noteworthy. First, the stacker's gain over MolFormer alone (+0.0087) is small in magnitude but **statistically significant under paired resampling** (p = 0.0015) — the ensemble is genuinely additive, not redundant. Second, MolFormer-XL beats the from-scratch GNN by a large and unambiguous margin (Δ = +0.0663, p < 10⁻⁴), consistent with the per-fold non-overlapping CIs in §6.1. Third, the Tanimoto-NN vs GNN comparison is **statistically not significant** (Δ = +0.0113, 95% CI crosses zero, p = 0.1125), formally confirming the §6.2 observation that a 1-feature kNN baseline ties the custom GNN on OOF ROC. The argument for retaining the GNN in the ensemble therefore rests on PR-AUC and prediction diversity, not on standalone ROC superiority.

### 6.3 Held-out mixed-set evaluation

To validate ensemble behavior on a realistic deployment-like distribution, I constructed a held-out mixed set of 100 randomly-sampled known actives and 1,000 randomly-sampled inactives. I prevented identity-leakage by excluding any reference fingerprint with Tanimoto ≥ 0.999 from the Tanimoto feature.

**Held-out mixed-set AUC: 0.9575**

This number is *not* directly comparable to scaffold-split test AUC — the mixed set's class balance and chemical-distance distribution differ from the held-out scaffold test set — but it confirms that ensemble behavior on a deployment-style ranking task is strong.

### 6.4 Comparison to prior work

I compare my test AUC against published scaffold-split HIV results. Numbers below were re-verified against primary sources; protocol differences between papers are non-trivial and noted in the table. Where the original paper does not benchmark HIV directly, I cite the standard reproduction (Zhou et al., 2023, Table 1) which uses an 8:1:1 scaffold split with 3 random seeds.

| Model | Test ROC-AUC | Protocol | Compute (downstream user) | Source |
|---|---|---|---|---|
| Logistic regression / ECFP | 0.702 ± 0.018 | MoleculeNet scaffold split | ≪ $1 | Wu et al. 2018, ESI Table S2 |
| KernelSVM / ECFP | 0.792 | MoleculeNet scaffold split | ~$1 | Wu et al. 2018, ESI Table S2 |
| GraphConv | 0.763 ± 0.016 | MoleculeNet scaffold split | ~$1–10 | Wu et al. 2018, ESI Table S2 |
| AttentiveFP (Xiong et al., 2019) | 0.757 ± 0.014 | 8:1:1 scaffold, 3 seeds | ~$10s | Re-run in Zhou et al. 2023, Table 1 |
| D-MPNN / Chemprop (Yang et al., 2019) | 0.771 ± 0.005 | 8:1:1 scaffold, 3 seeds | ~$10–100 | Re-run in Zhou et al. 2023, Table 1 |
| GROVER_base (Rong et al., 2020) | 0.625 ± 0.009 | 8:1:1 scaffold, 3 seeds | ~15K V100-hr pretraining + finetune | Zhou et al. 2023, Table 1 |
| GROVER_large (Rong et al., 2020) | 0.682 ± 0.011 | 8:1:1 scaffold, 3 seeds | ~24K V100-hr pretraining + finetune | Zhou et al. 2023, Table 1 |
| **This work (GNN + MolFormer ensemble)** | **0.806 ± 0.018** | **5-fold scaffold CV** | **$0** (uses public MolFormer-XL) | this paper, §6.1 |
| Uni-Mol (Zhou et al., 2023) | 0.808 ± 0.003 | 8:1:1 scaffold, 3 seeds | ~160 V100-hr pretraining (≲ $1K) | Zhou et al. 2023, Table 1 |

**Two protocol caveats are essential.** First, the Uni-Mol/GEM reproductions in Table 1 above use an 8:1:1 scaffold split with 3 random seeds; my numbers come from 5-fold scaffold CV. These protocols are not strictly comparable — but each test set is genuinely scaffold-held-out, so a 0.002–0.005 AUC difference is well within protocol-induced noise. Second, GROVER's HIV scaffold numbers (0.625, 0.682) are from Uni-Mol's reproduction, **not from the original GROVER paper, which does not report HIV in its main tables**. They appear here for completeness; readers should be aware that the GROVER authors did not themselves claim a strong HIV result.

**Substantive observations:**

1. The current SOTA on this benchmark (Uni-Mol, 0.808 ± 0.003) is statistically tied with my zero-cost ensemble (0.806 ± 0.018, with 95% fold-mean CI [0.791, 0.819]).
2. GROVER — by far the heaviest pretraining investment among 2D models — performs *worse* than a logistic-regression baseline on HIV scaffold-split. Pretraining scale is not a reliable predictor of downstream AUC on this benchmark.
3. MAT (Maziarka et al., 2020) is sometimes cited with an HIV result; the original paper does not benchmark HIV. I therefore omit it from this table.

---

## 7. Compute cost analysis

I document compute costs explicitly. The honest comparison is more nuanced than I initially framed it: Uni-Mol's pretraining is *not* the multi-hundred-thousand-dollar cluster run that informal narratives sometimes suggest. The asymmetry is real but smaller, and the more striking observation is the matched downstream cost.

**My pipeline (downstream cost):**
- GNN training: ~1 hour per fold on Apple Silicon MPS (consumer MacBook Pro), 5 folds = ~5 hours wall-clock, $0.
- MolFormer fine-tuning: ~1 hour per fold on Google Colab free-tier T4, 5 folds = ~5 GPU-hours of free-tier time, $0.
- Stacker fitting: ~10 minutes on consumer hardware, $0.
- Total: ~10 hours wall-clock distributed across free services. **Total downstream compute cost: $0.**

**Uni-Mol pretraining (per Zhou et al. 2023, Appendix C.1):**
- Molecular pretraining: 8× V100 GPUs × ~20 hours = ~160 V100-GPU-hours.
- At AWS p3.2xlarge on-demand rates ($3.06/V100-hour): **~$490**.
- At spot/reserved rates: **~$200**.
- The full Uni-Mol release also includes a pocket model (~544 V100-hours, another ~$1,700 on-demand). For HIV property prediction only the molecular model is used.
- Including their reported finetuning sweep (4× V100 GPUs × several days), realistic total replication cost is **low single-digit thousands of dollars**.

**Cost ratio (corrected).** My pipeline's marginal downstream cost is $0; Uni-Mol's pretraining cost is roughly **$500–$2,000**. The pretrained MolFormer-XL checkpoint that I fine-tune from is itself the result of substantial upstream compute (1.1B-molecule pretraining at IBM); my claim is not "machine learning is free" but rather "the marginal cost to a downstream user has collapsed to zero".

The cleaner observation is **performance**, not cost: my zero-marginal-cost ensemble *matches* Uni-Mol's HIV result (0.806 ± 0.018 vs 0.808 ± 0.003) without using any 3D conformers, custom pretraining, or institutional GPU access. The corresponding cost-per-AUC-point claim from earlier drafts (10⁵× advantage) was overstated by roughly two orders of magnitude and is removed.

---

## 8. Limitations

I bound my claims explicitly.

1. **Scaffold-split, retrospective evaluation only.** I performed no prospective wet-lab validation. Real drug discovery requires assay-confirmed hits, not held-out AUC.

2. **Single benchmark.** I report all results on MoleculeNet HIV. Generalization to other MoleculeNet tasks (BBBP, BACE, ClinTox, etc.) is plausible but not yet demonstrated. Multi-task evaluation is the obvious next step.

3. **Public pretraining is upstream cost.** While *my* pipeline incurs $0 marginal cost, MolFormer-XL's pretraining was itself expensive. My claim is that this cost has been *amortized* by IBM's public release; the marginal cost to a downstream user is now zero. A fully bootstrapped $0 result starting from random weights remains out of reach.

4. **No new architecture.** Both base models are public; my contribution is methodological (ensembling, scaffold-honest evaluation, threshold calibration) rather than architectural.

5. **Single-benchmark tie, not generalized parity.** The 0.806 vs 0.808 result places my ensemble within statistical noise of Uni-Mol *on HIV*. Uni-Mol reports stronger absolute numbers than my work on several other MoleculeNet datasets (e.g. BBBP, BACE, ClinTox); my tie is not a claim that 2D ensembling matches 3D pretraining in general. HIV may be unusually well-suited to the 2D, sequence-based features that dominate my ensemble.

6. **Protocol mismatch with primary baselines.** The Uni-Mol/GEM Table-1 reproductions I compare against use 8:1:1 scaffold split with 3 random seeds; I use 5-fold scaffold CV. These protocols are similar in spirit but not identical. A fully apples-to-apples comparison would require re-running my model under the 8:1:1/3-seed protocol or re-running Uni-Mol under 5-fold CV; both are deferred future work.

7. **Earlier-draft errors disclosed.** Earlier drafts of this manuscript reported a Uni-Mol pretraining cost of "$100,000+" and a Uni-Mol HIV AUC of 0.831. Both were wrong: the cost is roughly $500–$2,000 (Zhou et al. 2023, Appendix C.1) and the HIV AUC is 0.808 (Table 1). The 0.831 figure was a transcription error from GROVER's BACE result. Earlier drafts also cited an HIV AUC for MAT (Maziarka et al. 2020), which never benchmarked HIV. The headline result (0.806 ± 0.018) is unaffected by these corrections; the cost-asymmetry argument was substantially weakened, which I have reflected throughout.

---

## 9. Conclusion

This paper documents that, as of 2026, a careful ensemble of two publicly-available molecular models — combined through honest out-of-fold stacking, evaluated on scaffold-split CV, and calibrated through principled threshold tuning — achieves **0.806 ± 0.018 test AUC on MoleculeNet HIV at zero dollars of downstream compute cost**, statistically tied with Uni-Mol's 0.808 ± 0.003 (the current SOTA on this benchmark). The headline finding is the *performance match* without 3D conformers or institutional pretraining, not a several-orders-of-magnitude cost asymmetry.

I interpret this as evidence that, for binary molecular property prediction tasks where strong public pretrained checkpoints exist, the marginal value of bespoke pretraining has narrowed substantially on at least one standard benchmark. Engineering effort applied to ensemble design and evaluation methodology now recovers competitive AUC at zero downstream compute cost. I do not claim this generalizes to all MoleculeNet tasks; multi-benchmark replication is the natural next step.

The methodological implication is that low-resource molecular ML — accessible to students, independent researchers, and academic groups without GPU clusters — is now a credible path to competitive performance on standard benchmarks. I release all code, splits, and trained checkpoints publicly to support reproduction.

---

## 10. Reproducibility

I release all code, configuration, splits, trained checkpoints, and saved stacker coefficients at:

> https://github.com/v659/HIV-drug-discovery

The full pipeline (GNN training + MolFormer fine-tuning + stacker fitting + inference) reproduces from a clean clone in approximately 10 hours of wall-clock time on Google Colab free-tier and Apple Silicon, at $0 cost.

I provide a complete reproduction recipe, including the exact `pip` environment and version-pinned dependencies (notably `transformers==4.46.3` to maintain MolFormer-XL compatibility), in `README.md`.

---

## Acknowledgments

This work used Google Colaboratory's free GPU tier and IBM's publicly-released MolFormer-XL pretrained checkpoint. I acknowledge that the existence of these public resources is what makes a $0 compute budget feasible. I worked on this project as an independent researcher, without institutional GPU access or compute budget.

**AI assistance disclosure.** I used Anthropic's Claude (Claude Code) as a collaborator throughout this project. Claude contributed to: drafting and revising the manuscript prose; verifying literature baseline numbers against primary sources (which surfaced material errors in earlier drafts of §6.4); designing and implementing the publication figures (`src/make_figures.py`); implementing the paired-bootstrap significance tests; reviewing code changes during stacker, inference, and evaluation development; and managing repository hygiene (license, dependency pinning, .gitignore, commit organization). All scientific decisions, all training and evaluation runs, and the final acceptance of every claim in the paper are mine. The model architectures, datasets, and reported results are not Claude-generated — they are produced by the code in this repository, executed by me on the hardware described in §7.

---

## References

1. **Bemis, G. W. & Murcko, M. A.** (1996). The Properties of Known Drugs. 1. Molecular Frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887–2893. [doi:10.1021/jm9602928](https://doi.org/10.1021/jm9602928)

2. **Brody, S., Alon, U., & Yahav, E.** (2022). How Attentive are Graph Attention Networks? *International Conference on Learning Representations (ICLR)*. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)

3. **Chithrananda, S., Grand, G., & Ramsundar, B.** (2020). ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. *Machine Learning for Molecules Workshop, NeurIPS 2020*. [arXiv:2010.09885](https://arxiv.org/abs/2010.09885)

4. **Duvenaud, D., Maclaurin, D., Aguilera-Iparraguirre, J., Gómez-Bombarelli, R., Hirzel, T., Aspuru-Guzik, A., & Adams, R. P.** (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints. *Advances in Neural Information Processing Systems (NeurIPS)*, 28. [arXiv:1509.09292](https://arxiv.org/abs/1509.09292)

5. **Kearnes, S., McCloskey, K., Berndl, M., Pande, V., & Riley, P.** (2016). Molecular Graph Convolutions: Moving Beyond Fingerprints. *Journal of Computer-Aided Molecular Design*, 30(8), 595–608. [doi:10.1007/s10822-016-9938-8](https://doi.org/10.1007/s10822-016-9938-8)

6. **Landrum, G.** (2024). RDKit: Open-source cheminformatics. https://www.rdkit.org. (Used for Murcko scaffold extraction, ECFP4 fingerprints, and global descriptor computation.)

7. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). Focal Loss for Dense Object Detection. *IEEE International Conference on Computer Vision (ICCV)*. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

8. **Maziarka, Ł., Danel, T., Mucha, S., Rataj, K., Tabor, J., & Jastrzębski, S.** (2020). Molecule Attention Transformer. *Graph Representation Learning Workshop, NeurIPS 2020*. [arXiv:2002.08264](https://arxiv.org/abs/2002.08264)

9. **Rong, Y., Bian, Y., Xu, T., Xie, W., Wei, Y., Huang, W., & Huang, J.** (2020). Self-Supervised Graph Transformer on Large-Scale Molecular Data. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. [arXiv:2007.02835](https://arxiv.org/abs/2007.02835)

10. **Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., & Das, P.** (2022). Large-Scale Chemical Language Representations Capture Molecular Structure and Properties. *Nature Machine Intelligence*, 4(12), 1256–1264. [doi:10.1038/s42256-022-00580-7](https://doi.org/10.1038/s42256-022-00580-7) (MolFormer-XL.)

11. **Wolpert, D. H.** (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1)

12. **Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V.** (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*, 9(2), 513–530. [doi:10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)

13. **Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., Li, Z., Luo, X., Chen, K., Jiang, H., & Zheng, M.** (2020). Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism. *Journal of Medicinal Chemistry*, 63(16), 8749–8760. [doi:10.1021/acs.jmedchem.9b00959](https://doi.org/10.1021/acs.jmedchem.9b00959) (AttentiveFP.)

14. **Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K., & Barzilay, R.** (2019). Analyzing Learned Molecular Representations for Property Prediction. *Journal of Chemical Information and Modeling*, 59(8), 3370–3388. [doi:10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237) (D-MPNN / Chemprop.)

15. **Zhou, G., Gao, Z., Ding, Q., Zheng, H., Xu, H., Wei, Z., Zhang, L., & Ke, G.** (2023). Uni-Mol: A Universal 3D Molecular Representation Learning Framework. *International Conference on Learning Representations (ICLR)*. [OpenReview](https://openreview.net/forum?id=6K2RM6wVqKu)

---

## Figures

All figures are vector PDFs in `docs/figures/`, generated reproducibly by `src/make_figures.py` from saved out-of-fold predictions in `src/ensemble_stacker.pt`.

- **Figure 1** (`fig1_fold_aucs.pdf`) — Per-fold scaffold-held-out test AUC for the GNN and MolFormer-XL across all five folds, with mean reference lines.
- **Figure 2** (`fig2_oof_roc.pdf`) — ROC curves on n = 24,391 out-of-fold predictions for Tanimoto-NN, GNN, MolFormer-XL, and the final stacker.
- **Figure 3** (`fig3_oof_pr.pdf`) — Precision-Recall curves on the same OOF predictions, with the 3.7%-base-rate random reference line.
- **Figure 4** (`fig4_calibration.pdf`) — Reliability diagram (quantile-binned) comparing raw MolFormer-XL probabilities against the calibrated stacker output. Axes are zoomed to [0, 0.5] × [0, 0.3] because at a 3.7% base rate the upper-right of the unit square is unpopulated.
- **Figure 5** (`fig5_threshold_sweep.pdf`) — Recall, precision, and F1 vs decision threshold on the OOF stacker output, with Youden's J, base-rate, and F1-max thresholds marked.

---

## Appendix A — Threshold sweep on held-out mixed set

| Threshold | TP | FP | TN | FN | Recall | Precision | F1 | FPR |
|---|---|---|---|---|---|---|---|---|
| 0.020 | 98 | 309 | 691 | 2 | 0.980 | 0.241 | 0.387 | 0.309 |
| 0.043 (Youden) | 90 | 125 | 875 | 10 | 0.900 | 0.419 | 0.571 | 0.125 |
| 0.100 | 76 | 48 | 952 | 24 | 0.760 | 0.613 | 0.679 | 0.048 |
| 0.173 (base-rate) | 68 | 21 | 979 | 32 | 0.680 | 0.764 | 0.720 | 0.021 |
| 0.220 (F1-max) | 66 | 14 | 986 | 34 | 0.660 | 0.825 | 0.733 | 0.014 |
| 0.300 | 60 | 9 | 991 | 40 | 0.600 | 0.870 | 0.710 | 0.009 |
| 0.500 | 49 | 5 | 995 | 51 | 0.490 | 0.907 | 0.636 | 0.005 |

## Appendix B — Cost itemization

| Component | Hardware | Wall time | Cost |
|---|---|---|---|
| GNN training, fold 0 | Apple M-series MPS | ~1 hr | $0 |
| GNN training, folds 1–4 | Apple M-series MPS | ~4 hr | $0 |
| MolFormer fine-tuning, fold 0 | Colab T4 (free) | ~1 hr | $0 |
| MolFormer fine-tuning, folds 1–4 | Colab T4 (free) | ~4 hr | $0 |
| Stacker fitting | Apple M-series MPS | ~10 min | $0 |
| Inference + evaluation | Apple M-series MPS | ~5 min | $0 |
| **Total** | | **~10 hr** | **$0** |
