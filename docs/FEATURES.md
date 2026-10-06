# Feature catalogue

Exact feature lists of the trained models are in [`../experiments/feature_lists/`](../experiments/feature_lists/):
`lgb_m12_features.json` (stage 1, 71 features) and `lgb_s12a_features.json` (final stage 2, 84 features).

Notation: *S1* = the Source-1 entity of the pair, *record* = the Source-2/3 record of the pair, *core name* = the name
without legal form, honorifics, alias markers and filler words.

---

## Stage 1: pairwise features (71) — `src/features.py`

### Retrieval context (8)
| Feature | Meaning |
|---|---|
| `blk_cos` | blocking cosine similarity |
| `blk_rank_s1`, `blk_rank_p` | rank of the pair in the S1's list and in the record's list |
| `blk_n_s1`, `blk_n_p` | sizes of those lists |
| `blk_gap_s1max`, `blk_gap_pmax` | gap to the best cosine in the S1's / record's list |
| `blk_margin_p` | margin of this S1 over the record's next-best S1 (top feature by split gain in stage 1) |

### Name similarity (11)
| Feature | Meaning |
|---|---|
| `n_ratio`, `n_tsort`, `n_tset`, `n_partial`, `n_jw` | rapidfuzz ratio / token-sort / token-set / partial ratio and Jaro-Winkler on the core name |
| `n_join`, `n_join_partial` | the same on the *glued* name (spaces removed), which handles token splits and web domains |
| `n_join_full` | glued record name vs glued full S1 name (keeps the legal form, for domains such as `acmellc.com`) |
| `n_full` | ratio on the full name including the legal form |
| `n_sorted_lev` | normalised Levenshtein on alphabetically sorted tokens |
| `n_acronym` | the record name is the initials of the S1 core or full name |

### Address similarity (7)
| Feature | Meaning |
|---|---|
| `a_ratio`, `a_tsort`, `a_tset`, `a_partial` | rapidfuzz scores on the full normalised address |
| `ac_ratio`, `ac_tsort`, `ac_tset` | the same on the *core* address (per-country frequent tokens removed) |

### IDF-weighted token overlap (12)
For names (`n_*`) and addresses (`a_*`), using the token IDF of the country:

| Feature | Meaning |
|---|---|
| `*_w1`, `*_w2` | shared IDF mass as a share of the S1's / record's IDF mass |
| `*_wj` | weighted Jaccard |
| `*_common` | number of shared tokens |
| `*_maxidf` | IDF of the rarest shared token |
| `*_miss` | IDF mass not shared |

### House numbers (11) — Numba kernel `_numfeat`
| Feature | Meaning |
|---|---|
| `num_n1`, `num_n2` | count of numbers on each side |
| `num_f1`, `num_f2` | first (house) number on each side |
| `num_first_eq` | first numbers are equal |
| `num_first_dlev` | digit edit distance between the first numbers (noise drops or changes single digits) |
| `num_first_in` | the record's first number appears anywhere in the S1 address |
| `num_absdiff`, `num_lendiff` | numeric and digit-length difference to the closest number |
| `num_first_absdiff`, `num_first_lendiff` | the same for the first numbers (look-alikes sit at a nearby number with the same digit count) |

### Record and legal-form flags (10)
| Feature | Meaning |
|---|---|
| `p_is_domain`, `p_has_alias`, `p_name_indic`, `p_addr_empty`, `p_is_s3` | the record name is a web domain / an alias construction / in an Indic script; blank address; Source 3 |
| `p_name_unseen` | share of name tokens never seen in Source 1 (generated replacement names) |
| `p_marker_extra` | sister-company marker words in the record that the S1 name lacks |
| `legal_eq`, `legal_s1_empty`, `legal_p_empty` | legal forms agree; legal form missing on either side |

### Commonness and length (12)
| Feature | Meaning |
|---|---|
| `name_f1_s1`, `name_fp_s1`, `name_f1_p`, `name_fp_p` | how many Source-1 / pool rows of the country share the S1's / record's sorted name |
| `addr_f1_s1`, `addr_fp_s1`, `addr_f1_p`, `addr_fp_p` | the same for the full address |
| `n_len1`, `n_len2`, `a_len1`, `a_len2` | name and address lengths |

---

## Stage 2: reasoning over groups of records (84) — `src/stage2.py`, `src/stage2_emb.py`

Stage 2 receives the stage-1 probability `prob1`, `blk_cos`, 26 of the strongest stage-1 features, and the families
below. Every group statistic excludes the pair itself.

### Competition (13)
| Feature | Meaning |
|---|---|
| `g_rank_p`, `g_rank_s1` | rank of the pair among the record's owners / among the S1's records |
| `g_pmax`, `g_psum`, `g_p_strong` | best probability, total mass and number of strong (> 0.5) owners of the record |
| `g_smax`, `g_ssum`, `g_s_strong` | the same for the S1's records |
| `g_p_other`, `g_p_margin` | best *competing* owner of the record and this pair's margin over it |
| `g_ssum_excl`, `g_s_strong_excl`, `g_ssum_others` | the S1's mass counting only records for which it is the top owner; mass of its other records |

### Sibling agreement (9)
Similarity between this record and the S1's most confident other records ("siblings").

| Feature | Meaning |
|---|---|
| `sib_name_ratio_max`, `sib_name_tset_max`, `sib_addr_ratio_max`, `sib_addr_tset_max` | best name / address agreement with a sibling |
| `sib_name_w`, `sib_addr_w` | the same weighted by the sibling's probability |
| `sib_name_same_src`, `sib_addr_same_src` | best agreement with a sibling *from the same source* (copies share a per-source base record) |
| `sib_n` | number of siblings |

### Consistency of deviations (17)
How many other records of the S1 support this record's house number, name variant and address core, and how much
probability mass they carry. Split into same-source and other-source support, because a noisy number shared within one
source is usually a genuine copy while a number shared across sources is often a decoy group.

`cs_num_sup`, `cs_num_mass`, `cs_num_sup_same_src`, `cs_num_sup_other_src`, `cs_num_mass_same_src`,
`cs_num_mass_other_src`, `cs_s1num_mass`, `cs_s1num_mass_same_src`, `cs_s1num_mass_other_src`, `cs_num_is_s1`,
`cs_name_sup`, `cs_name_mass`, `cs_name_sup_same_src`, `cs_name_sup_other_src`, `cs_name_is_s1`, `cs_addr_sup`,
`cs_addr_mass`

### Signed offset and source slots (6)
| Feature | Meaning |
|---|---|
| `cs_num_sdiff`, `cs_num_srel` | signed (and relative) house-number difference to the S1's number |
| `sl_same_src_strong`, `sl_other_src_strong` | strong records already present in the same / the other source slot |
| `sl_same_src_mass` | probability mass already in the same source slot |
| `sl_comp_same_src_strong` | strong records in the same source slot of the competing owner |

### LaBSE multilingual similarity (11) — `src/emb_pairs.py`, `src/emb_sib.py`
LaBSE vectors (PCA-128, L2-normalised) of the normalised name and address.

| Feature | Meaning |
|---|---|
| `emb_cos`, `emba_cos` | name / address cosine |
| `emb_rank_s1`, `emba_rank_s1` | rank of the cosine among the S1's candidates |
| `emb_gap_s1`, `emba_gap_s1` | gap to the best cosine among the S1's candidates |
| `emb_gap_p`, `emba_gap_p` | gap to the best cosine among the record's competing owners |
| `emb_sib_max` | best name cosine to the S1's top-3 confident siblings (p ≥ 0.3) |
| `emb_sib_same_src` | the same, siblings from the same source only (5th most important stage-2 feature) |
| `emb_sib_w` | best sibling cosine × sibling probability |
