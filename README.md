# EEG-Signal-Anonymization

1) The family in pseudocode. Everything in "suppression/generalization" reduces to two atomic, many-to-one operations — suppress (delete or zero a coordinate/record) and generalize (coarsen a coordinate's alphabet by binning) — and both are lossy for the reason we formalized: f: A → B becomes non-injective, so I(B;S) ≤ I(A;S) by the data-processing inequality.
