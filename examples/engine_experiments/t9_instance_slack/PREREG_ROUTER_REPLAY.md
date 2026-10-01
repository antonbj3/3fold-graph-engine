# Frozen policy replay

Implement the already frozen 3.245 max-degree/mean-degree policy directly inside
witness construction. No training or selection is performed. Reuse held-out exact
references solely for verification; confirm every endpoint and first-sign update
matches the previously selected single-form trajectory. Charge native policy
selection and selected-form preparation; do not produce both forms at query time.
This replay validates implementation/cost of G3, not a second statistical test.
