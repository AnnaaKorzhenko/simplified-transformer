# new_verified

Contains the new verified pipeline artifacts:

- `data/` generated formula + dataset with alphabet size <= 6
- `verification/` external BLACK verification results and wrapper
- `training/` simplified-transformer layer comparison results
- `plots/` improved visualizations for 1/2/3 layer comparison

Reproduction commands:

```bash
python3 generate_diamond_star_dataset.py --output_dir generated_diamond_star --formula_id 1 --alphabet_size 6 --num_disjunctions 2 --num_conjunctions 1 --sequence_length 10 --num_positive 500 --num_negative 500 --seed 42
python3 verify_with_black.py --black-bin /opt/homebrew/bin/black -f generated_diamond_star/formulas/formula_1.json -d generated_diamond_star/datasets/dataset_1.csv -o black_verification_results.json
python3 compare_hard_layers.py --dataset_dir generated_diamond_star --formula_id 1 --epochs 100 --out_json results_hard_layers.json --out_plot plots_three_models/hard_layers_comparison.png
```

