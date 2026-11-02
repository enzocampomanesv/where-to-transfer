import os
import pandas as pd
from pathlib import Path

tier2_dir = Path(os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/transferLearning"))
for folder in ["informed_t1__layerwise_lr1e-04", "random__layerwise_lr1e-04"]:
    f = next((tier2_dir / folder).glob("*.csv"))
    df = pd.read_csv(f)
    print(f"\n{folder}")
    print(df[["strategy", "sampling", "sample_size"]].drop_duplicates().sort_values("sample_size"))