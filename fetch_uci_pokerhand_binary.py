import argparse
import csv
import os
import random
import urllib.request


# Use plain HTTP to avoid local SSL certificate issues on some machines.
UCI_BASE = "http://archive.ics.uci.edu/ml/machine-learning-databases/poker/"


def _download_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _row_to_sequence(row):
    # row: 11 ints: S1,R1,...,S5,R5,class
    suits_ranks = row[:10]
    toks = []
    for i in range(0, 10, 2):
        s = int(suits_ranks[i])
        r = int(suits_ranks[i + 1])
        toks.append(f"S{s}R{r}")
    return ",".join(toks)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch UCI poker-hand and convert to binary dataset")
    p.add_argument("--out_csv", default="datasets_poker_uci/uci_pokerhand_binary.csv")
    p.add_argument("--max_rows", type=int, default=200000, help="Max rows to write (after shuffle)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    train_url = UCI_BASE + "poker-hand-training-true.data"
    test_url = UCI_BASE + "poker-hand-testing.data"

    train_txt = _download_text(train_url)
    test_txt = _download_text(test_url)

    def parse_lines(txt: str):
        out = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 11:
                continue
            out.append([int(x) for x in parts])
        return out

    rows = parse_lines(train_txt) + parse_lines(test_txt)
    random.shuffle(rows)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "label"])
        for row in rows[: args.max_rows]:
            klass = int(row[10])  # 0..9
            # Binary label: 0 = high card only, 1 = pair or better
            label = 0 if klass == 0 else 1
            w.writerow([_row_to_sequence(row), label])

    print(f"Wrote {min(args.max_rows, len(rows))} rows to {args.out_csv}")


if __name__ == "__main__":
    main()

