import re, csv

def parse_results(log_file="recall_results_500.txt"):
    results = {}
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"✅ (\w+): ([0-9.]+)", line)
            if m:
                task, val = m.group(1), float(m.group(2))
                results[task] = val
    return results

def write_summary(results, out_csv="summary_table.csv"):
    if not results:
        print("❌ No results found.")
        return
    avg = sum(results.values()) / len(results)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "accuracy"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.3f}"])
        writer.writerow(["avg", f"{avg:.3f}"])
    print("=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
    print(f"avg: {avg:.3f}")
    print(f"[✅ Saved to {out_csv}]")

if __name__ == "__main__":
    results = parse_results()
    write_summary(results)
