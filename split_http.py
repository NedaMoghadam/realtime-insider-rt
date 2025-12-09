import pandas as pd

SOURCE = r"C:\path\to\your\http.csv"   # <-- CHANGE THIS
ROWS_PER_CHUNK = 500_000              # adjust if needed

def main():
    i = 0
    for chunk in pd.read_csv(SOURCE, chunksize=ROWS_PER_CHUNK):
        out_path = f"http_part_{i:03d}.csv"
        print(f"Writing {out_path} with {len(chunk)} rows")
        chunk.to_csv(out_path, index=False)
        i += 1

    print("Done! Created", i, "chunk files.")

if __name__ == "__main__":
    main()
