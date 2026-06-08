"""
Batch-test backend menggunakan satu folder dengan naming convention:
    r1, r2, r3, ...  → real
    f1, f2, f3, ...  → fake

Usage:
    1. Pastiin app.py jalan di terminal lain
    2. python test_batch.py "C:/Users/yourname/Downloads"
       atau
       python test_batch.py ~/Downloads
"""
import os, sys, glob, re, requests
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python test_batch.py /path/to/folder")
    print('Windows example: python test_batch.py "C:\\Users\\YourName\\Downloads"')
    print("Mac example:     python test_batch.py ~/Downloads")
    sys.exit(1)

folder = os.path.expanduser(sys.argv[1])
URL = "http://localhost:5000/analyze"

if not os.path.isdir(folder):
    print(f"❌ Folder tidak ditemukan: {folder}")
    sys.exit(1)

# Cari semua gambar
EXTS = ["jpg", "jpeg", "png", "webp"]
all_files = []
for ext in EXTS:
    all_files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
    all_files.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))

# Klasifikasi berdasarkan prefix nama file (r1.jpg → real, f1.jpg → fake)
pattern = re.compile(r"^([rf])(\d+)\.", re.IGNORECASE)
labeled_files = []
for fp in all_files:
    name = os.path.basename(fp)
    m = pattern.match(name)
    if m:
        prefix = m.group(1).lower()
        true_label = "real" if prefix == "r" else "fake"
        idx = int(m.group(2))
        labeled_files.append((fp, true_label, idx))

if not labeled_files:
    print(f"❌ Gak ada file dengan format r1.jpg / f1.jpg dst di {folder}")
    print("   File yang ada di folder:")
    for fp in sorted(all_files)[:10]:
        print(f"     {os.path.basename(fp)}")
    sys.exit(1)

# Sort: real dulu (sesuai index), terus fake
labeled_files.sort(key=lambda x: (x[1], x[2]))
print(f"✅ Ditemukan {len(labeled_files)} gambar berlabel di {folder}\n")

stats = Counter()
mistakes = []
current_label = None

for fp, true_label, idx in labeled_files:
    # Print header tiap ganti kategori
    if true_label != current_label:
        count = sum(1 for _, t, _ in labeled_files if t == true_label)
        print(f"\n=== {true_label.upper()} ({count} files) ===")
        print(f"{'file':<25} {'pred':<6} {'p_fake':>9}  {'ok'}")
        print("-" * 55)
        current_label = true_label

    try:
        with open(fp, "rb") as f:
            r = requests.post(URL, files={"image": f}, timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Backend gak nyala di {URL}")
        print("   Jalanin dulu: python app.py")
        sys.exit(1)

    if r.status_code != 200:
        print(f"  ERROR on {fp}: {r.text}")
        continue

    d = r.json()
    pred = "fake" if d["is_fake"] else "real"
    ok = pred == true_label
    stats[(true_label, pred)] += 1
    if not ok:
        mistakes.append((fp, true_label, pred, d["p_fake"]))
    mark = "OK" if ok else "WRONG"
    name = os.path.basename(fp)[:23]
    print(f"{name:<25} {pred:<6} {d['p_fake']:>8.2f}%  {mark}")

# Summary
print("\n" + "=" * 55)
print("CONFUSION MATRIX")
print("=" * 55)
print(f"{'':<10} {'pred_real':>12} {'pred_fake':>12}")
print(f"{'true_real':<10} {stats[('real','real')]:>12} {stats[('real','fake')]:>12}")
print(f"{'true_fake':<10} {stats[('fake','real')]:>12} {stats[('fake','fake')]:>12}")

total = sum(stats.values())
correct = stats[("real","real")] + stats[("fake","fake")]
if total:
    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

n_real = stats[("real","real")] + stats[("real","fake")]
n_fake = stats[("fake","real")] + stats[("fake","fake")]
if n_real:
    print(f"Real recall:  {stats[('real','real')]}/{n_real} = {100*stats[('real','real')]/n_real:.1f}%")
if n_fake:
    print(f"Fake recall:  {stats[('fake','fake')]}/{n_fake} = {100*stats[('fake','fake')]/n_fake:.1f}%")

if mistakes:
    print(f"\n{len(mistakes)} MISCLASSIFIED:")
    for fp, t, p, pf in mistakes:
        print(f"  {os.path.basename(fp):<25}  true={t:<5} pred={p:<5} p_fake={pf:.2f}%")