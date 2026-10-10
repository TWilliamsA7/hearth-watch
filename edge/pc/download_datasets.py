"""Download the extra fall datasets (OCCU, CAUCAFall, OF-Synthetic) - only the videos we use.

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\download_datasets.py                 (all three)
    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\download_datasets.py --only occu     (one of them)

Everything goes to D:\\hearth-data (--data). Standard library only. Safe to re-run: files that
are already there and complete are skipped. Labels for all datasets (OmniFall, see README Step 6)
go to <data>\\labels.

  occu       OCCU.zip on Zenodo is 10.8 GB, almost all raw Kinect frames; it also holds the 10
             ready-made colour videos (~16 MB). Zip files list their contents at the END, so we
             read that list with small HTTP range requests and download only those 10 videos,
             checking each one's CRC-32. -> <data>\\occu\\<person>\\<view>\\<name>_rgb.mp4
  caucafall  Mendeley's "Download all" is ~10 GB of still images; the 100 videos are 135 MB.
             We list the folders through Mendeley's public API and download only the .avi files,
             checking each SHA-256. -> <data>\\caucafall\\Subject.N\\<activity>\\<name>.avi
  of-syn     12,000 AI-generated clips in one 9.7 GB .tar on Hugging Face. A tar has no index, so
             it is streamed once from start to end, and only the clips matching the care-room
             filter are written (indoor; ages 18-34, 35-64, 65+; camera at eye height, high or
             top; 2,203 clips, ~1.2 GB). --syn-all keeps all 12,000.
             -> <data>\\of-syn\\<folder>\\<name>.mp4

Licenses: OCCU and CAUCAFall CC BY 4.0, OF-Synthetic CC BY-NC 4.0 (credit them; non-commercial).
"""

import argparse
import csv
import hashlib
import io
import json
import struct
import tarfile
import time
import urllib.request
import zlib
from pathlib import Path

UA = {"User-Agent": "hearth-dataset-fetch/1.0"}     # Mendeley refuses Python's default agent
OMNIFALL = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/"
LABELS = ["GMDCSA24", "le2i", "occu", "caucafall", "of-syn"]
OCCU_ZIP = "https://zenodo.org/api/records/15494102/files/OCCU.zip/content"
MENDELEY = "https://data.mendeley.com/public-api/datasets/7w7fccy7ky"
CAUCAFALL_VERSION = 4
SYN_TAR = OMNIFALL + "data_files/omnifall-synthetic_av1.tar"
SYN_AGES = {"young_adults_18_34", "middle_aged_35_64", "elderly_65_plus"}
SYN_CAMERAS = {"eye", "high", "top"}


def request(url, start=None, end=None):
    headers = dict(UA)
    if start is not None:
        headers["Range"] = "bytes=%d-%d" % (start, end)
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def fetch(url, start=None, end=None):
    with request(url, start, end) as r:
        return r.read()


def labels(data):
    folder = data / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    for name in LABELS:
        path = folder / ("%s.csv" % name)
        if not path.exists():
            path.write_bytes(fetch(OMNIFALL + "labels/%s.csv" % name))
    print("labels: %s in %s" % (", ".join(LABELS), folder))


# --- OCCU: a few files out of a remote 10.8 GB zip -------------------------------------------

def zip_listing(url):
    """Read a remote zip's central directory (its table of contents, at the end of the file)."""
    with request(url, 0, 0) as r:
        size = int(r.headers["Content-Range"].split("/")[1])
    tail = fetch(url, size - 65536, size - 1)
    end = tail.rfind(b"PK\x05\x06")                       # end-of-central-directory record
    cd_size, cd_offset = struct.unpack("<II", tail[end + 12:end + 20])
    loc = tail.rfind(b"PK\x06\x07")                       # zip64 locator (files > 4 GB)
    if loc >= 0:
        z64 = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        cd_size, cd_offset = struct.unpack("<QQ", fetch(url, z64 + 40, z64 + 55))
    cd, pos, entries = fetch(url, cd_offset, cd_offset + cd_size - 1), 0, []
    while cd[pos:pos + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[pos + 10:pos + 12])
        crc, csize, usize = struct.unpack("<III", cd[pos + 16:pos + 28])
        n, e, c = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        offset, = struct.unpack("<I", cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + n].decode("utf-8")
        extra = cd[pos + 46 + n:pos + 46 + n + e]
        # zip64: values that didn't fit in 32 bits are 0xFFFFFFFF here and stored in extra field 1
        x = 0
        while x + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[x:x + 4])
            if tag == 1:
                vals, v = extra[x + 4:x + 4 + length], 0
                if usize == 0xFFFFFFFF:
                    usize, = struct.unpack("<Q", vals[v:v + 8]); v += 8
                if csize == 0xFFFFFFFF:
                    csize, = struct.unpack("<Q", vals[v:v + 8]); v += 8
                if offset == 0xFFFFFFFF:
                    offset, = struct.unpack("<Q", vals[v:v + 8]); v += 8
            x += 4 + length
        entries.append({"name": name, "method": method, "crc": crc, "csize": csize,
                        "usize": usize, "offset": offset})
        pos += 46 + n + e + c
    return entries


def zip_extract(url, entry):
    header = fetch(url, entry["offset"], entry["offset"] + 29)
    n, e = struct.unpack("<HH", header[26:30])
    start = entry["offset"] + 30 + n + e
    raw = fetch(url, start, start + entry["csize"] - 1)
    if entry["method"] == 8:                              # deflate
        raw = zlib.decompressobj(-15).decompress(raw)
    elif entry["method"] != 0:                            # 0 = stored as is
        raise ValueError("unsupported zip compression %d for %s" % (entry["method"], entry["name"]))
    if zlib.crc32(raw) & 0xFFFFFFFF != entry["crc"] or len(raw) != entry["usize"]:
        raise ValueError("damaged download: %s (CRC/size mismatch)" % entry["name"])
    return raw


def occu(data, limit):
    print("occu: reading the zip's table of contents (no 10.8 GB download)...")
    videos = [e for e in zip_listing(OCCU_ZIP) if e["name"].endswith("_rgb.mp4")][: limit or None]
    for e in videos:
        parts = e["name"].split("/")                      # OCCU/jiayan/jiayan/view1/rgb/x.mp4
        out = data / "occu" / parts[1] / parts[3] / parts[-1]
        if out.exists() and out.stat().st_size == e["usize"]:
            print("  have %s" % out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zip_extract(OCCU_ZIP, e))
        print("  got  %s (%.1f MB, CRC ok)" % (out, e["usize"] / 1e6))
    print("occu: %d videos" % len(videos))


# --- CAUCAFall: only the .avi files, through Mendeley's API ----------------------------------

def caucafall(data, limit):
    folders = json.load(request("%s/folders/%d" % (MENDELEY, CAUCAFALL_VERSION)))
    by_id = {f["id"]: f for f in folders}

    def path(f):
        names = [f["name"]]
        while f.get("parent_id") in by_id:
            f = by_id[f["parent_id"]]
            names.insert(0, f["name"])
        return names

    leaves = sorted((path(f), f["id"]) for f in folders if len(path(f)) == 3)   # CAUCAFall/Subject.N/Activity
    got = 0
    for names, folder_id in leaves[: limit or None]:
        files = json.load(request("%s/files?folder_id=%s&version=%d" % (MENDELEY, folder_id, CAUCAFALL_VERSION)))
        for f in files:
            if not f["filename"].lower().endswith(".avi"):
                continue
            info = f["content_details"]
            out = data / "caucafall" / names[1] / names[2] / f["filename"]
            if out.exists() and out.stat().st_size == info["size"]:
                print("  have %s" % out)
                got += 1
                continue
            raw = fetch(info["download_url"])
            if hashlib.sha256(raw).hexdigest() != info["sha256_hash"]:
                raise ValueError("damaged download: %s (SHA-256 mismatch)" % f["filename"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            got += 1
            print("  got  %s (%.1f MB, SHA-256 ok)" % (out, len(raw) / 1e6))
    print("caucafall: %d videos" % got)


# --- OF-Synthetic: stream the 9.7 GB tar once, keep the care-room clips ----------------------

class Counting(io.RawIOBase):
    """Wraps the download so we can print how far through the 9.7 GB we are."""

    def __init__(self, stream, total):
        self.stream, self.total, self.done, self.last = stream, total, 0, 0.0

    def readable(self):
        return True

    def readinto(self, buf):
        chunk = self.stream.read(len(buf))
        buf[:len(chunk)] = chunk
        self.done += len(chunk)
        if time.time() - self.last > 5:
            self.last = time.time()
            print("  ... %.1f of %.1f GB streamed" % (self.done / 1e9, self.total / 1e9), flush=True)
        return len(chunk)


def of_syn(data, keep_all, limit):
    rows = list(csv.DictReader(open(data / "labels" / "of-syn.csv", newline="")))
    wanted = {r["path"].split("/")[-1] for r in rows
              if keep_all or (r["environment_category"] == "indoor" and r["age_group"] in SYN_AGES
                              and r["camera_elevation"] in SYN_CAMERAS)}
    if limit:
        wanted = set(sorted(wanted)[:limit])
    have = {p.stem for p in (data / "of-syn").rglob("*.mp4")} if (data / "of-syn").exists() else set()
    if wanted <= have:
        print("of-syn: all %d wanted clips already there" % len(wanted))
        return
    print("of-syn: streaming the 9.7 GB archive once, keeping %d clips (%d already there)..."
          % (len(wanted), len(wanted & have)))
    response = request(SYN_TAR)
    total = int(response.headers.get("Content-Length", 0))
    kept = 0
    with tarfile.open(fileobj=io.BufferedReader(Counting(response, total), 1 << 20), mode="r|") as tar:
        for member in tar:                                # in archive order, no going back
            stem = Path(member.name).stem
            if not member.isfile() or stem not in wanted:
                continue
            out = data / "of-syn" / Path(member.name).parent.name / Path(member.name).name
            if stem not in have:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tar.extractfile(member).read())
            kept += 1
            if limit and kept >= len(wanted):
                break
    print("of-syn: %d clips in %s" % (kept, data / "of-syn"))


def main():
    ap = argparse.ArgumentParser(description="Download OCCU, CAUCAFall and OF-Synthetic videos.")
    ap.add_argument("--data", default=r"D:\hearth-data", type=Path)
    ap.add_argument("--only", default="occu,caucafall,of-syn", help="comma list of datasets")
    ap.add_argument("--syn-all", action="store_true", help="keep all 12,000 synthetic clips")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos per dataset (a test)")
    args = ap.parse_args()
    start = time.time()
    labels(args.data)
    only = set(args.only.split(","))
    if "occu" in only:
        occu(args.data, args.limit)
    if "caucafall" in only:
        caucafall(args.data, args.limit)
    if "of-syn" in only:
        of_syn(args.data, args.syn_all, args.limit)
    print("done in %.0f s" % (time.time() - start))


if __name__ == "__main__":
    main()
