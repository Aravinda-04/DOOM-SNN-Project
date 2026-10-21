"""Automated asset setup for DOOM SNN Agent.

Downloads freedoom2.wad if missing and generates all jittered WADs and scenario configs.
Run once after cloning:
    python setup_assets.py
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FREEDOOM_URL = "https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip"
BASE_WAD = PROJECT_ROOT / "freedoom2.wad"


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100.0)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  [Downloading] {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  [Downloading] {mb_down:.1f} MB")
    sys.stdout.flush()


def ensure_freedoom2_wad() -> bool:
    if BASE_WAD.exists() and BASE_WAD.stat().st_size > 1_000_000:
        print(f"[OK] freedoom2.wad found ({BASE_WAD.stat().st_size / (1024*1024):.1f} MB).")
        return True

    print(f"[+] freedoom2.wad not found in {PROJECT_ROOT}.")
    print(f"    Fetching official FreeDoom 0.13.0 release from GitHub...")

    temp_zip = PROJECT_ROOT / "_freedoom_download.zip"
    try:
        req = urllib.request.Request(
            FREEDOOM_URL,
            headers={"User-Agent": "DOOM-SNN-Setup/1.0 (Python urllib)"},
        )
        with urllib.request.urlopen(req) as response, open(temp_zip, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64
            block_num = 0
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                block_num += 1
                download_progress(block_num, block_size, total_size)
        print()

        print("    Extracting freedoom2.wad...")
        with zipfile.ZipFile(temp_zip, "r") as zf:
            target_name = None
            for name in zf.namelist():
                if name.endswith("freedoom2.wad"):
                    target_name = name
                    break
            if not target_name:
                raise RuntimeError("freedoom2.wad not found inside downloaded archive.")
            with zf.open(target_name) as source, open(BASE_WAD, "wb") as dest:
                shutil.copyfileobj(source, dest)

        print(f"[OK] freedoom2.wad successfully extracted ({BASE_WAD.stat().st_size / (1024*1024):.1f} MB).")
        return True
    except Exception as exc:
        print(f"\n[ERROR] Failed to auto-download FreeDoom: {exc}")
        print("Please manually download freedoom2.wad from https://freedoom.github.io/download.html")
        print(f"and place it directly in: {PROJECT_ROOT}")
        return False
    finally:
        if temp_zip.exists():
            temp_zip.unlink(missing_ok=True)


def ensure_jittered_wads() -> bool:
    print("[+] Generating randomized jitter test WADs and scenario configs...")
    try:
        script = PROJECT_ROOT / "src" / "generate_random_jitter_wads.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(PROJECT_ROOT))
        print("[OK] Jittered WADs and configs successfully generated.")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to generate jitter WADs: {exc}")
        return False


def verify_vizdoom() -> None:
    print("[+] Verifying ViZDoom environment initialization...")
    try:
        import vizdoom as vzd
        game = vzd.DoomGame()
        cfg_path = PROJECT_ROOT / "scenarios" / "level01.cfg"
        game.load_config(str(cfg_path))
        game.set_window_visible(False)
        game.init()
        game.new_episode()
        state = game.get_state()
        game.close()
        print(f"[OK] ViZDoom successfully initialized FreeDoom2 MAP01 (Tic {state.tic}).")
    except Exception as exc:
        print(f"[WARN] ViZDoom initialization check had a warning/error: {exc}")


def main() -> None:
    print("=" * 65)
    print(" DOOM-SNN-Project: Asset & Environment Setup")
    print("=" * 65)
    wad_ok = ensure_freedoom2_wad()
    if not wad_ok:
        sys.exit(1)

    jitter_ok = ensure_jittered_wads()
    if not jitter_ok:
        sys.exit(1)

    verify_vizdoom()

    print("=" * 65)
    print(" All assets ready! You can now run visual evaluations:")
    print("   python src/evaluate_engine_state_policy.py \\")
    print("     --checkpoint exports/best-engine-state-spiking-snn-v2.pth \\")
    print("     --render --delay 0.02 --seed 0")
    print("=" * 65)


if __name__ == "__main__":
    main()
