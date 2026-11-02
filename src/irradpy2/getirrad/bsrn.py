"""
BSRN data downloader and parser.

This module provides a unified downloader and parser for BSRN shortwave radiation data.

Main capabilities
-----------------
- Download BSRN data by site and date range
- Parse radiation records from source files
- Generate standardized CSV output
- Sort final records by UTC timestamp and clean temporary files

Notes
-----
This module is built on top of shared utility functions defined in ``necessary_main.py``.
"""

import ftplib
import os
import gzip
import csv
import shutil
import stat
from datetime import datetime, timedelta, timezone
from .necessary_main import ensure_dir, init_str_date


class BSRNDownloader:
    HOST = "ftp.bsrn.awi.de"
    def __init__(self, site: str, username: str, password: str):
        self.site = site.lower()
        self.username = username
        self.password = password

    # Downloading utilities
    def _generate_month_ranges(self, start_date_str, end_date_str):
        """Generate monthly file identifiers within the given date range."""
        start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
        months = []
        cur = start_date.replace(day=1)

        while cur <= end_date:
            months.append(f"{cur.month:02d}{cur.year % 100:02d}")
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return months

    @staticmethod
    def _remove_readonly(func, path, exc_info):
        """Handle read-only files on Windows when removing temp directory."""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    def _download_file(self, ftp, remote_file, local_path):
        """Download one file from BSRN FTP."""
        try:
            with open(local_path, "wb") as lf:
                ftp.retrbinary(f"RETR {remote_file}", lf.write)
            return True
        except Exception:
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                return True
            return False

    def _safe_extract_gz(self, gz_path, output_dir):
        """Extract a compressed file into the target directory."""
        try:
            with gzip.open(gz_path, "rb") as gz:
                content = gz.read()

            base = os.path.basename(gz_path).replace(".gz", "")
            out_path = os.path.join(output_dir, base)

            with open(out_path, "wb") as f:
                f.write(content)

            return True

        except Exception:
            return False

    def download_and_extract(self, start, end, out_dir):
        """Download and extract monthly source files within the specified date range."""
        ensure_dir(out_dir)

        targets = [f"{self.site}{m}.dat.gz" for m in self._generate_month_ranges(start, end)]
        stats = {"target_files": len(targets), "available_files": 0, "missing_files": 0, "downloaded_files": 0, "extracted_files": 0, "ftp_error": None}

        try:
            with ftplib.FTP(self.HOST, timeout=60) as ftp:
                ftp.login(self.username, self.password)
                ftp.set_pasv(True)
                ftp.cwd(f"/{self.site}/")

                files = []
                ftp.retrlines("NLST", files.append)

                exist = [f for f in files if f in targets]
                missing = [f for f in targets if f not in exist]

                stats["available_files"] = len(exist)
                stats["missing_files"] = len(missing)

                for f in exist:
                    local_gz = os.path.join(out_dir, f)
                    ok = self._download_file(ftp, f, local_gz)

                    if not ok:
                        continue

                    stats["downloaded_files"] += 1

                    if self._safe_extract_gz(local_gz, out_dir):
                        stats["extracted_files"] += 1
                        os.remove(local_gz)

        except Exception as e:
            stats["ftp_error"] = str(e)

        return stats

    # Parsing utilities
    @staticmethod
    def _merge_lines(lines):
        """Merge pairs of data lines into a single record."""
        merged = []

        for i in range(0, len(lines) - 1, 2):
            l1, l2 = lines[i], lines[i + 1]
            m = (l1 + l2 + [None] * 21)[:21]
            merged.append(m)

        return merged

    @staticmethod
    def _parse_dat_file(fpath, start_dt, end_dt):
        """Parse a source file and extract radiation records."""
        lines = []
        has_c = False

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("*C0100"):
                    has_c = True
                    break

        prefer = "C" if has_c else "U"

        with open(fpath, "r", encoding="utf-8") as f:
            in_data = False

            for raw in f:
                line = raw.strip()

                if line.startswith("*U0100"):
                    in_data = (prefer == "U")
                    continue

                if line.startswith("*C0100"):
                    in_data = (prefer == "C")
                    continue

                if in_data and (line.startswith("*U") or line.startswith("*C")) and not (line.startswith("*U0100") or line.startswith("*C0100")):
                    in_data = False
                    continue

                if in_data and line:
                    lines.append(line.split())

        return BSRNDownloader._merge_lines(lines)

    # CSV utilities
    @staticmethod
    def _write_to_csv(merged, fpath, start_dt, end_dt, writer):
        """Write parsed radiation records into CSV."""
        row_count = 0

        for rec in merged:
            try:
                day = int(rec[0])
                minute = int(rec[1])

                part = os.path.basename(fpath)[3:7]
                month = int(part[:2])
                year = 2000 + int(part[2:])

                ts = datetime(year, month, day, tzinfo=timezone.utc) + timedelta(minutes=minute)

                if not (start_dt <= ts <= end_dt):
                    continue

                utc_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                row = [utc_str] + rec[2:21]

                writer.writerow(row)
                row_count += 1

            except Exception:
                continue

        return row_count

    def generate_csv(self, data_dir, start, end, out_path):
        """Generate the final CSV file from parsed BSRN data."""
        start_dt = datetime.strptime(start, "%Y%m%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end, "%Y%m%d").replace(hour=23, minute=59, tzinfo=timezone.utc)

        parsed_files = 0
        total_rows = 0

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "UTC_time",
                "ghi", "ghi_std", "ghi_min", "ghi_max",
                "dni", "dni_std", "dni_min", "dni_max",
                "dhi", "dhi_std", "dhi_min", "dhi_max",
                "longwave_down", "longwave_down_std", "longwave_down_min", "longwave_down_max",
                "temp_air", "rh", "pressure"
            ])

            if not os.path.exists(data_dir):
                return parsed_files, total_rows

            dat_files = sorted([d for d in os.listdir(data_dir) if d.endswith(".dat")])
            parsed_files = len(dat_files)

            for name in dat_files:
                p = os.path.join(data_dir, name)
                merged = self._parse_dat_file(p, start_dt, end_dt)
                total_rows += self._write_to_csv(merged, p, start_dt, end_dt, writer)

        return parsed_files, total_rows

    # Post-processing utilities
    @staticmethod
    def _sort_csv(csv_path):
        """Sort the generated CSV file by UTC timestamp."""
        if not os.path.exists(csv_path):
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = sorted(reader, key=lambda x: datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S"))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    @staticmethod
    def _clean_temp(temp_dir):
        """Remove the temporary working directory."""
        temp_dir = os.path.abspath(temp_dir)

        if not os.path.exists(temp_dir):
            return

        try:
            shutil.rmtree(temp_dir, onerror=BSRNDownloader._remove_readonly)
        except Exception:
            pass

    def run(self, start, end, save_path):
        """Run the complete BSRN download and conversion workflow."""
        save_path = os.path.abspath(save_path)
        save_dir = os.path.dirname(save_path)

        ensure_dir(save_dir)

        temp_dir = os.path.join(save_dir, "_bsrn_temp")

        try:
            self._clean_temp(temp_dir)

            print(f"[BSRN] site={self.site}, start={start}, end={end}")
            print(f"[BSRN] output={save_path}")

            download_stats = self.download_and_extract(start, end, temp_dir)

            if download_stats["ftp_error"]:
                print(f"[BSRN] ftp_error={download_stats['ftp_error']}")

            print(f"[BSRN] files target={download_stats['target_files']}, available={download_stats['available_files']}, downloaded={download_stats['downloaded_files']}, extracted={download_stats['extracted_files']}")

            parsed_files, total_rows = self.generate_csv(temp_dir, start, end, save_path)
            self._sort_csv(save_path)

            print(f"[BSRN] parsed_files={parsed_files}, rows={total_rows}")

            if total_rows == 0:
                print("[BSRN] warning=no valid records found")

            print(f"[BSRN] saved={save_path}")

        finally:
            self._clean_temp(temp_dir)


def download_bsrn(site, start, end, username, password, save_path):
    """
    Download BSRN shortwave radiation data and save the result as a CSV file.

    Parameters
    ----------
    site : str
        BSRN station identifier, for example ``"cab"`` or ``"qiq"``.
    start : str
        Start date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
    end : str
        End date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
    username : str
        BSRN FTP username.
    password : str
        BSRN FTP password.
    save_path : str
        Absolute output CSV file path.

    References
    ----------
    BSRN homepage:
    https://bsrn.awi.de/

    BSRN FTP data retrieval:
    https://bsrn.awi.de/data/data-retrieval-via-ftp/
    """
    if not save_path or not os.path.isabs(save_path):
        raise ValueError("save_path must be a non-empty absolute path.")

    start = init_str_date(start).replace("-", "")
    end = init_str_date(end).replace("-", "")

    d = BSRNDownloader(site, username, password)
    d.run(start, end, save_path)
