import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
DEFAULT_TAXON_FILE = PROJECT_ROOT / "Taxon_ids" / "taxon_families.txt"
INTERMEDIATE_DIR = PROJECT_ROOT / "Intermediate_txts"
DATA_DIR = PROJECT_ROOT / "data"
FAMILIES_DIR = DATA_DIR / "families"
SPECIES_DIR = DATA_DIR / "species"


def sanitize_label(label: str) -> str:
    """Make a label safe for use as a folder name."""
    cleaned = re.sub(r"[^\w\s\-',()]", "", label).strip()
    return cleaned or "Unknown"


def species_label(observation: dict) -> str | None:
    taxon = observation.get("taxon", {})
    if taxon.get("rank") != "species":
        return None
    label = taxon.get("preferred_common_name") or taxon.get("name")
    return sanitize_label(label) if label else None


def load_taxa(taxon_file: Path) -> list[tuple[str, str]]:
    taxa = []
    with taxon_file.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, taxon_id = line.split(": ", 1)
                taxa.append((name.strip(), taxon_id.strip()))
            except ValueError:
                print(f"Skipping malformed line: {line}")
    return taxa


def query_inaturalist(
    taxon_file: Path = DEFAULT_TAXON_FILE,
    per_page: int = 200,
    pages: int = 3,
    skip_existing: bool = True,
    timeout: int = 60,
    retries: int = 3,
) -> None:
    taxa = load_taxa(taxon_file)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    for group_name, taxon_id in taxa:
        output_name = sanitize_label(group_name)
        output_path = INTERMEDIATE_DIR / f"{output_name}.txt"
        if skip_existing and output_path.exists() and output_path.stat().st_size > 10:
            print(f"Skipping {group_name}: already fetched -> {output_path.name}")
            continue

        filtered_data = []
        for page in range(1, pages + 1):
            params = {
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "photos": "true",
                "per_page": per_page,
                "page": page,
            }

            response = None
            for attempt in range(1, retries + 1):
                try:
                    response = requests.get(OBSERVATIONS_URL, params=params, timeout=timeout)
                    break
                except requests.RequestException as exc:
                    print(f"Retry {attempt}/{retries} for {group_name} page {page}: {exc}")
                    time.sleep(2 * attempt)

            if response is None:
                print(f"Failed to fetch {group_name} (page {page}) after {retries} retries")
                break
            if response.status_code != 200:
                print(f"Error fetching {group_name} (page {page}): {response.status_code}")
                break

            observations = response.json().get("results", [])
            if not observations:
                break

            for observation in observations:
                if not observation.get("photos"):
                    continue

                family_label = sanitize_label(group_name)
                species = species_label(observation)
                taxon = observation.get("taxon", {})
                location = observation.get("location") or ""
                latitude, longitude = None, None
                if "," in location:
                    latitude, longitude = location.split(",", 1)

                filtered_data.append(
                    {
                        "family": family_label,
                        "species": species,
                        "scientific_name": taxon.get("name"),
                        "common name": taxon.get("preferred_common_name"),
                        "observation photo url": observation["photos"][0]["url"],
                        "date": observation.get("observed_on"),
                        "latitude": latitude,
                        "longitude": longitude,
                        "quality_grade": observation.get("quality_grade"),
                    }
                )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(filtered_data, file, indent=2)

        species_labels = {entry["species"] for entry in filtered_data if entry["species"]}
        print(
            f"{group_name}: saved {len(filtered_data)} observations "
            f"({len(species_labels)} species) -> {output_path.name}"
        )


def write_image(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return f"Skipped '{path}' because it already exists."
    path.write_bytes(content)
    return f"Saved image to {path}"


def download_single_image(image: dict, layout: str, retries: int = 3) -> list[str]:
    family = sanitize_label(image["family"])
    species = image.get("species")
    url = image["observation photo url"].replace("square", "original")
    photo_id = url.split("/")[-2]

    content = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=45)
            if response.status_code == 200:
                content = response.content
                break
            if attempt == retries:
                return [f"Failed to download {url} (status {response.status_code})"]
        except requests.RequestException as exc:
            if attempt == retries:
                return [f"Error downloading {url}: {exc}"]
            time.sleep(attempt)

    if content is None:
        return [f"Error downloading {url}: no content"]

    results = []
    if layout in {"family", "hierarchical"}:
        family_path = FAMILIES_DIR / family / f"{family}_{photo_id}.jpg"
        results.append(write_image(family_path, content))

    if layout in {"species", "hierarchical"} and species:
        species_path = SPECIES_DIR / family / species / f"{species}_{photo_id}.jpg"
        results.append(write_image(species_path, content))

    return results


def process_file(args: tuple[str, str]) -> None:
    file_name, layout = args
    with (INTERMEDIATE_DIR / file_name).open(encoding="utf-8") as file:
        data_list = json.load(file)

    families = {entry["family"] for entry in data_list}
    species = {entry["species"] for entry in data_list if entry["species"]}
    print(
        f"Processing {file_name}: {len(data_list)} images, "
        f"{len(families)} families, {len(species)} species"
    )
    download_images(data_list, layout=layout, max_workers=20)


def download_images(data_list: list[dict], layout: str = "hierarchical", max_workers: int = 20) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_image, image, layout): image for image in data_list
        }
        for future in as_completed(futures):
            for result in future.result():
                print(result)


def handle_urls(
    layout: str = "hierarchical",
    num_processes: int = 4,
    taxon_file: Path | None = DEFAULT_TAXON_FILE,
) -> None:
    if taxon_file and taxon_file.exists():
        allowed = {sanitize_label(name) + ".txt" for name, _ in load_taxa(taxon_file)}
        files = [
            file_name for file_name in allowed
            if (INTERMEDIATE_DIR / file_name).exists()
        ]
    else:
        files = [file.name for file in INTERMEDIATE_DIR.glob("*.txt")]

    print(f"Downloading {len(files)} taxon files", flush=True)
    with Pool(processes=num_processes) as pool:
        pool.map(process_file, [(file_name, layout) for file_name in files])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download iNaturalist bird images for training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Fetch observation metadata from iNaturalist.")
    query_parser.add_argument("--taxon-file", type=Path, default=DEFAULT_TAXON_FILE)
    query_parser.add_argument("--per-page", type=int, default=200)
    query_parser.add_argument("--pages", type=int, default=3)

    download_parser = subparsers.add_parser("download", help="Download images from Intermediate_txts/.")
    download_parser.add_argument(
        "--layout",
        choices=["family", "species", "hierarchical"],
        default="hierarchical",
        help="family=only family folders; species=only species folders; hierarchical=both",
    )
    download_parser.add_argument("--processes", type=int, default=4)
    download_parser.add_argument("--taxon-file", type=Path, default=DEFAULT_TAXON_FILE)

    args = parser.parse_args()

    if args.command == "query":
        query_inaturalist(taxon_file=args.taxon_file, per_page=args.per_page, pages=args.pages)
    elif args.command == "download":
        start = time.time()
        handle_urls(layout=args.layout, num_processes=args.processes, taxon_file=args.taxon_file)
        elapsed = time.time() - start
        print(f"Downloading images took {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
