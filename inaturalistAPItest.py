import json
import os, requests
from pathlib import Path
import re 
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# API endpoint for observations
url = "https://api.inaturalist.org/v1/observations"

# Parameters: you can filter by taxon_id, location, date, etc.
def query_inaturalist():
    with open('Taxon_ids/taxon_ids.txt', 'r') as file:
            content = file.read()

    lines = content.splitlines()
    bird_names = []
    taxon_ids = []
    for line in lines: 
        try: 
            bird_name, taxon_id = line.split(": ")
            bird_names.append(bird_name)
            taxon_ids.append(taxon_id)
        except ValueError:
            print(f"Skipping malformed line: {line}")

    print(bird_names)
    print(taxon_ids)
    iters = len(taxon_ids)
    for i in range(1,iters): 
        params = {
            "taxon_id": taxon_ids[i], 
            "per_page": 200  
        }

    # Send the GET request to the iNaturalist API
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Parse JSON response
            observations = data['results']
            
            ## Extract desired fields from each observation
            filtered_data = []
            for observation in observations:
                taxon = observation.get("taxon", {})
                if "photos" in observation and len(observation["photos"]) > 0: 
                        observation_photo_url = observation["photos"][0]["url"] 
                if "quality_grade" in observation and observation["quality_grade"]=="research": 
                    # Select specific fields: species name, date, and coordinates (lat, long)
                    filtered_observation = {
                        "species": observation['taxon']['name'],  # Species name
                        "common name": taxon.get("preferred_common_name", None),
                        "observation photo url": observation_photo_url,
                        "date": observation['observed_on'],  # Date of observation
                        "latitude": observation['location'].split(',')[0] if observation['location'] else None,  # Latitude
                        "longitude": observation['location'].split(',')[1] if observation['location'] else None,  # Longitude
                        "quality_grade": observation["quality_grade"]
                    }
                    filtered_data.append(filtered_observation)
            print()
            with open('Intermediate_txts/'+bird_names[i]+'.txt', 'w') as file:
                file.write(json.dumps(filtered_data, indent=2))
        else:
            print("Error:", response.status_code)

def process_file(file): 
    """worker function for one .txt file!"""
    animal_name = file.split(".")[0]
    os.makedirs("data/"+animal_name, exist_ok=True)
    with open("Intermediate_txts/"+file) as f:
        data_list= json.load(f)
    print(f"Processing {animal_name} with {len(data_list)} images")
    download_images(data_list, animal_name, max_workers=20)

def handle_urls(file_path_to_search, num_processes=4):
    """Run multiprocessing across all txt files."""
    files = [f for f in os.listdir(file_path_to_search) if f.endswith(".txt")]

    with Pool(processes=num_processes) as pool:
        pool.map(process_file, files)

def download_images(data_list, animal_name, max_workers=20): 
    """Use parallelization to download images"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        futures = {executor.submit(download_single_image, img, animal_name): img for img in data_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result)
    return results
            

def download_single_image(image, animal_name):
    """Download a single image-- WORKER FUNCTION!"""
    url = image['observation photo url'].replace("square", "original")
    split_url = url.split('/')
    file_path = "data/"+animal_name+"/"+animal_name+"_"+split_url[-2]+".jpg"

    if os.path.exists(file_path): 
        return f"Skipped '{file_path}' because it already exists."
    
    try: 
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code==200: 
            with open(file_path, 'wb') as fi: 
                fi.write(response.content)
                return f"Saved image to {file_path}"
        else: 
            return f"Failed to save '{file_path}' becasue the status code was: '{response.status_code}'"
    except Exception as e: 
        return f"Error downloading {file_path}: {e}"

def main():
    # query_inaturalist()
    start = time.time()
    handle_urls('Intermediate_txts', num_processes=4)
    end = time.time()
    print(f"Downloading images took {end-start:.2f} seconds or {end-start/60} minutes")

if __name__ == '__main__':
    main()