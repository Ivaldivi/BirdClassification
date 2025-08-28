import requests
import json
import os
from pathlib import Path
import re 

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


def download_photos(file_path_to_search):
    for root, dirs, files in os.walk(file_path_to_search):
        for file in files: 
            # print(file)
            if file =="Cranes.txt": 
                with open("Intermediate_txts/"+file) as f: 
                    # print(f.read())
                    openfile = f.read()
                    openfile = openfile.strip("\n")
    # requests.get(image_url, stream=True)

def main():
    # query_inaturalist()
    download_photos('Intermediate_txts')

if __name__ == '__main__':
    main()