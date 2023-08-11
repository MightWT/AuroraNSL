import requests
from bs4 import BeautifulSoup
import urllib
from urllib.parse import urlparse
from pathlib import Path

def filter_urls_by_extension(html_content, extension):
    soup = BeautifulSoup(html_content, 'html.parser')
    filtered_urls = []

    for link in soup.find_all('a'):
        href = link.get('href')

        if href:
            parsed_url = urlparse(href)
            file_extension = parsed_url.path.split('.')[-1]

            if file_extension.lower() == extension.lower():
                filtered_urls.append(href)

    return filtered_urls

base_url = "https://ssusi.jhuapl.edu/data_retriver?spc=f17&type=edr-aur&year=2014&Doy="

extension = 'nc'     # Desired file extension
start_doy = 1       # Starting value of Doy
end_doy = 365       # Ending value of Doy

# Create the "2014" folder if it doesn't exist
year_folder = Path.cwd() / "2014"
year_folder.mkdir(exist_ok=True)

for doy in range(start_doy, end_doy + 1):
    padded_doy = str(doy).zfill(3)   # Pad Doy to three digits


    url = base_url + padded_doy

    # Send a GET request to the URL and retrieve the HTML content
    response = requests.get(url)
    html_content = response.content

    # Filter URLs with the specified extension from the HTML content
    filtered_urls = filter_urls_by_extension(html_content, extension)
    
    # Print the filtered URLs
    print(f"URLs for Doy {padded_doy}:")
    for filtered_url in filtered_urls:
        file_name = Path(filtered_url).name

        with urllib.request.urlopen("https://ssusi.jhuapl.edu/"+filtered_url) as response:
            year_folder.joinpath(file_name).write_bytes(response.read())
        print(filtered_url)

    print("--------------------")
