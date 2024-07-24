import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import re
import csv
import threading
import time

def extract_and_save_table_text(url, csv_file_path):
    # Parse the URL to extract NotificationUser and Id
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    query_params = parse_qs(parsed_url.query)

    # Extract the required parts from the URL
    notification_user = path_parts[-1].split('.')[0]
    notification_id = query_params['Id'][0]

    # Create the directory structure
    directory_path = os.path.join(notification_user)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # File path to save the extracted text
    file_path = os.path.join(directory_path, f"{notification_id}.txt")


    title = None
    date = None
    table_text = ""
    supersede_links = {}

    # Retry mechanism
    for attempt in range(5):  # Try up to 5 times
        try:
            # Send a GET request to the URL
            response = requests.get(url, timeout=10)  # 10 seconds timeout

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the content with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find the table with the specific class and width
                table = soup.find('table', {'class': 'tablebg', 'width': '100%'})

                # Regex pattern to match date in format 'Month DD, YYYY'
                date_pattern = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b')

                if table:
                    # Extract the title from the second occurrence of the header
                    title_tags = soup.find_all('td', {'class': 'tableheader'})
                    if len(title_tags) > 1:
                        title = title_tags[1].get_text(strip=True)

                    # Extract all the text inside the table
                    table_text = table.get_text(separator='\n', strip=True)

                    # Find all text in the document to search for the date
                    all_text = soup.get_text(separator='\n', strip=True)
                    date_match = date_pattern.search(all_text)
                    if date_match:
                        date = date_match.group()

                    # Find all anchor tags within the table for supersede links
                    for paragraph in table.find_all('p'):
                        if 'supersede' in paragraph.get_text() or 'supersedes' in paragraph.get_text():
                            for a in paragraph.find_all('a', href=True):
                                supersede_links[paragraph.get_text()] = a['href']

                    # Write the table text to the file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(table_text)

                    # Append the extracted data to the CSV file
                    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([url, title, date, '; '.join(supersede_links.values()), notification_user, notification_id])

                    print(f"Table text saved to {file_path} and metadata saved to {csv_file_path}")
                    break

                else:
                    print("Table with specified class and width not found.")
                    break

            else:
                print(f"Failed to retrieve the page. Status code: {response.status_code}")
                time.sleep(5)  # Wait for 5 seconds before retrying

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying

def scrape_range(start_id, end_id, csv_file_path):
    # Base URL format
    base_url = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={}&Mode=0"

    # Read existing notification IDs from the CSV file
    existing_ids = set()
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                existing_ids.add(row[5])  # Notification ID is in the 6th column

    # Iterate over the range of IDs
    for notification_id in range(start_id, end_id - 1, -1):
        if str(notification_id) not in existing_ids:
            url = base_url.format(notification_id)
            extract_and_save_table_text(url, csv_file_path)

def main():
    csv_file_path = "notifications_metadata.csv"

    # Write CSV headers if the file does not exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["current_link", "title", "date", "supersedes_links", "notification_user", "notification_id"])

    # Define ranges for threading
    ranges = [(11000, 10500), (10499, 10000), (9999, 9500), (9499, 9000), (8999, 8500)]

    threads = []
    for start_id, end_id in ranges:
        thread = threading.Thread(target=scrape_range, args=(start_id, end_id, csv_file_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
