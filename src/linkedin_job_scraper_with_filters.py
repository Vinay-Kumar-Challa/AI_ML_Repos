import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import schedule
import logging
import urllib.parse

# Set up logging
logging.basicConfig(filename='job_scraper.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Define your filter conditions here
FILTERS = {
    'keywords': 'Data Engineer',  # Replace with your desired job title/keywords
    'location': 'United States',   # Replace with your desired location
    'time_posted': 'r7200',       # Jobs posted in last 24 hours (r86400 = 24 hours)
    # Add other filters as needed, e.g., 'f_E=2' for entry-level jobs
}

def build_linkedin_url(filters):
    """Build LinkedIn job search URL based on filter conditions."""
    base_url = 'https://www.linkedin.com/jobs/search?'
    params = {
        'keywords': filters.get('keywords', ''),
        'location': filters.get('location', ''),
        'f_TPR': filters.get('time_posted', '')
        # Add more params like 'f_E=2' for experience level if needed
    }
    # URL-encode parameters
    encoded_params = urllib.parse.urlencode({k: v for k, v in params.items() if v})
    return f"{base_url}{encoded_params}"

def scrape_linkedin_jobs():
    try:
        # Build URL with filters
        url = build_linkedin_url(FILTERS)
        logging.info(f"Scraping URL: {url}")

        # Make request to LinkedIn
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        job_cards = soup.find_all('div', class_='base-card')

        jobs = []
        for card in job_cards:
            try:
                title = card.find('h3', class_='base-search-card__title')
                title = title.text.strip() if title else 'N/A'

                company = card.find('h4', class_='base-search-card__subtitle')
                company = company.text.strip() if company else 'N/A'

                location = card.find('span', class_='job-search-card__location')
                location = location.text.strip() if location else 'N/A'
                
                job_type = card.find('span', class_='tvm__text tvm__text--low-emphasis')
                job_type = job_type.text.strip() if job_type else 'N/A'

                link = card.find('a', class_='base-card__full-link')
                link = link['href'] if link else 'N/A'

                jobs.append({
                    'title': title,
                    'company': company,
                    'location': location,
                    'Job Type':job_type,
                    'link': link,
                    'date_scraped': datetime.now().strftime('%Y-%m-%d')
                })
            except Exception as e:
                logging.error(f"Error parsing job card: {e}")
                continue

        # Save to CSV
        if jobs:
            df = pd.DataFrame(jobs)
            output_file = f'C:\\Users\\chvin\\Downloads\\linkedin_jobs_{datetime.now().strftime("%Y%m%d%H")}.csv'
            df.to_csv(output_file, index=False, encoding='utf-8')
            logging.info(f"Saved {len(jobs)} jobs to {output_file}")
        else:
            logging.warning("No jobs found")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

def job():
    logging.info("Starting daily job scrape")
    scrape_linkedin_jobs()
    logging.info("Daily job scrape completed")

# Schedule the job to run daily at 8:00 AM事例

# Main loop to keep the script running
if __name__ == "__main__":
    logging.info("Starting scheduler")
    job()
    schedule.every().day.at("08:00").do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute