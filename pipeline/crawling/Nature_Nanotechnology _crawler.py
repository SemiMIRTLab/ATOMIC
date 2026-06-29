import csv
import os
import logging
import time
import random
import requests
from bs4 import BeautifulSoup
import re
import argparse
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

class NatureNanotechnologyCrawler:
    """
    Nature Paper Crawler Tool
    Note: This tool is for academic research purposes only. 
    Please comply with Nature's terms of use and copyright regulations.
    """
    
    def __init__(self, download_path="./downloads_nnano", start_year=2010, max_articles=1000):
        self.csv_file = None
        self.csv_writer = None
        self.download_path = download_path
        self.start_year = start_year
        self.max_articles = max_articles
        self.driver = None
        self.setup_logging()
        self.setup_download_directory()
    
    # set up logging for rebust record
    def setup_logging(self):
        """Setup logging system"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/nature_nanotechnology_crawler.log', encoding='utf-8'),
                logging.StreamHandler()  # Also output to console
            ]
        )
        self.logger = logging.getLogger(__name__)
    # set up download dir
    def setup_download_directory(self):
        """Create download directory"""
        os.makedirs(self.download_path, exist_ok=True)
        self.logger.info(f"Download directory set: {self.download_path}")
    #Chrome Setting
    def get_chrome_options(self):
        """Configure Chrome browser options"""
        chrome_options = Options()
        prefs = {
            "download.default_directory": os.path.abspath(self.download_path),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        # Optional: headless mode
        # chrome_options.add_argument("--headless")
        return chrome_options
    
    def initialize_driver(self):
        """Initialize WebDriver"""
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.get_chrome_options())
            self.driver.maximize_window()
            self.logger.info("Browser started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Browser initialization failed: {e}")
            return False
    
    def navigate_to_nature(self):
        """Navigate to Nature search page"""
        search_url = (
            f"https://www.nature.com/search?"
            f"q=transmission%2Belectron%2Bmicroscopy&"
            f"journal=nnano&"
            f"article_type=research&"
            f"subject=materials-science%2C+nanoscience-and-technology&"
            f"order=relevance&"
            f"date_range=2010-2025"
        )
        
        try:
            self.driver.get(search_url)
            self.logger.info(f"Navigated to 2010~2025 search page")
            
            # Handle cookie consent popup
            self.handle_cookie_consent()
            return True
            
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            return False
    
    def handle_cookie_consent(self):
        """Handle cookie consent popup"""
        try:
            accept_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'accept')]"))
            )
            accept_button.click()
            self.logger.info("Cookie policy accepted")
        except:
            self.logger.info("No cookie consent popup found or already handled")

    # ================================================================================
    # get the number of result for that year
    def get_total_results(self):
        """Get total number of search results"""
        try:
            results_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-test='results-data'] > span:last-child"))
            )
            results_text = results_element.text
            results_number = int(results_text.split()[0])
            self.logger.info(f"Total search results: {results_number}")
            return results_number
        except Exception as e:
            self.logger.error(f"Cannot get total results: {e}")
            return 0
    
    # to wait for 2~4s
    def respectful_delay(self, min_delay=2, max_delay=4):
        """Add respectful delay to avoid overwhelming the server"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    
    def get_html(self):
        article_body = self.driver.find_element(
            # By.XPATH, '//*[@id="content"]/main/article/div[3]'
            By.XPATH, '//*[@id="content"]/main/article'
            
        )
        
        html = article_body.get_attribute("innerHTML")

        return html

    def is_open_access(self):
        try:
            self.driver.find_element(
                By.XPATH,
                "//a[@data-test='open-access']"
            )
            return True
        except:
            return False
    def parse_html(self,html):
        soup = BeautifulSoup(html, "lxml")
        url = self.driver.current_url
        open_access = self.is_open_access()
        article_id = url.rstrip("/").split("/")[-1]

        figures = soup.find_all("div", attrs={"data-test": "figure"})
        for fig in figures:
            fig_id = fig.get("id")  # figure-1, figure-2 ...

            # ===== title =====
            title_tag = fig.find("b", attrs={"data-test": "figure-caption-text"})
            title = title_tag.get_text(strip=True) if title_tag else ""

            # ===== caption (clean citation) =====
            caption_tag = fig.find("div", attrs={"data-test": "bottom-caption"})
            caption = ""
            if caption_tag:
                for sup in caption_tag.find_all("sup"):
                    if re.fullmatch(r"[0-9,\-–]+", sup.get_text(strip=True)):
                        sup.replace_with(" ")
                caption = caption_tag.get_text(" ", strip=True)

            # ===== image url =====
            img_tag = fig.select_one("a[data-test='img-link'] img")
            if not img_tag:
                continue

            img_url = img_tag.get("src", "")
            if img_url.startswith("//"):
                img_url = "https:" + img_url

            # ===== download image =====
            img_path = os.path.join("downloads_nnano", f"{article_id}_{fig_id}.jpg")

            try:
                r = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                r.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed to download {fig_id}: {e}")
                continue
            self.csv_writer.writerow({
                "ARTICLE_ID": article_id,
                "FIG_ID": fig_id,
                "TITLE": title,
                "CAPTION": caption,
                "IMAGE_PATH": img_path,
                "ARTICLE_URL": url,
                "OPEN_ACCESS": open_access,
            })

            
            self.logger.info(f"ARTICLE_ID {article_id}")
            self.logger.info(f"FIG_ID {fig_id}")
            # self.logger.info(f"TITLE: {title}")
            # self.logger.info(f"CAPTION: {caption}")
            # self.logger.info(f"IMAGE SAVED: {img_path}")
            # self.logger.info(f"ARTICLE_URL: {url}")
            # self.logger.info(f"OPEN_ACCESS: {open_access}")
            self.logger.info("=" * 80)
        self.csv_file.flush() 

    def process_article(self, article_index, year, global_count):
        """Process a single article"""
        try:
            # Locate article link
            article_xpath = f"//*[@id='search-article-list']/div/ul/li[{article_index}]/div/div/article/div[1]/div[2]/h3/a"
            article = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, article_xpath))
            )
            
            # Open article in new tab
            ActionChains(self.driver).key_down(Keys.CONTROL).click(article).key_up(Keys.CONTROL).perform()
            
            # Switch to new tab
            self.driver.switch_to.window(self.driver.window_handles[-1])
            
            html = self.get_html()
            self.parse_html(html)
            pass
            time.sleep(5)
            # Find and download PDF
            # download_success = self.find_and_download_pdf(year, global_count)
            
            # Close current tab and return to main page
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            
            # Respectful delay
            self.respectful_delay()
            
            # return download_success
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing article {article_index}: {e}")
            # Ensure return to main window
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
            return False
    
    def go_to_next_page(self):
        """Go to next page"""
        try:
            next_button = WebDriverWait(self.driver, 5).until(
                # EC.element_to_be_clickable((By.XPATH, "//li[@data-page='next']"))
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//li[@data-page='next']//a[contains(@class,'c-pagination__link')]"
                ))
            )
            
        except:
            self.logger.info("No more pages available")
            return False
        
        next_button.click()
        self.logger.info("Switched to next page")
        self.respectful_delay()
        return True
    
    def crawl_year(self, year):
        """Crawl articles for specified year"""
        if not self.navigate_to_nature():
            return False
        
        total_results = self.get_total_results()
        if total_results == 0:
            return False
        
        downloaded_count = 0
        processed_count = 0
        
        self.logger.info(f"Starting to crawl 2010~2025 data, total {total_results} articles")
        
        while downloaded_count < self.max_articles and processed_count < total_results:
            # Process articles on current page (usually 50 articles per page)
            for i in range(1, 51):  # 1-50
                if downloaded_count >= self.max_articles:
                    break
                
                processed_count += 1
                self.logger.info(f"Processing article {processed_count} of year {year}")
                
                success = self.process_article(i, year, downloaded_count + 1)
                if success:
                    downloaded_count += 1
                
                if processed_count >= total_results:
                    break
            
            # Try to go to next page
            if not self.go_to_next_page():
                break
        
        self.logger.info(f"Year 2010~2025 crawling completed, successfully downloaded {downloaded_count} articles")
        return True
    
    def run(self):
        """Main execution function"""
        if not self.initialize_driver():
            return False
        
        csv_path = "dataset_nnano.csv"
        file_exists = os.path.exists(csv_path)

        self.csv_file = open(csv_path, "a", newline="", encoding="utf-8-sig")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "ARTICLE_ID",
                "FIG_ID",
                "TITLE",
                "CAPTION",
                "IMAGE_PATH",
                "ARTICLE_URL",
                "OPEN_ACCESS",
            ],
        )

        if not file_exists:
            self.csv_writer.writeheader()

        try:
            # pass
            year = self.start_year
            # while True:
            self.logger.info(f"Starting to process year {year}")
            success = self.crawl_year(year)
            
            if not success:
                self.logger.info(f"Year 2010~2025 processing completed or error occurred")
                
                # year += 1
                
                # # Optional: set end year
                # if year > 2025:  
                #     break
                
        except KeyboardInterrupt:
            self.logger.info("Program interrupted by user")
        except Exception as e:
            self.logger.error(f"Program execution error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.csv_file:
            self.csv_file.close()
        if self.driver:
            self.driver.quit()
            self.logger.info("Browser closed")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Nature Paper Crawler Tool')
    parser.add_argument('--year', type=int, default=2010, help='Starting year (default: 2010)')
    parser.add_argument('--max-articles', type=int, default=1000, help='Maximum articles per year (default: 1000)')
    parser.add_argument('--output-dir', default='./downloads_nnano', help='Download directory (default: ./downloads_nnano)')
    return parser.parse_args()


def main():
    """Main program entry point"""
    print("Nature Paper Crawler")
    print("=" * 50)
    print("This tool is for academic research purposes only.")
    print("Please ensure you comply with Nature's terms of use.")
    print("=" * 50)
    
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Starting year: {args.year}")
    print(f"  Max articles per year: {args.max_articles}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 50)
    
    crawler = NatureNanotechnologyCrawler(
        download_path=args.output_dir,
        start_year=args.year,
        max_articles=args.max_articles
    )
    
    crawler.run()
    print("Program execution completed.")


if __name__ == "__main__":
    main()