# File: scripts/web_scraper.py
import requests
import os

def main():
    # Target URL
    url = "https://www.cnbc.com/world/?region=world"
    
    # (User-Agent)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "../data/raw_data")
    output_file = os.path.join(output_dir, "web_data.html")
    
    print(f"Fetching data from: {url}")
    
    try:
        # 1. Use Requests with HEADERS to collect data
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        # 2. Save collected data
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Data saved to: {output_file}")
        
        # 3. Print the first 10 lines (Checkpoint)
        print("\n--- First 10 lines of the file ---")
        with open(output_file, "r", encoding="utf-8") as f:
            for i in range(10):
                line = f.readline()
                if not line: break
                print(line.strip())
        print("----------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
