import requests
from base64 import BeautifulSoup
import os

def save_webpage_to_txt(url, filename="output.txt", folder="data"):
    """
    Fetches a webpage, extracts readable text (headings + paragraphs),
    and saves it into a .txt file inside the 'data' folder.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()  # check for errors
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract meaningful text (headings and paragraphs)
        text_parts = []
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            text = tag.get_text(strip=True)
            if len(text) > 40:  # ignore short junk
                text_parts.append(text)

        clean_text = "\n\n".join(text_parts)

        # Save to file
        file_path = os.path.join(folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"✅ Saved text from {url} → {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")

# ------------------------------
# Example usage:
if __name__ == "__main__":
    
    urls = ["https://www.gujaratvidyapith.org/department/information-communication-and-technology/department-of-computer-science"    ]

    for i, u in enumerate(urls, 1):
        save_webpage_to_txt(u, filename=f"page_{i}.txt")
