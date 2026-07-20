import os
import urllib.request
import ssl
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create unverified SSL context to bypass issues with misconfigured legal websites
ssl_context = ssl._create_unverified_context()

# Primary targets
SOURCES = [
    {
        "filename": "labour_act_2006_en.pdf",
        "title": "The Bangladesh Labour Act, 2006 (XLII of 2006)",
        "law_version_date": "2006-10-11",
        "amendments_included_through": "2006",
        "language": "en",
        "source_url": "https://www.doulah.net/bdlaws/Bangladesh%20Labor%20Act,%202006.pdf",
        "source_type": "primary"
    },
    {
        "filename": "labour_act_2006_en_mcci.pdf",
        "title": "The Bangladesh Labour Act, 2006 (English Upto 2018)",
        "law_version_date": "2018-10-24",
        "amendments_included_through": "2018",
        "language": "en",
        "source_url": "https://mccibd.org/wp-content/uploads/2021/09/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf",
        "source_type": "primary"
    },
    {
        "filename": "labour_act_2006_natlex.pdf",
        "title": "Bangladesh Labour Act, 2006 (ILO NATLEX version)",
        "law_version_date": "2006-10-11",
        "amendments_included_through": "2006",
        "language": "en",
        "source_url": "https://natlex.ilo.org/dyn/natlex2/natlex2/files/download/76402/BGD76402.pdf",
        "source_type": "primary"
    },
    {
        "filename": "labour_act_2006_bn.html",
        "title": "Bangladesh Labour Act, 2006 (Bangla, Minlaw bdlaws)",
        "law_version_date": "2006-10-11",
        "amendments_included_through": "2018",
        "language": "bn",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-952.html",
        "source_type": "primary"
    }
]

# Additional documents (Labour Rules 2015, ILO handbooks, etc.)
ADDITIONAL_SOURCES = [
    {
        "filename": "labour_rules_2015_en.pdf",
        "title": "Bangladesh Labour Rules, 2015 (English)",
        "law_version_date": "2015-09-15",
        "amendments_included_through": "2015",
        "language": "en",
        "source_url": "https://www.dife.gov.bd/site/page/f97b5e40-bb74-4b53-a55d-3d44cd7e2b7e/%E0%A6%86%E0%A6%8Plain-%E0%A6%93-%E0%A6%AC%E0%A6%BF%E0%A6%A7%E0%A6%BF%E0%A6%AE%E0%A6%BE%E0%A6%B2%E0%A6%BE", # DIFE law section
        "source_type": "secondary"
    }
]

def download_file(url: str, dest_path: str):
    logger.info(f"Downloading {url} to {dest_path}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            with open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
        logger.info(f"Successfully downloaded {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def main():
    raw_dir = "d:/000 NSU 4th/cse 596/shromic QA/labour-law-assistant/data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    downloaded_sources = []
    
    for src in SOURCES:
        dest = os.path.join(raw_dir, src["filename"])
        success = download_file(src["source_url"], dest)
        if success:
            downloaded_sources.append(src)
            
    # Save the sources.json
    sources_json_path = os.path.join(raw_dir, "sources.json")
    with open(sources_json_path, 'w', encoding='utf-8') as f:
        json.dump(downloaded_sources, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sources list metadata to {sources_json_path}")

if __name__ == "__main__":
    main()
