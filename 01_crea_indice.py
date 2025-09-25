# 01_crea_indice.py
# This script is responsible for creating or updating the index schema in Azure AI Search.
# It defines the structure of the data that will be stored, including the fields for content,
# ID, and the crucial vector field for semantic search.

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# --- Azure AI Search Configuration ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
API_VERSION = "2023-11-01" # Use a stable API version for consistency.

# Validate that all required environment variables are set.
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME]):
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

# Construct the full URL for the PUT request to the Azure AI Search API.
url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version={API_VERSION}"

# Set the required headers for the API request.
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_API_KEY
}

# --- Index Schema Definition ---
# This JSON object defines the structure of the search index.
index_body = {
    "name": AZURE_SEARCH_INDEX_NAME,
    "fields": [
        # The unique identifier for each document (chunk).
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        # The actual text content of the chunk.
        {"name": "content", "type": "Edm.String", "searchable": True},
        # The vector representation (embedding) of the content.
        {
            "name": "content_vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            # This MUST match the output dimensions of the embedding model.
            # 'paraphrase-multilingual-MiniLM-L12-v2' produces 384-dimension vectors.
            "dimensions": 384,
            "vectorSearchProfile": "my-hnsw-profile",
        },
    ],
    "vectorSearch": {
        # Defines the vector search configuration profiles.
        "profiles": [
            {"name": "my-hnsw-profile", "algorithm": "my-hnsw-config"}
        ],
        # Defines the vector search algorithms. HNSW is used for efficient similarity search.
        "algorithms": [
            {"name": "my-hnsw-config", "kind": "hnsw"}
        ],
    },
}

print(f"Sending PUT request to create/update index at: {url}")

# Send the request to Azure AI Search to create or update the index.
# A PUT request is idempotent, meaning it can be run multiple times safely.
response = requests.put(url, headers=headers, data=json.dumps(index_body))

# Check the response status code to confirm the result.
if response.status_code in [200, 201]:
    print("\nSUCCESS: The index was created or updated successfully.")
    print("The index is now configured for 384-dimension vectors.")
else:
    print(f"\nERROR: Azure responded with status code: {response.status_code}")
    print("Error details:")
    print(response.json())
