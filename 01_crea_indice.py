# 01_crea_indice.py
# This script is responsible for creating or updating the index schema in Azure AI Search.
# It defines the structure of the data that will be stored, including the fields for content,
# ID, and the crucial vector field for semantic search. This script should typically be
# run only once at the beginning of the project setup.

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# --- Azure AI Search Configuration ---
# Retrieve the necessary credentials and identifiers from environment variables.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
# Use a stable API version for consistency and to avoid breaking changes.
API_VERSION = "2023-11-01" 

# Validate that all required environment variables are set before proceeding.
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME]):
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

# Construct the full URL for the REST API endpoint to create/update an index.
url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version={API_VERSION}"

# Set the required headers for the API request, including the content type and the API key for authentication.
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_API_KEY
}

# --- Index Schema Definition ---
# This JSON object defines the structure (schema) of the search index. It specifies the fields,
# their types, and the vector search configuration.
index_body = {
    "name": AZURE_SEARCH_INDEX_NAME,
    "fields": [
        # The 'id' field is the unique identifier for each document (chunk).
        # 'key': True makes it the primary key.
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        
        # The 'content' field holds the actual text of the chunk.
        # 'searchable': True allows full-text search on this field.
        {"name": "content", "type": "Edm.String", "searchable": True},
        
        # The 'content_vector' field stores the vector embedding of the content.
        # 'type': 'Collection(Edm.Single)' specifies an array of single-precision floats.
        {
            "name": "content_vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            # 'dimensions' MUST match the output dimensions of the embedding model.
            # 'paraphrase-multilingual-MiniLM-L12-v2' produces 384-dimension vectors.
            "dimensions": 384,
            "vectorSearchProfile": "my-hnsw-profile",
        },
    ],
    "vectorSearch": {
        # 'profiles' define named configurations for vector search.
        "profiles": [
            {"name": "my-hnsw-profile", "algorithm": "my-hnsw-config"}
        ],
        # 'algorithms' define the vector search algorithms. HNSW (Hierarchical Navigable Small World)
        # is a high-performance algorithm used for approximate nearest neighbor search.
        "algorithms": [
            {"name": "my-hnsw-config", "kind": "hnsw"}
        ],
    },
}

# Keep the console output in Italian.
print(f"Sending PUT request to create/update index at: {url}")

# Send the request to Azure AI Search to create or update the index.
# A PUT request is idempotent, meaning it can be run multiple times safely. If the index
# already exists with this schema, Azure will confirm success. If it exists but with a
# different schema, it will be updated. If it doesn't exist, it will be created.
response = requests.put(url, headers=headers, data=json.dumps(index_body))

# Check the HTTP response status code to confirm the result.
# Status codes 200 (OK) or 201 (Created) indicate success.
if response.status_code in [200, 201]:
    print("\nSUCCESS: The index was created or updated successfully.")
    print("The index is now configured for 384-dimension vectors.")
else:
    # If something went wrong, print the error status code and the detailed error message from Azure.
    print(f"\nERROR: Azure responded with status code: {response.status_code}")
    print("Error details:")
    print(response.json())