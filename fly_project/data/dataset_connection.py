import os 
from dotenv import load_dotenv
from neuprint import Client,fetch_meta

load_dotenv()

token=os.getenv("NEUPRINT_TOKEN")

if not token:
    raise ValueError("TOKEN NOT FOUND ")

client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)



print("Available datasets on server:", list(client.fetch_datasets().keys()))

meta = fetch_meta(client=client)
print("\nSuccessfully connected to MaleCNS!")
print(f"Dataset Meta Keys:", list(meta.keys()) if isinstance(meta, dict) else "Connected")