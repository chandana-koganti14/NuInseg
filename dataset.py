import kagglehub

# Download latest version
path = kagglehub.dataset_download("ipateam/nuinsseg")

print("Path to dataset files:", path)