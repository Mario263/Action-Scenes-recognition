# import pandas as pd

# # Load the CSV file
# file_path = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Batman/VLM-testing/annotated.csv"  # Update this with your file path
# df = pd.read_csv(file_path)

# # Trim spaces and replace "none" with "No Action" in the "Action" column
# df["Action"] = df["Action"].str.strip().replace("none", "No Action")

# # Save the updated CSV
# output_path = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Batman/VLM-testing/updated_file.csv"
# df.to_csv(output_path, index=False)

# print(f"Updated file saved as {output_path}")
import pandas as pd

# Load the CSV file
file_path = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Batman/VLM-testing/updated_file.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Replace "/" with " and " in the "Action" column
df["Action"] = df["Action"].str.replace("/", " and ")

# Save the updated CSV
output_path = "updated_file.csv"
df.to_csv(output_path, index=False)

print(f"Updated file saved as {output_path}")