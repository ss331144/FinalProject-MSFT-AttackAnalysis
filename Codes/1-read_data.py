# -----------------------------------------
# 🧮 Data Processing and Statistics Libraries
# -----------------------------------------
import os  # Working with the file system
import pandas as pd  # Managing and analyzing tables

os.makedirs('data', exist_ok=True)

# List of Excel files to merge
Files = ['/Users/shryqb/PycharmProjects/new_project_original/file_1/Original_Data/Bulletin Search (2001 - 2008).xlsx',
         '/Users/shryqb/PycharmProjects/new_project_original/file_1/Original_Data/Bulletin Search (2008 - 2017).xlsx']  # List of train files

# Concatenate the Excel files into one DataFrame
Data_Frame = pd.concat([pd.read_excel(file) for file in Files], ignore_index=True)

# Enforcing order by reindexing
Data_Frame = Data_Frame.reset_index(drop=True)

save_dir = 'data'
excel_path = os.path.join(save_dir, 'Merged_Bulletin_Data.xlsx')
csv_path = os.path.join(save_dir, 'Merged_Bulletin_Data.csv')

# Save data to Excel
with pd.ExcelWriter(excel_path) as writer:
    Data_Frame.to_excel(writer, index=False, sheet_name='All Data')
Data_Frame.to_csv(csv_path, index=False)

# Load the Excel file and display the first 4 rows
df = pd.read_excel(excel_path)
print(df.head(4))
