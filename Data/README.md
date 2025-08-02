## 📑 Table of Contents – Project Structure Overview

1. [Project Structure Overview](#project-structure-overview)  
2. [Folder Summary](#folder-summary)  
3. [Folder Details](#folder-details)  
   - [Data/](#data)  
   - [Data Analysis/](#data-analysis)  
   - [Search Across Bulletins/](#search-across-bulletins)  
4. [Organizational Rationale](#organizational-rationale)



# Project Structure Overview

The **FinalProject-MSFT-AttackAnalysis** repository is organized to clearly separate raw data, supporting documentation, and additional files that enrich the analysis process.

| Folder Name               | Description                                                                                      |
|---------------------------|------------------------------------------------------------------------------------------------|
| `Data/`                   | Contains the main raw datasets with Microsoft security bulletins spanning multiple years.      |
| `Data Analysis/`          | Includes an Excel file that provides detailed explanations and descriptions of each data feature (metadata). |
| `Search Across Bulletins/`| Contains supplementary files downloaded from Microsoft's website that accompany the raw data, such as additional documents or reference materials. |

---

### Folder Details

- **Data/**  
  This folder stores the essential datasets used for analysis, including comprehensive Excel files like *Bulletin Search (2001 - 2008).xlsx* and *Bulletin Search (2008 - 2017).xlsx*, which list security bulletins issued by Microsoft over the years.

- **Data Analysis/**  
  Contains a single Excel file (or documentation) explaining each feature included in the datasets. This metadata file helps understand the meaning and relevance of the columns in the raw data.

- **Search Across Bulletins/**  
  This folder holds all related auxiliary files that came bundled with the raw data from Microsoft’s site. These are supporting materials but do not include any scripts or analysis code.

---

This structured approach keeps raw data, documentation, and supplementary files well organized, making it easier to navigate and maintain the project.
