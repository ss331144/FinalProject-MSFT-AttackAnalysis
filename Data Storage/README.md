# Storage 

1 - MongoDB Cloud : [MongoDB](mongodb+srv://user0:User0@cluster0.xrbfbxu.mongodb.net/final_project?retryWrites=true&w=majority)
### Guide :
In the **Query Filter** bar, enter your query, for example:

<img width="1113" height="42" alt="image" src="https://github.com/user-attachments/assets/13f24f2a-3bb9-4ba6-954b-1802ec8ad28c" />

   ```javascript
   { Severity: "Critical" }
   ```
Click **Find** to run the query.

<img width="48" height="27" alt="image" src="https://github.com/user-attachments/assets/37fabeaa-4e41-4b2f-aeea-41d8c421bf73" />

#### Example Queries
- Search by severity:
  ```javascript
  { Severity: "Critical" }
  ```
- This query retrieves documents from the microsoft_attack collection where: Severity is "Critical", the Year is 2015, and the Month is September, October, or November (9, 10, or 11).
```javascript
 db.microsoft_attack.find({
  $and: [
    { Severity: "Critical" },
    { Year: 2015 },
    { $or: [
      { Month: { $gte: 9 } }, 
      { Month: { $lte: 11 } }
    ]}
})
```


2 - Google Docs : [Google Docs](https://docs.google.com/spreadsheets/d/1l90t8_W-ONJ2Wz4msPEY7h-4dhggCJ8_/edit?gid=474301616#gid=474301616)

3 - One Drive : [One Drive](https://yvcstudents-my.sharepoint.com/:x:/r/personal/314741851_students_yvc_ac_il/_layouts/15/Doc.aspx?sourcedoc=%7B61D19296-267D-4630-AC6D-19C23C48DE6F%7D&file=Merged_Bulletin_Data.xlsx&action=default&mobileredirect=true)

---

**Important:**  
To run the code from the Google Colab notebook, you must have **MySQL installed on your local computer** and run the relevant services from your personal computer.

---

Thank you for your understanding.
