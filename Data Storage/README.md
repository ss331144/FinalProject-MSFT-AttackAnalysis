# Storage 

1 - MongoDB Cloud : [MongoDB](https://cloud.mongodb.com/v2/689e1601b7c6de55cb2cccbe#/explorer/689e169d8c995c677442b540/final_project/microsoft_attack/find)
## Guide :
In the **Query Filter** bar, enter your query, for example:
   ```javascript
   { Severity: "Critical" }
   ```
Click **Find** to run the query.
### Example Queries
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
      { Month: { $gte: 9 } }, // חודשים גדולים או שווים ל-9 (ספטמבר או אחריו)
      { Month: { $lte: 11 } } // חודשים קטנים או שווים ל-11 (נובמבר או לפני)
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
