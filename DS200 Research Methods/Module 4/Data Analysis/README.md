# DS200-M4

## MGNREGA CSV Dataset

### Overview

This dataset contains district-wise and year-wise records from the Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA) scheme. It tracks key performance indicators, fund utilization, employment statistics, and more to monitor the progress and effectiveness of the program.

### Data Structure

- **Format**: CSV (Comma-Separated Values)
- **Rows**: Each row records data for a specific district and year
- **Columns**:
    - `State`: State name for the record
    - `District`: Corresponding district name
    - `Year`: Reporting year
    - `Households`: Number of registered households
    - `Employment Provided`: Number of households actually provided with work
    - `Persondays Generated`: Total days of employment generated
    - `Funds Released`: Monetary allocation or expenditure
    - Other columns may include gender, social category, project types

### Usage

- Statistical and analytical research on rural employment trends
- Visualization of scheme performance across districts and years
- Policy evaluation and academic studies

### Example Row

| State      | District   | Year | Households | Employment Provided | Persondays Generated | Funds Released |
|------------|------------|------|------------|--------------------|---------------------|---------------|
| StateName  | DistrictA  | 2023 | 1200       | 900                | 15000               | 3,000,000     |

### Notes

- The dataset is suitable for large-scale data analysis and visualization in Python, R, MATLAB, or other numerical tools.
- See the header row for a full list of fields and examine initial rows to understand data layout.

---

For questions or feedback, contact the data author or refer to the official MGNREGA documentation.
