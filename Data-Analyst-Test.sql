-- Question 1
# Solving Q1 with SQL code:
    
-- Step 1: Remove duplicates and count total unique customers
WITH unique_customers AS (
    SELECT DISTINCT customerID
    FROM customer_activity
),

-- Step 2: Identify customers who had a second visit
customers_with_second_visit AS (
    SELECT DISTINCT customerID
    FROM customer_activity
    WHERE secondVisitDate IS NOT NULL
)
-- Step 3: Calculate total unique customers and customers with second visit
SELECT
    (SELECT COUNT(*) FROM customers_with_second_visit) AS Customers_With_Second_Visit,
    (SELECT COUNT(*) FROM unique_customers) AS Total_Unique_Customers,
    ((SELECT COUNT(*) FROM customers_with_second_visit) * 100.0 / (SELECT COUNT(*) FROM unique_customers)) AS Percentage_Returned;




-- Question 2

# Solving Q2 with SQL code:

-- Step 1: Create a CTE (Common Table Expression) named RankedStories

WITH RankedStories AS (
    
    -- Step 2: Select section, headline, and count the number of pageviews per headline in each section
    SELECT 
        section,
        headline,
        COUNT(*) AS pageviews,
        
        -- Step 3: Assign a rank to each headline within its section based on the number of pageviews, in descending order
        ROW_NUMBER() OVER (PARTITION BY section ORDER BY COUNT(*) DESC) AS rank_num
        
    FROM 
        customer_activity
    
    -- Step 4: Group the results by section and headline to aggregate the pageviews
    GROUP BY 
        section, headline
)

-- Step 5: Select the top 3 headlines per section based on pageviews
SELECT 
    section,
    headline,
    pageviews

FROM 
    RankedStories

-- Step 6: Filter the results to include only the top 3 ranked headlines per section
WHERE 
    rank_num <= 3

-- Step 7: Order the final results by section and within each section by pageviews in descending order
ORDER BY 
    section, pageviews DESC;



# Solving Q2 with Python code:


import pandas as pd

# Load the data from the Excel file

file_path = 'C:/Users/adil_/Desktop/Excel_file/Data Analyst Take Home Exam Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Data Analyst Take Home Exam Dat')

# Grouping the data by 'section' and 'headline' to find the top three best-performing stories in each section by pageviews

top_stories = df.groupby(['section', 'headline']).size().reset_index(name='pageviews')

# Sorting the stories by section and then by pageviews in descending order
top_stories_sorted = top_stories.sort_values(by=['section', 'pageviews'], ascending=[True, False])

# Selecting the top three stories per section
top_stories_per_section = top_stories_sorted.groupby('section').head(3)

# Displaying the results
print(top_stories_per_section)
    


-- Question 3
# Solving Q3 with SQL code:
    
SELECT 
    section,
    COUNT(DISTINCT customerID) AS unique_visitors
FROM 
    customer_activity
WHERE 
    section IN ('WSJ_Tech', 'WSJ_Markets')
GROUP BY 
    section;

-- Calculate total visits for Tech and Markets sections
SELECT 
    section,
    SUM(totalVisits) AS total_visits
FROM 
    customer_activity
WHERE 
    section IN ('WSJ_Tech', 'WSJ_Markets')
GROUP BY 
    section;

-- Calculate average visits per unique visitor for Tech and Markets sections
SELECT 
    section,
    SUM(totalVisits) / COUNT(DISTINCT customerID) AS avg_visits_per_visitor
FROM 
    customer_activity
WHERE 
    section IN ('WSJ_Tech', 'WSJ_Markets')
GROUP BY 
    section;



# Solving Q3 with Python code:

import pandas as pd

# File path
file_path = 'C:/Users/adil_/Desktop/Excel_file/Data Analyst Take Home Exam Data.xlsx'

# Read the data from the Excel file
df = pd.read_excel(file_path)

# Filter the data to only include 'Tech' and 'Markets' sections
tech_markets_df = df[df['section'].isin(['WSJ_Tech', 'WSJ_Markets'])]

# Count the number of unique customer IDs for each section to find the number of unique visitors
unique_visitors = tech_markets_df.groupby('section')['customerID'].nunique()

# Calculate the total visits for each section
total_visits = tech_markets_df.groupby('section')['totalVisits'].sum()

# Calculate the average number of visits per unique visitor for each section
avg_visits_per_visitor = total_visits / unique_visitors

# Display the results
print("Unique Visitors:\n", unique_visitors)
print("\nTotal Visits:\n", total_visits)
print("\nAverage Visits per Visitor:\n", avg_visits_per_visitor)

