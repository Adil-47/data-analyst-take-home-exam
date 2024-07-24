-- Question 1

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



-- Question 3

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
    your_table_name
WHERE 
    section IN ('WSJ_Tech', 'WSJ_Markets')
GROUP BY 
    section;

-- Calculate average visits per unique visitor for Tech and Markets sections
SELECT 
    section,
    SUM(totalVisits) / COUNT(DISTINCT customerID) AS avg_visits_per_visitor
FROM 
    your_table_name
WHERE 
    section IN ('WSJ_Tech', 'WSJ_Markets')
GROUP BY 
    section;
