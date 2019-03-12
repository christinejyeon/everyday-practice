/* Question 1-a DAU */
SELECT DATE(event_timestamp) AS Date, COUNT(DISTINCT user_id) AS DAU
FROM events
WHERE event_name='gameStarted'
GROUP BY Date

/* Question 1-b Weekly Stickiness */
SELECT a.Date_Daily, a.Date_Weekly, a.Num_Users_d, b.Num_Users_w, CAST(a.Num_Users_d AS FLOAT)/CAST(b.Num_Users_w AS FLOAT) AS Weekly_Stickiness
FROM (SELECT DATE(event_timestamp) AS Date_Daily,
STRFTIME('%W', event_timestamp) AS Date_Weekly,
COUNT( DISTINCT user_id) AS Num_Users_d
FROM events
WHERE event_name='gameStarted'
GROUP BY Date_Daily) a
LEFT JOIN (SELECT
STRFTIME('%W', event_timestamp) AS Date_Weekly,
COUNT( DISTINCT user_id) AS Num_Users_w
FROM events
WHERE event_name='gameStarted'
GROUP BY Date_Weekly) b
ON b.Date_Weekly = a.Date_Weekly
ORDER BY a.Date_Daily, a.Date_Weekly

/* Question 2-a Daily Revenue */
SELECT DATE(a.date) AS Daily,
ROUND((CASE WHEN b.transaction_value IS NULL THEN 0 ELSE b.transaction_value END)-SUM(a.cost),2) AS Daily_Revenue
FROM (SELECT * FROM acquisition) a
LEFT JOIN (
SELECT DATE(event_timestamp) AS Date_Daily, SUM(CASE WHEN transaction_value IS NULL THEN 0 ELSE transaction_value END) AS transaction_value
FROM events
GROUP BY Date_Daily) b
ON a.date=b.Date_Daily
GROUP BY a.date

/* Question 2-b Daily Conversion Rate */
SELECT Date, ROUND(Num_Transaction*1.0/DAU*1.0,2) AS Daily_Conversion_Rate
FROM (SELECT DATE(event_timestamp) AS Date,
SUM(CASE WHEN event_name='gameStarted' THEN 1 ELSE 0 END) DAU,
SUM(CASE WHEN event_name='transaction' THEN 1 ELSE 0 END) Num_Transaction
FROM events
GROUP BY Date)

/* Question 3 Average Daily Playtime */
SELECT user_id, TIME(AVG(STRFTIME('%s',playtime)), 'unixepoch')
FROM (
SELECT a.user_id, a.cal_date, TIME(STRFTIME('%s',b.ts_ended)-STRFTIME('%s',a.ts_started), 'unixepoch') AS playtime
FROM (SELECT DISTINCT user_id, TIME(SUM(STRFTIME('%s', event_timestamp)), 'unixepoch') AS ts_started, DATE(event_timestamp) AS cal_date
FROM events
WHERE event_name='gameStarted'
GROUP BY user_id, cal_date
ORDER BY user_id, cal_date) a
INNER JOIN
(SELECT DISTINCT user_id, TIME(SUM(STRFTIME('%s', event_timestamp)), 'unixepoch') AS ts_ended, DATE(event_timestamp) AS cal_date
FROM events
WHERE event_name='gameEnded'
GROUP BY user_id, cal_date
ORDER BY user_id, cal_date)b
ON b.user_id = a.user_id AND b.cal_date = a.cal_date
ORDER BY a.user_id)
GROUP BY user_id

/* Question 4-a CPI Per Acquisition Channel */
SELECT a.source, SUM(a.cost)/b.Num_Install AS CPI
FROM (SELECT * FROM acquisition) a
LEFT JOIN (
SELECT DATE(event_timestamp) AS Date_Daily, SUM(CASE WHEN event_name='install' THEN 1 ELSE 0 END) AS Num_Install
FROM events) b
ON a.date=b.Date_Daily
GROUP BY a.source
