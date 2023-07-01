# Tracking Changes in KPI Importance Over Time

# Abstract
We seek to demonstrate that:<br><br>
1. machine readable financial data from SEC filing companies can be used to determine what the most important line item or key performance indicator (KPI) is for a given stock<br><br>
2. a given KPI's importance can change in trend over time.<br><br>
Example: AMZN has historically been a revenue more than earnings story, but in the past few quarters, cost cutting-driven eps has begun to matter to the stock's performance.
<br><br>

# Problem statement
What is the most important KPI for a given stock?  <br><br>
Data Science in fundamental investing seeks to discover predictive investing signal with statistical correlates among fundamental / alternative data.  <br><br>
However, while Data Scientists usually have domain knowledge, they often don't know individual stock characteristics as well as traditional portfolio managers and analysts.<br><br>
A common risk to the credibility and subsequent acceptance of Data Science conclusions is the response: "that's a good r squared but everyone knows XYZ stock doesn't trade on that metric."

# Background
Insync Analytics provides human-centered street consensus and financial modeling solutions for institutional investors globally, with detailed, accurate historicals and estimates data for financial metrics and KPIs going back 10 years for 4,000+ companies not available elsewhere.

# Solution
The approach is to rank order the relationship between the hundreds of company specific KPIs in our fundamental dataset and the post-earnigs five day alpha of the stock price.  <br><br>
Alpha in this case is defined as the five day beta adjusted performance of the stock versus the S&P 500 index.  <br><br>
For example: if the beta is 1 and the market is down 5% 5 days after earnings and the stock is down 5%, nothing happened.<br><br>
Importantly for this analysis, beta is calculated on a rolling basis, so the beta for each given earnings report period is "as of" that report period.  <br><br>
The beta rolling window is a standard 60 days.  <br><br>

# Conclusion
Looking specifically at AMZN, we can note that historically EPS surprises do not appear to be correlated with stock movement subsequent to the earnings announcement date.
![image](https://github.com/jed-gore/insync/assets/39496491/f7838320-ca41-4c5c-a3ed-3b17ccd71f16)

So this begs the question: what DOES matter to AMZN stock?<br><br>
Given that AMZN historically has been a revenue growth story, we might assume revenue would be the key KPI.<br><br>
However our regression work suggests otherwise:<br><br>
In fact it appears that margins and quarterly change in earnings matter most.
WHy the disconnect?
![image](https://github.com/jed-gore/insync/assets/39496491/1b2772df-a1ed-406f-871b-ea9c8bd9eadf)

The counterintuitive result can be explained by looking at the <b>history</b> of the relationships, or the rolling regression of KPI vs stock alpha post earnings.<br><br>
Earnings and margins have only recently begun to trend upwards in importance.<br>

![image](https://github.com/jed-gore/insync/assets/39496491/84663121-3daa-4ef6-b59e-3e676aba56c3)<br>
![image](https://github.com/jed-gore/insync/assets/39496491/8356a319-bbea-4d80-a419-3fd55587dda7)

SO when the fundamental portfolio manager rightly points out: "AMZN does not trade on earnings" now the data scientist can reply: "That was true, but it trades on earnings and per customer metrics now."
<br><br>
Noticing these key regime changes mark the difference between outperforming one's investing peers and not.




