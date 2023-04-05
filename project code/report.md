# Summary section

In this project, I did a sentiment analysis on 497 companies with 10k and calculated the return of these companies within 10 days of their filing date in 2022.

# Data Section

## What's the sample?

In this project, I use 10k files,sp 500 file, crsp_2022, 2021_ccm_cleaned. 

## How are the return variables built and modified? (Mechanical description.) 

First of all, I convert the format of filing_date into datetime format, and I merge the crsp_2022 and filing_date dataframe into a new dataframe which named merged_df using a right join.
```
df1['filing_date'] = pd.to_datetime(df1['filing_date'])
merged_df=pd.merge(ret,df1,left_on='ticker',right_on='Symbol',how='right')
merged_df
```

After merged, I add a new column which called count and set the value in the columns is -1. Set a variable which equals to 1
```
merged_df['Count'] = -1
temp_value=1
```

Starting a loop that iterates over each row in the merged_df DataFrame.
```
for index,row in merged_df.iterrows():
    if row['date']==row['filing_date']:
        merged_df.loc[index,'Count']=0
        temp_value=1
    elif row['date']>row['filing_date']:
        merged_df.loc[index,'Count']=temp_value
        temp_value += 1
    else:
        merged_df.loc[index,'Count']=-2
        temp_value=1
```

In this for loop, index representing the index of each row and row representing the actual values in each row.
```
if row['date']==row['filing_date']:
        merged_df.loc[index,'Count']=0
        temp_value=1
```
This line checks if the value in the 'date' column is equal to the value in the 'filing_date' column for the current row. If it is, the value in count changes to 0.

```
elif row['date']>row['filing_date']:
        merged_df.loc[index,'Count']=temp_value
        temp_value += 1
```
This code checks if the value in the 'date' column is less than the value in the 'filing_date' column for the current row. If yes, the value in count for the current row changes to temp_value, and the value for temp_value add 1.

```
 else:
        merged_df.loc[index,'Count']=-2
        temp_value=1
```
This code is executed if neither of the previous conditions are met, the the value in count for the current row changes to -2.

For the last step of this part:

```
# Filter to only include rows where days_since_filing is between 0 and 2
filtered_0_2=merged_df[(merged_df['Count']>=0)&(merged_df['Count']<=2)]
#print(filtered_0_2)
# Group by ticker and calculate cumulative return
cumulative_returns_0_2=filtered_0_2.groupby('ticker')['ret'].apply(lambda x:np.prod(1+x)-1)
```
These two line filter dataframe ' merged_df' include rows where the value in the 'Count' column is between 0 and 2 (inclusive), using formula to calculate the cumulative return for the 3 days.

```
# Filter to only include rows where days_since_filing is between 3 and 10
filtered_3_10=merged_df[(merged_df['Count']>=3)&(merged_df['Count']<=10)]
# Group by ticker and calculate cumulative return
cumulative_returns_3_10=filtered_3_10.groupby('ticker')['ret'].apply(lambda x: np.prod(1+x)-1)
```
These two line do the same steps with above, but it filter day t+3 to day t+10 and calculate. 

```
return3_10=pd.DataFrame(cumulative_returns_3_10).rename(columns={'ret':'ret_310'})
return0_2=pd.DataFrame(cumulative_returns_0_2).rename(columns={'ret':'ret_02'})

cumulative_returns=return0_2.merge(return3_10,how='right',on='ticker',indicator=True,validate='1:1')
cumulative_returns
```
Convert the result above into dataframe and merge them together. 

## How are the sentiment variables are built and modified? (Mechanical description.)

Those sentiment variable are built and modified using NEAR_regex, using 're.findall' to check the total number of times the words in this list appear in this text.

### Integrity Words sentiment analysis
```
positive_sentiment= ['(trustworthy|honest|transparent|ethical|reliable|accountable|authentic|principled|dependable|credible|sincere|genuine)']
negative_sentiment= ['(deceptive|fraudulent|dishonest|corrupt|unethical|unreliable|unaccountable|inauthentic|unprincipled|suspicious|dubious|misleading)']

new1=[]
new2=[]
for index,row in df.iterrows():
    sentence = row['html']
    
    integrity_positive=(len(re.findall(NEAR_regex(positive_sentiment),sentence))/len(sentence.split()))
    
    integrity_negative=(len(re.findall(NEAR_regex(negative_sentiment),sentence))/len(sentence.split()))
    new1.append(integrity_positive)
    new2.append(integrity_negative)
    
df['integrity_positive']=new1
df['integrity_negative']=new2
```

### Risk words sentiment analysis
```
positive_sentiment= ['(opportunity|growth|advantage|benefit|strength|robust|stable|prosperous|potential|advantageous)']
negative_sentiment= ['(risk|uncertainly|challenge|vulnerability|weakness|fluctuation|exposure|instability|threat|liability)']

new3=[]
new4=[]
for index,row in df.iterrows():
    sentence = row['html']
    
    risk_positive=(len(re.findall(NEAR_regex(positive_sentiment),sentence))/len(sentence.split()))
    
    risk_negative=(len(re.findall(NEAR_regex(negative_sentiment),sentence))/len(sentence.split()))
    new3.append(risk_positive)
    new4.append(risk_negative)
    
df['risk_positive']=new3
df['risk_negative']=new4
```

### Market words sentiment analysis
```
positive_sentiment= ['(expansionary|synergisti|innovative|competitive|promising|progressive|potent|thriving)']
negative_sentiment= ['(reduce|decline|ambiguous|eroding|challenge)']

new5=[]
new6=[]
for index,row in df.iterrows():
    sentence = row['html']
    
    market_positive=(len(re.findall(NEAR_regex(positive_sentiment),sentence))/len(sentence.split()))
    
    market_negative=(len(re.findall(NEAR_regex(negative_sentiment),sentence))/len(sentence.split()))
    new5.append(market_positive)
    new6.append(market_negative)
    
df['market_positive']=new5
df['market_negative']=new6
```

## Reason why I choose the three topics for the “contextual sentiment” measures

Choosing integrity, risk, and market as the focus of sentiment analysis for a 10-K filing can be useful for several reasons:

Integrity: Examining the language used by a company in its 10-K filing can reveal clues about its ethical values and commitment to transparency. An analysis of the sentiment around this topic can help investors assess the level of trust they can place in the company's management and decision-making.

Risk: The 10-K filing typically includes a discussion of risks faced by the company, such as market competition, economic conditions, and regulatory changes. Analyzing the sentiment around these risks can help investors gauge the level of concern or optimism the company has about these challenges.

Market: Understanding the sentiment of a company's discussion of the market can help investors get a sense of how the company views its competitive environment and potential opportunities for growth. This information can be valuable when making investment decisions.

## Show and discuss summary stats of your final analysis sample

![](plots/final1.png)

![](plots/final2.png)

## Smell tests

The contextual sentiment not fishy, and all the words are related to the financial situation. Depends on the words, I want to know if a company is financially healthy, and I want to know if I should invest in the company.

All variables have unique values, so these contextual sentiment words can be analyzed.

## caveats about the sample and/or data

For the analysis_sample, because it lost some return value in crsp file, they have some 'NA' in columns 'ret_02' and 'ret_310'. It will reflect the analysis for this company.

# Result

## Scatter Plots

![plot1.png](attachment:5c9cc8c6-b833-4ec6-a61d-cc0f1f5d0e05.png)

![plots/plot2.png](attachment:bc0f7b7d-7531-4da0-91b2-7ba1437ea2eb.png)

![plots/plot3.png](attachment:523fae1e-30a2-407b-a683-78263de465cc.png)

![plots/plot4.png](attachment:fa69cd9a-9791-4d31-ad07-585cbd45475e.png)

![plots/plot5.png](attachment:219725b7-a96f-4647-8c17-43804c40e35a.png)

![plots/plot6.png](attachment:ea2f117e-7e4f-429b-927f-cc5226ef78bf.png)

![plots/plot7.png](attachment:ced702c2-f3bf-4bd1-9d03-589e5e7e218e.png)

![plots/plot8.png](attachment:f883313f-659d-4439-90e2-ec3cb51e1545.png)

![plots/plot9.png](attachment:c6ac445c-efe7-42e7-ad89-c1deba937cce.png)

![plots/plot10.png](attachment:bf4e54ce-e540-41f9-ae2e-b5161c8071f8.png)

## Four discussion topics

### 1.
#### positive 
From the scatter plot, we can found that most of the day t to day t+2 data are in the middle of the range of sentiment score at 0.02 to 0.08 and return range at -0.2 to 0.2. Most of day t+3 to day t+10 data are located same place.
However, red points have more outliers than blue points.

#### negative
From the scatter plot, we can found that most of the day t to day t+2 data are in the middle of the range of sentiment score at 0.01 to 0.25 and return range at -0.2 to 0.2. Most of day t+3 to day t+10 data are located same place.
However, red points have more outliers than blue points.


### 2.
After comparison/contrast conflicts with Table 3 of the Garcia, Hu, and Rohrer paper (ML_JFE.pdf, in the repo), I found there is little correlation between sentiment and return. I think the reason why they do so many more firms and years and additional controls in their study is because they want to know will some words really reflect the return.

### 3. 
They do have a relationship with returns that look "different enough" from zero. 
Integrity: The language used by companies to discuss their ethical and moral standards can impact the sentiment of investors and may have a relationship with returns. Companies that are perceived as having high integrity may be viewed more favorably by investors, leading to higher stock prices and potentially higher returns. Conversely, companies that are perceived as having low integrity may be viewed more skeptically by investors, leading to lower stock prices and potentially lower returns.

Risk: The language used by companies to describe potential risks and uncertainties may also impact investor sentiment and have a relationship with returns. Companies that are perceived as having a higher level of risk may be viewed more skeptically by investors, leading to lower stock prices and potentially lower returns. Conversely, companies that are perceived as having lower risk may be viewed more favorably by investors, leading to higher stock prices and potentially higher returns.

Market: The sentiment of the overall market can also impact investor sentiment and have a relationship with returns. During times of positive market sentiment, investors may be more optimistic about the economy and the prospects of companies, leading to higher stock prices and potentially higher returns. Conversely, during times of negative market sentiment, investors may be more cautious and risk-averse, leading to lower stock prices and potentially lower returns.

In terms of an economic argument for why sentiment in these contexts can be value relevant, it's important to consider the impact of investor sentiment on market efficiency. If investors are making investment decisions based on sentiment rather than fundamental factors such as financial performance and industry trends, this can lead to market inefficiencies and potentially mispricing of stocks. However, if sentiment is correlated with returns, it may be possible for investors to capitalize on sentiment-driven market inefficiencies and generate higher returns. Additionally, understanding the sentiment of investors and the overall market can be useful for predicting future market trends and making informed investment decisions. 

### 4. 
It didn't have much difference between ML_sentiment with return. Because  the 10k is as objective a document as possible, the company analyzes its own factors for the previous year, including its financial situation and risks.
