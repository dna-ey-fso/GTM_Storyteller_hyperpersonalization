import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import numpy as np
import time

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')


nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)





# Tutorial 1 pages

# introduction page of tutorial 1
def page1():
    st.title("Introduction tutorial 1: ")
    st.write("Two different people investing in the same stocks can have completely different experiences.")
    st.markdown("**Customer 1:** Bert is a non-experienced investor who has deposited a small amount of money and is personally investing for entertainment and to learn a little about how the stock markets work. Learning and having fun is his top priority; performance of his portfolio comes second.")
    st.markdown("**Customer 2:** Ann works in finance and knows a lot about the stock market; she has no need for additional information. She wants to get a concise and clear overview of how her portfolio performs and wants to understand financials of the companies. ")
    st.markdown("**Portfolio (all the data that is used as input fot the NLP models):**")

    st.markdown('It appears that your portfolio had a negative return of **-3.62% over the last 60 days**')
    data = [['Apple', 'AAPL', '20%', '-7.5%', 'Apple iPhone 15 trial production commences in China - GSMArena.com news', 'Information Technology', 1.27, 21.83, 2.60, 5.52, 41.88, 5.56, 16.48], ['Microsoft', 'MSFT', '20%', '-1.28%', 'Microsoft is looking at OpenAIs GPT for Word, Outlook, and PowerPoint', 'Information Technology', 0.93, 25.00,2.1, 8.83, 10.24, 8.52, 17.17],['Amazon', 'AMZN', '20%', '-26.46%', 'Amazon to axe 18,000 jobs as it cuts costs', 'Consumer Cyclical', 1.17, 49.51, 5.58, 1.94, 7.70, 2.07, 20.55], ['Google', 'GOOG', '20%', '-10.18%', 'Google Messages bakes Material You theming into its account switcher', 'Information Technology',1.07, 17.18, 1.3, 4.33, 4.69, 3.89, 11.65 ],['Visa', 'V', '10%', '16.65%', 'New Digital Nomad Visa Calculator Takes The Hassle Out Of Choosing Where To Work And Live', 'Financial Services', 0.94, 26.74, 1.70, 16.26, 13.28, 16.25, 24.38], ['Johnson & Johnson', 'JNJ', '10%', '7.89%', 'Johnson & Johnson Files for IPO of Consumer Health Division Kenvue', 'Healthcare', 0.57, 16.67, 3.74, 4.82, 6.08, 4.70, 15.27]]

    df = pd.DataFrame(data, columns=['Stock name', 'Ticker', 'Weight in fund', 'Changed value in the last 60 days', 'news', 'sector','beta(5y Montly)', 'Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales (ttm)', 'Price/Book (mrq)', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA' ])

    st.table(df)

    st.markdown("**the wold index raised 9.88% and the s&p500 raised 7.25% during last 60 days**")




def page2():
   st.title("Customer 1: Bart")
     

   col1, col2 = st.columns(2)

   with col1:
      image = Image.open("image.png")
      st.image(image, width=200, )

   st.empty() # creates an empty space

   with col2:
         # Add image
      st.markdown("**Balance:** 1170$")
      st.markdown("**Invested:** 1000$")
      st.markdown("**Cash:** 0$")
      st.markdown("**Total:** 1170$")
      st.markdown("**Return:** +17%")

   st.markdown("**Summary of your portfolio during the last 60 days:**")
   st.markdown("Your portfolio had a performance of -3.62% during the last 60 days. It is composed of different stocks from different sectors such as Information Technology, Consumer Cyclical, Financial Services and Healthcare. Some of the notable stocks are Apple, Microsoft, Amazon, Google, Visa and Johnson & Johnson, which have a weight of 20% each in the portfolio except for Visa and Johnson & Johnson that have a weight of 10% each. The most significant change in value for the last 60 days for the stocks in the portfolio is Amazon with a change of -26.46% and Visa with a change of 16.65%. The world index raised 9.88% and S&P500 raised 7.25% during the last 60 days. Your portfolio underperformed the market during this period.")

   sentiment = nlp("Your portfolio had a performance of -3.62% during the last 60 days. It is composed of different stocks from different sectors such as Information Technology, Consumer Cyclical, Financial Services and Healthcare. Some of the notable stocks are Apple, Microsoft, Amazon, Google, Visa and Johnson & Johnson, which have a weight of 20% each in the portfolio except for Visa and Johnson & Johnson that have a weight of 10% each. The most significant change in value for the last 60 days for the stocks in the portfolio is Amazon with a change of -26.46% and Visa with a change of 16.65%. The world index raised 9.88% and S&P500 raised 7.25% during the last 60 days. Your portfolio underperformed the market during this period.")

   st.markdown('Sentiment analysis : {}'.format(sentiment[0]['label']))

   with st.expander("Full report"):
      st.markdown("The portfolio had a performance of -3.62% during this period.")
      st.markdown("The portfolio is composed of different stocks from different sectors such as Information Technology, Consumer Cyclical, Financial Services and Healthcare. The portfolio is well diversified across different sectors which is a good practice for risk management.")
      st.markdown("The portfolio is composed of notable stocks such as Apple, Microsoft, Amazon, Google, Visa, and Johnson & Johnson which have a weight of 20% each in the portfolio except for Visa and Johnson & Johnson which have a weight of 10% each. The weight of each stock in the portfolio is well balanced, which is also a good practice for risk management.")
      st.markdown("The most significant change in value for the last 60 days for the stocks in the portfolio is Amazon with a change of -26.46%. This can be attributed to the company's decision to axe 18,000 jobs as it cuts costs. This news had a negative impact on the stock's performance.")
      st.markdown("On the other hand, Visa had a change of 16.65% which can be attributed to the company's new digital nomad visa calculator which takes the hassle out of choosing where to work and live. This news had a positive impact on the stock's performance.")
      st.markdown("Apple had a change of -7.5% which can be attributed to the company's trial production of the iPhone 15 in China. This news had a negative impact on the stock's performance.")
      st.markdown("Microsoft had a change of -1.28% which can be attributed to the company's interest in OpenAi's GPT for Word, Outlook, and PowerPoint. This news had a negative impact on the stock's performance.")
      st.markdown("Google had a change of -10.18% which can be attributed to the company's Google Messages bakes Material You theming into its account switcher. This news had a negative impact on the stock's performance.")
      st.markdown("Johnson & Johnson had a change of 7.89% which can be attributed to the company's IPO of Consumer Health Division Kenvue. This news had a positive impact on the stock's performance.")
      st.markdown("It's also important to note that the world index raised 9.88% and S&P500 raised 7.25% during the last 60 days. This means that the investor's portfolio underperformed the market during this period.")
      st.markdown("In conclusion, the investor's portfolio had a negative performance of -10.38% during the last 60 days. The portfolio is well diversified across different sectors and stocks, which is a good practice for risk management. The most significant change in value for the last 60 days for the stocks in the portfolio is Amazon with a change of -26.46% and Visa with a change of 16.65%. The investor's portfolio underperformed the market during this period. It's important for the investor to review the portfolio and make any necessary adjustments to improve its performance in the future.")




   chart_data = pd.DataFrame(
    [1,3,4,2,0,-2,-2,-1,2,2,4,5,4,3,5,6,7,5,6,7,9,6,5,7,8,9,10,14,16,15,19,21,21,23,21,21,21,24,24,18],
    columns=['Portfolio'])

   st.line_chart(chart_data)

   st.markdown("**Possible questions to which you want answers:**")
   with st.expander("What factors may have contributed to the underperformance of your portfolio compared to the broader market indices such as the S&P 500 and the World index during the last 60 days?"):
      st.markdown("The world index raised 9.88% and S&P500 raised 7.25% during the last 60 days.")
      st.markdown("This means that your portfolio underperformed the market during this period. Several factors could have contributed to the underperformance of the portfolio during the last 60 days.")
      st.markdown("One possibility is that the portfolio's sector allocations did not align with the sectors that performed well during that time period. For example, the portfolio's Information Technology and Consumer Cyclical sectors - which make up 40% of the portfolio - had negative returns (-7.5% and -26.46% respectively) while the broader market indices had positive returns. Additionally, the portfolio's allocation to those sectors might be overweighted or in the wrong side of market.")
      st.markdown("Another factor could be the specific stocks within the portfolio underperformed compared to the broader market. For example, Amazon had a large negative return of -26.46%, which could have had a significant impact on the portfolio's overall performance.Additionally, the Financial Services sector, which makes up only 10% of the portfolio, had a positive return of 16.65%, which might not have been enough to offset the negative returns in other sectors. The healthcare sector also had a positive return which could have helped to mitigate the negative returns of other sectors in the portfolio.")
      st.markdown("It's also worth noting that the World index and S&P 500 index have positive returns 9.88% and 7.25% respectively while the portfolio had a negative return of -3.62%, which could indicate the portfolio's performance may have been impacted by factors such as country or regional market performance or the performance of other industries not represented in the portfolio.")

  
   
   with st.expander("Why did I underperformed the markets"):
      st.markdown("Based on the information provided, it appears that the portfolio underperformed the markets during the last 60 days due to the negative returns of the Information Technology and Consumer Cyclical sectors. These two sectors make up 40% of the portfolio, and both had negative returns of -7.5% and -26.46% respectively. Additionally, companies like Amazon, Google and Apple which are high weightage stock in the portfolio had substantial negative returns of -26.46%, -10.18% and -7.5% respectively. This could be the primary factor for underperformance of the portfolio. It is also possible that the portfolio's stock selection, allocation, or sector weightings differed from those of the broader market indices, which contributed to the underperformance. Additionally, the general market conditions and economic environment during this period may have had an impact on the performance of the portfolio, and that may have led to the underperformance as well. It's also worth noting that past performance is not indicative of future performance and the portfolio's returns may be affected by various factors such as market conditions, individual stock performance, and economic conditions. It's recommended that you consult with a financial advisor to understand how this performance fits into your overall investment strategy and take an informed decision for any re-allocation or modifications required for your portfolio.")










# ______________ Ann haar pagina 



def page3():
   st.title("Customer 2: Ann")
   
   col1, col2 = st.columns(2)

   with col1:
      image = Image.open("image2.png")
      st.image(image, width=200, )

   st.empty() # creates an empty space

   with col2:
         # Add image
      st.markdown("**Balance:** 1170$")
      st.markdown("**Invested:** 1000$")
      st.markdown("**Cash:** 0$")
      st.markdown("**Total:** 1170$")
      st.markdown("**Return:** +17%")

   #vanaf hier beginnen aanpassen
   st.markdown("**Summary of your portfolio during the last 60 days:**")
   st.markdown("The portfolio has experienced a negative return of -3.62% over the last 60 days. This underperformance can be attributed to poor performance from several of the portfolio's holdings, particularly Amazon, Google and Apple.")
   st.markdown("Apple, one of the portfolio's largest holdings, had a negative return of -7.5% over the last 60 days which has a significant impact on the overall performance of the portfolio. This decline, along with poor performance of other large holdings such as Amazon and Google, had a significant impact on the portfolio's overall performance.")
   st.markdown("It is also worth noting that the portfolio is heavily invested in the Information Technology sector, with Apple, Microsoft, Google and Amazon representing 80% of the portfolio. This high concentration in a single sector increases the portfolio's risk. Additionally, the portfolio's beta is relatively high, with an average of 1.1 over the last 5 years, indicating that it has higher volatility than the overall market. Furthermore, the forward P/E ratio of the portfolio is relatively high, with an average of 22.36, suggesting that the portfolio may be overvalued.")


   sentiment = nlp("The portfolio has experienced a negative return of -3.62% over the last 60 days. This underperformance can be attributed to poor performance from several of the portfolio's holdings, particularly Amazon, Google and Apple. Apple, one of the portfolio's largest holdings, had a negative return of -7.5% over the last 60 days which has a significant impact on the overall performance of the portfolio. This decline, along with poor performance of other large holdings such as Amazon and Google, had a significant impact on the portfolio's overall performance. It is also worth noting that the portfolio is heavily invested in the Information Technology sector, with Apple, Microsoft, Google and Amazon representing 80% of the portfolio. This high concentration in a single sector increases the portfolio's risk. Additionally, the portfolio's beta is relatively high, with an average of 1.1 over the last 5 years, indicating that it has higher volatility than the overall market. Furthermore, the forward P/E ratio of the portfolio is relatively high, with an average of 22.36, suggesting that the portfolio may be overvalued.")
   st.markdown('Sentiment analysis: {}'.format(sentiment[0]['label']))

   with st.expander("Full report"):
      st.markdown("First, it is important to note that the portfolio experienced a negative return of -3.62% over the last 60 days. This underperformance can primarily be attributed to poor performance from several of the portfolio's largest holdings, specifically Amazon, Google and Apple.")
      st.markdown("When looking at the portfolio's sector allocation, it is heavily invested in the Information Technology sector, with Apple, Microsoft, Google and Amazon representing 80% of the portfolio. This high concentration in a single sector increases the portfolio's risk, as the performance of the sector will have a disproportionate impact on the overall performance of the portfolio.")
      st.markdown("Additionally, the portfolio's beta is relatively high, with an average of 1.1 over the last 5 years, indicating that it has higher volatility than the overall market. This means that the portfolio is likely to experience larger declines during market downturns compared to the overall market.")
      st.markdown("Furthermore, the forward P/E ratio of the portfolio is relatively high, with an average of 22.36. This suggests that the portfolio may be overvalued and that investors may be paying a high price for each dollar of earnings. This could be a cause of concern, as it might indicate that the price of the portfolio's stocks are not aligned with their underlying fundamentals.")
      st.markdown("Another ratio to consider is the Price to Sales (P/S) ratio, which compares the market value of a company to its revenue. This ratio can indicate how much investors are willing to pay for each dollar of revenue generated")

      



   chart_data = pd.DataFrame([1,3,4,2,0,-2,-2,-1,2,2,4,5,4,3,5,6,7,5,6,7,9,6,5,7,8,9,10,14,16,15,19,21,21,23,21,21,21,24,24,18],columns=['Portfolio'])

   st.line_chart(chart_data)

   st.markdown("**Possible questions to which you want answers:**")
   with st.expander("Can you suggest any potential adjustments or changes that could be made to the portfolio to improve performance or reduce risk?"):
      st.markdown("1. Rebalancing the portfolio: The portfolio is heavily invested in the Information Technology sector, with Apple, Microsoft, Google and Amazon representing 80% of the portfolio. This high concentration in a single sector increases the portfolio's risk, as the performance of the sector will have a disproportionate impact on the overall performance of the portfolio. Rebalancing the portfolio to diversify across sectors and industries may help to reduce risk.")
      st.markdown("2. Reducing the portfolio's beta: As mentioned before, the average beta of the portfolio is of 1.06 over the last 5 years, indicating that it has higher volatility than the overall market. To reduce the portfolio's beta, you could reduce the number of high-beta stocks in the portfolio and increase the number of low-beta stocks.")
      st.markdown("3. Adjusting the forward P/E ratio: The portfolio has a relatively high forward P/E ratio, with an average of 22.36, which suggests that the portfolio may be overvalued and that investors may be paying a high price for each dollar of earnings. To improve performance, you could adjust the portfolio by selling overvalued stocks and buying undervalued stocks.")
      st.markdown("4. Adding more value stocks: Value stocks, which are stocks that are undervalued relative to their fundamentals, tend to have lower valuations, such as lower P/E and P/B ratios, and higher dividends compared to growth stocks. Adding more value stocks to the portfolio could provide a more defensive position and potentially better returns over the long term.")
      st.markdown("5. Monitoring the news and fundamental data: Keeping track of the news and fundamental data of the stocks in the portfolio can help to identify")
      
   with st.expander("What is the portfolio's volatility in relation to the overall market?"):
      st.markdown("The average beta of the portfolio is of 1.06 over the last 5 years which means it is more volatile than the overall market.")











# Create a sidebar
st.sidebar.title("Modern investment platform")
st.sidebar.markdown("**Tutorial 1:**")

# Add buttons to the sidebar
if st.sidebar.button("Introduction"):
    page1()
if st.sidebar.button("Person 1"):
    page2()
if st.sidebar.button("Person 2"):
    page3()
