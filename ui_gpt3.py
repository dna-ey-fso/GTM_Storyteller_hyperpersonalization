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
      st.markdown("**Balance:** 963,8 $")
      st.markdown("**Invested:** 1000 $")
      st.markdown("**Cash:** 0$")
      st.markdown("**Total:** 963,8 $")
      st.markdown("**Return:** -3.62 %")

   st.markdown("**Summary of your portfolio during the last 60 days:**")
   st.markdown("Your portfolio of stocks had a return of -3.62% over the last 60 days. This was worse than the world index which gained 9.88% and the S&P500 which gained 7.25%. Some of the stocks in your portfolio had good news, like Johnson & Johnson, which gained 7.89%, but others had bad news, like Amazon, which lost 26.46%.")

   sentiment = nlp("Your portfolio of stocks had a return of -3.62% over the last 60 days. This was worse than the world index which gained 9.88% and the S&P500 which gained 7.25%. Some of the stocks in your portfolio had good news, like Johnson & Johnson, which gained 7.89%, but others had bad news, like Amazon, which lost 26.46%.")

   st.markdown('The performance seems: {}'.format(sentiment[0]['label']))

   with st.expander("Full report"):
      st.markdown("The investor's portfolio has underperformed compared to the world index and S&P 500 in the last 60 days. The portfolio consists of six stocks, namely Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), Google (GOOG), Visa (V) and Johnson & Johnson (JNJ), each having a weight of 20%, 10%, 20%, 20%, 10% and 10% respectively. Apple has seen a decline of 7.5%, Microsoft has seen a decline of 1.28%, Amazon has seen a decline of 26.46%, Google has seen a decline of 10.18%, Visa has seen an increase of 16.65%, and Johnson & Johnson has seen an increase of 7.89%. These individual performances have caused the portfolio's overall performance to decline by -3.62%. The world index has increased by 9.88% and the S&P 500 has increased by 7.25% in the same period.")
 



   chart_data = pd.DataFrame(
    [1,3,4,2,0,-2,-2,-1,2,2,4,5,4,3,5,6,7,5,6,7,9,6,5,7,8,9,10,14,16,15,19,21,21,23,21,21,21,24,24,18,16,14,14,15,12,13,12,12,9,6,11,5,1,-1,0, -1,-3],
    columns=['Portfolio'])

   st.line_chart(chart_data)

   st.markdown("**Possible questions to which you want answers:**")
   #question 1
   with st.expander("What stocks make up the investor's portfolio?"):
      st.markdown("The investor's portfolio consists of Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), Google (GOOG), Visa (V) and Johnson & Johnson (JNJ).")
      
   #question 2
   with st.expander("What are the respective weights of each stock in the portfolio?"):
      st.markdown("The respective weights of each stock in the portfolio are 20%, 10%, 20%, 20%, 10% and 10%, respectively.")

   #question 3
   with st.expander("What has been the performance of the portfolio in the last 60 days?"):
      st.markdown("The portfolio's overall performance has declined by -3.62% in the last 60 days.")
   #question 4
   with st.expander("How has the performance of the portfolio compared to the world index and S&P 500 in the last 60 days?"):
      st.markdown("The portfolio has underperformed compared to the world index and S&P 500 in the last 60 days. The world index has increased by 9.88% and the S&P 500 has increased by 7.25% in the same period.")

   #question 5
   with st.expander("What has been the performance of each stock in the portfolio in the last 60 days?"):
      st.markdown("Apple has seen a decline of 7.5%, Microsoft has seen a decline of 1.28%, Amazon has seen a decline of 26.46%, Google has seen a decline of 10.18%, Visa has seen an increase of 16.65%, and Johnson & Johnson has seen an increase of 7.89% in the last 60 days.")









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
      st.markdown("**Balance:** 963,8 $")
      st.markdown("**Invested:** 1000 $")
      st.markdown("**Cash:** 0$")
      st.markdown("**Total:** 963,8 $")
      st.markdown("**Return:** -3.62 %")

   #vanaf hier beginnen aanpasseo
   st.markdown("**Summary of your portfolio during the last 60 days:**")
   st.markdown("Over the last 60 days, the investor's portfolio had a negative return of -3.62%. The portfolio was comprised of Apple (AAPL) at 20%, Microsoft (MSFT) at 20%, Amazon (AMZN) at 20%, Google (GOOG) at 20%, Visa (V) at 10%, and Johnson & Johnson (JNJ) at 10%. Apple had a negative return of -7.5%, Microsoft had a negative return of -1.28%, Amazon had a negative return of -26.46%, Google had a negative return of -10.18%, Visa had a positive return of 16.65%, and Johnson & Johnson had a positive return of 7.89%. The world index rose 9.88% and the S&P500 rose 7.25% during the same period.")

   sentiment = nlp("Over the last 60 days, the investor's portfolio had a negative return of -3.62%. The portfolio was comprised of Apple (AAPL) at 20%, Microsoft (MSFT) at 20%, Amazon (AMZN) at 20%, Google (GOOG) at 20%, Visa (V) at 10%, and Johnson & Johnson (JNJ) at 10%. Apple had a negative return of -7.5%, Microsoft had a negative return of -1.28%, Amazon had a negative return of -26.46%, Google had a negative return of -10.18%, Visa had a positive return of 16.65%, and Johnson & Johnson had a positive return of 7.89%. The world index rose 9.88% and the S&P500 rose 7.25% during the same period.")
   st.markdown('The performance seems: {}'.format(sentiment[0]['label']))

   with st.expander("Full report"):
      st.markdown("Over the last 60 days, the investor's portfolio had a negative return of -3.62%, compared to the world index which rose 9.88% and the S&P500 which rose 7.25%. The portfolio was comprised of Apple (AAPL) at 20%, Microsoft (MSFT) at 20%, Amazon (AMZN) at 20%, Google (GOOG) at 20%, Visa (V) at 10%, and Johnson & Johnson (JNJ) at 10%. Apple had a negative return of -7.5%, Microsoft had a negative return of -1.28%, Amazon had a negative return of -26.46%, Google had a negative return of -10.18%, Visa had a positive return of 16.65%, and Johnson & Johnson had a positive return of 7.89%. The news for each company was varied, with Apple involved in a trial production of its new iPhone 15, Microsoft looking at OpenAI's GPT for Word, Outlook, and PowerPoint, Amazon announcing it was axing 18,000 jobs, Google Messages baking Material You theming into its account switcher, Visa introducing a new Digital Nomad Visa Calculator, and Johnson & Johnson filing for an IPO of its Consumer Health Division Kenvue.")
      

   chart_data = pd.DataFrame([1,3,4,2,0,-2,-2,-1,2,2,4,5,4,3,5,6,7,5,6,7,9,6,5,7,8,9,10,14,16,15,19,21,21,23,21,21,21,24,24,18,16,14,14,15,12,13,12,12,9,6,11,5,1,-1,0, -1,-3],columns=['Portfolio'])

   st.line_chart(chart_data)

   st.markdown("**Possible questions to which you want answers:**")
   #question 1
   with st.expander("What was the overall return of the investor's portfolio over the last 60 days?"):
      st.markdown("The overall return of the investor's portfolio over the last 60 days was -3.62%.")
      
   #question 2
   with st.expander("What were the specific returns of each of the stocks in the portfolio?"):
      st.markdown("The specific returns of each of the stocks in the portfolio were Apple (AAPL) at -7.5%, Microsoft (MSFT) at -1.28%, Amazon (AMZN) at -26.46%, Google (GOOG) at -10.18%, Visa (V) at 16.65%, and Johnson & Johnson (JNJ) at 7.89%.")

   #question 3
   with st.expander("What was the news related to each of the stocks in the portfolio?"):
      st.markdown("The news related to each of the stocks in the portfolio was Apple involved in a trial production of its new iPhone 15, Microsoft looking at OpenAI's GPT for Word, Outlook, and PowerPoint, Amazon announcing it was axing 18,000 jobs, Google Messages baking Material You theming into its account switcher, Visa introducing a new Digital Nomad Visa Calculator, and Johnson & Johnson filing for an IPO of its Consumer Health Division Kenvue.")
   #question 4
   with st.expander("How did the world index and the S&P500 perform over the same period?"):
      st.markdown("The world index rose 9.88% and the S&P500 rose 7.25% over the same period.")

   #question 5
   with st.expander("What is the forward P/E, PEG Ratio (5 yr expected), Price/Sales (ttm), Price/Book (mrq), Enterprise Value/Revenue, and Enterprise Value/EBITDA for each of the stocks in the portfolio?"):
      st.markdown("The forward P/E, PEG Ratio (5 yr expected), Price/Sales (ttm), Price/Book (mrq), Enterprise Value/Revenue, and Enterprise Value/EBITDA for each of the stocks in the portfolio is Apple (AAPL) at 21.83, 2.60, 5.52, 41.88, 5.56, 16.48; Microsoft (MSFT) at 25.00, 2.1, 8.83, 10.24, 8.52, 17.17; Amazon (AMZN) at 49.51, 5.58, 1.94, 7.70, 2.07, 20.55; Google (GOOG) at 17.18, 1.3, 4.33, 4.69, 3.89, 11.65; Visa (V) at 26.74, 1.70, 16.26, 13.28, 16.25, 24.38; and Johnson & Johnson (JNJ) at 16.67, 3.74, 4.82, 6.08, 4.70, 15.27.")











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
