import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import numpy as np
import time
import os
import openai



# API key for gpt 3  !! Do not overkill please I only have 18$ for free :D !!
openai.api_key = "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"



#Part of sentiment analysis of the performance
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)



def portfolio_performance():
   #Get portfolio performance in real life -> so look at all the stocks their portfolio and their weight in the fund
   performance_value = -3.62

   text = "It appears that your portfolio had a negative return of {performance:.2f} % over the last 60 days"
   return text.format(performance = performance_value)

def portfolio_data_beginner():
   #get performance of s&p500 and world index over the same period in time -> insert thin into the text
   #Do an api call
   #Get the data for all the stocks of the porfolio
   #Put it in a panda framework
   #return the data
   data = [['Stock name', 'Ticker', 'Weight in fund', 'Changed value in the last 60 days', 'news', 'sector' ],['Apple', 'AAPL', '20%', '-7.5%', 'Apple iPhone 15 trial production commences in China - GSMArena.com news', 'Information Technology'], ['Microsoft', 'MSFT', '20%', '-1.28%', 'Microsoft is looking at OpenAIs GPT for Word, Outlook, and PowerPoint', 'Information Technology'],['Amazon', 'AMZN', '20%', '-26.46%', 'Amazon to axe 18,000 jobs as it cuts costs', 'Consumer Cyclical'], ['Google', 'GOOG', '20%', '-10.18%', 'Google Messages bakes Material You theming into its account switcher', 'Information Technology'],['Visa', 'V', '10%', '16.65%', 'New Digital Nomad Visa Calculator Takes The Hassle Out Of Choosing Where To Work And Live', 'Financial Services'], ['Johnson & Johnson', 'JNJ', '10%', '7.89%', 'Johnson & Johnson Files for IPO of Consumer Health Division Kenvue', 'Healthcare']]

   #df = pd.DataFrame(data, columns=['Stock name', 'Ticker', 'Weight in fund', 'Changed value in the last 60 days', 'news', 'sector','beta(5y Montly)', 'Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales (ttm)', 'Price/Book (mrq)', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA' ])

   return data

def portfolio_data_advanced_investor():
   #Hierbij worden ook andere extra financial parameters in kaart gebracht

   #get performance of s&p500 and world index over the same period in time -> insert thin into the text
   #Do an api call
   #Get the data for all the stocks of the porfolio
   #Put it in a panda framework
   #return the data
   data = [['Stock name', 'Ticker', 'Weight in fund', 'Changed value in the last 60 days', 'news', 'sector','beta(5y Montly)', 'Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales (ttm)', 'Price/Book (mrq)', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA' ],['Apple', 'AAPL', '20%', '-7.5%', 'Apple iPhone 15 trial production commences in China - GSMArena.com news', 'Information Technology', 1.27, 21.83, 2.60, 5.52, 41.88, 5.56, 16.48], ['Microsoft', 'MSFT', '20%', '-1.28%', 'Microsoft is looking at OpenAIs GPT for Word, Outlook, and PowerPoint', 'Information Technology', 0.93, 25.00,2.1, 8.83, 10.24, 8.52, 17.17],['Amazon', 'AMZN', '20%', '-26.46%', 'Amazon to axe 18,000 jobs as it cuts costs', 'Consumer Cyclical', 1.17, 49.51, 5.58, 1.94, 7.70, 2.07, 20.55], ['Google', 'GOOG', '20%', '-10.18%', 'Google Messages bakes Material You theming into its account switcher', 'Information Technology',1.07, 17.18, 1.3, 4.33, 4.69, 3.89, 11.65 ],['Visa', 'V', '10%', '16.65%', 'New Digital Nomad Visa Calculator Takes The Hassle Out Of Choosing Where To Work And Live', 'Financial Services', 0.94, 26.74, 1.70, 16.26, 13.28, 16.25, 24.38], ['Johnson & Johnson', 'JNJ', '10%', '7.89%', 'Johnson & Johnson Files for IPO of Consumer Health Division Kenvue', 'Healthcare', 0.57, 16.67, 3.74, 4.82, 6.08, 4.70, 15.27]]

   #df = pd.DataFrame(data, columns=['Stock name', 'Ticker', 'Weight in fund', 'Changed value in the last 60 days', 'news', 'sector','beta(5y Montly)', 'Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales (ttm)', 'Price/Book (mrq)', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA' ])

   return data

def index_performance():
   snp500_performance = 9.88
   global_performance = 7.25

   text = "The wold index raised {performance1:.2f}% and the s&p500 raised {performance2:.2f}% during last 60 days"
   return text.format(performance1 = snp500_performance, performance2 = global_performance)

def get_balance():
   #Get portfolio performance in real life -> so look at all the stocks their portfolio and their weight in the fund
   performance_value = -3.62
   percentage = (100+performance_value)/100
   balance = 1000*percentage

   return balance

def get_return():
   #Get portfolio performance in real life -> so look at all the stocks their portfolio and their weight in the fund
   performance_value = -3.62
   return performance_value

def get_gtp3_brieftext(input_text):
   #api call for gpt3
   response = openai.Completion.create(
      model="text-davinci-003",
      prompt = input_text, 
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
   return response.choices[0].text.strip()


def get_gtp3_longtext(input_text):
   #api call for gpt3
   response = openai.Completion.create(
      model="text-davinci-003",
      prompt = input_text, 
      temperature=0.7,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
   return response.choices[0].text.strip()

def get_questions(text_data):
   response = openai.Completion.create(
      model="text-davinci-003",
      prompt = text_data, 
      temperature=0.7,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
   text = response.choices[0].text.strip()
   questions = text.split("\n")

 

   #returns a list 
   return questions
   


# Tutorial 1 pages

# introduction page of tutorial 1
def page1(portfolio_performance_text, portfolio_data_text, index_performance_text):
    st.title("Introduction tutorial 1: ")
    st.write("Two different people investing in the same stocks can have completely different experiences.")
    st.markdown("**Customer 1:** Bert is a non-experienced investor who has deposited a small amount of money and is personally investing for entertainment and to learn a little about how the stock markets work. Learning and having fun is his top priority; performance of his portfolio comes second.")
    st.markdown("**Customer 2:** Ann works in finance and knows a lot about the stock market; she has no need for additional information. She wants to get a concise and clear overview of how her portfolio performs and wants to understand financials of the companies. ")
    st.markdown("**Portfolio (all the data that is used as input fot the NLP models):**")

   #print finance info
    st.markdown(portfolio_performance_text)
    df = pd.DataFrame(portfolio_data_text)
    st.table(df)
    st.markdown(index_performance_text)








#page 2 of the streamlit ui -> with Bart his portfolio (non experienced investor)
def page2(portfolio_performance_text, portfolio_data_text, index_performance_text):
   st.title("Customer 1: Bart")
   col1, col2 = st.columns(2)

   with col1:
      image = Image.open("image.png")
      st.image(image, width=200, )

   st.empty() # creates an empty space

   with col2:
         # Get real time value of investments (Started with 1000$ in this example)
      balance = get_balance()
      return_portfolio = get_return()
      st.markdown("**Balance:** {} $".format(balance))
      st.markdown("**Invested:** 1000 $")
      st.markdown("**Cash:** 0$")
      st.markdown("**Total:** {} $".format(balance))
      st.markdown("**Return:** {} %".format(return_portfolio))

   st.markdown("**Summary of your portfolio:**")

   #Send financial data + correct input data to the gpt3 api

   #api call to get the brief summary of the portfolio performance
      #transform the list into a string 
   portfolio_data_text = str(portfolio_data_text)
      #add al the texts together to one big text
   input_text_summary = portfolio_performance_text + portfolio_data_text + index_performance_text
      #add the extra message for gpt 3 to it
   input_text_brief_summary = "Can you give me a brief summary of the perfomance of this portfolio :" + input_text_summary + "."
      # api call brief summary
   output_text_brief_summary = get_gtp3_brieftext(input_text_brief_summary)

       #add the extra message for gpt 3 to it
   input_text_long_summary = "Can you give me an extended summary of the perfomance of this portfolio :" + input_text_summary + "."
      # api call extended summary
   output_text_long_summary = get_gtp3_longtext(input_text_long_summary)


   #streamlit show brief summarry
   st.markdown(output_text_brief_summary)

   sentiment = nlp(output_text_brief_summary)

   st.markdown('The performance seems: {}'.format(sentiment[0]['label']))

   with st.expander("Full report"):
      st.markdown(output_text_long_summary)
 



   chart_data = pd.DataFrame(
    [1,3,4,2,0,-2,-2,-1,2,2,4,5,4,3,5,6,7,5,6,7,9,6,5,7,8,9,10,14,16,15,19,21,21,23,21,21,21,24,24,18,16,14,14,15,12,13,12,12,9,6,11,5,1,-1,0, -1,-3, return_portfolio],
    columns=['Portfolio'])



   #Get 5 question that are interesting:
   question_list = get_questions("Can you give me 5 questions that you can can answer with the next information that are interestings for a beginner investor + the answer you would give:" + input_text_summary)


   st.line_chart(chart_data)

   st.markdown("**Possible questions to which you want answers:**")
   #question 1
   with st.expander(question_list[0]):
      st.markdown(question_list[1])
      
   #question 2
   with st.expander(question_list[3]):
      st.markdown(question_list[4])

   #question 3
   with st.expander(question_list[6]):
      st.markdown(question_list[7])
   #question 4
   with st.expander(question_list[9]):
      st.markdown(question_list[10])

   #question 5
   with st.expander(question_list[12]):
      st.markdown(question_list[13])

   st.markdown(question_list)



# ______________ Ann haar pagina 
def page3(portfolio_performance_text, portfolio_data_text, index_performance_text):
   st.title("Customer 2: Ann")
   
   col1, col2 = st.columns(2)

   with col1:
      image = Image.open("image2.png")
      st.image(image, width=200, )

   st.empty() # creates an empty space

   with col2:
         # Get real time value of investments (Started with 1000$ in this example)
      balance = get_balance()
      return_portfolio = get_return()
      st.markdown("**Balance:** {} $".format(balance))
      st.markdown("**Invested:** 1000 $")
      st.markdown("**Cash:** 0$")
      st.markdown("**Total:** {} $".format(balance))
      st.markdown("**Return:** {} %".format(return_portfolio))


   
   st.markdown("**Summary of your portfolio:**")

   #Send financial data + correct input data to the gpt3 api

   #api call to get the brief summary of the portfolio performance
      #transform the list into a string 
   portfolio_data_text = str(portfolio_data_text)
      #add al the texts together to one big text
   input_text_summary = portfolio_performance_text + portfolio_data_text + index_performance_text
      #add the extra message for gpt 3 to it
   input_text_brief_summary = "Can you give me a brief summary of the perfomance of this portfolio for an professional investor:" + input_text_summary + "."
      # api call brief summary
   output_text_brief_summary = get_gtp3_brieftext(input_text_brief_summary)

       #add the extra message for gpt 3 to it
   input_text_long_summary = "Can you give me an extended summary of the perfomance of this portfolio for an professional investor:" + input_text_summary + "."
      # api call extended summary
   output_text_long_summary = get_gtp3_longtext(input_text_long_summary)

      #streamlit show brief summarry
   st.markdown(output_text_brief_summary)

   sentiment = nlp(output_text_brief_summary)

   st.markdown('The performance seems: {}'.format(sentiment[0]['label']))

   with st.expander("Full report"):
      st.markdown(output_text_long_summary)



   chart_data = pd.DataFrame([1,3,4,2,0,-2,-2,-1,2,2,4,5,4,3,5,6,7,5,6,7,9,6,5,7,8,9,10,14,16,15,19,21,21,23,21,21,21,24,24,18,16,14,14,15,12,13,12,12,9,6,11,5,1,-1,0, -1,-3],columns=['Portfolio'])



   st.line_chart(chart_data)


   #Get 5 question that are interesting:
   question_list = get_questions("Can you give me 5 questions that you can can answer with the next information that are interestings for a prefossional investor + the answer you would give:" + input_text_summary)


   st.markdown("**Possible questions to which you want answers:**")
   #question 1


   question1 = str(question_list[0])
   question2 = str(question_list[1])
   question3 = str(question_list[3])
   question4 = str(question_list[4])
   question5 = str(question_list[6])
   question6 = str(question_list[7])
   question7 = str(question_list[9])
   question8 = str(question_list[10])
   question9 = str(question_list[12])
   question10 = str(question_list[13])

   with st.expander(question1):
      st.markdown(question2)
      
   #question 2
   with st.expander(question3):
      st.markdown(question4)

   #question 3
   with st.expander(question5):
      st.markdown(question6)
   #question 4
   with st.expander(question7):
      st.markdown(question8)

   #question 5
   with st.expander(question9):
      st.markdown(question10)

   st.markdown(question_list)







# Create a sidebar
st.sidebar.title("Modern investment platform")
st.sidebar.markdown("**Tutorial 1:**")

portfolio_performance_text = portfolio_performance()
#call the create data method:
portfolio_beginner_data_text = portfolio_data_beginner()
portfolio_advanced_data_text = portfolio_data_advanced_investor()

#data = get_data()
index_performance_text = index_performance()

# Add buttons to the sidebar
if st.sidebar.button("Introduction"):
    page1(portfolio_performance_text, portfolio_advanced_data_text, index_performance_text)
if st.sidebar.button("Person 1"):
    page2(portfolio_performance_text, portfolio_beginner_data_text, index_performance_text)
if st.sidebar.button("Person 2"):
    page3(portfolio_performance_text, portfolio_advanced_data_text, index_performance_text)
