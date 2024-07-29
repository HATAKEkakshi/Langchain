import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


prompt="""you are youtube video summarizer. you will be taking the transcript text and summarizing the entire video and providing the important summary in points within 250 words. Please provide the summary of the text"""
## getting the transcript from the youtube video transcript
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split('=')[1]
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
    
        transcript=""
        for i in transcript_text:
            transcript+=" "+i["text"]
        return transcript
    except Exception as e:
        raise e
   
## generating the summary
def generate_gemini_content(transcript_text,prompt):
    
    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

st.title("Youtube transcript to detailed notes converter")
youtube_link=st.text_input("Enter youtube video link :")

if youtube_link:
    video_id=youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg",use_column_width=True)
if st.button("Get detailed notes"):
    transcript_text=extract_transcript_details(youtube_link)
    print("transcript_text:",transcript_text)
    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        print(summary)
        st.markdown("## Detailed Notes :")
        st.write(summary)