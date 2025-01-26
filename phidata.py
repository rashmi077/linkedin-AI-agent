import os 
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from datetime import datetime


# Load environment
load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

## New Generation Agent
web_search_agent=Agent(
    name="AI News Linkedin Curator",
    role="Create a professional Linkedin post about the latest AI developments",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=[
        "Format news as a compelling linkedin post",
          "Include 3-5 key  AI news developments",
          "Write in a professional engaging tone",
          "Use bullet points for readability",
          "Include relevent hashtags",
          "Provide source links for credibility",
          "Highlight the broader impact of AI development",
          "End with an engagement prompt ",
          "Ensure content is suitable for professional networking audience",
          "Keep the total lenght under 3000 characters",
    ],
    show_tools_calls=True,
    markdown=True,
)
# New Relevence Agent
news_relevance_agent=Agent(
    name="News Relevance Validator",
    role="Creitically evaluate AI news for social media posting ",
    model=Groq(id ="llama-3.3-70b-versatile"),

instructions=[
    "Carefully asses the generated AI news content",
    "Determine if the content is suitable for Linkedin posting ",
    """Check for:
        -Professionalism 
        -Current relevance
        - Potenial impact
        -Absence of controversial content""",
        """Provide a structured evalution with:
            -Suitablity score (0-10)
            -Posting recommendation(Yes/No)
            -Specific reasons for evaluation""",
        "If not suitable,explain specfic reasons ",
        "Suggest potenial modifications if needed ",
        "Respond with 'no' in the posting recommendation if conent is not suitable"
    ],

    show_tools_calls=True ,
    markdown=True
    

)

def main():
    #Generate AI news content
    news_response=web_search_agent.run("5 latest significant AI news developments with sources",steam=False)

    # validate the generated news content
    validation_response=news_relevance_agent.run(
        f"Evaluate the following AI content for linkedin posting suitability:\n\n{news_response.content}",
        stream=False
    )
    #Check if validation recommends not posting 
    news_content=news_response.content
    if "< function=duckduckgo_news" in validation_response.content:
        news_content = "" 
     
    else:
        news_content=news_response.content 
 
    return {
        "news_content":news_content,
        "validation":validation_response.content
    }
if __name__== '__main__':
    result=main()
    print("generated News:")
    print(result['news_content'])
    print("\nValidation Result:")
    print(result['validation'])