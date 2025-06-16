from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
import httpx
import json
from datetime import datetime
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.responses import StreamingResponse, JSONResponse
import math
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from bson import ObjectId


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler('retirement_api.log', maxBytes=10*1024*1024, backupCount=5)
    ]
)

logger = logging.getLogger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chart_data: Optional[Dict[str, Any]] = None
    contains_chart: bool = False

class SessionCreate(BaseModel):
    user_id: str
    session_name: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    session_name: str
    created_at: datetime
    updated_at: datetime

class MessageResponse(BaseModel):
    id: str
    session_id: str
    user_message: str
    ai_response: str
    chart_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    contains_chart: bool = False
    timestamp: datetime

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None
    current_age: int
    retirement_age: int
    income: float
    salary_growth: float = 0.02
    investment_return: float = 0.05
    contribution_rate: float = 0.1
    pension_multiplier: float = 0.015
    end_age: int = 90
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    ai_retirement_data: Optional[List[Dict]] = None

class UserProfileCreate(BaseModel):
    name: str
    email: Optional[str] = None
    current_age: int
    retirement_age: int
    income: float
    salary_growth: float = 0.02
    investment_return: float = 0.05
    contribution_rate: float = 0.1
    pension_multiplier: float = 0.015
    end_age: int = 90
    social_security_base: float = 18000.0
    pension_base: float = 8000.0
    four01k_base: float = 10000.0
    other_base: float = 4000.0
    defined_benefit_base: float = 14000.0
    defined_benefit_yearly_increase: float = 300.0
    inflation: float = 0.02  # New field for inflation rate
    beneficiary_included: bool = False  # New field
    beneficiary_life_expectancy: Optional[int] = None


class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    current_age: Optional[int] = Field(default=None, ge=18)
    retirement_age: Optional[int] = None
    income: Optional[float] = None
    salary_growth: Optional[float] = None
    investment_return: Optional[float] = None
    contribution_rate: Optional[float] = None
    pension_multiplier: Optional[float] = None
    end_age: Optional[int] = None
    social_security_base: Optional[float] = None
    pension_base: Optional[float] = None
    four01k_base: Optional[float] = None
    other_base: Optional[float] = None
    defined_benefit_base: Optional[float] = None
    defined_benefit_yearly_increase: Optional[float] = None
    inflation: Optional[float] = None
    beneficiary_included: Optional[bool] = None
    beneficiary_life_expectancy: Optional[int] = None

    
class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None
    current_age: int
    retirement_age: int
    income: float
    salary_growth: float
    investment_return: float
    contribution_rate: float
    pension_multiplier: float
    end_age: int
    created_at: datetime
    updated_at: datetime
    social_security_base: float
    pension_base: float
    four01k_base: float
    other_base: float
    defined_benefit_base: float
    defined_benefit_yearly_increase: float
    inflation: float
    beneficiary_included: bool
    beneficiary_life_expectancy: Optional[int]
    retirement_data: List[Dict]
    ai_retirement_data: Optional[List[Dict]] = None

class AIPreferencesRequest(BaseModel):
    user_id: str
    session_id: str

class AISuggestion(BaseModel):
    icon: str
    title: str
    description: str
    questions: List[str]

class AIPreferencesResponse(BaseModel):
    suggestions: List[AISuggestion]

class UserInfo(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None

class LyzrChatRequest(BaseModel):
    user_id: str

class ChatCompResponse(BaseModel):
    session_id: str
    structured_data: Optional[dict] = None

# FastAPI app
app = FastAPI(title="Retirement Planning API", version="1.0.0")
logger.info("FastAPI application initialized")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
logger.info(f"MongoDB URL configured: {MONGODB_URL}")

try:
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client.retirement_chatbot
    logger.info("MongoDB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB client: {str(e)}")
    raise

# Collections
sessions_collection = db.sessions
messages_collection = db.messages
user_profiles_collection = db.user_profiles
logger.info("MongoDB collections configured")

# External API configuration
LYZR_API_KEY = "sk-default-OtrntyKynW1jJaFWjmLbAWpMaXCS3XOM"
LYZR_BASE_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
PENSION_AGENT_ID = "6846d83862d8a0cca7618626"
RETIREMENT_AGENT_ID = "6846d27f62d8a0cca7618607"
logger.info("External API configuration loaded")


def convert_object_ids(data):
    """Recursively convert all ObjectId instances to strings within the dict."""
    if isinstance(data, dict):
        return {k: convert_object_ids(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_object_ids(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data 

def convert_datetimes(data):
    """Recursively convert all datetime instances to ISO 8601 strings within the dict."""
    if isinstance(data, dict):
        return {k: convert_datetimes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_datetimes(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    return data 

async def call_lyzr_api(agent_id: str, session_id: str, user_id: str, message: str) -> dict:
    """Call the Lyzr API"""
    logger.info(f"Starting Lyzr API call - Agent ID: {agent_id}, Session ID: {session_id}, User ID: {user_id}")
    logger.debug(f"Message content: {message[:100]}..." if len(message) > 100 else f"Message content: {message}")
    
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": LYZR_API_KEY,
            }
            
            logger.info(f"Making HTTP request to: {LYZR_BASE_URL}")
            logger.debug(f"Request payload keys: {list(payload.keys())}")
            
            response = await client.post(
                LYZR_BASE_URL,
                json=payload,
                headers=headers,
                timeout=180.0
            )
            
            logger.info(f"Received response with status code: {response.status_code}")
            response.raise_for_status()
            
            response_data = response.json()
            logger.info(f"Received response: {response_data}")

            logger.info("Successfully parsed JSON response from Lyzr API")
            logger.debug(f"Response data keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
            
            return response_data
            
        except httpx.RequestError as e:
            logger.error(f"HTTP request error in Lyzr API call: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error in Lyzr API call: {e.response.status_code} - {str(e)}")
            raise HTTPException(status_code=e.response.status_code, detail=f"API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Lyzr API call: {str(e)}")
            raise

async def stream_lyzr_api(agent_id: str, session_id: str, user_id: str, message: str):
    """Stream response from Lyzr API"""
    logger.info(f"Starting Lyzr API stream - Agent ID: {agent_id}, Session ID: {session_id}, User ID: {user_id}")
    logger.debug(f"Message content: {message[:100]}..." if len(message) > 100 else f"Message content: {message}")
    
    LYZR_STREAM_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/stream/"
    
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": LYZR_API_KEY,
            }
            
            logger.info(f"Making streaming HTTP request to: {LYZR_STREAM_URL}")
            logger.debug(f"Request payload keys: {list(payload.keys())}")
            
            async with client.stream(
                "POST",
                LYZR_STREAM_URL,
                json=payload,
                headers=headers,
                timeout=600.0
            ) as response:
                logger.info(f"Received streaming response with status code: {response.status_code}")
                response.raise_for_status()
                
                async for chunk in response.aiter_text():
                    if chunk:
                        logger.debug(f"Received stream chunk: {chunk[:100]}...")
                        yield chunk
                        
        except httpx.RequestError as e:
            logger.error(f"HTTP request error in Lyzr API stream: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error in Lyzr API stream: {e.response.status_code} - {str(e)}")
            raise HTTPException(status_code=e.response.status_code, detail=f"API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Lyzr API stream: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
def get_latest_retirement_data(user_profile: dict) -> List[Dict]:
    """Get the latest retirement data (AI-generated if available, otherwise manual)"""
    ai_data = user_profile.get('ai_retirement_data', [])
    manual_data = user_profile.get('retirement_data', [])
    
    # If AI data exists and is more recent, use it
    if ai_data:
        logger.info("Using AI-generated retirement data")
        return ai_data
    else:
        logger.info("Using manually calculated retirement data")
        return manual_data
    
def create_pension_prompt(user_message: str, user_profile: dict):
    """Create pension prompt using user profile data"""
    logger.info("Creating pension prompt with user profile data")
    
    # Prepare form data for prompt
    form_data = {
        "retireAge": user_profile['retirement_age'],
        "lifeExpectancy": user_profile['end_age'],
        "beneficiaryIncluded": user_profile['beneficiary_included'],
        "beneficiaryLifeExpectancy": user_profile.get('beneficiary_life_expectancy', "Not specified"),
        "salaryGrowth": user_profile['salary_growth'],
        "investmentReturn": user_profile['investment_return'],
        "inflation": user_profile['inflation'],
    }
    
    # Get existing pension data from profile
    latest_retirement_data = get_latest_retirement_data(user_profile)
    
    prompt = f"""
    Respond only in ENGLISH language

    You are a pension planning assistant for users in the USA. Your role is to answer questions about pensions, provide investment planning advice, and suggest chart updates when relevant.

    Current context:
    - Form data: {json.dumps(form_data)}
    - Pension data: {json.dumps(latest_retirement_data)}

    **Chart Information:**
    The current chart is a stacked bar chart displaying the projected annual retirement income from different sources (Social Security, Pension, 401k, Other, Defined Benefit) for each year from the retirement age to the end of the projection period. The projection period extends to the maximum of the user's life expectancy or the beneficiary's life expectancy if a beneficiary is included.

    **How the Chart Data is Calculated:**
    The income projections are based on the following assumptions from the form data:
    - Retirement age: {form_data.get('retireAge', 'Not specified')}
    - Life expectancy: {form_data.get('lifeExpectancy', 'Not specified')}
    - Beneficiary included: {form_data.get('beneficiaryIncluded', 'Not specified')}
    - Beneficiary life expectancy: {form_data.get('beneficiaryLifeExpectancy', 'Not specified')}
    - Salary growth rate: {form_data.get('salaryGrowth', 'Not specified')}
    - Investment return rate: {form_data.get('investmentReturn', 'Not specified')}
    - Inflation rate: {form_data.get('inflation', 'Not specified')}

    The initial amounts for each income source at the retirement age (age = retireAge) are currently set as:
    - Social Security: ${user_profile['social_security_base']} (starts at age 62)
    - Pension: ${user_profile['pension_base']}
    - 401k: ${user_profile['four01k_base']} (starts at age 58)
    - Other: ${user_profile['other_base']} (starts at age 58)
    - Defined Benefit: ${user_profile['defined_benefit_base']} (starts at age 62)

    For each subsequent year after retirement, these amounts grow as follows:
    - Social Security grows with the inflation rate (starting at age 62).
    - Pension grows with the salary growth rate.
    - 401k grows with the investment return rate until age 65, then decreases by 5% per year.
    - Other grows with the investment return rate (starting at age 58).
    - Defined Benefit increases by ${user_profile['defined_benefit_yearly_increase']} per year starting at age 62.

    The calculation for each income source at a given age is:
    - Social Security: age < 62 ? 0 : {user_profile['social_security_base']} * (1 + inflation)^(age - 62)
    - Pension: {user_profile['pension_base']} * (1 + salaryGrowth)^(age - retireAge)
    - 401k: age < 58 ? 0 : (age <= 65 ? {user_profile['four01k_base']} * (1 + investmentReturn)^(age - 58) : {user_profile['four01k_base']} * (1 + investmentReturn)^(65 - 58) * (0.95)^(age - 65))
    - Other: age < 58 ? 0 : {user_profile['other_base']} * (1 + investmentReturn)^(age - 58)
    - Defined Benefit: age < 62 ? 0 : {user_profile['defined_benefit_base']} + (age - 62) * {user_profile['defined_benefit_yearly_increase']}

    User question: {user_message}

    Provide your entire response as a single JSON object with the following structure:
    {{
      "text_response": "Your textual answer here",
      "chart_data": null or {{"type": "pension_data", "data": [ {{ "age": number, "Social Security": number, "Pension": number, "401k": number, "Other": number, "Defined Benefit": number }}, ... ]}},
      "contains_chart": true or false
    }}

    **Instructions for Chart Updates:**

    - Set "contains_chart" to true only if the user's question implies a need to visualize data or update the chart.
    - If "contains_chart" is true, "chart_data" must be a JSON string containing the chart data in the specified format.
    - If "contains_chart" is false, set "chart_data" to null.
    - Provide a "text_response" with reasoning to user query.
    """ 
    
    logger.info(f"Pension prompt created, total length: {len(prompt)}")
    return prompt

async def save_ai_retirement_data(user_id: str, ai_retirement_data: List[Dict]):
    """Save AI-generated retirement data to user profile"""
    logger.info(f"Saving AI retirement data for user: {user_id}")
    
    try:
        update_result = await user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "ai_retirement_data": ai_retirement_data,
                "updated_at": datetime.utcnow()
            }}
        )
        
        if update_result.modified_count > 0:
            logger.info(f"AI retirement data saved successfully for user: {user_id}")
        else:
            logger.warning(f"No document updated for user: {user_id}")
    except Exception as e:
        logger.error(f"Error saving AI retirement data: {str(e)}")
        raise
    
async def save_message(session_id: str, user_message: str, ai_response: str, chart_data: Optional[Dict[str, Any]] = None, contains_chart: bool = False):
    """Save a message to the database"""
    logger.info(f"Saving message to database - Session ID: {session_id}")
    logger.debug(f"User message length: {len(user_message)}")
    logger.debug(f"AI response length: {len(ai_response)}")
    logger.debug(f"Contains chart: {contains_chart}")
    
    try:
        message_doc = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_message": user_message,
            "ai_response": ai_response,
            "chart_data": chart_data,
            "contains_chart": contains_chart,
            "timestamp": datetime.utcnow()
        }
        
        logger.debug(f"Generated message ID: {message_doc['id']}")
        
        result = await messages_collection.insert_one(message_doc)
        logger.info(f"Message saved successfully with MongoDB ID: {result.inserted_id}")
        
        return message_doc
    except Exception as e:
        logger.error(f"Failed to save message to database: {str(e)}")
        raise

def estimate_ss(income):
    monthly = income / 12
    AIME = min(monthly, 11300)
    PIA = 0
    if AIME <= 1174:
        PIA = AIME * 0.9
    elif AIME <= 7078:
        PIA = 1174 * 0.9 + (AIME - 1174) * 0.32
    else:
        PIA = 1174 * 0.9 + (7078 - 1174) * 0.32 + (AIME - 7078) * 0.15
    return PIA * 12

def calculate_ss(age, base_ss, FRA=67):
    if age < 62:
        return 0
    if age < FRA:
        reduction = 0.067 * (FRA - age)
        return base_ss * (1 - reduction)
    elif age <= 70:
        increase = 0.08 * (age - FRA)
        return base_ss * (1 + increase)
    else:
        return base_ss * 1.32

def calculate_401k_balance(age, income, contribution_rate, investment_return, retirement_age, balance=0):
    for a in range(25, age):
        if a < retirement_age:
            balance += income * contribution_rate
            balance *= (1 + investment_return)
    return balance

def calculate_other_investments(age, investment_return, retirement_age, base=200):
    if age < retirement_age:
        return base * pow(1 + investment_return, age - 25)
    else:
        return base * pow(1 + investment_return, retirement_age - 25) * pow(0.95, age - retirement_age)

def calculate_pension(age, income, retirement_age, salary_growth, pension_multiplier, start_work_age=25):
    if age < retirement_age:
        return 0
    years_worked = retirement_age - start_work_age
    final_avg_salary = income * pow(1 + salary_growth, years_worked)
    return years_worked * final_avg_salary * pension_multiplier

def generate_retirement_data(user_profile: dict):
    """Generate retirement data matching prompt formulas"""
    years = list(range(user_profile['retirement_age'], user_profile['end_age'] + 1))
    retirement_data = []
    max_ss_benefit = 58476.0  

    for age in years:
        try:
            # Social Security calculation
            ss_value = 0
            if age >= 62:
                ss_value = user_profile['social_security_base'] * (1 + user_profile['inflation']) ** (age - 62)
                if math.isinf(ss_value) or math.isnan(ss_value):
                    ss_value = max_ss_benefit
                ss_value = min(ss_value, max_ss_benefit)

            # Pension calculation
            pension_value = user_profile['pension_base'] * \
                        (1 + user_profile['salary_growth']) ** (age - user_profile['retirement_age'])

            # 401k calculation
            four01k_value = 0
            if age >= 58:
                if age <= 65:
                    four01k_value = user_profile['four01k_base'] * \
                                (1 + user_profile['investment_return']) ** (age - 58)
                else:
                    base_at_65 = user_profile['four01k_base'] * \
                                (1 + user_profile['investment_return']) ** (65 - 58)
                    four01k_value = base_at_65 * (0.95) ** (age - 65)

            # Other calculation
            other_value = 0
            if age >= 58:
                other_value = user_profile['other_base'] * \
                            (1 + user_profile['investment_return']) ** (age - 58)

            # Defined Benefit calculation
            defined_benefit_value = 0
            if age >= 62:
                defined_benefit_value = user_profile['defined_benefit_base'] + \
                                    user_profile['defined_benefit_yearly_increase'] * (age - 62)

            retirement_data.append({
                "age": age,
                "Social Security": round(ss_value, 2),
                "Pension": round(pension_value, 2),
                "401k": round(four01k_value, 2),
                "Other": round(other_value, 2),
                "Defined Benefit": round(defined_benefit_value, 2)
            })
        except (OverflowError, ValueError) as e:
            logger.error(f"Error in generating retirement data: {e}")
            continue

    return retirement_data

async def get_user_chat_history(user_id: str, session_id: str) -> str:
    """Get formatted chat history for a user session"""
    try:
        cursor = messages_collection.find({
            "session_id": session_id
        }).sort("timestamp", 1).limit(10)
        
        chat_history = []
        async for message in cursor:
            chat_history.append(f"User: {message['user_message']}")
            chat_history.append(f"Assistant: {message['ai_response']}")
        
        return "\n".join(chat_history) if chat_history else "No previous chat history"
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return "Error retrieving chat history"

async def generate_personalized_suggestions(user_id: str, session_id: str, user_profile: dict) -> List[AISuggestion]:
    """Generate personalized AI suggestions based on user profile and chat history"""
    try:
        # Get chat history
        chat_history = await get_user_chat_history(user_id, session_id)
        
        # Create prompt for AI agent
        prompt = f"""
        Based on the user's retirement profile and chat history, generate 5 personalized retirement planning suggestions.
        
        User Profile:
        - Name: {user_profile.get('name', 'User')}
        - Current Age: {user_profile.get('current_age', 30)}
        - Retirement Age: {user_profile.get('retirement_age', 65)}
        - Current Income: ${user_profile.get('income', 700):,}
        - Investment Return Rate: {user_profile.get('investment_return', 0.06)*100}%
        - Contribution Rate: {user_profile.get('contribution_rate', 0.1)*100}%
        
        Recent Chat History:
        {chat_history}
        
        Generate personalized suggestions with titles, descriptions, and 3 specific questions for each category:
       1. Optimize Retirement Plan
       2. Investment Recommendations
       3. Smart Insights
       4. Generate Report
       5. Ask Questions
        
        Make the suggestions specific to this user's profile and chat history. Format as JSON with structure:
        [
            {{"icon": "calculator", "title": "Optimize Retirement Plan", "description": "...", "questions": ["...", "...", "..."]}},
            {{"icon": "trending-up", "title": "Investment Recommendations", "description": "...", "questions": ["...", "...", "..."]}},
            {{"icon": "lightbulb", "title": "Smart Insights", "description": "...", "questions": ["...", "...", "..."]}},
            {{"icon": "file-text", "title": "Generate Report", "description": "...", "questions": ["...", "...", "..."]}},
            {{"icon": "help-circle", "title": "Ask Questions", "description": "...", "questions": ["...", "...", "..."]}}
        ]
        """
        
        # Call the AI agent to generate suggestions
        api_response = await call_lyzr_api(
            agent_id="684a8fcfe5203d8a7b64825e",
            session_id=session_id,
            user_id=user_id,
            message=prompt
        )
        
        # Parse the response and create suggestions
        try:
            suggestions_data = json.loads(api_response.get("response", "[]"))
            suggestions = []
            
            for suggestion in suggestions_data:
                suggestions.append(AISuggestion(
                    icon=suggestion.get("icon", "help-circle"),
                    title=suggestion.get("title", ""),
                    description=suggestion.get("description", ""),
                    questions=suggestion.get("questions", [])
                ))
            
            return suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI suggestions response: {str(e)}")
            return get_default_suggestions()
    except Exception as e:
        logger.error(f"Error generating personalized suggestions: {str(e)}")
        return get_default_suggestions()
    
def get_default_suggestions() -> List[AISuggestion]:
    """Return default suggestions if personalized generation fails"""
    return [
        AISuggestion(
            icon="calculator",
            title="Optimize Retirement Plan",
            description="Get AI suggestions to improve your retirement strategy",
            questions=[
                "Please analyze my current retirement plan and provide specific recommendations to optimize my retirement income.",
                "What adjustments should I make to my retirement savings rate to reach my income goals more effectively?",
                "How can I rebalance my retirement portfolio to reduce risk while maintaining growth potential?"
            ]
        ),
        AISuggestion(
            icon="trending-up",
            title="Investment Recommendations",
            description="Discover better investment options based on your profile",
            questions=[
                "Based on my retirement planning data, what investment strategies would you recommend to maximize my retirement income?",
                "What are the best low-cost index funds or ETFs I should consider for my retirement portfolio?",
                "How should I adjust my investment strategy as I get closer to retirement age?"
            ]
        ),
        AISuggestion(
            icon="lightbulb",
            title="Smart Insights",
            description="Understand key factors affecting your retirement income",
            questions=[
                "What are the key factors that most significantly impact my retirement income projections?",
                "How does inflation affect my retirement planning and what strategies can protect against it?",
                "What are the tax implications of my current retirement strategy and how can I optimize for tax efficiency?"
            ]
        ),
        AISuggestion(
            icon="file-text",
            title="Generate Report",
            description="Create a detailed retirement planning report",
            questions=[
                "Please generate a comprehensive retirement planning report based on my current assumptions.",
                "Create a detailed analysis of my retirement income sources and their sustainability over time.",
                "Generate a risk assessment report for my retirement plan including potential scenarios."
            ]
        ),
        AISuggestion(
            icon="help-circle",
            title="Ask Questions",
            description="Get answers about retirement planning strategies",
            questions=[
                "I have questions about retirement planning strategies. Can you help me understand the best practices?",
                "What are the most common retirement planning mistakes and how can I avoid them?",
                "How do I plan for healthcare costs in retirement and what options should I consider?"
            ]
        )
    ]

def sanitize_response_data(data):
    """Recursively sanitize all out-of-range float values in the data."""
    if isinstance(data, dict):
        return {k: sanitize_response_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_response_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None  # or 0, depending on your application's needs
    return data

# API Routes

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Retirement Planning API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_pension(request: ChatRequest):
    """Chat endpoint for pension planning"""
    logger.info(f"Pension chat request received - Session ID: {request.session_id}, User ID: {request.user_id}")
    logger.debug(f"Message: {request.message[:100]}..." if len(request.message) > 100 else f"Message: {request.message}")
    
    try:
        # Retrieve user profile
        logger.info("Retrieving user profile")
        user_profile = await user_profiles_collection.find_one({"user_id": request.user_id})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Create the pension prompt
        logger.info("Creating pension prompt")
        prompt = create_pension_prompt(request.message, user_profile)
        logger.info("Pension prompt created successfully")
        
        # Call Lyzr API
        logger.info("Calling Lyzr API for pension planning")
        api_response = await call_lyzr_api(
            agent_id=PENSION_AGENT_ID,
            session_id=request.session_id,
            user_id=request.user_id,
            message=prompt
        )
        logger.info("Lyzr API call completed successfully")
        
        # Parse the structured response
        logger.info("Parsing structured response from API")
        text_response = ""
        chart_data_raw = None
        contains_chart = False

        try:
            # First attempt to parse the entire response as JSON
            structured_response = json.loads(api_response["response"])
            logger.info("Successfully parsed JSON response directly")
            text_response = structured_response.get("text_response", "")
            chart_data_raw = structured_response.get("chart_data")
            contains_chart = structured_response.get("contains_chart", False)
        except json.JSONDecodeError as json_error:
            logger.warning(f"Initial JSON parse failed: {str(json_error)}, trying to extract from markdown")
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', api_response["response"])
            if json_match:
                logger.info("Found JSON in markdown code block")
                try:
                    structured_response = json.loads(json_match.group(1).strip())
                    logger.info("Successfully parsed JSON from markdown")
                    
                    # FIXED: Always try to get text_response from the structured JSON first
                    text_response = structured_response.get("text_response", "")
                    
                    # Only if text_response is empty in JSON, then look for text outside JSON blocks
                    if not text_response:
                        # Get text before the JSON block
                        text_before_json = api_response["response"].split('```json')[0].strip()
                        
                        if text_before_json:
                            text_response = text_before_json
                        else:
                            # Check if there's text after the JSON block
                            parts = api_response["response"].split('```')
                            if len(parts) > 2:
                                text_after_json = parts[2].strip()
                                if text_after_json:
                                    text_response = text_after_json
                    
                    chart_data_raw = structured_response.get("chart_data")
                    contains_chart = structured_response.get("contains_chart", False)
                    
                except json.JSONDecodeError as markdown_json_error:
                    logger.error(f"Failed to parse JSON from markdown: {str(markdown_json_error)}")
                    text_response = api_response["response"].split('```json')[0].strip()
            else:
                logger.warning("No JSON found in markdown, using raw response")
                text_response = api_response["response"].strip()

        # If text_response is still empty, provide a default response
        if not text_response and not contains_chart:
            text_response = "I understand your question about pension planning. Let me help you with that."
            logger.info("Using default text response as parsed response was empty")

        logger.info(f"Parsed response - Text length: {len(text_response)}, Contains chart: {contains_chart}")
        
        chart_data = None
        if contains_chart and chart_data_raw:
            logger.info("Parsing chart data from response")
            try:
                # Parse chart_data_raw if it's a string
                if isinstance(chart_data_raw, str):
                    parsed_chart_data = json.loads(chart_data_raw)
                else:
                    parsed_chart_data = chart_data_raw
                
                logger.info(f"Chart data parsed successfully, type: {type(parsed_chart_data)}")
                
                # Save AI-generated retirement data if it's in the expected format
                if isinstance(parsed_chart_data, dict) and parsed_chart_data.get("type") == "pension_data":
                    ai_retirement_data = parsed_chart_data.get("data", [])
                    if ai_retirement_data:
                        logger.info("Saving AI-generated retirement data")
                        await save_ai_retirement_data(request.user_id, ai_retirement_data)
                elif isinstance(parsed_chart_data, list):
                    # If it's a list, assume it's retirement data
                    logger.info("Saving AI-generated retirement data (list format)")
                    await save_ai_retirement_data(request.user_id, parsed_chart_data)
                
                # Handle different chart data formats for response
                if isinstance(parsed_chart_data, list):
                    # If it's a list, wrap it in a dictionary structure
                    chart_data = {
                        "data": parsed_chart_data,
                        "type": "line_chart",
                        "title": "Retirement Income Projection"
                    }
                    logger.info(f"Converted list chart data to dict format with {len(parsed_chart_data)} records")
                elif isinstance(parsed_chart_data, dict):
                    # If it's already a dict, use it directly
                    chart_data = parsed_chart_data
                    logger.info("Using chart data as dict directly")
                else:
                    logger.warning(f"Unexpected chart data type: {type(parsed_chart_data)}")
                    chart_data = None
                    contains_chart = False
                    
            except json.JSONDecodeError as chart_error:
                logger.error(f"Failed to parse chart data: {str(chart_error)}")
                chart_data = None
                contains_chart = False
            except Exception as chart_parse_error:
                logger.error(f"Unexpected error parsing chart data: {str(chart_parse_error)}")
                chart_data = None
                contains_chart = False
        
        # Save message to database - save the raw parsed data for database
        logger.info("Saving message to database")
        database_chart_data = None
        if contains_chart and chart_data_raw:
            try:
                # For database, save the raw data (could be list or dict)
                if isinstance(chart_data_raw, str):
                    database_chart_data = json.loads(chart_data_raw)
                else:
                    database_chart_data = chart_data_raw
            except Exception as db_chart_error:
                logger.error(f"Error preparing chart data for database: {str(db_chart_error)}")
                database_chart_data = None
        
        await save_message(
            session_id=request.session_id,
            user_message=request.message,
            ai_response=text_response,
            chart_data=database_chart_data,
            contains_chart=contains_chart
        )
        logger.info("Message saved successfully")
        
        # Create response with properly formatted chart_data for API
        response = ChatResponse(
            response=text_response,
            session_id=request.session_id,
            chart_data=chart_data,
            contains_chart=contains_chart
        )

        logger.info("Pension chat request completed successfully")
        return response
    except HTTPException as http_error:
        logger.error(f"HTTP exception in pension chat: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in pension chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_comparison_prompt(data: dict) -> str:
    """Creates the final AI prompt based on structured user and AI data"""
    try:
        # Ensure the data can be serialized to JSON
        json_data = json.dumps(data, indent=2, default=str)
        return f"""
        {json_data}
        """.strip()
    except Exception as e:
        logger.error(f"Error generating comparison prompt: {str(e)}")
        return f"Analyze the retirement planning data for user and provide comprehensive insights."

@app.post("/compare-retirement", response_model=ChatCompResponse)
async def compare_retirement_projection(request: LyzrChatRequest):
    """Compare user-calculated vs AI-generated retirement data using Lyzr"""
    logger.info(f"Comparison request from user: {request.user_id}")

    try:
        # Retrieve user profile
        logger.info("Retrieving user profile")
        user_profile = await user_profiles_collection.find_one({"user_id": request.user_id})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Convert ObjectId to string and datetime to ISO format string for JSON serialization
        user_profile = convert_object_ids(user_profile)
        user_profile = convert_datetimes(user_profile)
        
        logger.debug(f"User profile retrieved for user: {user_profile.get('name', 'Unknown')}")

        # Extract retirement data
        logger.info("Extracting retirement data")
        retirement_data = user_profile.get('retirement_data', [])
        ai_retirement_data = user_profile.get('ai_retirement_data', [])
        
        # Prepare data for Lyzr API
        prompt_data = {
            "user_id": request.user_id,
            "retirement_data": retirement_data,
            "ai_retirement_data": ai_retirement_data
        }
        
        # Serialize data to JSON for the message field
        try:
            message = json.dumps(prompt_data, indent=2, default=str)
            logger.debug(f"Serialized message length: {len(message)}")
        except Exception as e:
            logger.error(f"Error serializing retirement data: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to serialize retirement data")

        # Prepare payload for Lyzr
        lyzr_payload = {
            "user_id": request.user_id,
            "agent_id": "684be1140a55475675b60d0b",
            "session_id": request.user_id,  # Generate a new session ID if not provided
            "message": message
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": LYZR_API_KEY
        }

        logger.info(f"Making request to Lyzr API with agent_id: {lyzr_payload['agent_id']}")

        # Call Lyzr with proper error handling
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                response = await client.post(LYZR_BASE_URL, headers=headers, json=lyzr_payload)
                
                logger.info(f"Lyzr API response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Lyzr API error response: {response.text}")
                
                response.raise_for_status()
                lyzr_result = response.json()
                
                logger.info("Lyzr API call successful")
                
            except httpx.TimeoutException as e:
                logger.error(f"Lyzr API timeout: {str(e)}")
                raise HTTPException(status_code=408, detail="Request timeout to external API")
            except httpx.HTTPStatusError as e:
                logger.error(f"Lyzr API HTTP error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f"External API error: {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Lyzr API request error: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to connect to external API")
        
        raw_response = lyzr_result.get("response", "").strip()
        logger.info(f"Received response from Lyzr, length: {len(raw_response)}")

        # Try to extract structured JSON from markdown block
        import re
        structured_data = None
        
        # First try to find JSON in markdown code blocks
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', raw_response)
        if json_match:
            try:
                structured_data = json.loads(json_match.group(1).strip())
                logger.info("Successfully extracted structured data from markdown JSON block")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode structured JSON from markdown: {str(e)}")
        else:
            try:
                structured_data = json.loads(raw_response)
                logger.info("Successfully parsed entire response as JSON")
            except json.JSONDecodeError:
                logger.info("Response is not JSON format, treating as plain text")

        try:
            await save_message(
                session_id=lyzr_payload['session_id'],
                user_message="Compare retirement projections",
                ai_response=raw_response,
                chart_data=structured_data,
                contains_chart=bool(structured_data)
            )
            logger.info("Comparison interaction saved to database")
        except Exception as save_error:
            logger.warning(f"Failed to save message to database: {str(save_error)}")

        return ChatCompResponse(
            session_id=lyzr_payload['session_id'],
            structured_data=structured_data
        )
    except HTTPException as http_error:
        logger.error(f"HTTP exception in compare retirement: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compare retirement: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat-news", response_model=ChatResponse)
async def chat_retirement(request: ChatRequest):
    """Chat endpoint for retirement planning"""
    logger.info(f"Retirement chat request received - Session ID: {request.session_id}, User ID: {request.user_id}")
    logger.debug(f"Message: {request.message[:100]}..." if len(request.message) > 100 else f"Message: {request.message}")
    
    try:
        # Retrieve user profile
        logger.info("Retrieving user profile")
        user_profile = await user_profiles_collection.find_one({"user_id": request.user_id})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Create personalized prompt with user details
        logger.info("Creating personalized prompt with user details")
        prompt = f"""
        User Profile:
        - Name: {user_profile.get('name', 'User')}
        - Current Age: {user_profile.get('current_age', 30)}
        - Retirement Age: {user_profile.get('retirement_age', 65)}
        - Current Income: ${user_profile.get('income', 700):,}
        - Salary Growth Rate: {user_profile.get('salary_growth', 0.02)*100}%
        - Investment Return Rate: {user_profile.get('investment_return', 0.05)*100}%
        - Contribution Rate: {user_profile.get('contribution_rate', 0.1)*100}%
        - Inflation Rate: {user_profile.get('inflation', 0.02)*100}%
        - Beneficiary Included: {user_profile.get('beneficiary_included', False)}
        - Beneficiary Life Expectancy: {user_profile.get('beneficiary_life_expectancy', 'Not specified')}
        - Social Security Base: ${user_profile.get('social_security_base', 18000):,}
        - Pension Base: ${user_profile.get('pension_base', 800):,}
        - 401k Base: ${user_profile.get('four01k_base', 100):,}
        - Other Investments Base: ${user_profile.get('other_base', 400):,}
        - Defined Benefit Base: ${user_profile.get('defined_benefit_base', 14000):,}
        - Defined Benefit Yearly Increase: ${user_profile.get('defined_benefit_yearly_increase', 300):,}
        
        User Question: {request.message}
        
        Provide a detailed and personalized response based on the user's profile. Ensure the response is relevant to their retirement planning needs.
        """
        
        logger.info("Calling Lyzr API for retirement planning")
        async with httpx.AsyncClient() as client:
            try:
                payload = {
                    "user_id": request.user_id,
                    "agent_id": "684adedfaa4e01a01dd89a72",
                    "session_id": request.session_id,
                    "message": prompt
                }
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": LYZR_API_KEY
                }
                logger.info(f"Making HTTP request with payload: {payload}")
                
                response = await client.post(
                    LYZR_BASE_URL,
                    json=payload,
                    headers=headers,
                    timeout=600.0
                )
                
                logger.info(f"Received response with status code: {response.status_code}")
                response.raise_for_status()
                
                response_data = response.json()
                logger.info(f"Received response: {response_data}")
                
                response_text = response_data.get("response", "")
                logger.info(f"Retrieved response text, length: {len(response_text)}")
                
            except httpx.RequestError as e:
                logger.error(f"HTTP request error in Lyzr API call: {str(e)}")
                raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP status error in Lyzr API call: {e.response.status_code} - {str(e)}")
                raise HTTPException(status_code=e.response.status_code, detail=f"API error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in Lyzr API call: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info("Saving retirement message to database")
        await save_message(
            session_id=request.session_id,
            user_message=request.message,
            ai_response=response_text,
            chart_data=None,
            contains_chart=False
        )
        logger.info("Retirement message saved successfully")
        
        response = ChatResponse(
            response=response_text,
            session_id=request.session_id,
        )
        
        logger.info("Retirement chat request completed successfully")
        return response
    except HTTPException as http_error:
        logger.error(f"HTTP exception in retirement chat: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in retirement chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-news-stream")
async def chat_retirement_stream(request: ChatRequest):
    """Streaming chat endpoint for retirement planning"""
    logger.info(f"Retirement stream chat request received - Session ID: {request.session_id}, User ID: {request.user_id}")
    logger.debug(f"Message: {request.message[:100]}..." if len(request.message) > 100 else f"Message: {request.message}")
    
    try:
        # Retrieve user profile
        logger.info("Retrieving user profile")
        user_profile = await user_profiles_collection.find_one({"user_id": request.user_id})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Create personalized prompt with user details
        logger.info("Creating personalized prompt with user details")
        prompt = f"""
        User Profile:
        - Name: {user_profile.get('name', 'User')}
        - Current Age: {user_profile.get('current_age', 30)}
        - Retirement Age: {user_profile.get('retirement_age', 65)}
        - Current Income: ${user_profile.get('income', 700):,}
        - Salary Growth Rate: {user_profile.get('salary_growth', 0.02)*100}%
        - Investment Return Rate: {user_profile.get('investment_return', 0.05)*100}%
        - Contribution Rate: {user_profile.get('contribution_rate', 0.1)*100}%
        - Inflation Rate: {user_profile.get('inflation', 0.02)*100}%
        - Beneficiary Included: {user_profile.get('beneficiary_included', False)}
        - Beneficiary Life Expectancy: {user_profile.get('beneficiary_life_expectancy', 'Not specified')}
        - Social Security Base: ${user_profile.get('social_security_base', 18000):,}
        - Pension Base: ${user_profile.get('pension_base', 800):,}
        - 401k Base: ${user_profile.get('four01k_base', 100):,}
        - Other Investments Base: ${user_profile.get('other_base', 400):,}
        - Defined Benefit Base: ${user_profile.get('defined_benefit_base', 14000):,}
        - Defined Benefit Yearly Increase: ${user_profile.get('defined_benefit_yearly_increase', 300):,}
        
        User Question: {request.message}
        
        Provide a detailed and personalized response based on the user's profile. Ensure the response is relevant to their retirement planning needs.
        """
        
        async def stream_response():
            logger.info("Starting to stream response from Lyzr API")
            response_text = ""
            async for chunk in stream_lyzr_api(
                agent_id="684adedfaa4e01a01dd89a72",
                session_id=request.session_id,
                user_id=request.user_id,
                message=prompt
            ):
                response_text += chunk
                yield chunk
            
            logger.info(f"Streamed response collected, total length: {len(response_text)}")
            
            logger.info("Saving retirement stream message to database")
            await save_message(
                session_id=request.session_id,
                user_message=request.message,
                ai_response=response_text,
                chart_data=None,
                contains_chart=False
            )
            logger.info("Retirement stream message saved successfully")
        
        logger.info("Returning streaming response")
        return StreamingResponse(
            content=stream_response(),
            media_type="text/plain"
        )
    except HTTPException as http_error:
        logger.error(f"HTTP exception in retirement stream chat: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in retirement stream chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create a new session"""
    logger.info(f"Creating new session for user ID: {request.user_id}")
    
    try:
        session_id = str(uuid.uuid4())
        session_name = request.session_name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session_doc = {
            "session_id": session_id,
            "user_id": request.user_id,
            "session_name": session_name,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await sessions_collection.insert_one(session_doc)
        logger.info(f"Created new session: {session_id}")
        return SessionResponse(**session_doc)
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", response_model=List[SessionResponse])
async def get_all_sessions(user_id: Optional[str] = None):
    """Get all sessions, optionally filtered by user_id"""
    logger.info(f"Retrieving sessions, user_id filter: {user_id}")
    
    try:
        query = {}
        if user_id:
            query["user_id"] = user_id
            logger.debug(f"Applied user_id filter: {user_id}")
        
        cursor = sessions_collection.find(query).sort("updated_at", -1)
        sessions = []
        
        session_count = 0
        async for session in cursor:
            sessions.append(SessionResponse(**session))
            session_count += 1
        
        logger.info(f"Retrieved {session_count} sessions")
        return sessions
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}", response_model=List[MessageResponse])
async def get_session_messages(session_id: str):
    """Get all messages for a specific session"""
    logger.info(f"Retrieving messages for session ID: {session_id}")
    
    try:
        cursor = messages_collection.find({"session_id": session_id}).sort("timestamp", 1)
        messages = []
        
        message_count = 0
        async for message in cursor:
            # Convert chart_data from list to a suitable dictionary structure if necessary
            if isinstance(message['chart_data'], list):
                message['chart_data'] = {"type": "chart_data", "data": message['chart_data']} # Or another suitable conversion

            messages.append(MessageResponse(**message))
            message_count += 1
        
        logger.info(f"Retrieved {message_count} messages for session {session_id}")
        return messages
    except Exception as e:
        logger.error(f"Error retrieving messages for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages"""
    logger.info(f"Deleting session and messages for session ID: {session_id}")
    
    try:
        logger.info("Deleting messages for session")
        message_result = await messages_collection.delete_many({"session_id": session_id})
        logger.info(f"Deleted {message_result.deleted_count} messages")
        
        logger.info("Deleting session document")
        session_result = await sessions_collection.delete_one({"session_id": session_id})
        
        if session_result.deleted_count == 0:
            logger.warning(f"Session {session_id} not found for deletion")
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Session {session_id} deleted successfully")
        return {"message": "Session deleted successfully"}
    except HTTPException as http_error:
        logger.error(f"HTTP exception in delete session: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user-profile", response_model=UserProfileResponse)
async def create_user_profile(request: UserProfileCreate):
    """Create a new user profile, ensuring no duplicate email addresses"""
    logger.info(f"Creating user profile for: {request.name}, email: {request.email}")
    
    try:
        if request.email:
            existing_user = await user_profiles_collection.find_one({"email": request.email})
            if existing_user:
                logger.warning(f"Attempted to create duplicate user profile with email: {request.email}")
                raise HTTPException(status_code=400, detail="User profile with this email already exists")
        
        user_id = str(uuid.uuid4())
        
        user_doc = {
            "user_id": user_id,
            "name": request.name,
            "email": request.email,
            "current_age": request.current_age,
            "retirement_age": request.retirement_age,
            "income": request.income,
            "salary_growth": request.salary_growth,
            "investment_return": request.investment_return,
            "contribution_rate": request.contribution_rate,
            "pension_multiplier": request.pension_multiplier,
            "end_age": request.end_age,
            "social_security_base": request.social_security_base,
            "pension_base": request.pension_base,
            "four01k_base": request.four01k_base,
            "other_base": request.other_base,
            "defined_benefit_base": request.defined_benefit_base,
            "defined_benefit_yearly_increase": request.defined_benefit_yearly_increase,
            "inflation": request.inflation,
            "beneficiary_included": request.beneficiary_included,
            "beneficiary_life_expectancy": request.beneficiary_life_expectancy,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "retirement_data": [],
            "ai_retirement_data": None
        }
        
        result = await user_profiles_collection.insert_one(user_doc)
        logger.info(f"User profile created with ID: {user_id}")

        await calculate_retirement_data(user_id)
        
        return UserProfileResponse(**user_doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.put("/user-profile/{user_id}", response_model=UserProfileResponse)
async def update_user_profile(user_id: str, request: UserProfileUpdate):
    """Update user profile"""
    logger.info(f"Updating user profile for ID: {user_id}")
    
    try:
        # Create an update document with only the fields provided
        update_data = {k: v for k, v in request.dict(exclude_unset=True).items()}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")
        
        update_data["updated_at"] = datetime.utcnow()

        result = await user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        updated_doc = await user_profiles_collection.find_one({"user_id": user_id})

        return UserProfileResponse(**convert_object_ids(updated_doc))
    except HTTPException as http_error:
        logger.error(f"HTTP exception: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/ai-retirement-data/{user_id}")
async def get_ai_retirement_data(user_id: str):
    """Get AI-generated retirement data for a user"""
    logger.info(f"Retrieving AI retirement data for user: {user_id}")
    
    try:
        user_doc = await user_profiles_collection.find_one({"user_id": user_id})
        
        if not user_doc:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        ai_retirement_data = user_doc.get('ai_retirement_data', [])
        manual_retirement_data = user_doc.get('retirement_data', [])
        
        return {
            "user_id": user_id,
            "ai_retirement_data": ai_retirement_data,
            "manual_retirement_data": manual_retirement_data,
            "has_ai_data": bool(ai_retirement_data),
            "latest_data_source": "ai" if ai_retirement_data else "manual"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving AI retirement data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-profile/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(user_id: str):
    """Get user profile by ID"""
    logger.info(f"Retrieving user profile for ID: {user_id}")
    
    try:
        user_doc = await user_profiles_collection.find_one({"user_id": user_id})
        
        if not user_doc:
            raise HTTPException(status_code=404, detail="User profile not found")

        # Convert ObjectId in the document to String
        sanitized_user_doc = convert_object_ids(user_doc)
        sanitized_user_doc = convert_datetimes(sanitized_user_doc)
        
        return JSONResponse(content=sanitized_user_doc)
    except HTTPException as http_error:
        logger.error(f"HTTP exception: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Error retrieving user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
    
@app.get("/retirement-calculation/{user_id}")
async def calculate_retirement_data(user_id: str):
    """Calculate and store retirement data"""
    try:
        user_doc = await user_profiles_collection.find_one({"user_id": user_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        retirement_data = generate_retirement_data(user_doc)
        
        update_result = await user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "retirement_data": retirement_data,
                "updated_at": datetime.utcnow()
            }}
        )
        
        if update_result.modified_count == 0:
            logger.warning(f"Failed to update retirement data for user: {user_id}")
        
        return {
            "user_id": user_id,
            "retirement_data": retirement_data
        }
    except Exception as e:
        logger.error(f"Retirement calculation error: {str(e)}")
        raise

@app.post("/user-sessions/{user_id}")
async def create_user_session(user_id: str, session_name: Optional[str] = None):
    """Create a new session for a specific user"""
    logger.info(f"Creating session for user ID: {user_id}")
    
    try:
        user_doc = await user_profiles_collection.find_one({"user_id": user_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        session_request = SessionCreate(
            user_id=user_id,
            session_name=session_name
        )
        
        return await create_session(session_request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai-preferences", response_model=AIPreferencesResponse)
async def get_ai_preferences(request: AIPreferencesRequest):
    """Get personalized AI suggestions for a user"""
    logger.info(f"Getting AI preferences for user: {request.user_id}, session: {request.session_id}")
    
    try:
        user_doc = await user_profiles_collection.find_one({"user_id": request.user_id})
        
        if not user_doc:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        suggestions = await generate_personalized_suggestions(
            request.user_id, 
            request.session_id, 
            user_doc
        )
        
        return AIPreferencesResponse(suggestions=suggestions)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users", response_model=List[UserInfo])
async def get_all_users():
    """Get all users with their IDs, names, and emails"""
    logger.info("Retrieving all users")
    
    try:
        cursor = user_profiles_collection.find({}, {"user_id": 1, "name": 1, "email": 1, "_id": 0})
        users = []
        
        user_count = 0
        async for user in cursor:
            users.append(UserInfo(**user))
            user_count += 1
        
        logger.info(f"Retrieved {user_count} users")
        return users
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    
    try:
        logger.info("Testing database connection")
        await db.command("ping")
        logger.info("Database connection successful")
        
        result = {"status": "healthy", "database": "connected"}
        logger.info("Health check completed successfully")
        return result
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        result = {"status": "unhealthy", "database": "disconnected", "error": str(e)}
        return result

if __name__ == "__main__":
    logger.info("Starting application server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=800) 