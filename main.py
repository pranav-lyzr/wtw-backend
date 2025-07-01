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
import asyncio

# Configure logging
# logging.disable(logging.CRITICAL)
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
    country: Optional[str] = "USA"
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
    country: Optional[str] = "USA"
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
    country: Optional[str] = None
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
    country: Optional[str] = "USA"
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
LYZR_API_KEY = "sk-default-gx2ux63PhW8Fmpt7gS5uPgff7BlEl81L"
LYZR_BASE_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
PENSION_AGENT_ID = "6850133251da0258f4744a30"
RETIREMENT_AGENT_ID = "6846d27f62d8a0cca7618607"
USER_ID = "workspace1@wtw.com"
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
                "user_id": USER_ID,
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
                timeout=6000.0
            )
            
            logger.info(f"Received response with status code: {response.status_code}")
            # Log the raw response immediately
            raw_response = response.text[:2000] + "..." if len(response.text) > 2000 else response.text
            logger.info(f"Raw Lyzr API response: {raw_response}")
            
            response.raise_for_status()
            
            try:
                response_data = response.json()
                # Log the parsed response
                response_log = json.dumps(response_data, indent=2)[:2000] + "..." if len(json.dumps(response_data)) > 2000 else json.dumps(response_data)
                logger.info(f"Received response: {response_log}")
                logger.info(f"Response data keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
                return response_data
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse JSON response: {str(json_error)}")
                logger.error(f"Raw response content: {raw_response}")
                raise HTTPException(status_code=500, detail=f"Invalid JSON response from API: {str(json_error)}")
            
        except httpx.RequestError as e:
            logger.error(f"HTTP request error in Lyzr API call: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error in Lyzr API call: {e.response.status_code} - {str(e)}")
            error_response = e.response.text[:2000] + "..." if len(e.response.text) > 2000 else e.response.text
            logger.error(f"Error response content: {error_response}")
            raise HTTPException(status_code=e.response.status_code, detail=f"API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Lyzr API call: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

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
    
    manual_data = user_profile.get('retirement_data', [])
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

    prompt = append_persona_instructions(
        user_profile.get('email'),
        prompt
    )
    
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

# def generate_retirement_data(user_profile: dict):
#     """Generate retirement data matching prompt formulas"""
#     years = list(range(user_profile['retirement_age'], user_profile['end_age'] + 1))
#     retirement_data = []
#     max_ss_benefit = 58476.0  

#     for age in years:
#         try:
#             # Social Security calculation
#             ss_value = 0
#             if age >= 62:
#                 ss_value = user_profile['social_security_base'] * (1 + user_profile['inflation']) ** (age - 62)
#                 if math.isinf(ss_value) or math.isnan(ss_value):
#                     ss_value = max_ss_benefit
#                 ss_value = min(ss_value, max_ss_benefit)

#             # Pension calculation
#             pension_value = user_profile['pension_base'] * \
#                         (1 + user_profile['salary_growth']) ** (age - user_profile['retirement_age'])

#             # 401k calculation
#             four01k_value = 0
#             if age >= 58:
#                 if age <= 65:
#                     four01k_value = user_profile['four01k_base'] * \
#                                 (1 + user_profile['investment_return']) ** (age - 58)
#                 else:
#                     base_at_65 = user_profile['four01k_base'] * \
#                                 (1 + user_profile['investment_return']) ** (65 - 58)
#                     four01k_value = base_at_65 * (0.95) ** (age - 65)

#             # Other calculation
#             other_value = 0
#             if age >= 58:
#                 other_value = user_profile['other_base'] * \
#                             (1 + user_profile['investment_return']) ** (age - 58)

#             # Defined Benefit calculation
#             defined_benefit_value = 0
#             if age >= 62:
#                 defined_benefit_value = user_profile['defined_benefit_base'] + \
#                                     user_profile['defined_benefit_yearly_increase'] * (age - 62)

#             retirement_data.append({
#                 "age": age,
#                 "Social Security": round(ss_value, 2),
#                 "Pension": round(pension_value, 2),
#                 "401k": round(four01k_value, 2),
#                 "Other": round(other_value, 2),
#                 "Defined Benefit": round(defined_benefit_value, 2)
#             })
#         except (OverflowError, ValueError) as e:
#             logger.error(f"Error in generating retirement data: {e}")
#             continue

#     return retirement_data


def normalize_rate(rate: float) -> float:
    """Normalize a rate to decimal form (e.g., 5 -> 0.05, 0.05 -> 0.05)"""
    return rate / 100 if rate >= 1 else rate

def generate_retirement_data(user_profile: dict):
    """Generate retirement data matching prompt formulas"""
    years = list(range(user_profile['retirement_age'], user_profile['end_age'] + 1))
    retirement_data = []
    max_ss_benefit = 58476.0  

    # Normalize rates
    inflation = normalize_rate(user_profile.get('inflation', 0.02))
    salary_growth = normalize_rate(user_profile.get('salary_growth', 0.02))
    investment_return = normalize_rate(user_profile.get('investment_return', 0.05))

    for age in years:
        try:
            # Social Security calculation
            ss_value = 0
            if age >= 62:
                ss_value = user_profile['social_security_base'] * (1 + inflation) ** (age - 62)
                if math.isinf(ss_value) or math.isnan(ss_value):
                    ss_value = max_ss_benefit
                ss_value = min(ss_value, max_ss_benefit)

            # Pension calculation
            pension_value = user_profile['pension_base'] * \
                        (1 + salary_growth) ** (age - user_profile['retirement_age'])

            # 401k calculation
            four01k_value = 0
            if age >= 58:
                if age <= 65:
                    four01k_value = user_profile['four01k_base'] * \
                                (1 + investment_return) ** (age - 58)
                else:
                    base_at_65 = user_profile['four01k_base'] * \
                                (1 + investment_return) ** (65 - 58)
                    four01k_value = base_at_65 * (0.95) ** (age - 65)

            # Other calculation
            other_value = 0
            if age >= 58:
                other_value = user_profile['other_base'] * \
                            (1 + investment_return) ** (age - 58)

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

def generate_retirement_data_denmark(user_profile: dict):
    """Generate Denmark-style retirement data including Firmapensionsordning"""
    years = list(range(user_profile['retirement_age'], user_profile['end_age'] + 1))
    retirement_data = []
    
    # Folkepension: 2024 base values (DKK)
    folkepension_base = 81000 + 90000  # Basic amount + supplement
    atp_base = 25000  # Average ATP payout (DKK)
    currency_factor = 1  # Use 1 for DKK, convert if needed

    # Normalize rates
    inflation = normalize_rate(user_profile.get('inflation', 0.02))
    salary_growth = normalize_rate(user_profile.get('salary_growth', 0.02))
    investment_return = normalize_rate(user_profile.get('investment_return', 0.05))
    contribution_rate = normalize_rate(user_profile.get('contribution_rate', 0.1))
    firmapension_contribution_rate = normalize_rate(user_profile.get('firmapension_contribution_rate', 0.05))

    # Step 1: Adjust Folkepension base to value at age 67
    years_until_67 = max(0, 67 - user_profile['current_age'])
    retirement_year_adjusted_folkepension = folkepension_base * (1 + inflation) ** years_until_67

    for age in years:
        try:
            # Folkepension: apply inflation from 2024 to 67, then grow yearly from 67
            folkepension = 0
            if age >= 67:
                folkepension = retirement_year_adjusted_folkepension * (1 + inflation) ** (age - 67)

            # ATP: Adjust from retirement year onwards
            atp_value = atp_base * (1 + inflation) ** (age - user_profile['retirement_age'])

            # Occupational Pension
            occ_pension = contribution_rate * user_profile['income'] * \
                          ((1 + salary_growth) ** (age - user_profile['current_age'])) * \
                          (1 + investment_return) ** (age - user_profile['retirement_age'])

            # Firmapension (employer-sponsored)
            firmapension_value = firmapension_contribution_rate * user_profile['income'] * \
                                 ((1 + salary_growth) ** (age - user_profile['current_age'])) * \
                                 (1 + investment_return) ** (age - user_profile['retirement_age'])

            # Private Pension (similar to 'Other')
            private_value = user_profile['other_base'] * (1 + investment_return) ** (age - user_profile['retirement_age'])

            retirement_data.append({
                "age": age,
                "Folkepension": round(folkepension * currency_factor, 2),
                "ATP": round(atp_value * currency_factor, 2),
                "Occupational Pension": round(occ_pension * currency_factor, 2),
                "Firmapension": round(firmapension_value * currency_factor, 2),
                "Private Pension": round(private_value * currency_factor, 2),
            })

        except (OverflowError, ValueError) as e:
            logger.error(f"Error in Denmark retirement data generation: {e}")
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

def append_persona_instructions(email: Optional[str], prompt: str) -> str:
    """Append persona-specific instructions to the prompt based on user email"""
    logger.info(f"Appending persona instructions for email: {email}")
    
    if email == "smallresponse@gmail.com":
        persona_instruction = """
        Provide a concise response in 2-3 sentences, keeping the answer brief and to the point.
        """
    elif email == "naiveuser@gmail.com":
        persona_instruction = """
        Provide a detailed response and explain all technical terms in simple language, assuming the user has no prior financial knowledge.
        """
    elif email == "shortbulletpoints@gmail.com":
        persona_instruction = """
        Provide the response in 3-5 concise bullet points, keeping each point short and clear.
        """
    elif "briefbullets" in email:
        persona_instruction = """
        **Response Style: Brief / Bullets**
        - Provide a concise response in 3-5 bullet points.
        - Use short, clear sentences.
        - Focus on key information, avoiding unnecessary details.
        - Example:
          - Main point 1
          - Main point 2
          - Main point 3
        """
    elif "plainlanguage" in email:
        persona_instruction = """
        **Response Style: Plain-language (Non-finance)**
        - Use simple, everyday words that anyone can understand.
        - Provide a concise response in 3-5 bullet points.
        - Avoid financial jargon or complex terms.
        - Explain concepts as if speaking to someone with no financial background. Explain financial works when required.
        - Keep the response friendly and approachable.
        - Example: Instead of "asset allocation," say "how your money is split between different investments."
        """
    elif "financepro" in email:
        persona_instruction = """
        **Response Style: Technical (Finance Pro)**
        - Provide a concise response in 3-5 bullet points.
        - Use precise financial terminology and industry-specific language.
        - Include detailed calculations or methodologies where relevant.
        - Assume the user has advanced knowledge of finance and retirement planning.
        - Provide in-depth analysis suitable for a financial professional.
        - Example: Reference terms like "net present value," "compound annual growth rate," or "actuarial assumptions."
        """
    else:
        persona_instruction = ""
    
    if persona_instruction:
        logger.debug(f"Applying persona instruction: {persona_instruction.strip()}")
        prompt += f"\n\n**Persona-Specific Instructions:**\n{persona_instruction}"
    
    return prompt

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
            agent_id="68519f1ee8762a5908ab2930",
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

def create_master_prompt(user_message: str, user_profile: dict):
    """Create master prompt for the parent agent"""
    logger.info("Creating master prompt with user profile data")
    
    # Prepare form data for prompt
    form_data = {
        "name": user_profile.get('name', 'User'),
        "current_age": user_profile.get('current_age', 30),
        "retirement_age": user_profile.get('retirement_age', 65),
        "life_expectancy": user_profile.get('end_age', 85),
        "income": user_profile.get('income', 70000),
        "salary_growth": user_profile.get('salary_growth', 0.02),
        "investment_return": user_profile.get('investment_return', 0.05),
        "inflation": user_profile.get('inflation', 0.02),
        "beneficiary_included": user_profile.get('beneficiary_included', False),
        "beneficiary_life_expectancy": user_profile.get('beneficiary_life_expectancy', 'Not specified'),
    }
    
    # Get existing pension data from profile
    latest_retirement_data = get_latest_retirement_data(user_profile)
    
    prompt = f"""
You are a Parent Agent for retirement planning. Analyze the user query and provide appropriate response coordination.

**User Profile:**
- Form data: {json.dumps(form_data)}
- Pension data: {json.dumps(latest_retirement_data)}

**User Question:** {user_message}

**Response Requirements:**
Respond with ONLY this JSON structure:
{{
  "text_response": "Your detailed answer here with proper disclaimers and source citations if needed",
  "chart_data": null or {{"type": "pension_data", "data": [{{"age": number, "Social Security": number, "Pension": number, "401k": number, "Other": number, "Defined Benefit": number}}, ...]}},
  "contains_chart": true or false
}}

**Instructions:**
1. Analyze if query needs visualization (income projections, comparisons) → set contains_chart to true
2. For WTW Advisor responses: Include banner "⚠️ *Educational only – not WTW advice*" and cite sources
3. For News/Market queries: Provide complete chart data when applicable
4. Personalize response based on user profile (age: {form_data['current_age']}, retirement age: {form_data['retirement_age']})
5. Use pension calculations from provided data for accuracy
6. Keep response clear, concise, and professionally formatted
"""
    
    logger.info(f"Master prompt created, total length: {len(prompt)}")
    return prompt


# API Routes

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Retirement Planning API"}


# Modified /chat endpoint to handle Denmark-specific Lyzr API call
@app.post("/chat", response_model=ChatResponse)
async def chat_retirement_unified(request: ChatRequest):
    """Unified chat endpoint for retirement planning - routes to appropriate specialized agents"""
    logger.info(f"Unified retirement chat request - Session ID: {request.session_id}, User ID: {request.user_id}")
    logger.debug(f"Message: {request.message[:100]}..." if len(request.message) > 100 else f"Message: {request.message}")
    
    try:
        # Retrieve user profile with proper error handling
        logger.info("Retrieving user profile")
        user_profile = await user_profiles_collection.find_one({"user_id": request.user_id})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Ensure user_profile is a dict, not a list
        if isinstance(user_profile, list):
            if len(user_profile) > 0:
                user_profile = user_profile[0]
            else:
                raise HTTPException(status_code=404, detail="User profile is empty")
        
        # Safely extract profile data with defaults
        country = user_profile.get('country', 'USA').lower() if isinstance(user_profile, dict) else 'usa'
        
        # Prepare form data for context with safe extraction
        form_data = {
            "retireAge": user_profile.get('retirement_age', 65),
            "lifeExpectancy": user_profile.get('end_age', 85),
            "beneficiaryIncluded": user_profile.get('beneficiary_included', False),
            "beneficiaryLifeExpectancy": user_profile.get('beneficiary_life_expectancy', "Not specified"),
            "salaryGrowth": user_profile.get('salary_growth', 0.02),
            "investmentReturn": user_profile.get('investment_return', 0.05),
            "inflation": user_profile.get('inflation', 0.02),
        }
        
        latest_retirement_data = user_profile.get('retirement_data', [])
        
        logger.info("Retrieving recent chat history")
        recent_messages = []
        cursor = messages_collection.find({"session_id": request.session_id}).sort("timestamp", -1).limit(3)
        async for message in cursor:
            recent_messages.append({
                "user_message": message['user_message'],
                "ai_response": message['ai_response'],
                "timestamp": message['timestamp'].isoformat()
            })
        recent_messages.reverse()

        # Determine agent ID based on country
        agent_id = "6851461baf3ce50cc6e2e4b3"  # Default master agent for USA
        if country == 'denmark':
            agent_id = "685927b4a83b7ec9a60665f9"  # Denmark-specific agent
            logger.info(f"Using Denmark-specific agent ID: {agent_id}")
            
            # Create Denmark-specific master prompt
            master_prompt = f"""
            **USER PROFILE (Denmark):**
            - Name: {user_profile.get('name', 'User')}
            - Age: {user_profile.get('current_age', 30)} → Retirement: {user_profile.get('retirement_age', 65)}
            - Income: DKK {user_profile.get('income', 500000):,} | Growth: {normalize_rate(user_profile.get('salary_growth', 0.02))*100}%
            - Investment Return: {normalize_rate(user_profile.get('investment_return', 0.05))*100}% | Inflation: {normalize_rate(user_profile.get('inflation', 0.02))*100}%
            - Contribution Rate: {normalize_rate(user_profile.get('contribution_rate', 0.1))*100}%
            - Folkepension Base: DKK 171000
            - ATP Base: DKK {user_profile.get('pension_base', 25000):,}
            - Occupational Pension Base: DKK {user_profile.get('pension_base', 8000):,}
            - Firmapension Contribution Rate: {user_profile.get('firmapension_contribution_rate', 0.05)*100}% (if applicable)
            - Private Pension Base: DKK {user_profile.get('other_base', 4000):,}

            **CURRENT CONTEXT:**
            - Form Data: {json.dumps(form_data)}
            - Pension Graph Data: {json.dumps(latest_retirement_data)}
            - Recent Conversations: {json.dumps(recent_messages, default=str)}

            **USER QUESTION:** {request.message}

            Provide personalized response for this specific Danish user with exact calculations. Keep the response more analytical   
            """
        else:
            # Create USA-specific master prompt
            master_prompt = f"""
            **USER PROFILE:**
            - Name: {user_profile.get('name', 'User')}
            - Age: {user_profile.get('current_age', 30)} → Retirement: {user_profile.get('retirement_age', 65)}
            - Income: ${user_profile.get('income', 70000):,} | Growth: {normalize_rate(user_profile.get('salary_growth', 0.02))*100}%
            - Investment Return: {normalize_rate(user_profile.get('investment_return', 0.05))*100}% | Inflation: {normalize_rate(user_profile.get('inflation', 0.02))*100}%
            - Social Security: ${user_profile.get('social_security_base', 18000):,} | Pension: ${user_profile.get('pension_base', 800):,}
            - 401k: ${user_profile.get('four01k_base', 100):,} | Other: ${user_profile.get('other_base', 400):,}
            - Defined Benefit: ${user_profile.get('defined_benefit_base', 14000):,} (+${user_profile.get('defined_benefit_yearly_increase', 300):,}/year)

            **CURRENT CONTEXT:**
            - Form Data: {json.dumps(form_data)}
            - Pension Graph Data: {json.dumps(latest_retirement_data)}
            - Recent Conversations: {json.dumps(recent_messages, default=str)}

            **USER QUESTION:** {request.message}

            Use user data to get more personalized response for this USA client. Keep the response more analytical   
            """

        master_prompt = append_persona_instructions(user_profile.get('email'), master_prompt)
        
        logger.info(f"Calling Lyzr API with agent ID: {agent_id}") 
        api_response = await call_lyzr_api(
            agent_id=agent_id,
            session_id=request.session_id,
            user_id=request.user_id,
            message=master_prompt
        )
        logger.info("Lyzr API call completed successfully")
        logger.info(f"Received response: {api_response}")

        raw_api_response = api_response

        
        # Enhanced LLM response parsing - with better error handling and logging
        logger.info("Parsing LLM response with comprehensive format handling")
        text_response = ""
        chart_data_raw = None
        contains_chart = False

        def parse_llm_response(response_content):
            """Comprehensive LLM response parser handling all possible formats"""
            # Ensure we have valid response content
            if not response_content or not isinstance(response_content, str):
                logger.error("Invalid or empty response content from API")
                return {
                    'text_response': "Empty or invalid API response",
                    'chart_data_raw': None,
                    'contains_chart': False,
                    'parse_method': 'error'
                }
            
            logger.info(f"Raw response length: {len(response_content)}")
            logger.debug(f"Raw response preview: {response_content[:500]}...")
            
            try:
                # Strategy 1: Try direct JSON parsing
                try:
                    parsed = json.loads(response_content.strip())
                    logger.info("✓ Successfully parsed as direct JSON")
                    return handle_parsed_json(parsed, "direct_json")
                except json.JSONDecodeError:
                    pass  # Move to next strategy

                # Strategy 2: Extract JSON from markdown code blocks
                import re
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                matches = re.findall(json_pattern, response_content, re.IGNORECASE)
                if matches:
                    try:
                        parsed = json.loads(matches[0].strip())
                        logger.info("✓ Successfully parsed JSON from markdown code block")
                        return handle_parsed_json(parsed, "markdown_json")
                    except json.JSONDecodeError:
                        pass  # Move to next strategy

                # Strategy 3: Look for JSON-like structures
                json_like_pattern = r'\{[\s\S]*\}'
                matches = re.findall(json_like_pattern, response_content, re.DOTALL)
                if matches:
                    try:
                        parsed = json.loads(matches[0].strip())
                        logger.info("✓ Successfully parsed JSON-like structure")
                        return handle_parsed_json(parsed, "json_like")
                    except json.JSONDecodeError:
                        pass  # Move to next strategy

                # Strategy 4: Key-value extraction
                extracted_data = {}
                # Text response patterns
                text_patterns = [
                    r'"text_response"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'text_response\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'Response\s*:\s*(.+?)(?=\n\n|\n$|$)',
                    r'Answer\s*:\s*(.+?)(?=\n\n|\n$|$)',
                ]
                for pattern in text_patterns:
                    match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
                    if match:
                        extracted_data['text_response'] = match.group(1).strip()
                        break

                # Chart flag patterns
                chart_flag_patterns = [
                    r'"contains_chart"\s*:\s*(true|false)',
                    r'contains_chart\s*:\s*(true|false)',
                    r'Chart\s*:\s*(yes|no|true|false)',
                ]
                for pattern in chart_flag_patterns:
                    match = re.search(pattern, response_content, re.IGNORECASE)
                    if match:
                        flag_val = match.group(1).lower()
                        extracted_data['contains_chart'] = flag_val in ['true', 'yes']
                        break

                # Chart data patterns
                chart_data_patterns = [
                    r'"chart_data"\s*:\s*(\[[\s\S]*?\])',
                    r'chart_data\s*:\s*(\[[\s\S]*?\])',
                    r'"chart_data"\s*:\s*(\{[\s\S]*?\})',
                ]
                for pattern in chart_data_patterns:
                    match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            extracted_data['chart_data_raw'] = json.loads(match.group(1).strip())
                        except json.JSONDecodeError:
                            extracted_data['chart_data_raw'] = match.group(1).strip()
                        break

                if extracted_data:
                    logger.info("✓ Successfully extracted data using regex patterns")
                    return {
                        'text_response': extracted_data.get('text_response', ''),
                        'chart_data_raw': extracted_data.get('chart_data_raw'),
                        'contains_chart': extracted_data.get('contains_chart', False),
                        'parse_method': 'regex_extraction'
                    }

                # Strategy 5: Last resort - treat entire response as text
                logger.warning("All JSON parsing strategies failed, treating as plain text")
                return {
                    'text_response': response_content.strip(),
                    'chart_data_raw': None,
                    'contains_chart': False,
                    'parse_method': 'plain_text'
                }
                
            except Exception as e:
                logger.error(f"Error during parsing: {str(e)}")
                logger.info(f"Response content snippet: {response_content[:500]}...")
                return {
                    'text_response': response_content.strip(),
                    'chart_data_raw': None,
                    'contains_chart': False,
                    'parse_method': 'error_fallback'
                }

        def handle_parsed_json(parsed_data, parse_method):
            """Handle parsed JSON data regardless of structure"""
            logger.info(f"Handling parsed JSON data using method: {parse_method}")
            
            # Handle different data types
            if isinstance(parsed_data, list):
                if parsed_data:
                    # Try to extract the first dictionary if available
                    if isinstance(parsed_data[0], dict):
                        parsed_data = parsed_data[0]
                        logger.info("Converted list to first dict element")
                    else:
                        # Treat as chart data
                        return {
                            'text_response': '',
                            'chart_data_raw': parsed_data,
                            'contains_chart': True,
                            'parse_method': parse_method
                        }
                else:
                    logger.warning("Empty list in parsed data")
                    return {
                        'text_response': '',
                        'chart_data_raw': None,
                        'contains_chart': False,
                        'parse_method': parse_method
                    }
            
            if not isinstance(parsed_data, dict):
                logger.error(f"Unexpected parsed data type: {type(parsed_data)}")
                return {
                    'text_response': f"Unexpected response format: {type(parsed_data).__name__}",
                    'chart_data_raw': None,
                    'contains_chart': False,
                    'parse_method': parse_method
                }
            
            # Extract response fields
            text_response = parsed_data.get('text_response', '') or \
                            parsed_data.get('response', '') or \
                            parsed_data.get('message', '') or \
                            parsed_data.get('answer', '') or \
                            parsed_data.get('content', '')
            
            chart_data_raw = parsed_data.get('chart_data') or \
                             parsed_data.get('data') or \
                             parsed_data.get('chart')
            
            contains_chart = parsed_data.get('contains_chart', False) or \
                             parsed_data.get('has_chart', False) or \
                             parsed_data.get('include_chart', False)
            
            # Auto-detect chart if chart_data exists
            if chart_data_raw and not contains_chart:
                contains_chart = True
                logger.info("Auto-detected chart presence from chart_data")
            
            return {
                'text_response': str(text_response).strip() if text_response else '',
                'chart_data_raw': chart_data_raw,
                'contains_chart': contains_chart,
                'parse_method': parse_method
            }

        # Parse the response with detailed error handling
        logger.info("Parsing LLM response")
        if not isinstance(api_response, dict) or "response" not in api_response:
            logger.error(f"Invalid API response structure: {type(api_response)}")
            raise HTTPException(status_code=500, detail="Invalid API response structure")
        
        response_content = api_response["response"]
        if not response_content:
            logger.error("Empty response content from API")
            raise HTTPException(status_code=500, detail="Empty response from AI service")
        
        # Parse the response
        parsed_result = parse_llm_response(response_content)
        
        text_response = parsed_result['text_response']
        chart_data_raw = parsed_result['chart_data_raw']
        contains_chart = parsed_result['contains_chart']
        parse_method = parsed_result['parse_method']
        
        logger.info(f"Response parsed successfully using method: {parse_method}")
        logger.info(f"Text response length: {len(text_response)}, Contains chart: {contains_chart}")
        
        # Validate that we got meaningful content
        if not text_response and not contains_chart:
            logger.error("No meaningful content extracted from LLM response")
            logger.error(f"Full response content: {response_content[:1000]}...")
            raise HTTPException(status_code=500, detail=raw_api_response) 

                
     

        logger.info(f"Final parsed response - Text length: {len(text_response)}, Contains chart: {contains_chart}")
        
        # Handle chart data with better error handling
        chart_data = None
        if contains_chart and chart_data_raw:
            logger.info("Processing chart data from response")
            try:
                # Parse chart_data_raw if it's a string
                if isinstance(chart_data_raw, str):
                    parsed_chart_data = json.loads(chart_data_raw)
                else:
                    parsed_chart_data = chart_data_raw
                
                logger.info(f"Chart data parsed successfully, type: {type(parsed_chart_data)}")
                
                # Save AI-generated retirement data
                if isinstance(parsed_chart_data, dict) and parsed_chart_data.get("type") == "pension_data":
                    ai_retirement_data = parsed_chart_data.get("data", [])
                    if ai_retirement_data and isinstance(ai_retirement_data, list):
                        logger.info("Saving AI-generated retirement data")
                        await save_ai_retirement_data(request.user_id, ai_retirement_data)
                elif isinstance(parsed_chart_data, list):
                    logger.info("Saving AI-generated retirement data (list format)")
                    await save_ai_retirement_data(request.user_id, parsed_chart_data)
                
                # Format chart data for response
                if isinstance(parsed_chart_data, list):
                    chart_data = {
                        "data": parsed_chart_data,
                        "type": "line_chart",
                        "title": "Retirement Income Projection"
                    }
                    logger.info(f"Converted list chart data to dict format with {len(parsed_chart_data)} records")
                elif isinstance(parsed_chart_data, dict):
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
        
        # Save message to database with safe chart data handling
        logger.info("Saving message to database")
        database_chart_data = None
        if contains_chart and chart_data_raw:
            try:
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
        
        # Create final response
        response = {
            "response": text_response,
            "session_id": request.session_id,
            "chart_data": chart_data,
            "contains_chart": contains_chart,
            "raw_api_response": raw_api_response
        }

        logger.info("Unified retirement chat request completed successfully")
        return response
        
    except HTTPException as http_error:
        logger.error(f"HTTP exception in unified retirement chat: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in unified retirement chat: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")    

@app.post("/chat_pension", response_model=ChatResponse)
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
            "agent_id": "68519f01ed2a4cfbefdf1481",
            "session_id": request.user_id,  # Generate a new session ID if not provided
            "message": message
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": LYZR_API_KEY
        }

        logger.info(f"Making request to Lyzr API with agent_id: {lyzr_payload['agent_id']}")

        # Call Lyzr with proper error handling
        async with httpx.AsyncClient(timeout=600.0) as client:
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

        # In the /chat-news endpoint, replace the existing append_persona_instructions call
        prompt = append_persona_instructions(
            user_profile.get('email'),
            prompt
        )
        
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
        
        prompt = append_persona_instructions(
            user_profile.get('email'),
            prompt
        )
                
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
            "country": request.country if request.country else "USA",
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
    
@app.delete("/user-profile/{user_id}")
async def delete_user_profile(user_id: str):
    """Delete a user profile and all associated sessions and messages"""
    logger.info(f"Deleting user profile for ID: {user_id}")
    
    try:
        # Check if user exists
        user_doc = await user_profiles_collection.find_one({"user_id": user_id})
        if not user_doc:
            logger.warning(f"User profile not found for ID: {user_id}")
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Delete all sessions for the user
        logger.info(f"Deleting sessions for user ID: {user_id}")
        session_result = await sessions_collection.delete_many({"user_id": user_id})
        logger.info(f"Deleted {session_result.deleted_count} sessions")
        
        # Delete all messages for the user's sessions
        logger.info(f"Deleting messages for user ID: {user_id}")
        message_result = await messages_collection.delete_many({"session_id": {"$in": [
            session["session_id"] async for session in sessions_collection.find({"user_id": user_id}, {"session_id": 1})
        ]}})
        logger.info(f"Deleted {message_result.deleted_count} messages")
        
        # Delete the user profile
        logger.info(f"Deleting user profile for ID: {user_id}")
        user_result = await user_profiles_collection.delete_one({"user_id": user_id})
        
        if user_result.deleted_count == 0:
            logger.warning(f"Failed to delete user profile for ID: {user_id}")
            raise HTTPException(status_code=404, detail="User profile not found")
        
        logger.info(f"User profile {user_id} and associated data deleted successfully")
        return {"message": "User profile and associated data deleted successfully"}
    
    except HTTPException as http_error:
        logger.error(f"HTTP exception in delete user profile: {http_error.detail}")
        raise
    except Exception as e:
        logger.error(f"Error deleting user profile {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retirement-calculation/{user_id}")
async def calculate_retirement_data(user_id: str):
    """Calculate and store retirement data"""
    try:
        user_doc = await user_profiles_collection.find_one({"user_id": user_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Select retirement calculation based on country
        if user_doc.get('country', 'USA').lower() == 'denmark':
            logger.info(f"Generating Denmark retirement data for user: {user_id}")
            retirement_data = generate_retirement_data_denmark(user_doc)
        else:
            logger.info(f"Generating USA retirement data for user: {user_id}")
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
        raise HTTPException(status_code=400, detail=str(e))


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