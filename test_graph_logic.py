#!/usr/bin/env python3
"""
Test script to verify graph keyword detection and naive user detection logic
"""

import re
from typing import Optional

def check_for_graph_keywords(message: str) -> bool:
    """Check if user message contains graph/chart related keywords"""
    import re
    
    # Keywords that indicate user wants to see graphs/charts
    graph_keywords = [
        r'\bgraph\b', r'\bchart\b', r'\bplot\b', r'\bvisualization\b', r'\bvisual\b',
        r'\bshow\s+me\b', r'\bdisplay\b', r'\bsee\b', r'\bview\b', r'\bpresent\b',
        r'\bdiagram\b', r'\bfigure\b', r'\bimage\b', r'\bpicture\b', r'\bdraw\b',
        r'\bcreate\s+(a\s+)?(graph|chart|plot)\b', r'\bgenerate\s+(a\s+)?(graph|chart|plot)\b',
        r'\bretirement\s+(graph|chart|plot)\b', r'\bincome\s+(graph|chart|plot)\b',
        r'\bprojection\s+(graph|chart|plot)\b', r'\bforecast\s+(graph|chart|plot)\b',
        r'\bcan\s+you\s+(show|display|create|generate)\b', r'\bwould\s+you\s+(show|display|create|generate)\b',
        r'\bplease\s+(show|display|create|generate)\b', r'\bI\s+want\s+to\s+see\b',
        r'\bI\s+would\s+like\s+to\s+see\b', r'\bcan\s+I\s+see\b', r'\bcould\s+I\s+see\b',
        r'\bretirement\s+projection\b', r'\bincome\s+projection\b', r'\bfinancial\s+projection\b',
        r'\bbreakdown\b', r'\banalysis\b', r'\bcomparison\b', r'\boverview\b', r'\bsummary\b',
        r'\bdata\s+visualization\b', r'\bdata\s+chart\b', r'\bdata\s+graph\b'
    ]
    
    message_lower = message.lower()
    
    for pattern in graph_keywords:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return True
    
    return False

def is_financially_naive_user(email: Optional[str]) -> bool:
    """Check if user is financially naive based on email"""
    if not email:
        return False
    
    email_lower = email.lower()
    naive_indicators = [
        'naive', 'naiveuser', 'beginner', 'newbie', 'novice', 
        'plainlanguage', 'simple', 'basic', 'starter', 'learning',
        'student', 'firsttime', 'newcomer', 'amateur'
    ]
    
    for indicator in naive_indicators:
        if indicator in email_lower:
            return True
    
    return False

def test_graph_keywords():
    """Test graph keyword detection"""
    print("=== Testing Graph Keyword Detection ===")
    
    test_messages = [
        "Can you show me a graph of my retirement income?",
        "I want to see a chart of my projections",
        "Please create a plot of my financial data",
        "What is my retirement age?",
        "How much should I save?",
        "Can you display a breakdown of my income sources?",
        "I would like to see a comparison of different scenarios",
        "Show me the data visualization",
        "What are the best investment options?",
        "Can you generate a retirement projection chart?",
        "I need help understanding my pension",
        "Please present an overview of my financial situation"
    ]
    
    for message in test_messages:
        has_graph = check_for_graph_keywords(message)
        print(f"'{message}' -> Graph keywords: {has_graph}")

def test_naive_user_detection():
    """Test naive user detection"""
    print("\n=== Testing Naive User Detection ===")
    
    test_emails = [
        "naiveuser@gmail.com",
        "beginner@example.com",
        "financepro@company.com",
        "student@university.edu",
        "plainlanguage@test.com",
        "expert@financial.com",
        "newcomer@startup.com",
        "professional@bank.com",
        "firsttime@user.com",
        "amateur@learning.com"
    ]
    
    for email in test_emails:
        is_naive = is_financially_naive_user(email)
        print(f"'{email}' -> Naive user: {is_naive}")

def test_combined_logic():
    """Test the combined logic for chart API calls"""
    print("\n=== Testing Combined Logic ===")
    
    test_cases = [
        {
            "email": "naiveuser@gmail.com",
            "message": "What is my retirement age?",
            "expected": "Skip chart API (naive user, no graph keywords)"
        },
        {
            "email": "naiveuser@gmail.com",
            "message": "Can you show me a graph of my retirement income?",
            "expected": "Call chart API (naive user, has graph keywords)"
        },
        {
            "email": "financepro@company.com",
            "message": "What is my retirement age?",
            "expected": "Call chart API (literate user, always call)"
        },
        {
            "email": "financepro@company.com",
            "message": "Can you show me a graph of my retirement income?",
            "expected": "Call chart API (literate user, always call)"
        }
    ]
    
    for case in test_cases:
        email = case["email"]
        message = case["message"]
        expected = case["expected"]
        
        is_naive = is_financially_naive_user(email)
        has_graph = check_for_graph_keywords(message)
        
        if is_naive:
            should_call = has_graph
            result = "Call chart API" if should_call else "Skip chart API"
        else:
            should_call = True
            result = "Call chart API"
        
        print(f"Email: {email}")
        print(f"Message: {message}")
        print(f"Naive: {is_naive}, Has graph keywords: {has_graph}")
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        print("---")

if __name__ == "__main__":
    test_graph_keywords()
    test_naive_user_detection()
    test_combined_logic() 